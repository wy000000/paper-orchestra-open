# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict
from pydantic import BaseModel
from google.genai import types  # type: ignore

from methods.prompts.literature_review_agent import (
    literature_review_agent_writter_prompt,
)
from utils.gemini_utils import call_gemini_with_contents, parse_gemini_latex_results
from utils.scholar_utils import s2_title_search
from utils.academic_search import AcademicPaper, search_academic_databases
from utils import model_config


class CandidatePaper(BaseModel):
    title: str
    year: int
    reason: str


class DiscoveryResult(BaseModel):
    section_name: str
    candidates: List[CandidatePaper]


class RankedCandidate(BaseModel):
    id: int
    reason: str = ""


class CandidateRanking(BaseModel):
    candidates: List[RankedCandidate]


class PaperData(BaseModel):
    citation_key: str
    title: str
    authors: List[str]
    venue: str
    year: int
    abstract: str
    citation_count: int
    found_in_section: str
    reason: str
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    publication_date: Optional[str] = None


class WriterOutput(BaseModel):
    sections_written: List[str]
    latex_code: str


class HybridLiteratureAgent:
    def __init__(
        self,
        idea_path: str,
        experimental_log_path: str,
        latex_template_path: str,
        conference_guidelines_path: str,
        output_dir: str,
        model_name: str = "gemini-3.1-pro-preview",
        max_workers: int = 5,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model_name = model_name
        self.max_workers = max_workers
        self.context_paths = {
            "idea.md": idea_path,
            "experimental_log.md": experimental_log_path,
            "template.tex": latex_template_path,
            "conference_guidelines.md": conference_guidelines_path,
        }
        self.context_files = self._load_context_files()
        self.google_search_tool = types.Tool(google_search=types.GoogleSearch())
        self.discovery_config = model_config.get_literature_discovery_config()

        self.registry_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self._discovery_error_lock = threading.Lock()
        self._failed_discovery_sources = set()

    def _safe_print(self, message: str):
        """Thread-safe printing to prevent garbled output."""
        with self.print_lock:
            print(message)

    def run(
        self,
        outline_path: str,
        cutoff_date: Optional[str] = None,
    ):
        if not cutoff_date:
            cutoff_date = datetime.datetime.now().strftime("%Y-%m")

        self._safe_print(
            f"🚀 Starting Fast Hybrid Literature Agent (Cutoff: {cutoff_date}, Threads: {self.max_workers})..."
        )

        with open(outline_path, "r") as f:
            outline = json.load(f)

        outline_output_path = f"{self.output_dir}/outline_v1.json"
        unique_papers_registry: Dict[str, PaperData] = {}
        processed_tasks = set()

        # --- PHASE 1: DISCOVERY & RETRIEVAL ---
        self._safe_print(f"📚 Phase 1: Analyzing Plan & Executing Search Tasks...")

        search_tasks = self._collect_search_tasks(outline)
        self._safe_print(
            f"    -> Identified {len(search_tasks)} distinct search tasks."
        )

        self._execute_tasks_in_parallel(
            search_tasks, outline, unique_papers_registry, processed_tasks, cutoff_date
        )

        self._safe_print(
            f"\n✨ Final Count: {len(unique_papers_registry)} unique enriched papers retrieved."
        )

        # --- PHASE 2: ASSET GENERATION ---
        final_papers_list = list(unique_papers_registry.values())
        bibtex_str = self._generate_bibtex(final_papers_list)
        citation_map = self._generate_citation_map(final_papers_list)

        self._safe_print("🔄 Injecting citations back into Outline...")
        updated_outline = self._inject_citations_into_outline(
            outline, unique_papers_registry
        )

        # --- PHASE 3: WRITING ---
        self._safe_print("✍️  Phase 3: Writing Introduction & Related Work...")
        latex_content = self._synthesize_content(
            updated_outline, final_papers_list, cutoff_date
        )

        # --- PHASE 4: SAVING ---
        self._save_outputs(
            latex_content,
            bibtex_str,
            citation_map,
            updated_outline,
            outline_output_path,
        )
        self._safe_print(
            f"✅ Done! Literature Review Outputs saved to {self.output_dir}/"
        )

    def _execute_tasks_in_parallel(
        self, tasks, outline, registry, processed_tasks_set, cutoff_date
    ):
        """Run a batch of search tasks concurrently."""
        active_tasks = []
        for task in tasks:
            task_id = f"{task['section']}:{task['focus']}"
            if task_id in processed_tasks_set:
                continue
            active_tasks.append(task)
            processed_tasks_set.add(task_id)

        if not active_tasks:
            return

        if model_config.uses_third_party(model_config.LIGHT_ROLE):
            self._execute_academic_discovery(
                active_tasks, outline, registry, cutoff_date
            )
            return

        self._safe_print(f"   🔎 Phase 1a: Google Candidates Discovery (MaxPool=10)...")
        all_candidates_with_tasks = []

        with ThreadPoolExecutor(max_workers=10) as executor:  # Fast for Google Search!
            future_to_task = {
                executor.submit(
                    self._process_search_task_google, task, outline, cutoff_date
                ): task
                for task in active_tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    cands = future.result()
                    if cands:
                        for cand_item in cands:
                            all_candidates_with_tasks.append(
                                {"cand": cand_item, "task": task}
                            )
                except Exception as exc:
                    self._safe_print(
                        f"      ❌ Google Search failed for '{task['focus'][:20]}...': {exc}"
                    )

        self._safe_print(
            f"   📊 Phase 1b: Semantic Scholar Enrichment (Rate-limited, sequential)..."
        )
        import time

        for item in all_candidates_with_tasks:
            # wait 1s per query to avoid SS rate limits!
            time.sleep(1.0)
            self._enrich_and_register_paper(
                item["cand"], item["task"], registry, cutoff_date
            )

    # --- Academic-database discovery (no Google Search grounding) ---

    def _execute_academic_discovery(
        self, tasks, outline, registry, cutoff_date
    ):
        """Phase 1 replacement that queries scholarly APIs directly."""
        config = self.discovery_config
        self._safe_print(
            "   🔬 Phase 1a: Academic API Discovery "
            f"(sources: {', '.join(config.sources)})..."
        )

        max_workers = min(max(1, self.max_workers), 8, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._discover_from_academic_apis, task, outline, cutoff_date
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    records = future.result()
                except Exception as exc:
                    self._safe_print(
                        f"      ❌ Academic discovery failed for "
                        f"'{task['focus'][:30]}...': {exc}"
                    )
                    continue

                for record in records:
                    self._register_academic_paper(
                        record, task["section"], registry, cutoff_date
                    )

    def _discover_from_academic_apis(self, task, outline, cutoff_date) -> List[Dict]:
        """Plans queries, searches the configured databases, and ranks hits."""
        focus = (task.get("focus") or "").strip()
        if not focus:
            return []

        self._safe_print(
            f"   🔎 Academic Search [{task['section']}]: '{focus[:70]}'"
        )

        config = self.discovery_config
        queries = self._plan_academic_queries(task, outline, cutoff_date)

        merged: Dict[str, AcademicPaper] = {}
        for query in queries:
            papers = search_academic_databases(
                query=query,
                sources=config.sources,
                limit_per_source=config.max_results_per_source,
                cutoff_date=cutoff_date,
                timeout=config.timeout,
                source_options=config.source_options,
                on_error=self._on_academic_search_error,
            )
            for paper in papers:
                key = self._normalize_title(paper.title)
                if not key:
                    continue
                if key in merged:
                    merged[key].merge(paper)
                else:
                    merged[key] = paper

        if not merged:
            self._safe_print(f"      ⚠️ No API results for '{focus[:50]}'")
            return []

        candidates = sorted(
            merged.values(),
            key=lambda paper: (paper.citation_count, paper.year or 0),
            reverse=True,
        )[: config.max_candidates]

        selected = self._rank_academic_candidates(task, candidates, cutoff_date)
        return [paper.as_dict() for paper in selected]

    def _plan_academic_queries(self, task, outline, cutoff_date) -> List[str]:
        """Asks the light model for a few tool-free, API-friendly queries."""
        config = self.discovery_config
        focus = (task.get("focus") or "").strip()
        core_problem = (
            outline.get("intro_related_work_plan", {})
            .get("introduction_strategy", {})
            .get("problem_gap_hypothesis", "N/A")
        )

        prompt = f"""
        You are planning keyword queries for academic literature databases (OpenAlex, Crossref, arXiv, DBLP).
        These APIs match titles, abstracts and keywords, so do NOT use boolean operators or quotation marks.

        SEARCH TASK: {focus}
        CONTEXT: {task.get('context', '')}
        BROADER PROJECT PROBLEM: {core_problem}
        CUTOFF DATE: papers published before {cutoff_date}

        Write up to {config.max_queries} short English queries (3-8 words each) covering
        different angles of the task. Prefer technical terms that appear in paper titles.

        Return JSON only, following this schema:
        {{"queries": ["query one", "query two"]}}
        """

        queries: List[str] = []
        try:
            response = call_gemini_with_contents(
                model_name="gemini-3-flash-preview",
                model_role=model_config.LIGHT_ROLE,
                contents=[prompt],
                generation_configs={"temperature": 0.2},
                check_parsed_response_not_none=False,
            )
            parsed = response.get("parsed_response") or {}
            raw_queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
            for raw_query in raw_queries:
                if isinstance(raw_query, dict):
                    raw_query = raw_query.get("query", "")
                query = str(raw_query).strip()
                if query and query not in queries:
                    queries.append(query)
        except Exception as exc:
            self._safe_print(
                f"      ⚠️ Query planning failed, using the task text: {exc}"
            )

        if not queries:
            queries = [focus]

        return queries[: config.max_queries]

    def _rank_academic_candidates(
        self, task, candidates: List[AcademicPaper], cutoff_date
    ) -> List[AcademicPaper]:
        """Lets the light model pick the relevant subset of real API records."""
        if not candidates:
            return []

        is_targeted = task.get("search_type") == "targeted"
        limit = 1 if is_targeted else 15

        if len(candidates) == 1:
            candidates[0].reason = f"Only database match for: {task['focus']}"
            return candidates

        payload = []
        for index, paper in enumerate(candidates):
            authors = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            payload.append(
                {
                    "id": index,
                    "title": paper.title,
                    "year": paper.year,
                    "venue": paper.venue,
                    "citations": paper.citation_count,
                    "authors": authors,
                    "abstract": (paper.abstract or "")[:300],
                }
            )

        instruction = (
            "Return EXACTLY 1 candidate: the primary paper that introduced or "
            "defined the requested method, dataset, or baseline."
            if is_targeted
            else "Select the 10-15 most relevant and influential papers. "
            "Do not force results; drop unrelated papers."
        )

        prompt = f"""
        Below are real bibliographic records returned by academic databases for this task.
        Select the papers that best match the task. Do not invent papers and do not
        include records that are only loosely related.

        SEARCH TASK: {task.get('focus', '')}
        CONTEXT: {task.get('context', '')}
        CUTOFF DATE: {cutoff_date}
        INSTRUCTION: {instruction}

        RECORDS:
        {json.dumps(payload, ensure_ascii=False)}

        Return JSON only, following this schema:
        {CandidateRanking.model_json_schema()}
        """

        try:
            response = call_gemini_with_contents(
                model_name="gemini-3-flash-preview",
                model_role=model_config.LIGHT_ROLE,
                contents=[prompt],
                generation_configs={"temperature": 0.1},
                check_parsed_response_not_none=False,
            )
            parsed = response.get("parsed_response") or {}
            raw_candidates = (
                parsed.get("candidates", []) if isinstance(parsed, dict) else []
            )

            selected: List[AcademicPaper] = []
            seen_ids = set()
            for item in raw_candidates:
                if isinstance(item, int):
                    item = {"id": item, "reason": ""}
                if not isinstance(item, dict):
                    continue
                try:
                    paper_id = int(item.get("id"))
                except (TypeError, ValueError):
                    continue
                if paper_id in seen_ids or not 0 <= paper_id < len(candidates):
                    continue
                seen_ids.add(paper_id)
                reason = str(item.get("reason") or "").strip()
                candidates[paper_id].reason = reason or (
                    f"Relevant to {task['section']}."
                )
                selected.append(candidates[paper_id])

            if selected:
                return selected[:limit]
        except Exception as exc:
            self._safe_print(
                f"      ⚠️ Candidate ranking failed, keeping database order: {exc}"
            )

        fallback_pool = candidates
        if is_targeted:
            # Prefer the database's own relevance order for a single-paper match.
            fallback_pool = sorted(
                candidates,
                key=lambda paper: (paper.best_rank, -paper.citation_count),
            )
        fallback = fallback_pool[: min(limit, len(fallback_pool))]
        for paper in fallback:
            paper.reason = paper.reason or (
                f"Top database match for {task.get('section', 'this section')}."
            )
        return fallback

    def _on_academic_search_error(self, source: str, message: str) -> None:
        with self._discovery_error_lock:
            if source in self._failed_discovery_sources:
                return
            self._failed_discovery_sources.add(source)
        self._safe_print(f"      ⚠️ {message}")

    def _register_academic_paper(self, record, section, registry, cutoff_date):
        """Registers an API record, using S2 only when the abstract is missing."""
        title = (record.get("title") or "").strip()
        if not title:
            return

        norm_title = self._normalize_title(title)
        with self.registry_lock:
            if norm_title in registry:
                return

        if not self._is_paper_allowed(
            {
                "publicationDate": record.get("publication_date"),
                "year": record.get("year"),
            },
            cutoff_date,
        ):
            return

        abstract = (record.get("abstract") or "").strip()
        if not abstract:
            # DBLP/Crossref sometimes omit abstracts. Try Semantic Scholar as a
            # best-effort fill, but keep the record even if that fails: the
            # bibliographic data from the academic API is still valid.
            import time

            time.sleep(1.0)
            s2_data = s2_title_search(
                title,
                int(record.get("year") or 0) or None,
                cutoff_date,
            )
            if s2_data and s2_data.get("abstract"):
                record = dict(record)
                record["abstract"] = s2_data["abstract"]
                record["citation_count"] = max(
                    int(record.get("citation_count") or 0),
                    int(s2_data.get("citationCount") or 0),
                )
                if not record.get("venue"):
                    record["venue"] = s2_data.get("venue") or ""
                if not record.get("journal"):
                    record["journal"] = (s2_data.get("journal") or {}).get("name")
                abstract = record["abstract"].strip()

        if len(abstract) > 1500:
            abstract = abstract[:1500] + "... [Truncated]"

        sources = record.get("sources") or []
        venue = record.get("venue") or ("arXiv" if "arxiv" in sources else "Unknown")

        full_obj = PaperData(
            citation_key=self._generate_key(
                [{"name": author} for author in record.get("authors") or []],
                record.get("year"),
                title,
            ),
            title=title,
            authors=list(record.get("authors") or []),
            venue=venue,
            year=int(record.get("year") or 0),
            abstract=abstract,
            citation_count=int(record.get("citation_count") or 0),
            found_in_section=section,
            reason=record.get("reason") or "",
            journal=record.get("journal"),
            volume=record.get("volume"),
            pages=record.get("pages"),
            publication_date=record.get("publication_date"),
        )

        with self.registry_lock:
            if norm_title not in registry:
                registry[norm_title] = full_obj
                self._safe_print(
                    f"      -> 📚 Added API paper: {full_obj.title}"
                )

    def _collect_search_tasks(self, outline: Dict) -> List[Dict]:

        tasks = []
        plan = outline.get("intro_related_work_plan", {})

        # 1. Introduction Macro-Tasks
        intro_strat = plan.get("introduction_strategy", {})
        for direction in intro_strat.get("search_directions", []):
            tasks.append(
                {
                    "section": "Introduction",
                    "focus": direction,
                    "context": f"Hook: {intro_strat.get('hook_hypothesis', '')}. Gap: {intro_strat.get('problem_gap_hypothesis', '')}",
                    "search_type": "exploration",
                }
            )

        # 2. Related Work Micro-Tasks
        rw_strat = plan.get("related_work_strategy", {})
        for sub in rw_strat.get("subsections", []):
            queries = sub.get("limitation_search_queries", [])
            if not queries:
                queries = [
                    f"{sub.get('subsection_title', '')} {sub.get('methodology_cluster', '')}"
                ]

            for q in queries:
                tasks.append(
                    {
                        "section": f"Related Work: {sub.get('subsection_title', 'General')}",
                        "focus": q,
                        "context": f"Mission: {sub.get('sota_investigation_mission', '')}. Hypothesis: {sub.get('limitation_hypothesis', '')}",
                        "search_type": "exploration",
                    }
                )

        # 3. Targeted Tasks (Must-Haves from Section Plan)
        blueprint = outline.get("section_plan", [])
        for section in blueprint:
            section_title = section.get("section_title", "Unknown Section")
            for subsection in section.get("subsections", []):
                sub_title = subsection.get("subsection_title", "General")
                bullets = subsection.get("content_bullets", [])
                context_str = " ".join(bullets)

                hints = subsection.get("citation_hints", [])
                for hint in hints:
                    tasks.append(
                        {
                            "section": f"{section_title} - {sub_title}",
                            "focus": hint,
                            "context": f"Must-have citation for section covering: {context_str}",
                            "search_type": "targeted",
                        }
                    )

        return tasks

    def _inject_citations_into_outline(
        self, outline: Dict, registry: Dict[str, PaperData]
    ) -> Dict:
        # Group keys by section name
        section_to_keys = {}
        for p in registry.values():
            sec = p.found_in_section
            if sec not in section_to_keys:
                section_to_keys[sec] = []
            section_to_keys[sec].append(p.citation_key)

        plan = outline.get("intro_related_work_plan", {})

        # Introduction
        intro_strat = plan.get("introduction_strategy", {})
        intro_keys = section_to_keys.get("Introduction", [])
        intro_strat["citation_candidates"] = list(set(intro_keys))

        # Related Work
        rw_strat = plan.get("related_work_strategy", {})
        for sub in rw_strat.get("subsections", []):
            key = f"Related Work: {sub.get('subsection_title', 'General')}"
            sub["citation_candidates"] = section_to_keys.get(key, [])

        # Other Sections
        blueprint = outline.get("section_plan", [])
        for section in blueprint:
            section_title = section.get("section_title", "Unknown Section")
            for sub in section.get("subsections", []):
                sub_title = sub.get("subsection_title", "General")
                key = f"{section_title} - {sub_title}"
                sub["citation_candidates"] = section_to_keys.get(key, [])
                if "citation_hints" in sub:
                    del sub["citation_hints"]

        return outline

    def _process_search_task_google(self, task, outline, cutoff_date) -> list:
        if task.get("search_type") == "targeted":
            self._safe_print(
                f"   🎯 Targeted Search [{task['section']}]: '{task['focus']}'"
            )
        else:
            self._safe_print(
                f"   🔎 Exploration [{task['section']}]: '{task['focus']}'"
            )

        retry_count = 0
        while retry_count < 3:
            try:
                candidates = self._discover_candidates(task, outline, cutoff_date)
                if candidates:
                    return candidates
                retry_count += 1
            except Exception as e:
                retry_count += 1
        return []

    def _enrich_and_register_paper(self, cand, task, registry, cutoff_date):
        cand_title = getattr(cand, "title", None) or cand.get("title")
        cand_year = getattr(cand, "year", None) or cand.get("year")
        cand_reason = getattr(cand, "reason", None) or cand.get("reason", "")

        norm_title = self._normalize_title(cand_title)

        with self.registry_lock:
            if norm_title in registry:
                return

        try:
            s2_data = s2_title_search(cand_title, cand_year, cutoff_date)

            if s2_data:
                if not self._is_paper_allowed(s2_data, cutoff_date):
                    return

                if not s2_data.get("abstract", ""):
                    return

                authors_list = [a["name"] for a in s2_data.get("authors", [])]
                j_info = s2_data.get("journal") or {}

                abstract_text = s2_data.get("abstract") or ""
                if len(abstract_text) > 1500:
                    abstract_text = abstract_text[:1500] + "... [Truncated]"

                full_obj = PaperData(
                    citation_key=self._generate_key(
                        s2_data.get("authors", []),
                        s2_data.get("year"),
                        s2_data["title"],
                    ),
                    title=s2_data["title"],
                    authors=authors_list,
                    venue=s2_data.get("venue") or "arXiv",
                    year=s2_data.get("year") or cand_year,
                    abstract=abstract_text,
                    citation_count=s2_data.get("citationCount") or 0,
                    found_in_section=task["section"],
                    reason=cand_reason,
                    journal=j_info.get("name"),
                    volume=j_info.get("volume"),
                    pages=j_info.get("pages"),
                    publication_date=s2_data.get("publicationDate"),
                )

                final_norm_title = self._normalize_title(full_obj.title)

                with self.registry_lock:
                    if final_norm_title not in registry:
                        registry[final_norm_title] = full_obj
                        self._safe_print(
                            f"      -> 📚 Added enriched paper: {full_obj.title}"
                        )
        except Exception as e:
            pass

    def _is_paper_allowed(self, s2_data: Dict, cutoff_date_str: str) -> bool:
        if not cutoff_date_str:
            return True

        c_parts = [int(x) for x in cutoff_date_str.split("-")]
        c_y = c_parts[0]
        c_m = c_parts[1] if len(c_parts) > 1 else 12
        c_d = c_parts[2] if len(c_parts) > 2 else 1

        p_y, p_m, p_d = None, None, None
        if s2_data.get("publicationDate"):
            try:
                p_parts = [int(x) for x in str(s2_data["publicationDate"]).split("-")]
                p_y = p_parts[0]
                p_m = p_parts[1] if len(p_parts) > 1 else None
                p_d = p_parts[2] if len(p_parts) > 2 else None
            except:
                pass

        if p_y is None and s2_data.get("year"):
            p_y = int(s2_data["year"])

        if p_y is None:
            return True
        if p_y > c_y:
            return False
        if p_y < c_y:
            return True
        if p_m is None:
            return True
        if p_m > c_m:
            return False
        if p_m < c_m:
            return True
        if p_d is None:
            return True
        if p_d > c_d:
            return False

        return True

    def _discover_candidates(self, task, outline, cutoff_date) -> List[CandidatePaper]:
        intro_plan = outline.get("intro_related_work_plan", {}).get(
            "introduction_strategy", {}
        )
        core_problem = intro_plan.get("problem_gap_hypothesis", "N/A")

        if task.get("search_type") == "targeted":
            prompt = f"""
            Identify the SPECIFIC academic paper described by this hint: "{task['focus']}".
            
            CONTEXT: {task['context']}
            CUTOFF DATE: Published BEFORE {cutoff_date}
            
            INSTRUCTIONS:
            1. Use Google Search to find the EXACT title and year of this paper.
            2. If the hint refers to a dataset, model architecture, or baseline (e.g., "research paper introducing..."), you MUST find the primary, original paper that introduced it.
            3. Return EXACTLY 1 candidate paper that matches best.

            YOU MUST return the results in json format (```json content```) following this schema:
            {DiscoveryResult.model_json_schema()}
            """
        else:
            prompt = f"""
            Find 10-15 highly relevant, influential academic papers matching the search task. Do not force results; only return strictly relevant papers.
            
            SEARCH TASK: {task['focus']}
            CONTEXT FOR TASK: {task['context']}
            BROADER PROJECT PROBLEM: {core_problem}
            CUTOFF DATE: Published BEFORE {cutoff_date}
            
            INSTRUCTIONS:
            1. Use Google Search to find REAL papers (ArXiv, CVPR, NeurIPS, ICML, ICLR, etc.).
            2. Only return papers published in top-tier conferences or journals, or highly-cited works, or the exact state-of-the-art relevant to the task.

            YOU MUST return the results in json format (```json content```) following this schema:
            {DiscoveryResult.model_json_schema()}
            """

        response_dict = call_gemini_with_contents(
            model_name="gemini-3-flash-preview",
            model_role=model_config.LIGHT_ROLE,
            contents=[prompt],
            generation_configs={
                "tools": [self.google_search_tool],
                "temperature": 0.1 if task.get("search_type") == "targeted" else 0.4,
            },
        )

        if response_dict["parsed_response"]:
            return response_dict["parsed_response"].get("candidates", [])
        return []

    def _synthesize_content(self, outline, papers: List[PaperData], cutoff_date: str):
        paper_keys = [p.citation_key for p in papers]
        papers_payload = [p.model_dump() for p in papers]

        context_payload = {
            "template.tex": self.context_files.get("template.tex", ""),
            "intro_related_work_plan": outline.get("intro_related_work_plan", {}),
            "project_idea": self.context_files.get("idea.md", ""),
            "project_experimental_log": self.context_files.get(
                "experimental_log.md", ""
            ),
            "citation_checklist": paper_keys,
            "collected_papers": papers_payload,
        }

        writer_system_prompt = literature_review_agent_writter_prompt.format(
            paper_count=len(papers),
            min_cite_paper_count=int(len(papers) * 0.9),
            cutoff_date=cutoff_date,
        )

        user_prompt = "Generate the LaTeX for Introduction and Related Work sections."
        response_dict = call_gemini_with_contents(
            model_name=self.model_name,
            contents=[user_prompt, json.dumps(context_payload)],
            generation_configs={"system_instruction": writer_system_prompt},
            result_parsing_func=parse_gemini_latex_results,
        )
        return response_dict["parsed_response"]

    def _generate_key(self, authors_raw, year, title):
        if not authors_raw:
            first_author = "Unknown"
        elif isinstance(authors_raw[0], dict):
            first_author = authors_raw[0]["name"].split()[-1]
        else:
            first_author = str(authors_raw[0]).split()[-1]

        clean_author = re.sub(r"[^a-zA-Z]", "", first_author).capitalize()
        year_str = str(year) if year else "2024"
        clean_title_str = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower())
        words = clean_title_str.split()
        stopwords = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "for",
            "to",
            "of",
            "and",
            "is",
            "are",
            "with",
            "by",
            "study",
        }
        meaningful = [w.capitalize() for w in words if w not in stopwords]
        title_part = "".join(meaningful[:2]) if meaningful else "Paper"
        return f"{clean_author}{year_str}{title_part}"

    def _generate_bibtex(self, papers: List[PaperData]) -> str:
        entries = []
        seen_keys = set()
        for p in papers:
            base_key = p.citation_key
            unique_key = base_key
            suffix = "a"
            while unique_key in seen_keys:
                unique_key = f"{base_key}{suffix}"
                suffix = chr(ord(suffix) + 1)
            p.citation_key = unique_key
            seen_keys.add(unique_key)

            author_str = " and ".join(p.authors) if p.authors else "Unknown"
            entry_type = "article" if p.journal else "inproceedings"

            entry = f"@{entry_type}{{{p.citation_key},\n  title={{{p.title}}},\n  author={{{author_str}}},\n"
            if p.journal:
                entry += f"  journal={{{p.journal}}},\n"
            else:
                entry += f"  booktitle={{{p.venue}}},\n"
            entry += f"  year={{{p.year}}}"
            if p.volume:
                entry += f",\n  volume={{{p.volume}}}"
            if p.pages:
                entry += f",\n  pages={{{p.pages}}}"
            entry += "\n}"
            entries.append(entry)
        return "\n\n".join(entries)

    def _generate_citation_map(self, papers: List[PaperData]) -> Dict[str, Dict]:
        return {
            p.citation_key: {
                "citation_key": p.citation_key,
                "title": p.title,
                "authors": p.authors,
                "venue": p.venue,
                "year": p.year,
                "abstract": p.abstract.strip(),
            }
            for p in papers
        }

    def _save_outputs(self, latex, bibtex, citation_map, outline, outline_path):
        with open(
            f"{self.output_dir}/updated_template.tex", "w", encoding="utf-8"
        ) as f:
            f.write(latex)
        with open(f"{self.output_dir}/references.bib", "w", encoding="utf-8") as f:
            f.write(bibtex)
        with open(f"{self.output_dir}/citation_map.json", "w", encoding="utf-8") as f:
            json.dump(citation_map, f, indent=4)
        with open(outline_path, "w", encoding="utf-8") as f:
            json.dump(outline, f, indent=4)

    def _load_context_files(self) -> Dict[str, str]:
        loaded = {}
        for name, path in self.context_paths.items():
            if os.path.exists(path):
                with open(path, "r") as f:
                    loaded[name] = f.read()
            else:
                loaded[name] = ""
        return loaded

    def _normalize_title(self, title):
        return re.sub(r"[^a-z0-9]", "", title.lower())
