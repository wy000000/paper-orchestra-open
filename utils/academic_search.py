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

"""Direct academic-database search backends for literature discovery.

The original discovery phase asks Gemini (with the Google Search tool) to name
papers.  Google Search grounding is Gemini-only and generativelanguage/
Vertex AI are not directly reachable from some networks (for example mainland
China without a proxy).  This module queries scholarly databases instead and
returns real bibliographic records, so the LLM only has to plan queries and
rank candidates.

Every backend below is free and needs no API key (Semantic Scholar accepts an
optional key).  Configure which ones to use through
``model_config.yaml -> literature_discovery.sources``.
"""

import calendar
import datetime
import html
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import requests

from utils import model_config

DEFAULT_SOURCES = ("openalex", "crossref", "arxiv", "dblp")
VALID_SOURCES = (
    "openalex",
    "crossref",
    "arxiv",
    "dblp",
    "semantic_scholar",
    "pubmed",
)

USER_AGENT = "PaperOrchestra/1.0 (literature discovery; academic APIs)"

# Minimum seconds between two requests to the same API.  These are
# deliberately conservative: the agent runs many search tasks concurrently.
_SOURCE_INTERVALS = {
    "openalex": 0.12,
    "crossref": 0.30,
    "arxiv": 3.0,
    "dblp": 1.0,
    "semantic_scholar": 1.1,
    "pubmed": 0.40,
}


class _RateLimiter:
    """Thread-safe, per-source minimum-interval limiter."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if delay:
            time.sleep(delay)


_LIMITERS = {}
_LIMITERS_LOCK = threading.Lock()


def _get_limiter(source: str, min_interval: Optional[float]) -> _RateLimiter:
    interval = (
        _SOURCE_INTERVALS.get(source, 1.0)
        if min_interval is None
        else float(min_interval)
    )
    key = (source, round(interval, 3))
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = _RateLimiter(interval)
            _LIMITERS[key] = limiter
    return limiter


_HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "application/json, application/xml, text/xml;q=0.9, */*;q=0.8"
    ),
}

_logged_warnings = set()
_warn_lock = threading.Lock()


def _warn_once(message: str) -> None:
    with _warn_lock:
        if message in _logged_warnings:
            return
        _logged_warnings.add(message)
    print(f"WARNING: {message}")


@dataclass
class AcademicPaper:
    """Normalized bibliographic record shared by all backends."""

    title: str
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    venue: str = ""
    abstract: str = ""
    citation_count: int = 0
    source: str = ""
    sources: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    url: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    reason: str = ""
    best_rank: int = 999

    def merge(self, other: "AcademicPaper") -> None:
        """Fills gaps from a duplicate record and keeps the richer values."""
        if not self.abstract and other.abstract:
            self.abstract = other.abstract
        elif (
            other.abstract
            and len(other.abstract) > len(self.abstract)
            and len(self.abstract) < 200
        ):
            self.abstract = other.abstract

        if not self.authors and other.authors:
            self.authors = other.authors
        if not self.venue and other.venue:
            self.venue = other.venue
        if self.year is None:
            self.year = other.year
        if not self.doi and other.doi:
            self.doi = other.doi
        if not self.url and other.url:
            self.url = other.url
        if not self.publication_date and other.publication_date:
            self.publication_date = other.publication_date
        if not self.journal and other.journal:
            self.journal = other.journal
        if not self.volume and other.volume:
            self.volume = other.volume
        if not self.pages and other.pages:
            self.pages = other.pages

        self.citation_count = max(self.citation_count, other.citation_count)
        self.best_rank = min(self.best_rank, other.best_rank)
        for source in other.sources or [other.source]:
            if source and source not in self.sources:
                self.sources.append(source)

    def as_dict(self) -> Dict:
        return asdict(self)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_int(value) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(str(value)[:4]) if len(str(value)) >= 4 else int(value)
    except (TypeError, ValueError):
        return None


def _cutoff_end_date(cutoff_date: Optional[str]) -> Optional[str]:
    """Turns ``YYYY-MM`` into the last day of that month (``YYYY-MM-DD``)."""
    if not cutoff_date:
        return None
    parts = str(cutoff_date).split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 12
        day = int(parts[2]) if len(parts) > 2 else calendar.monthrange(year, month)[1]
        return datetime.date(year, month, day).isoformat()
    except (ValueError, IndexError):
        return None


def _http_get(
    source: str,
    url: str,
    params: Dict,
    timeout: float,
    options: Optional["model_config.SourceOptions"] = None,
    headers: Optional[Dict] = None,
    attempts: int = 2,
):
    """GET with per-source throttling and a light retry on 429/5xx."""
    limiter = _get_limiter(source, options.min_interval if options else None)
    request_headers = dict(_HTTP_HEADERS)
    if headers:
        request_headers.update(headers)

    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        limiter.wait()
        try:
            response = requests.get(
                url, params=params, timeout=timeout, headers=request_headers
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
                continue
            raise

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            body = response.text or ""
            if "json" in content_type or body.lstrip()[:1] in ("{", "["):
                try:
                    return response.json()
                except ValueError as exc:
                    raise RuntimeError(f"{source}: invalid JSON response") from exc
            return body

        last_error = RuntimeError(f"{source}: HTTP {response.status_code}")
        if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
            retry_after = response.headers.get("Retry-After", "")
            try:
                wait = min(float(retry_after), 5.0)
            except (TypeError, ValueError):
                wait = 2.0 + attempt
            time.sleep(wait)
            continue
        raise last_error

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{source}: request failed")


def _openalex_abstract(inverted_index: Optional[Dict]) -> str:
    if not inverted_index:
        return ""
    positions = []
    for word, indexes in inverted_index.items():
        for index in indexes or []:
            positions.append((index, word))
    return " ".join(word for _, word in sorted(positions))


def _search_openalex(
    query: str,
    limit: int,
    cutoff_date: Optional[str],
    options: "model_config.SourceOptions",
    timeout: float,
) -> List[AcademicPaper]:
    params = {
        "search": query,
        "per-page": max(1, min(limit, 50)),
        "select": (
            "title,publication_year,publication_date,authorships,"
            "primary_location,abstract_inverted_index,cited_by_count,doi,type"
        ),
    }
    if options.mailto:
        params["mailto"] = options.mailto
    end_date = _cutoff_end_date(cutoff_date)
    if end_date:
        params["filter"] = f"to_publication_date:{end_date}"

    payload = _http_get(
        "openalex",
        "https://api.openalex.org/works",
        params,
        timeout,
        options=options,
    )
    papers = []
    for item in (payload or {}).get("results", []):
        title = _clean_text(item.get("title") or item.get("display_name"))
        if not title:
            continue

        location = item.get("primary_location") or {}
        source_info = location.get("source") or {}
        venue = _clean_text(source_info.get("display_name"))
        journal = venue if source_info.get("type") == "journal" else None
        doi = (item.get("doi") or "").replace("https://doi.org/", "") or None

        papers.append(
            AcademicPaper(
                title=title,
                year=_safe_int(item.get("publication_year")),
                authors=[
                    _clean_text((authorship.get("author") or {}).get("display_name"))
                    for authorship in item.get("authorships", [])
                    if (authorship.get("author") or {}).get("display_name")
                ],
                venue=venue or "OpenAlex",
                abstract=_openalex_abstract(item.get("abstract_inverted_index")),
                citation_count=int(item.get("cited_by_count") or 0),
                source="openalex",
                sources=["openalex"],
                doi=doi,
                url=("https://doi.org/" + doi) if doi else location.get("landing_page_url"),
                publication_date=item.get("publication_date"),
                journal=journal,
            )
        )
    return papers


def _search_crossref(
    query: str,
    limit: int,
    cutoff_date: Optional[str],
    options: "model_config.SourceOptions",
    timeout: float,
) -> List[AcademicPaper]:
    params = {
        "query.bibliographic": query,
        "rows": max(1, min(limit, 50)),
    }
    if options.mailto:
        params["mailto"] = options.mailto
    end_date = _cutoff_end_date(cutoff_date)
    if end_date:
        params["filter"] = f"until-pub-date:{end_date}"

    payload = _http_get(
        "crossref",
        "https://api.crossref.org/works",
        params,
        timeout,
        options=options,
    )
    items = ((payload or {}).get("message") or {}).get("items", [])
    papers = []
    for item in items:
        titles = item.get("title") or []
        title = _clean_text(titles[0] if titles else "")
        if not title:
            continue

        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        date_parts = issued[0] if issued else []
        year = _safe_int(date_parts[0]) if date_parts else None
        publication_date = None
        if date_parts and date_parts[0]:
            try:
                publication_date = datetime.date(
                    int(date_parts[0]),
                    int(date_parts[1]) if len(date_parts) > 1 and date_parts[1] else 1,
                    int(date_parts[2]) if len(date_parts) > 2 and date_parts[2] else 1,
                ).isoformat()
            except (TypeError, ValueError):
                publication_date = None

        authors = []
        for author in item.get("author") or []:
            name = " ".join(
                part
                for part in [author.get("given"), author.get("family")]
                if part
            ) or author.get("name")
            if name:
                authors.append(_clean_text(name))

        container = _clean_text((item.get("container-title") or [""])[0])
        doi = item.get("DOI")
        is_journal = item.get("type") == "journal-article"

        papers.append(
            AcademicPaper(
                title=title,
                year=year,
                authors=authors,
                venue=container or "Crossref",
                abstract=_clean_text(item.get("abstract")),
                citation_count=int(item.get("is-referenced-by-count") or 0),
                source="crossref",
                sources=["crossref"],
                doi=doi,
                url=item.get("URL") or (("https://doi.org/" + doi) if doi else None),
                publication_date=publication_date,
                journal=container if is_journal else None,
                volume=item.get("volume"),
                pages=item.get("page"),
            )
        )
    return papers


def _search_arxiv(
    query: str,
    limit: int,
    cutoff_date: Optional[str],
    options: "model_config.SourceOptions",
    timeout: float,
) -> List[AcademicPaper]:
    safe_query = query.replace('"', " ").strip()
    params = {
        "search_query": f"all:{safe_query}",
        "start": 0,
        "max_results": max(1, min(limit, 50)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    text = _http_get(
        "arxiv",
        "https://export.arxiv.org/api/query",
        params,
        max(timeout, 25.0),
        options=options,
        attempts=1,
    )
    if not isinstance(text, str):
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RuntimeError(f"arxiv: invalid XML response ({exc})") from exc

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        if not title:
            continue

        published = entry.findtext("atom:published", default="", namespaces=ns)
        journal_ref = _clean_text(
            entry.findtext("arxiv:journal_ref", default="", namespaces=ns)
        )
        doi = _clean_text(entry.findtext("arxiv:doi", default="", namespaces=ns))

        papers.append(
            AcademicPaper(
                title=title,
                year=_safe_int(published),
                authors=[
                    name
                    for name in (
                        _clean_text(
                            author.findtext(
                                "atom:name", default="", namespaces=ns
                            )
                        )
                        for author in entry.findall("atom:author", ns)
                    )
                    if name
                ],
                venue=journal_ref or "arXiv",
                abstract=_clean_text(
                    entry.findtext("atom:summary", default="", namespaces=ns)
                ),
                citation_count=0,
                source="arxiv",
                sources=["arxiv"],
                doi=doi or None,
                url=entry.findtext("atom:id", default="", namespaces=ns) or None,
                publication_date=published[:10] if published else None,
                journal=journal_ref or None,
            )
        )
    return papers


def _search_dblp(
    query: str,
    limit: int,
    cutoff_date: Optional[str],
    options: "model_config.SourceOptions",
    timeout: float,
) -> List[AcademicPaper]:
    params = {
        "q": query,
        "format": "json",
        "h": max(1, min(limit, 50)),
    }
    payload = _http_get(
        "dblp",
        "https://dblp.org/search/publ/api",
        params,
        timeout,
        options=options,
    )
    hits = (((payload or {}).get("result") or {}).get("hits") or {}).get("hit", [])
    if isinstance(hits, dict):
        hits = [hits]

    papers = []
    for hit in hits:
        info = hit.get("info") if isinstance(hit, dict) else None
        if isinstance(info, list):
            info = info[0] if info else None
        if not isinstance(info, dict):
            continue

        title = _clean_text(info.get("title"))
        if not title:
            continue

        raw_authors = (info.get("authors") or {}).get("author", [])
        if isinstance(raw_authors, (dict, str)):
            raw_authors = [raw_authors]
        authors = []
        for author in raw_authors:
            name = author.get("text") if isinstance(author, dict) else author
            if name:
                authors.append(_clean_text(name))

        venue = _clean_text(info.get("venue"))
        papers.append(
            AcademicPaper(
                title=title,
                year=_safe_int(info.get("year")),
                authors=authors,
                venue=venue or "DBLP",
                abstract="",
                citation_count=0,
                source="dblp",
                sources=["dblp"],
                doi=info.get("doi"),
                url=info.get("ee") or info.get("url"),
                publication_date=None,
                journal=venue if info.get("type") == "Journal Articles" else None,
            )
        )
    return papers


def _search_semantic_scholar(
    query: str,
    limit: int,
    cutoff_date: Optional[str],
    options: "model_config.SourceOptions",
    timeout: float,
) -> List[AcademicPaper]:
    api_key = options.api_key or model_config.get_semantic_scholar_api_key()
    headers = {"X-API-KEY": api_key} if api_key else {}
    params = {
        "query": query,
        "limit": max(1, min(limit, 50)),
        "fields": (
            "title,authors,venue,year,abstract,citationCount,journal,"
            "publicationDate,externalIds,url"
        ),
    }
    payload = _http_get(
        "semantic_scholar",
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params,
        timeout,
        options=options,
        headers=headers,
    )
    papers = []
    for item in (payload or {}).get("data", []):
        title = _clean_text(item.get("title"))
        if not title:
            continue
        journal_info = item.get("journal") or {}
        external_ids = item.get("externalIds") or {}
        papers.append(
            AcademicPaper(
                title=title,
                year=_safe_int(item.get("year")),
                authors=[
                    _clean_text(author.get("name"))
                    for author in item.get("authors", [])
                    if author.get("name")
                ],
                venue=_clean_text(item.get("venue")) or "Semantic Scholar",
                abstract=_clean_text(item.get("abstract")),
                citation_count=int(item.get("citationCount") or 0),
                source="semantic_scholar",
                sources=["semantic_scholar"],
                doi=external_ids.get("DOI"),
                url=item.get("url"),
                publication_date=item.get("publicationDate"),
                journal=_clean_text(journal_info.get("name")) or None,
                volume=journal_info.get("volume"),
                pages=journal_info.get("pages"),
            )
        )
    return papers


def _search_pubmed(
    query: str,
    limit: int,
    cutoff_date: Optional[str],
    options: "model_config.SourceOptions",
    timeout: float,
) -> List[AcademicPaper]:
    common = {"db": "pubmed", "tool": options.tool or "paper_orchestra"}
    if options.email:
        common["email"] = options.email
    if options.api_key:
        common["api_key"] = options.api_key

    search_params = dict(common)
    search_params.update(
        {
            "term": query,
            "retmode": "json",
            "retmax": max(1, min(limit, 50)),
            "sort": "relevance",
        }
    )
    payload = _http_get(
        "pubmed",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        search_params,
        timeout,
        options=options,
    )
    ids = (((payload or {}).get("esearchresult") or {}).get("idlist") or [])
    if not ids:
        return []

    fetch_params = dict(common)
    fetch_params.update({"id": ",".join(ids), "retmode": "xml"})
    text = _http_get(
        "pubmed",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        fetch_params,
        timeout,
        options=options,
    )
    if not isinstance(text, str):
        return []

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    papers = []
    for article in root.findall(".//PubmedArticle"):
        title_element = article.find(".//ArticleTitle")
        title = _clean_text(
            "".join(title_element.itertext()) if title_element is not None else ""
        )
        if not title:
            continue

        abstract_parts = []
        for abstract in article.findall(".//Abstract/AbstractText"):
            label = abstract.get("Label")
            value = _clean_text("".join(abstract.itertext()))
            if value:
                abstract_parts.append(f"{label}: {value}" if label else value)

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName", default="")
            initials = author.findtext("Initials", default="")
            collective = author.findtext("CollectiveName", default="")
            name = collective or " ".join(part for part in [last, initials] if part)
            if name:
                authors.append(_clean_text(name))

        pub_date = article.find(".//JournalIssue/PubDate")
        year = None
        publication_date = None
        if pub_date is not None:
            year = _safe_int(pub_date.findtext("Year"))
            month = pub_date.findtext("Month") or "1"
            day = pub_date.findtext("Day") or "1"
            month_lookup = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            }
            month_number = month_lookup.get(month, None)
            if month_number is None:
                month_number = _safe_int(month) or 1
            if year:
                try:
                    publication_date = datetime.date(
                        year, month_number, _safe_int(day) or 1
                    ).isoformat()
                except ValueError:
                    publication_date = f"{year:04d}"

        doi = None
        for article_id in article.findall(".//ArticleIdList/ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = _clean_text(article_id.text)
                break
        doi = doi or _clean_text(
            article.findtext(".//ELocationID[@EIdType='doi']", default="")
        ) or None

        pmid = _clean_text(article.findtext(".//PMID", default=""))
        papers.append(
            AcademicPaper(
                title=title,
                year=year,
                authors=authors,
                venue=_clean_text(article.findtext(".//Journal/Title", default=""))
                or "PubMed",
                abstract=" ".join(abstract_parts),
                citation_count=0,
                source="pubmed",
                sources=["pubmed"],
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                publication_date=publication_date,
                journal=_clean_text(article.findtext(".//Journal/Title", default=""))
                or None,
            )
        )
    return papers


_SEARCHERS = {
    "openalex": _search_openalex,
    "crossref": _search_crossref,
    "arxiv": _search_arxiv,
    "dblp": _search_dblp,
    "semantic_scholar": _search_semantic_scholar,
    "pubmed": _search_pubmed,
}


def _merge_records(records: Iterable[AcademicPaper]) -> List[AcademicPaper]:
    merged: Dict[str, AcademicPaper] = {}
    order: List[str] = []
    for record in records:
        if not record or not record.title:
            continue
        key = _normalize_title(record.title)
        if not key:
            continue
        if key in merged:
            merged[key].merge(record)
        else:
            merged[key] = record
            order.append(key)
    papers = [merged[key] for key in order]
    # Prefer well-cited work but keep the relative order stable for ties.
    papers.sort(key=lambda paper: (paper.citation_count, paper.year or 0), reverse=True)
    return papers


def search_source(
    source: str,
    query: str,
    limit: int = 10,
    cutoff_date: Optional[str] = None,
    timeout: float = 20.0,
    options: Optional["model_config.SourceOptions"] = None,
) -> List[AcademicPaper]:
    """Runs one backend. Returns [] for unknown sources."""
    searcher = _SEARCHERS.get(source)
    if searcher is None:
        raise ValueError(
            f"Unknown literature discovery source '{source}'. "
            f"Use one of: {', '.join(VALID_SOURCES)}."
        )
    return searcher(
        query,
        limit,
        cutoff_date,
        options or model_config.SourceOptions(),
        timeout,
    )


def _passes_cutoff(paper: AcademicPaper, cutoff_date: Optional[str]) -> bool:
    """Cheap pre-filter before records reach the agent (which re-checks)."""
    if not cutoff_date or paper.year is None:
        return True
    parts = str(cutoff_date).split("-")
    try:
        cutoff_year = int(parts[0])
        cutoff_month = int(parts[1]) if len(parts) > 1 else 12
    except (ValueError, IndexError):
        return True

    if paper.year > cutoff_year:
        return False
    if paper.year < cutoff_year:
        return True
    if paper.publication_date:
        try:
            paper_month = int(str(paper.publication_date).split("-")[1])
            return paper_month <= cutoff_month
        except (ValueError, IndexError):
            return True
    return True


def search_academic_databases(
    query: str,
    sources: Sequence[str] = DEFAULT_SOURCES,
    limit_per_source: int = 10,
    cutoff_date: Optional[str] = None,
    timeout: float = 20.0,
    source_options: Optional[Dict[str, "model_config.SourceOptions"]] = None,
    on_error: Optional[Callable[[str, str], None]] = None,
) -> List[AcademicPaper]:
    """Queries several databases in parallel and merges duplicate titles.

    ``source_options`` maps a source name to its dedicated settings (mailto,
    API key, result cap, timeout, rate limit). Individual backend failures are
    reported through ``on_error(source, message)`` and never abort the whole
    search: a single blocked API should not stop discovery.
    """
    active_sources = [source for source in sources if source in _SEARCHERS]
    if not active_sources or not query.strip():
        return []

    source_options = source_options or {}
    results: List[AcademicPaper] = []
    with ThreadPoolExecutor(max_workers=min(len(active_sources), 6)) as executor:
        future_to_source = {}
        for source in active_sources:
            options = source_options.get(source) or model_config.SourceOptions()
            future_to_source[
                executor.submit(
                    search_source,
                    source,
                    query,
                    options.max_results or limit_per_source,
                    cutoff_date,
                    options.timeout or timeout,
                    options,
                )
            ] = source

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                source_results = future.result()
                for position, paper in enumerate(source_results, start=1):
                    paper.best_rank = min(paper.best_rank, position)
                results.extend(source_results)
            except Exception as exc:
                message = f"{source} search failed for '{query[:60]}': {exc}"
                if on_error is not None:
                    on_error(source, message)
                else:
                    _warn_once(message)

    return [
        paper
        for paper in _merge_records(results)
        if _passes_cutoff(paper, cutoff_date)
    ]
