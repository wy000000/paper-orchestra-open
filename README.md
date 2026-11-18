# <div align="center">🎻 PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing</div>
<div align="center">Yiwen Song<sup>1</sup>, Yale Song<sup>1</sup>, Tomas Pfister<sup>1</sup>, and Jinsung Yoon<sup>1</sup></div>
<div align="center"><sup>1</sup>Google Cloud AI Research</div>
<br><br>

PaperOrchestra is a multi-agent framework for automated AI research paper writing. It flexibly transforms unconstrained pre-writing materials (ideas and experimental logs) into submission-ready LaTeX manuscripts, including comprehensive literature synthesis and generated visuals, such as plots and conceptual diagrams.

Acting like an orchestrated team of specialized agents, it handles outline generation, literature review, section writing, content refinement, and plotting to close the gap between raw research materials and submission ready papers.

<div align="center">
  <img src="assets/overview.png" alt="PaperOrchestra Overview" width="90%"/>
</div>

## Project Origin

This repository is a modified version of [PaperOrchestra](https://github.com/google-research/paper-orchestra) by Google Cloud AI Research, the codebase behind the paper *PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing* (arXiv:2604.05018). Credit for the original design, agents, prompts, and templates belongs to the original authors; see [Citation](#citation).

A summary of the modifications made in this fork is provided in [Changes from the Original](#changes-from-the-original).

## Changes from the Original

This fork keeps the original multi-agent pipeline, prompts, and LaTeX templates intact. The changes are limited to infrastructure and configuration:

| Area | Change |
|------|--------|
| Third-party model endpoints | Added an optional `model_config.yaml` mechanism (new `utils/model_config.py` and `model_config.example.yaml`) that routes the `main`, `light`, and `image` model roles to third-party or self-hosted APIs. Supports `openai` and `gemini` API styles, `${ENV_VAR}` references in API keys, and the `PAPER_ORCHESTRA_CONFIG` environment variable for a config file outside the project root. |
| LLM call routing | Updated `utils/gemini_utils.py`, `utils/llm_backend_utils.py`, and `utils/openai_utils.py` so calls are dispatched through the configured endpoint. Calls that require Google Search grounding automatically fall back to native Gemini (with a warning) when the `light` role uses an OpenAI-compatible endpoint. |
| Lightweight role wiring | The `light` role now powers candidate paper discovery in `methods/agents/literature_review_agent.py` and citation-title extraction in `utils/content_parsing_utils.py`. |
| Academic-database discovery | New `utils/academic_search.py` queries OpenAlex, Crossref, arXiv, DBLP, Semantic Scholar and PubMed directly, so literature discovery can run without Google Search grounding. It is used automatically when the `light` role is enabled; the `literature_discovery` section selects sources. Records come from the APIs; the `light` model only plans queries and ranks candidates. |
| Semantic Scholar API key | The key can now be supplied via `model_config.yaml` (`semantic_scholar.api_key`) in addition to `.env`; removed the startup log lines that printed the key (`utils/scholar_utils.py`, `autoraters/citation_f1.py`). |
| Image MIME detection | `utils/paper_banana_utils.py` now detects image MIME types from magic bytes (PNG/JPEG/GIF/WebP/BMP) instead of hard-coding `image/jpeg`. |
| CLI startup validation | `paper_writing_cli.py` validates the config and prints the active endpoints on startup, exiting early on invalid config. |
| Dependencies | Added `PyYAML` to `requirements.txt`. |
| Repository hygiene | `.gitignore` now excludes `model_config.yaml`, `*.exe`, `doc/`, and `runs/`; local sample outputs and the bundled `pandoc.exe` were removed from version control. |
| Documentation | This README documents the third-party endpoint setup under [Installation](#installation). |

When no config file is present, or all roles are disabled, behaviour is identical to the original repository: keys and models still come from `.env` and the CLI flags. See [Installation](#installation) for configuration details.

## Key Features

- **Multi-Agent Pipeline**: Orchestrates specialized agents (Outline, Literature Review, Section Writing, Content Refinement, Plotting) for end-to-end paper generation.
- **Comprehensive Literature Synthesis**: Automatically searches, prioritizes, and integrates relevant literature into the manuscript.
- **Visuals Generation**: Generates plots and supports the inclusion of conceptual diagrams.
- **LaTeX Automation**: Produces submission-ready LaTeX manuscripts adhering to target conference templates.

## Installation

1.  **Environment Setup**: The project relies on a Conda environment named `paper_orchestra`. You can set up a similar environment with Python 3.11 and install the required dependencies.
    ```bash
    conda create -n paper_orchestra python=3.11
    conda activate paper_orchestra
    pip install -r requirements.txt
    ```

2.  **Environment Variables**: Create a `.env` file in the root of the project and add your API keys:
    ```env
    # Add keys if you have them, otherwise leave them empty
    # You need to specify related keys if you want to use the model
    # For example, if you want to use Gemini models, you need to specify either VERTEX_AI_PROJECT and VERTEX_AI_LOCATION, or GEMINI_API_KEY
    # If you want to use OpenAI models, you need to specify OPENAI_API_KEY
    # SEMANTIC_SCHOLAR_API_KEY is optional and can also be set in model_config.yaml
    
    SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
    OPENAI_API_KEY=your_openai_api_key
    VERTEX_AI_PROJECT=your_vertex_ai_project
    VERTEX_AI_LOCATION=your_vertex_ai_location
    GEMINI_API_KEY=your_gemini_api_key

    # if you want to run demo and send emails to yourself at the end, set these 2 fields as well
    SMTP_EMAIL=your_email
    SMTP_PASSWORD=your_password
    ```

3.  **Optional: Third-party Model Endpoints and API Settings**: PaperOrchestra can route its three logical model roles to third-party or self-hosted APIs. Copy the template and edit it:
    ```bash
    cp model_config.example.yaml model_config.yaml
    ```

    ```yaml
    main:   # writer / reflection / plotting LLMs
      enabled: true
      base_url: "https://api.example.com/v1"
      model_name: "gpt-5"
      api_key: "${MAIN_API_KEY}"
      api_style: "openai"

    light:  # lightweight tasks (flash-level calls)
      enabled: false
      base_url: "https://api.example.com/v1"
      model_name: "gpt-5-mini"
      api_key: "${LIGHT_API_KEY}"
      api_style: "openai"

    image:  # plotting agent image generation
      enabled: false
      base_url: "https://api.example.com/v1"
      model_name: "gpt-image-1"
      api_key: "${IMAGE_API_KEY}"
      api_style: "openai"

    literature_discovery:  # used whenever the `light` role is enabled
      sources: ["openalex", "crossref", "arxiv", "dblp"]  # | semantic_scholar | pubmed
      timeout: 20              # global defaults, applied to every source
      max_results_per_source: 10
      max_candidates: 40
      max_queries: 3

      openalex:                # no key required
        mailto: "${OPENALEX_MAILTO}"   # optional polite-pool email
        max_results: 10
        timeout: 20
        min_interval: 0.12
      crossref:                # no key required
        mailto: "${CROSSREF_MAILTO}"   # optional polite-pool email
        max_results: 10
        timeout: 20
        min_interval: 0.30
      arxiv:                   # no key required
        max_results: 10
        timeout: 25
        min_interval: 3.0
      dblp:                    # no key required
        max_results: 10
        timeout: 20
        min_interval: 1.0
      semantic_scholar:        # optional but recommended key (HTTP 429)
        api_key: "${SEMANTIC_SCHOLAR_API_KEY}"
        max_results: 10
        timeout: 20
        min_interval: 1.1
      pubmed:                  # no key required
        email: "${PUBMED_EMAIL}"    # optional contact info
        tool: "paper_orchestra"
        api_key: "${NCBI_API_KEY}"  # optional; 3 -> 10 requests/second
        max_results: 10
        timeout: 20
        min_interval: 0.40

    semantic_scholar:  # optional; anonymous access works with stricter limits
      api_key: "${SEMANTIC_SCHOLAR_API_KEY}"
    ```

    * `model_config.yaml` is gitignored; only the example file is committed.
    * `api_key` supports `${ENV_VAR}` references. If it is empty, the loader falls back to `MAIN_API_KEY`, `LIGHT_API_KEY`, or `IMAGE_API_KEY`.
    * `api_style` is optional and defaults to `openai`. Use `gemini` for Gemini-compatible endpoints (include the API version in `base_url`, e.g. `https://relay.example.com/v1beta`).
    * Disabled roles or a missing file keep the original `.env` / CLI behaviour.
    * When a role is enabled, its `model_name` overrides the corresponding CLI model flag.
    * The `semantic_scholar.api_key` value takes precedence over `SEMANTIC_SCHOLAR_API_KEY` in `.env`; `literature_discovery.semantic_scholar.api_key` (if set) takes precedence over both. Anonymous Semantic Scholar access works but is heavily rate-limited; a free key is recommended.
    * The `light` role currently powers two kinds of calls: candidate paper discovery in the literature agent (`methods/agents/literature_review_agent.py`) and citation-title extraction in `autoraters/citation_f1.py`. It does not write, reflect, plot, evaluate PDFs, or generate images.
    * **The discovery backend is decided by `light`.** With `light.enabled: false`, discovery uses native Gemini + Google Search grounding (the original behaviour). With `light.enabled: true`, discovery queries academic databases instead, and `light` is only used for tool-free query planning and ranking. In that setup no Gemini/Vertex credentials are needed, and `api_style: openai` works for any OpenAI-compatible endpoint.
    * **Academic-database discovery.** The `literature_discovery` section only selects and tunes the databases; it has no on/off switch. Titles, authors, venues, abstracts and DOIs come from the APIs, not from the LLM. Sources (all free; only Semantic Scholar has an optional key):
      * `openalex` — multidisciplinary, broad coverage, abstracts + citation counts; set `mailto` for the polite pool.
      * `crossref` — authoritative DOI metadata for journals/conferences; abstracts are patchy, may be slow or return HTTP 429.
      * `arxiv` — CS/physics/math preprints, very recent, abstracts; no citation counts, ~3s between requests.
      * `dblp` — most accurate CS conference/journal bibliography; no abstracts or citation counts, may be blocked by a bot challenge (skipped with a warning).
      * `semantic_scholar` — multidisciplinary, abstracts + citation counts; anonymous access is heavily rate-limited, so a free key is recommended.
      * `pubmed` — biomedical/life sciences; abstracts, DOI and PMID. Skip it for non-biomedical topics.
    * **Each source has its own optional block** (same key as its name under `literature_discovery`). Every block accepts `max_results`, `timeout` and `min_interval` (seconds between requests); `mailto` is only used by `openalex`/`crossref`, `email`/`tool`/`api_key` by `pubmed`, and `api_key` by `semantic_scholar`. Global `timeout`, `max_results_per_source`, `max_candidates` and `max_queries` act as defaults/limits. Environment fallbacks: `OPENALEX_MAILTO`, `CROSSREF_MAILTO`, `SEMANTIC_SCHOLAR_API_KEY`, `PUBMED_EMAIL`, `NCBI_API_KEY` (or `PUBMED_API_KEY`).
    * Set `PAPER_ORCHESTRA_CONFIG=/path/to/file.yaml` to use a config file outside the project root.

## Dataset

The dataset used for paper generation is **not included** in this code repository. It will be released separately at a later date.
Once available, you should download the dataset and place it under the `datasets/` directory.

## Usage

You can run the paper generation pipeline using the provided bash script or directly via the Python CLI.

### Using the Bash Script
```bash
./paper_writing_cli.sh
```

### Using the Python CLI directly
```bash
python paper_writing_cli.py \
  --raw_materials_dir /path_to_your_raw_materials/ \
  --latex_template_dir /path_to_your_latex_template/ \
  --output_dir /path_to_your_output_folder/ \
  --use_plotting true
```

### Key Arguments
*   `--raw_materials_dir`: (Required) Directory containing the raw materials (e.g., idea and experimental log).
*   `--latex_template_dir`: (Required) Directory containing the LaTeX template (e.g., `templates/cvpr2025`).
*   `--output_dir`: Directory to output the generated paper. If not specified, an automatic folder with a timestamp is created.
*   `--use_plotting`: Enable the plotting agent workflow to generate figures from code (Default: `False`).
*   `--writer_model_name`: LLM for writer and literature agents.
*   `--reflection_model_name`: LLM for reflection agents.

### Interactive Demo

We provide an interactive Streamlit demo for the paper generation pipeline. For instructions on how to run and use it, please refer to the [README.md](frontend/README.md) in the `frontend` directory.

## Project Structure

*   `autoraters/`: Contains scripts for automated evaluation.
*   `methods/`: Core logic for agents and pipeline execution.
*   `templates/`: LaTeX templates for target conferences, you can add your own template by creating a new subfolder under this directory.
*   `utils/`: Utility functions for API calls, PDF handling, etc.

## Citation

If you find this repo or our paper helpful, please cite it as follows:

```bibtex
@article{song2026paperorchestra,
  title={PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing},
  author={Song, Yiwen and Song, Yale and Pfister, Tomas and Yoon, Jinsung},
  journal={arXiv preprint arXiv:2604.05018},
  year={2026}
}
```

# Disclaimer

This is not an officially supported Google product.
