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

"""Optional third-party model endpoint configuration.

PaperOrchestra uses three logical model roles:

* ``main``  - the writer/reflection/plotting LLMs.
* ``light`` - lightweight tasks currently backed by ``gemini-3-flash-preview``.
* ``image`` - the image generation model used by the plotting agent.

Each role can be pointed at a third-party (or self-hosted) endpoint through
``model_config.yaml``.  When a role is disabled, missing, or the config file is
absent, the original Gemini/OpenAI behaviour configured through ``.env`` and the
CLI flags is preserved.

A non-role ``literature_discovery`` section selects the academic databases
used when the ``light`` role is enabled; otherwise Gemini + Google Search is
used for discovery.
"""

import os
import re
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - only triggered on broken installs
    yaml = None

try:
    import dotenv

    dotenv.load_dotenv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
    )
except ImportError:  # pragma: no cover - python-dotenv is in requirements
    pass

MAIN_ROLE = "main"
LIGHT_ROLE = "light"
IMAGE_ROLE = "image"

VALID_ROLES = (MAIN_ROLE, LIGHT_ROLE, IMAGE_ROLE)
VALID_API_STYLES = ("openai", "gemini")

SEMANTIC_SCHOLAR_SECTION = "semantic_scholar"
LITERATURE_DISCOVERY_SECTION = "literature_discovery"

# Top-level YAML sections that are not model roles.
NON_ROLE_SECTIONS = {SEMANTIC_SCHOLAR_SECTION, LITERATURE_DISCOVERY_SECTION}

DEFAULT_DISCOVERY_SOURCES = ("openalex", "crossref", "arxiv", "dblp")
VALID_DISCOVERY_SOURCES = (
    "openalex",
    "crossref",
    "arxiv",
    "dblp",
    "semantic_scholar",
    "pubmed",
)


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_config.yaml",
)

_CONFIG_PATH_ENV = "PAPER_ORCHESTRA_CONFIG"

# Fallback environment variables used when ``api_key`` is empty in the YAML.
_ROLE_API_KEY_ENV = {
    MAIN_ROLE: "MAIN_API_KEY",
    LIGHT_ROLE: "LIGHT_API_KEY",
    IMAGE_ROLE: "IMAGE_API_KEY",
}

_ENV_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_TRUE_STRINGS = {"true", "1", "yes", "on", "enabled"}


def _mask_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


@dataclass(frozen=True)
class ModelEndpoint:
    """Resolved configuration for one logical model role."""

    role: str
    base_url: str
    model_name: str
    api_key: str
    api_style: str = "openai"

    @property
    def masked_api_key(self) -> str:
        return _mask_secret(self.api_key)


@dataclass(frozen=True)
class SourceOptions:
    """Dedicated settings for one literature-discovery database."""

    max_results: Optional[int] = None
    timeout: Optional[float] = None
    min_interval: Optional[float] = None
    mailto: str = ""
    email: str = ""
    tool: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class LiteratureDiscoveryConfig:
    """How the literature agent discovers candidate papers.

    Academic discovery is active whenever the ``light`` role is enabled: that
    role is only used for tool-free query planning and ranking, so any
    OpenAI-compatible endpoint is enough.  This section only selects and tunes
    the databases that are queried.
    """

    sources: tuple = DEFAULT_DISCOVERY_SOURCES
    timeout: float = 20.0
    max_results_per_source: int = 10
    max_candidates: int = 40
    max_queries: int = 3
    source_options: Dict[str, SourceOptions] = field(default_factory=dict)


_config_lock = threading.Lock()
_config_cache: Optional[Dict[str, Optional[ModelEndpoint]]] = None
_config_cache_path: Optional[str] = None
_raw_config_cache: Optional[dict] = None
_raw_config_cache_path: Optional[str] = None


def get_config_path() -> str:
    """Returns the YAML config path, honouring PAPER_ORCHESTRA_CONFIG."""
    configured_path = os.environ.get(_CONFIG_PATH_ENV)
    path = configured_path or DEFAULT_CONFIG_PATH
    return os.path.abspath(os.path.expanduser(path))


def _expand_env_refs(value, field_name: str, required: bool = True):
    """Expands ``${VAR}`` references.

    When ``required`` is False, references to unset variables expand to an empty
    string instead of raising (useful for optional settings such as API keys).
    """
    if not isinstance(value, str):
        return value

    def _replace(match: "re.Match[str]") -> str:
        env_name = match.group(1)
        if env_name not in os.environ:
            if required:
                raise ValueError(
                    f"model_config.yaml: environment variable '{env_name}' "
                    f"referenced by '{field_name}' is not set."
                )
            return ""
        return os.environ[env_name]

    return _ENV_REF_PATTERN.sub(_replace, value).strip()


def _is_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return bool(value)


def _build_endpoint(role: str, raw_config) -> Optional[ModelEndpoint]:
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"model_config.yaml: role '{role}' must be a mapping with "
            "'enabled', 'base_url', 'model_name' and 'api_key'."
        )

    if not _is_enabled(raw_config.get("enabled", False)):
        return None

    base_url = _expand_env_refs(raw_config.get("base_url", ""), f"{role}.base_url")
    model_name = _expand_env_refs(
        raw_config.get("model_name", ""), f"{role}.model_name"
    )
    api_key = _expand_env_refs(raw_config.get("api_key", ""), f"{role}.api_key")
    api_style = _expand_env_refs(
        raw_config.get("api_style", "openai"), f"{role}.api_style"
    )
    api_style = (api_style or "openai").lower()

    if not api_key:
        api_key = os.environ.get(_ROLE_API_KEY_ENV[role], "")

    missing = []
    if not base_url:
        missing.append("base_url")
    if not model_name:
        missing.append("model_name")
    if not api_key:
        missing.append("api_key")
    if missing:
        raise ValueError(
            f"model_config.yaml: role '{role}' is enabled but missing "
            f"{', '.join(missing)}. Set the value directly, reference an "
            f"environment variable with ${{VAR}}, or set "
            f"{_ROLE_API_KEY_ENV[role]} for the API key."
        )

    if api_style not in VALID_API_STYLES:
        raise ValueError(
            f"model_config.yaml: role '{role}' has unsupported api_style "
            f"'{api_style}'. Use one of: {', '.join(VALID_API_STYLES)}."
        )

    return ModelEndpoint(
        role=role,
        base_url=base_url.rstrip("/"),
        model_name=model_name,
        api_key=api_key,
        api_style=api_style,
    )


def _load_raw_config(force_reload: bool = False) -> dict:
    """Loads and caches the raw YAML mapping."""
    global _raw_config_cache, _raw_config_cache_path

    config_path = get_config_path()
    with _config_lock:
        if (
            _raw_config_cache is not None
            and not force_reload
            and _raw_config_cache_path == config_path
        ):
            return _raw_config_cache

    if not os.path.exists(config_path):
        raw_config = {}
    else:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to read model_config.yaml. "
                "Install it with `pip install pyyaml`."
            )

        with open(config_path, "r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}

        if not isinstance(raw_config, dict):
            raise ValueError(
                f"{config_path} must contain a YAML mapping with roles "
                f"({', '.join(VALID_ROLES)}) and optional sections "
                f"({', '.join(sorted(NON_ROLE_SECTIONS))})."
            )

        unknown_roles = set(raw_config) - set(VALID_ROLES) - NON_ROLE_SECTIONS
        if unknown_roles:
            print(
                f"WARNING: model_config.yaml: ignoring unknown role(s): "
                f"{', '.join(sorted(unknown_roles))}"
            )

    with _config_lock:
        _raw_config_cache = raw_config
        _raw_config_cache_path = config_path
    return raw_config


def load_model_config(force_reload: bool = False) -> Dict[str, Optional[ModelEndpoint]]:
    """Loads and caches the model endpoint configuration."""
    global _config_cache, _config_cache_path

    config_path = get_config_path()
    with _config_lock:
        if (
            _config_cache is not None
            and not force_reload
            and _config_cache_path == config_path
        ):
            return _config_cache

    raw_config = _load_raw_config(force_reload=force_reload)
    endpoints = {
        role: _build_endpoint(role, raw_config.get(role)) for role in VALID_ROLES
    }

    with _config_lock:
        _config_cache = endpoints
        _config_cache_path = config_path
    return endpoints


def get_semantic_scholar_api_key() -> str:
    """Returns the Semantic Scholar API key from YAML or the environment.

    The key is optional: anonymous requests work but share a heavily
    rate-limited pool. Precedence is the nested per-source section
    (``literature_discovery.semantic_scholar.api_key``), then the top-level
    ``semantic_scholar.api_key`` used by the enrichment path, then
    ``SEMANTIC_SCHOLAR_API_KEY`` from ``.env``/the environment.
    """
    raw_config = _load_raw_config()

    discovery = raw_config.get(LITERATURE_DISCOVERY_SECTION)
    if isinstance(discovery, dict):
        nested = discovery.get("semantic_scholar")
        if isinstance(nested, dict):
            nested_key = _expand_env_refs(
                nested.get("api_key", "") or "",
                "literature_discovery.semantic_scholar.api_key",
                required=False,
            )
            if nested_key:
                return nested_key

    section = raw_config.get(SEMANTIC_SCHOLAR_SECTION)
    if section is None:
        return os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    if not isinstance(section, dict):
        raise ValueError(
            "model_config.yaml: 'semantic_scholar' must be a mapping with an "
            "'api_key' field."
        )

    api_key = _expand_env_refs(
        section.get("api_key", ""),
        "semantic_scholar.api_key",
        required=False,
    )
    return api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")


def _parse_discovery_sources(value) -> tuple:
    if value is None:
        return DEFAULT_DISCOVERY_SOURCES
    if isinstance(value, str):
        candidates = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        candidates = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError(
            "model_config.yaml: 'literature_discovery.sources' must be a list "
            "of database names or a comma-separated string."
        )

    sources = []
    for source in candidates:
        source = source.lower()
        if source not in VALID_DISCOVERY_SOURCES:
            raise ValueError(
                f"model_config.yaml: 'literature_discovery.sources' contains "
                f"unsupported source '{source}'. Use one of: "
                f"{', '.join(VALID_DISCOVERY_SOURCES)}."
            )
        if source not in sources:
            sources.append(source)

    return tuple(sources) or DEFAULT_DISCOVERY_SOURCES


def _positive_int(value, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"model_config.yaml: 'literature_discovery.{field_name}' must be "
            "an integer."
        )
    return parsed if parsed > 0 else default


def _optional_positive_int(value, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"model_config.yaml: '{field_name}' must be an integer."
        )
    if parsed <= 0:
        raise ValueError(
            f"model_config.yaml: '{field_name}' must be greater than 0."
        )
    return parsed


def _optional_positive_float(value, field_name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"model_config.yaml: '{field_name}' must be a number."
        )
    if parsed <= 0:
        raise ValueError(
            f"model_config.yaml: '{field_name}' must be greater than 0."
        )
    return parsed


def _parse_source_options(source: str, raw_options) -> SourceOptions:
    if raw_options is None:
        return SourceOptions()
    if not isinstance(raw_options, dict):
        raise ValueError(
            f"model_config.yaml: 'literature_discovery.{source}' must be a "
            "mapping of source-specific settings."
        )

    prefix = f"literature_discovery.{source}"
    return SourceOptions(
        max_results=_optional_positive_int(
            raw_options.get("max_results"), f"{prefix}.max_results"
        ),
        timeout=_optional_positive_float(
            raw_options.get("timeout"), f"{prefix}.timeout"
        ),
        min_interval=_optional_positive_float(
            raw_options.get("min_interval"), f"{prefix}.min_interval"
        ),
        mailto=_expand_env_refs(
            raw_options.get("mailto", "") or "",
            f"{prefix}.mailto",
            required=False,
        ),
        email=_expand_env_refs(
            raw_options.get("email", "") or "",
            f"{prefix}.email",
            required=False,
        ),
        tool=str(raw_options.get("tool") or "").strip(),
        api_key=_expand_env_refs(
            raw_options.get("api_key", "") or "",
            f"{prefix}.api_key",
            required=False,
        ),
    )


def _source_env_fallbacks(source: str, options: SourceOptions) -> SourceOptions:
    """Fills optional identity fields from the per-source environment vars."""
    updates = {}
    if source == "openalex" and not options.mailto:
        updates["mailto"] = os.environ.get("OPENALEX_MAILTO", "")
    elif source == "crossref" and not options.mailto:
        updates["mailto"] = os.environ.get("CROSSREF_MAILTO", "")
    elif source == "semantic_scholar" and not options.api_key:
        updates["api_key"] = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    elif source == "pubmed":
        if not options.email:
            updates["email"] = os.environ.get("PUBMED_EMAIL", "")
        if not options.api_key:
            updates["api_key"] = os.environ.get(
                "NCBI_API_KEY", ""
            ) or os.environ.get("PUBMED_API_KEY", "")
        if not options.tool:
            updates["tool"] = "paper_orchestra"

    return replace(options, **updates) if updates else options


def get_literature_discovery_config() -> LiteratureDiscoveryConfig:
    """Returns the literature-discovery settings from ``model_config.yaml``.

    The section is optional and only selects/tunes databases.  Whether
    academic discovery is used at all is decided by the ``light`` role: an
    enabled third-party ``light`` endpoint means no Gemini + Google Search is
    available, so discovery goes through these APIs instead.
    """
    raw_config = _load_raw_config()
    section = raw_config.get(LITERATURE_DISCOVERY_SECTION)

    if section is None:
        return LiteratureDiscoveryConfig()

    if not isinstance(section, dict):
        raise ValueError(
            "model_config.yaml: 'literature_discovery' must be a mapping with "
            "'sources', optional tuning fields and per-source sections."
        )

    sources = _parse_discovery_sources(section.get("sources"))
    source_options = {}
    for source in VALID_DISCOVERY_SOURCES:
        options = _parse_source_options(source, section.get(source))
        if source == "semantic_scholar" and not options.api_key:
            # The S2 key is shared with the enrichment path; keep accepting
            # the legacy top-level `semantic_scholar` section as well.
            options = replace(options, api_key=get_semantic_scholar_api_key())
        source_options[source] = _source_env_fallbacks(source, options)

    timeout_value = section.get("timeout", 20.0)
    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError):
        raise ValueError(
            "model_config.yaml: 'literature_discovery.timeout' must be a number "
            "of seconds."
        )
    if timeout <= 0:
        timeout = 20.0

    return LiteratureDiscoveryConfig(
        sources=sources,
        timeout=timeout,
        max_results_per_source=_positive_int(
            section.get("max_results_per_source"),
            "max_results_per_source",
            10,
        ),
        max_candidates=_positive_int(
            section.get("max_candidates"), "max_candidates", 40
        ),
        max_queries=_positive_int(section.get("max_queries"), "max_queries", 3),
        source_options=source_options,
    )


def reload_model_config() -> Dict[str, Optional[ModelEndpoint]]:
    """Forces the next access to re-read the YAML file."""
    return load_model_config(force_reload=True)


def get_endpoint(role: str) -> Optional[ModelEndpoint]:
    """Returns the endpoint for a role, or ``None`` when it is disabled."""
    return load_model_config().get(role)


def uses_third_party(role: str) -> bool:
    """Whether a third-party endpoint is enabled for the given role."""
    return get_endpoint(role) is not None


def resolve_backend(model_name: str, role: str = MAIN_ROLE) -> str:
    """Resolves which backend a call should use.

    Returns one of ``third_party``, ``openai`` or ``gemini``.  A configured
    third-party endpoint always takes precedence over the model-name heuristic.
    """
    if uses_third_party(role):
        return "third_party"

    name = (model_name or "").lower()
    if "gpt" in name or "o1" in name or "o3" in name:
        return "openai"
    return "gemini"


def print_active_endpoints() -> None:
    """Prints the enabled third-party endpoints and integration status."""
    endpoints = load_model_config()
    active = {role: endpoint for role, endpoint in endpoints.items() if endpoint}

    if active:
        print("Third-party model endpoints:")
        for role in VALID_ROLES:
            endpoint = active.get(role)
            if endpoint is None:
                continue
            print(
                f"  {role:<5} -> {endpoint.base_url} | model={endpoint.model_name} "
                f"| style={endpoint.api_style} | api_key={endpoint.masked_api_key}"
            )
    else:
        print(
            "No third-party model endpoints enabled "
            "(using .env / CLI model names)."
        )

    semantic_scholar_key = get_semantic_scholar_api_key()
    if semantic_scholar_key:
        print(f"Semantic Scholar API key: {_mask_secret(semantic_scholar_key)}")
    else:
        print(
            "Semantic Scholar API key: not set "
            "(anonymous access, stricter rate limits)."
        )

    discovery = get_literature_discovery_config()
    if uses_third_party(LIGHT_ROLE):
        print(
            "Literature discovery: academic APIs "
            f"({', '.join(discovery.sources)})"
        )
    else:
        print("Literature discovery: Google Search grounding (Gemini).")

