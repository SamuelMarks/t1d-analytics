"""Centralized LLM provider configuration, prompt construction, and streaming helpers."""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from typing_extensions import TypedDict, TypeGuard

from t1d_analytics.api_schemas import (
    ModelInfo,
    ModelsResponse,
    ProvidersStatusResponse,
    ProviderStatus,
)
from t1d_analytics.errors import MissingApiKeyError, MissingSdkError

logger = logging.getLogger(__name__)


@runtime_checkable
class _TextPartProtocol(Protocol):
    """Protocol for a Google/Gemini content part containing text."""

    @property
    def text(self) -> Optional[str]:
        """Return text fragment."""
        ...


@runtime_checkable
class _CandidateContentProtocol(Protocol):
    """Protocol for a Google/Gemini candidate content block."""

    @property
    def parts(self) -> Optional[Sequence[_TextPartProtocol]]:
        """Return sequence of content parts."""
        ...


@runtime_checkable
class _CandidateProtocol(Protocol):
    """Protocol for a Google/Gemini streaming candidate."""

    @property
    def content(self) -> Optional[_CandidateContentProtocol]:
        """Return candidate content block."""
        ...


@runtime_checkable
class _DeltaTextProtocol(Protocol):
    """Protocol for a streaming delta or choice item exposing `text`."""

    @property
    def text(self) -> Optional[str]:
        """Return delta or choice text."""
        ...


@runtime_checkable
class _DeltaContentProtocol(Protocol):
    """Protocol for a streaming delta or message item exposing `content`."""

    @property
    def content(self) -> Optional[str]:
        """Return delta or message content."""
        ...


@runtime_checkable
class _ChoiceWithDeltaProtocol(Protocol):
    """Protocol for a streaming choice item exposing `delta`."""

    @property
    def delta(
        self,
    ) -> Optional[Union[_DeltaContentProtocol, _DeltaTextProtocol]]:
        """Return choice delta."""
        ...


@runtime_checkable
class _ChoiceWithMessageProtocol(Protocol):
    """Protocol for a streaming choice item exposing `message`."""

    @property
    def message(self) -> Optional[_DeltaContentProtocol]:
        """Return choice message."""
        ...


@runtime_checkable
class OpenAiChunkProtocol(Protocol):
    """Protocol for an OpenAI/Ollama/any-llm streaming chunk exposing `choices`."""

    @property
    def choices(
        self,
    ) -> Optional[
        Sequence[
            Union[
                _ChoiceWithDeltaProtocol,
                _ChoiceWithMessageProtocol,
                _DeltaTextProtocol,
            ]
        ]
    ]:
        """Return sequence of streaming or completion choices."""
        ...


@runtime_checkable
class GoogleChunkProtocol(Protocol):
    """Protocol for a Google/Gemini streaming chunk exposing `candidates`."""

    @property
    def candidates(self) -> Optional[Sequence[_CandidateProtocol]]:
        """Return sequence of streaming candidates."""
        ...


@runtime_checkable
class AnthropicChunkProtocol(Protocol):
    """Protocol for an Anthropic streaming chunk exposing `delta`."""

    @property
    def delta(
        self,
    ) -> Optional[Union[_DeltaTextProtocol, _DeltaContentProtocol]]:
        """Return top-level streaming delta."""
        ...


DirectTextChunkProtocol = _DeltaTextProtocol
OpenAiStreamChunkProtocol = OpenAiChunkProtocol
GeminiStreamChunkProtocol = GoogleChunkProtocol
AnthropicStreamChunkProtocol = AnthropicChunkProtocol
LlmStreamChunkProtocol = OpenAiChunkProtocol
LlmDeltaChoicesContainerProtocol = OpenAiChunkProtocol
LlmMessageChoicesContainerProtocol = OpenAiChunkProtocol
GeminiCandidateChunkProtocol = GoogleChunkProtocol
AnthropicDeltaChunkProtocol = AnthropicChunkProtocol


@runtime_checkable
class _HttpStatusProtocol(Protocol):
    """Protocol for an HTTP response exposing a `status` integer code."""

    @property
    def status(self) -> int:
        """Return HTTP status code."""
        ...


class _UrlResponseProtocol(Protocol):
    """Protocol for an HTTP response context manager returned by `urlopen`."""

    def read(self) -> bytes:
        """Read response body bytes."""
        ...

    def __enter__(self) -> "_UrlResponseProtocol":
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[
            Union[
                str,
                int,
                float,
                bool,
                None,
            ]
        ],
    ) -> Optional[bool]:
        """Exit context manager."""
        ...


class _AnyLlmModelProtocol(Protocol):
    """Protocol for a model descriptor returned by any_llm.list_models."""

    @property
    def id(self) -> str:
        """Return model identifier."""
        ...


class _AnyLlmListModelsFn(Protocol):
    """Callable protocol for any_llm.list_models."""

    def __call__(
        self, provider: str, api_key: Optional[str] = None
    ) -> Sequence[_AnyLlmModelProtocol]:
        """Return models for the specified provider."""
        ...


StreamScalar = Union[str, int, float, bool, None]


class StreamDeltaMappingDict(TypedDict, total=False):
    """Typed dictionary for a nested `delta` or `message` mapping inside a stream chunk."""

    content: Optional[str]
    text: Optional[str]
    role: Optional[str]


class StreamChoiceMappingDict(TypedDict, total=False):
    """Typed dictionary for a choice item inside a `choices` sequence."""

    delta: StreamDeltaMappingDict
    message: StreamDeltaMappingDict
    text: Optional[str]
    index: int


StreamChoiceElement = Union[
    StreamScalar,
    StreamChoiceMappingDict,
    Mapping[str, Union[StreamScalar, Mapping[str, StreamScalar]]],
]

StreamMappingChunk = Mapping[
    str,
    Union[
        StreamScalar,
        Sequence[StreamChoiceElement],
        Mapping[str, StreamScalar],
    ],
]


@runtime_checkable
class LlmCompletionMessageProtocol(Protocol):
    """Protocol for a message object inside an LLM completion choice."""

    @property
    def content(self) -> Optional[str]:
        """Return message text content."""
        ...


@runtime_checkable
class LlmCompletionChoiceProtocol(Protocol):
    """Protocol for a completion choice object exposing a `message`."""

    @property
    def message(self) -> LlmCompletionMessageProtocol:
        """Return completion message."""
        ...


@runtime_checkable
class LlmCompletionResponseProtocol(Protocol):
    """Protocol for a synchronous LLM completion response exposing `choices`."""

    @property
    def choices(self) -> Sequence[LlmCompletionChoiceProtocol]:
        """Return completion choices."""
        ...


ObjectChunkInput = Union[
    LlmCompletionResponseProtocol,
    OpenAiStreamChunkProtocol,
    GeminiStreamChunkProtocol,
    AnthropicStreamChunkProtocol,
    DirectTextChunkProtocol,
]

StreamChunkInput = Union[
    str,
    StreamMappingChunk,
    ObjectChunkInput,
]


@runtime_checkable
class LlmTransportProtocol(Protocol):
    """Protocol for an LLM transport client exposing a `completion` method."""

    def completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        stream: bool = False,
    ) -> Union[LlmCompletionResponseProtocol, StreamChunkInput]:
        """Execute a chat completion call."""
        ...


class AnyLlmFactoryProtocol(Protocol):
    """Callable protocol for creating an `LlmTransportProtocol` for a given provider."""

    def __call__(
        self, provider: str, api_key: Optional[str] = None
    ) -> LlmTransportProtocol:
        """Create and return an LLM transport instance."""
        ...


PROVIDER_ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "cohere": "COHERE_API_KEY",
    "xai": "XAI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "huggingface": "HF_TOKEN",
    "azure": "AZURE_API_KEY",
    "azureopenai": "AZURE_OPENAI_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "vertexai": "GOOGLE_APPLICATION_CREDENTIALS",
    "lmstudio": "LM_STUDIO_API_BASE",
    "llamacpp": "LLAMACPP_API_BASE",
    "vllm": "VLLM_API_BASE",
    "llamafile": "LLAMAFILE_API_BASE",
    "mzai": "ANY_LLM_KEY",
}

PROVIDER_ALT_ENV_KEYS: Dict[str, Tuple[str, ...]] = {
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "huggingface": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
    "azure": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
    "azureopenai": ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_BEARER_TOKEN_BEDROCK"),
    "vertexai": ("GOOGLE_APPLICATION_CREDENTIALS", "VERTEXAI_PROJECT"),
    "lmstudio": ("LM_STUDIO_API_BASE", "LMSTUDIO_API_KEY"),
    "llamacpp": ("LLAMACPP_API_BASE", "LLAMACPP_API_KEY"),
    "vllm": ("VLLM_API_BASE", "VLLM_API_KEY"),
    "llamafile": ("LLAMAFILE_API_BASE", "LLAMAFILE_API_KEY"),
}

CLOUD_MODEL_CATALOG: Dict[str, List[str]] = {
    "openai": ["openai/gpt-4o", "openai/gpt-4o-mini"],
    "anthropic": ["anthropic/claude-3-5-sonnet", "anthropic/claude-3-5-haiku"],
    "google": ["google/gemini-1.5-pro", "google/gemini-1.5-flash"],
}

DEFAULT_CHOICE_OLLAMA_MODELS: Tuple[str, ...] = (
    "gemma:2b",
    "llama3.2:1b",
    "phi4-mini",
)

DEFAULT_CHOICE_HF_REPOS: Tuple[str, ...] = (
    "google/gemma-4-E4B",
    "microsoft/Phi-4-mini-flash-reasoning",
    "meta-llama/Llama-3.2-1B",
)

_FAST_SMALL_OLLAMA_ORDER: Tuple[str, ...] = (
    "gemma:2b",
    "llama3.2:1b",
    "phi4-mini",
    "phi4-mini:latest",
)

_HEAVY_OLLAMA_SUBSTRINGS: Tuple[str, ...] = (
    "26b",
    "27b",
    "32b",
    "70b",
)


def _ollama_model_sort_key(model_name: str) -> Tuple[int, int, str]:
    """
    Compute a deterministic ranking key for an Ollama model name.

    Ranks lightweight CPU-friendly models first (`tier 0`) and heavy 26B/27B/70B
    models last (`tier 2`).

    Args:
    ----
        model_name: Ollama model name or `ollama/<tag>` identifier.

    Returns:
    -------
        Tuple[int, int, str]: `(tier, priority_index, lower_name)` sort key.

    """
    cleaned = model_name.strip().lower()
    if cleaned.startswith("ollama/"):
        cleaned = cleaned[7:]
    if any(heavy in cleaned for heavy in _HEAVY_OLLAMA_SUBSTRINGS):
        return (2, 999, cleaned)
    for idx, preferred in enumerate(_FAST_SMALL_OLLAMA_ORDER):
        if cleaned == preferred or cleaned.startswith(f"{preferred}:"):
            return (0, idx, cleaned)
    return (1, 500, cleaned)


def sort_ollama_models(models: Sequence[str]) -> List[str]:
    """
    Deterministically sort Ollama model names so fast/small models appear first and 26B/27B/70B models appear last.

    Args:
    ----
        models: Sequence of Ollama model identifier strings.

    Returns:
    -------
        List[str]: Ranked list of model names.

    """
    return sorted(models, key=_ollama_model_sort_key)


def is_filesystem_path_token(token: str) -> bool:
    r"""
    Check whether a raw environment variable token looks like a filesystem directory path rather than a model ID.

    Prevents `OLLAMA_MODELS=/mnt/data/ollama_models` or `C:\ollama_models` from being misparsed as a model name.

    Args:
    ----
        token: Candidate model token string.

    Returns:
    -------
        bool: True if the token is a filesystem path, False otherwise.

    """
    stripped = token.strip()
    if not stripped:
        return True
    if stripped.startswith(("/", "\\", ".")):
        return True
    if (
        len(stripped) >= 2
        and stripped[0].isalpha()
        and stripped[1] == ":"
        and (len(stripped) == 2 or stripped[2] in ("\\", "/"))
    ):
        return True
    return "\\" in stripped or (
        stripped.count("/") >= 2
        and not stripped.startswith(
            ("hf.co/", "huggingface/", "together/", "fireworks/", "openrouter/")
        )
    )


def read_configured_ollama_model_list() -> List[str]:
    """
    Read configured Ollama model names from `OLLAMA_MODEL_LIST` (or non-path `OLLAMA_MODELS`).

    Ignores any token in `OLLAMA_MODELS` that looks like a filesystem path (such as `/mnt/data/ollama_models`).

    Returns
    -------
        List[str]: Sanitized list of configured Ollama model names.

    """
    raw = os.environ.get("OLLAMA_MODEL_LIST", "").strip()
    if not raw:
        raw = os.environ.get("OLLAMA_MODELS", "").strip()
    if not raw:
        return []
    models: List[str] = []
    for part in raw.split(","):
        tok = part.strip()
        if tok and not is_filesystem_path_token(tok):
            models.append(tok)
    return models


def resolve_installed_ollama_model_tag(
    requested_model: str,
    installed_models: Sequence[str],
) -> Optional[str]:
    """
    Match a requested Ollama model identifier directly against installed `/api/tags` model names.

    Args:
    ----
        requested_model: Model identifier (e.g., `'gemma:2b'`, `'ollama/gemma:2b'`, `'phi4-mini'`).
        installed_models: Sequence of model names reported by Ollama `/api/tags`.

    Returns:
    -------
        Optional[str]: Exact installed model tag if present in `installed_models`, else None.

    """
    cleaned = requested_model.strip()
    if cleaned.lower().startswith("ollama/"):
        cleaned = cleaned[7:]
    lower_req = cleaned.lower()

    installed_by_lower = {
        m.strip().lower(): m.strip() for m in installed_models if m.strip()
    }
    base_req = lower_req[:-7] if lower_req.endswith(":latest") else lower_req
    for cand in (lower_req, f"{base_req}:latest", base_req):
        if cand in installed_by_lower:
            return installed_by_lower[cand]
    return None


ANY_LLM_PROVIDER_CATALOG: Dict[str, List[str]] = {
    **CLOUD_MODEL_CATALOG,
    "ollama": list(DEFAULT_CHOICE_OLLAMA_MODELS),
    "gemini": ["gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash"],
    "mistral": ["mistral/mistral-large-latest", "mistral/codestral-latest"],
    "groq": ["groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"],
    "deepseek": ["deepseek/deepseek-chat", "deepseek/deepseek-reasoner"],
    "together": ["together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
    "fireworks": ["fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct"],
    "openrouter": ["openrouter/openai/gpt-4o-mini"],
    "cohere": ["cohere/command-r-plus"],
    "xai": ["xai/grok-2-latest"],
    "cerebras": ["cerebras/llama3.1-70b"],
    "sambanova": ["sambanova/Meta-Llama-3.1-70B-Instruct"],
    "perplexity": ["perplexity/sonar-pro"],
    "huggingface": [
        "huggingface/google/gemma-4-E4B",
        "huggingface/microsoft/Phi-4-mini-flash-reasoning",
        "huggingface/meta-llama/Llama-3.2-1B",
        "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
    ],
    "azure": ["azure/gpt-4o"],
    "azureopenai": ["azureopenai/gpt-4o"],
    "bedrock": ["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"],
    "vertexai": ["vertexai/gemini-1.5-pro"],
    "lmstudio": ["lmstudio/local-model"],
    "llamacpp": ["llamacpp/local-model"],
    "vllm": ["vllm/local-model"],
    "llamafile": ["llamafile/local-model"],
    "mzai": ["mzai/gpt-4o-mini"],
}


def resolve_ollama_base_url(
    raw_host: Optional[str] = None,
    default_host: str = "http://127.0.0.1:11434",
) -> str:
    """
    Purely resolve and normalize the Ollama host URL without mutating `os.environ`.

    Strips trailing `/v1` or `/api` path suffixes and replaces `0.0.0.0` bind addresses
    with `127.0.0.1` so client connections always target a routable loopback endpoint.

    Args:
    ----
        raw_host: Optional explicit host URL override.
        default_host: Fallback host URL if neither `raw_host` nor `OLLAMA_HOST` is set.

    Returns:
    -------
        str: Normalized base URL starting with http:// or https:// and without trailing slash.

    """
    candidate = (
        raw_host if raw_host is not None else os.environ.get("OLLAMA_HOST")
    ) or default_host
    host = candidate.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    host = host.replace("://0.0.0.0", "://127.0.0.1")
    for suffix in ("/v1", "/api"):
        if host.lower().endswith(suffix):
            host = host[: -len(suffix)].rstrip("/")
    return host


def configure_ollama_env(
    raw_host: Optional[str] = None,
    default_host: str = "http://127.0.0.1:11434",
) -> str:
    """
    Explicitly initialize `OLLAMA_HOST` and `OLLAMA_API_BASE` in `os.environ` and return the resolved URL.

    Args:
    ----
        raw_host: Optional explicit host URL override.
        default_host: Fallback host URL if neither `raw_host` nor `OLLAMA_HOST` is set.

    Returns:
    -------
        str: Normalized base URL stored in `os.environ`.

    """
    host = resolve_ollama_base_url(raw_host=raw_host, default_host=default_host)
    os.environ["OLLAMA_HOST"] = host
    os.environ["OLLAMA_API_BASE"] = host
    return host


normalize_ollama_host = configure_ollama_env


def resolve_provider_and_model(
    model_name: str, provider: Optional[str] = None
) -> Tuple[str, str]:
    """
    Resolve the LLM provider identifier and model name from a composite model string.

    Supports both 'provider/model' and Mozilla any-llm 'provider:model' syntax when the
    prefix matches a known any-llm provider, as well as 'sql' and 'hf.co/<org>/<repo>'
    Ollama Hugging Face model tags.

    Args:
    ----
        model_name: Model identifier or 'provider/model' string (e.g., 'openai/gpt-4o').
        provider: Optional explicit provider override.

    Returns:
    -------
        Tuple[str, str]: A tuple of (resolved_provider, actual_model).

    """
    cleaned = model_name.strip()
    if cleaned.lower() == "sql" and not provider:
        return ("sql", "sql")
    resolved_provider = provider or "ollama"
    actual_model = cleaned
    if not provider:
        if cleaned.startswith("hf.co/") or cleaned in DEFAULT_CHOICE_HF_REPOS:
            return ("ollama", cleaned)
        if cleaned.startswith("ollama/") and (
            cleaned[7:].startswith("hf.co/") or cleaned[7:] in DEFAULT_CHOICE_HF_REPOS
        ):
            return ("ollama", cleaned[7:])
        if ":" in cleaned and cleaned.split(":", 1)[0].lower() in PROVIDER_ENV_KEYS:
            prefix, remainder = cleaned.split(":", 1)
            resolved_provider, actual_model = prefix.lower(), remainder
        elif "/" in cleaned:
            resolved_provider, actual_model = cleaned.split("/", 1)
    return resolved_provider, actual_model


def is_provider_configured(provider: str) -> bool:
    """
    Check if the specified LLM provider has credentials configured in the environment.

    Args:
    ----
        provider: Provider identifier string (e.g., 'ollama', 'openai', 'anthropic', 'google', 'groq').

    Returns:
    -------
        bool: True if configured, False otherwise.

    """
    prov = provider.lower().strip()
    if prov == "ollama":
        return True
    if prov in PROVIDER_ALT_ENV_KEYS:
        return any(bool(os.environ.get(k)) for k in PROVIDER_ALT_ENV_KEYS[prov])
    if prov in PROVIDER_ENV_KEYS:
        return bool(os.environ.get(PROVIDER_ENV_KEYS[prov]))
    return False


def get_effective_api_key(
    provider: str, api_key: Optional[str] = None
) -> Optional[str]:
    """
    Resolve the effective API key for a cloud provider from explicit argument or environment.

    Args:
    ----
        provider: Provider identifier string.
        api_key: Optional explicit API key override.

    Returns:
    -------
        Optional[str]: Resolved API key or None.

    """
    if api_key:
        return api_key
    prov = provider.lower().strip()
    if prov in PROVIDER_ALT_ENV_KEYS:
        for env_key in PROVIDER_ALT_ENV_KEYS[prov]:
            val = os.environ.get(env_key)
            if val:
                return val
    if prov in PROVIDER_ENV_KEYS:
        return os.environ.get(PROVIDER_ENV_KEYS[prov])
    return None


def validate_provider_readiness(provider: str) -> None:
    """
    Validate that the requested LLM provider has its required SDK and API credentials configured.

    Args:
    ----
        provider: Provider identifier (e.g. 'ollama', 'openai', 'anthropic', 'google').

    Raises:
    ------
        MissingSdkError: If `any-llm-sdk` is not installed for a non-Ollama provider.
        MissingApiKeyError: If the required provider API key environment variable is not set.

    """
    prov = provider.lower().strip()
    if prov != "ollama":
        try:
            import any_llm

            _ = any_llm
        except ImportError as exc:
            raise MissingSdkError(
                f"Provider '{provider}' requires any-llm-sdk. Install with: pip install any-llm-sdk[{provider}]",
                params={"provider": provider},
            ) from exc
        if prov in PROVIDER_ENV_KEYS and not is_provider_configured(prov):
            env_var = PROVIDER_ENV_KEYS[prov]
            raise MissingApiKeyError(
                f"Provider '{provider}' requires environment variable '{env_var}' to be set.",
                params={"provider": provider, "env_var": env_var},
            )


def parse_configured_models_string(raw_models: str) -> List[Tuple[str, str]]:
    """
    Parse a comma-separated list of configured any-llm models into (provider, full_model_name) pairs.

    Accepts 'provider/model' or 'provider:model' entries. Bare model names default to 'ollama'.

    Args:
    ----
        raw_models: Comma-separated model list string.

    Returns:
    -------
        List[Tuple[str, str]]: List of (provider, canonical 'provider/model' or bare model) tuples.

    """
    results: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for token in raw_models.split(","):
        item = token.strip()
        if not item or is_filesystem_path_token(item):
            continue
        prov, mod = resolve_provider_and_model(item)
        prov_clean = prov.lower().strip()
        if prov_clean == "sql":
            canonical = "sql"
        elif prov_clean == "ollama" and (
            "/" not in item
            or item.startswith("hf.co/")
            or item in DEFAULT_CHOICE_HF_REPOS
        ):
            canonical = mod
        else:
            canonical = f"{prov_clean}/{mod}"
        if canonical not in seen:
            seen.add(canonical)
            results.append((prov_clean, canonical))
    return results


def _discover_additional_provider_models(
    active_catalog: Mapping[str, Sequence[str]],
    seen: set[str],
    list_models_fn: Optional[_AnyLlmListModelsFn],
) -> List[Tuple[str, str, bool]]:
    """
    Discover models for non-primary any-llm providers that have configured credentials.

    Args:
    ----
        active_catalog: Primary cloud catalog mapping already processed.
        seen: Mutable set of discovered model names.
        list_models_fn: Optional callable to query live SDK models.

    Returns:
    -------
        List[Tuple[str, str, bool]]: Additional `(provider, model_name, is_configured)` tuples.

    """
    extra_discovered: List[Tuple[str, str, bool]] = []
    for prov, prov_models in ANY_LLM_PROVIDER_CATALOG.items():
        if (
            prov in active_catalog
            or prov in ("gemini", "ollama")
            or not is_provider_configured(prov)
        ):
            continue
        if list_models_fn is not None:
            try:
                sdk_models = list_models_fn(
                    provider=prov,
                    api_key=get_effective_api_key(prov),
                )
                for sdk_m in sdk_models:
                    full_name = (
                        sdk_m.id
                        if sdk_m.id.startswith(f"{prov}/")
                        else f"{prov}/{sdk_m.id}"
                    )
                    if full_name not in seen:
                        seen.add(full_name)
                        extra_discovered.append((prov, full_name, True))
            except Exception as exc:
                logger.debug(f"any_llm.list_models failed for {prov}: {exc}")

        for m in prov_models:
            if m not in seen:
                seen.add(m)
                extra_discovered.append((prov, m, True))
    return extra_discovered


def discover_configured_any_llm_models(
    extra_configured_models: Optional[str] = None,
    list_models_fn: Optional[_AnyLlmListModelsFn] = None,
    cloud_catalog: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Tuple[str, str, bool]]:
    """
    Discover all models configured for Mozilla any-llm across environment variables and catalogs.

    Combines standard cloud catalog models (OpenAI, Anthropic, Google), any additional
    providers configured via environment variables (e.g., GROQ_API_KEY, MISTRAL_API_KEY),
    explicitly configured model strings in T1D_ANY_LLM_MODELS / ANY_LLM_CONFIGURED_MODELS,
    and optional live model discovery via `list_models_fn`.

    Args:
    ----
        extra_configured_models: Optional comma-separated model string from client headers.
        list_models_fn: Optional callable to query live models (e.g., `any_llm.list_models`).
        cloud_catalog: Optional primary cloud catalog mapping overriding `CLOUD_MODEL_CATALOG`.

    Returns:
    -------
        List[Tuple[str, str, bool]]: List of (provider, model_name, is_configured) tuples.

    """
    discovered: List[Tuple[str, str, bool]] = []
    seen: set[str] = set()
    active_catalog = cloud_catalog if cloud_catalog is not None else CLOUD_MODEL_CATALOG

    for prov, prov_models in active_catalog.items():
        conf = is_provider_configured(prov)
        for m in prov_models:
            seen.add(m)
            discovered.append((prov, m, conf))

    discovered.extend(
        _discover_additional_provider_models(active_catalog, seen, list_models_fn)
    )

    env_models_str = ",".join(
        filter(
            None,
            [
                ",".join(read_configured_ollama_model_list()),
                os.environ.get("T1D_ANY_LLM_MODELS", ""),
                os.environ.get("ANY_LLM_CONFIGURED_MODELS", ""),
                extra_configured_models or "",
            ],
        )
    )
    for prov, canonical_model in parse_configured_models_string(env_models_str):
        if canonical_model not in seen:
            seen.add(canonical_model)
            conf = is_provider_configured(prov) or bool(extra_configured_models)
            discovered.append((prov, canonical_model, conf))

    return discovered


def is_mapping_chunk(
    val: Union[
        StreamChunkInput,
        StreamChoiceElement,
        Sequence[StreamChoiceElement],
    ],
) -> TypeGuard[StreamMappingChunk]:
    """
    Narrow a streaming chunk or nested element to a string-keyed Mapping.

    Args:
    ----
        val: Candidate chunk or nested choice/delta value.

    Returns:
    -------
        bool: True if `val` is a Mapping.

    """
    return isinstance(val, Mapping)


def is_sequence_of_choices(
    val: Union[
        str,
        int,
        float,
        bool,
        Sequence[StreamChoiceElement],
        Mapping[str, Union[str, int, float, bool, None]],
        None,
    ],
) -> TypeGuard[Sequence[StreamChoiceElement]]:
    """
    Narrow a mapping value to a non-string Sequence of choice elements.

    Args:
    ----
        val: Raw `choices` field from a mapping chunk.

    Returns:
    -------
        bool: True if `val` is a non-string Sequence.

    """
    return isinstance(val, Sequence) and not isinstance(val, str)


def _extract_delta_or_message_str(
    item: Optional[
        Union[
            _DeltaContentProtocol,
            _DeltaTextProtocol,
            LlmCompletionMessageProtocol,
        ]
    ],
    prefer_content: bool = True,
) -> str:
    """
    Extract a non-empty string from a delta or message protocol instance.

    Args:
    ----
        item: Delta or message object.
        prefer_content: Whether to check `.content` before `.text`.

    Returns:
    -------
        str: Extracted text or empty string.

    """
    if item is None:
        return ""
    content_val: Optional[str] = (
        item.content if isinstance(item, _DeltaContentProtocol) else None
    )
    text_val: Optional[str] = (
        item.text if isinstance(item, _DeltaTextProtocol) else None
    )
    primary = content_val if prefer_content else text_val
    secondary = text_val if prefer_content else content_val
    chosen = primary or secondary
    return chosen if isinstance(chosen, str) and chosen else ""


def _extract_google_chunk_delta(chunk: StreamChunkInput) -> str:
    """
    Extract text delta from a Google/Gemini streaming chunk object.

    Args:
    ----
        chunk: Streaming chunk protocol instance.

    Returns:
    -------
        str: Extracted text or empty string.

    """
    if isinstance(chunk, GeminiStreamChunkProtocol):
        for cand in chunk.candidates or []:
            if isinstance(cand, _CandidateProtocol) and cand.content is not None:
                content = cand.content
                if isinstance(content, _CandidateContentProtocol):
                    for part in content.parts or []:
                        if isinstance(part, _TextPartProtocol):
                            text_val = part.text
                            if isinstance(text_val, str) and text_val:
                                return text_val

    if isinstance(chunk, DirectTextChunkProtocol):
        text_attr = chunk.text
        if isinstance(text_attr, str) and text_attr:
            return text_attr
    return ""


def _extract_anthropic_chunk_delta(chunk: StreamChunkInput) -> str:
    """
    Extract text delta from an Anthropic streaming chunk object.

    Args:
    ----
        chunk: Streaming chunk protocol instance.

    Returns:
    -------
        str: Extracted text or empty string.

    """
    if isinstance(chunk, AnthropicStreamChunkProtocol) and chunk.delta is not None:
        extracted = _extract_delta_or_message_str(chunk.delta, prefer_content=False)
        if extracted:
            return extracted

    if isinstance(chunk, OpenAiStreamChunkProtocol):
        for c in chunk.choices or []:
            if isinstance(c, _ChoiceWithDeltaProtocol) and c.delta is not None:
                extracted = _extract_delta_or_message_str(c.delta, prefer_content=False)
                if extracted:
                    return extracted
    return ""


def _extract_openai_chunk_delta(chunk: StreamChunkInput) -> str:
    """
    Extract text delta from an OpenAI/Ollama/any-llm streaming chunk object.

    Args:
    ----
        chunk: Streaming chunk protocol instance.

    Returns:
    -------
        str: Extracted text or empty string.

    """
    for c in (
        chunk.choices
        if isinstance(chunk, OpenAiStreamChunkProtocol) and chunk.choices is not None
        else ()
    ):
        if isinstance(c, _ChoiceWithDeltaProtocol) and c.delta is not None:
            extracted = _extract_delta_or_message_str(c.delta, prefer_content=True)
            if extracted:
                return extracted

        if isinstance(c, _ChoiceWithMessageProtocol) and c.message is not None:
            extracted = _extract_delta_or_message_str(c.message, prefer_content=True)
            if extracted:
                return extracted

        if isinstance(c, DirectTextChunkProtocol):
            text_val = c.text
            if isinstance(text_val, str) and text_val:
                return text_val

    if isinstance(chunk, DirectTextChunkProtocol):
        direct_text = chunk.text
        if isinstance(direct_text, str) and direct_text:
            return direct_text

    return ""


def _extract_mapping_chunk_delta(chunk: StreamMappingChunk) -> str:
    """
    Extract token delta string from a dictionary/mapping streaming chunk.

    Args:
    ----
        chunk: Mapping chunk payload.

    Returns:
    -------
        str: Extracted token string or empty string.

    """
    choices_raw = chunk.get("choices")
    if is_sequence_of_choices(choices_raw):
        for choice in choices_raw:
            if is_mapping_chunk(choice):
                delta = choice.get("delta")
                if is_mapping_chunk(delta) and delta.get("content"):
                    return str(delta["content"])
                if choice.get("text"):
                    return str(choice["text"])
    msg = chunk.get("message")
    if is_mapping_chunk(msg) and msg.get("content"):
        return str(msg["content"])
    if chunk.get("response"):
        return str(chunk["response"])
    if chunk.get("text"):
        return str(chunk["text"])
    return ""


def extract_stream_delta(chunk: StreamChunkInput, provider: str = "ollama") -> str:
    """
    Extract token delta string from provider-specific streaming response chunk.

    Accurately inspects delta objects across ollama, openai, anthropic, google, and other any-llm providers.

    Args:
    ----
        chunk: Raw response chunk from any-llm or provider SDK.
        provider: Provider identifier ('ollama', 'openai', 'anthropic', 'google', etc.).

    Returns:
    -------
        str: Extracted token content string, or empty string.

    """
    if isinstance(chunk, str):
        return chunk
    if is_mapping_chunk(chunk):
        return _extract_mapping_chunk_delta(chunk)

    prov = provider.lower()
    if prov in ("google", "gemini"):
        return _extract_google_chunk_delta(chunk)
    if prov == "anthropic":
        return _extract_anthropic_chunk_delta(chunk)
    return _extract_openai_chunk_delta(chunk)


class _AnyLlmTransportAdapter:
    """Adapter bridging `any_llm.AnyLLM` to `LlmTransportProtocol`."""

    def __init__(self, provider: str, api_key: Optional[str] = None) -> None:
        """
        Initialize the underlying `AnyLLM` client instance.

        Args:
        ----
            provider: Resolved LLM provider identifier.
            api_key: Optional provider API key.

        """
        from any_llm import AnyLLM

        if provider.lower() == "ollama":
            configure_ollama_env()
        self._client = (
            AnyLLM.create(provider, api_key=api_key)
            if api_key
            else AnyLLM.create(provider)
        )

    def completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        stream: bool = False,
    ) -> Union[LlmCompletionResponseProtocol, StreamChunkInput]:
        """
        Delegate chat completion to the underlying `AnyLLM` instance.

        Args:
        ----
            model: Model identifier string.
            messages: Sequence of chat message mappings.
            stream: Whether to request streaming token chunks.

        Returns:
        -------
            Union[LlmCompletionResponseProtocol, StreamChunkInput]: Completion response or stream iterator.

        """
        completion_fn: Callable[
            ...,
            Union[LlmCompletionResponseProtocol, StreamChunkInput],
        ] = self._client.completion
        normalized_messages = [dict(m) for m in messages]
        if stream:
            return completion_fn(
                model=model,
                messages=normalized_messages,
                stream=True,
            )
        return completion_fn(
            model=model,
            messages=normalized_messages,
        )


def create_any_llm_transport(
    provider: str, api_key: Optional[str] = None
) -> LlmTransportProtocol:
    """
    Create an `LlmTransportProtocol` backed by `any_llm.AnyLLM`.

    Args:
    ----
        provider: Resolved LLM provider identifier.
        api_key: Optional provider API key.

    Returns:
    -------
        LlmTransportProtocol: Configured transport adapter.

    """
    return _AnyLlmTransportAdapter(provider, api_key=api_key)


def complete_llm_prompt(
    prompt: str,
    model_name: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    llm_factory: Optional[AnyLlmFactoryProtocol] = None,
) -> str:
    """
    Execute a synchronous LLM chat completion using `any-llm` and return the stripped text.

    Args:
    ----
        prompt: The user prompt string to send to the model.
        model_name: Model identifier or `provider/model` string.
        api_key: Optional explicit API key for cloud providers.
        provider: Optional explicit provider override.
        llm_factory: Optional injectable factory `(provider, api_key) -> LlmTransportProtocol`.

    Returns:
    -------
        str: Stripped completion text from the first response choice.

    """
    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)
    if resolved_provider == "ollama":
        normalize_ollama_host()
    effective_api_key = get_effective_api_key(resolved_provider, api_key)

    active_factory: AnyLlmFactoryProtocol = (
        llm_factory if llm_factory is not None else create_any_llm_transport
    )
    llm = (
        active_factory(resolved_provider, api_key=effective_api_key)
        if effective_api_key
        else active_factory(resolved_provider)
    )

    response = llm.completion(
        model=actual_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_stream_delta(response, resolved_provider).strip()


def compact_schema_for_prompt(
    schema: str,
    max_tables: int = 8,
    max_sample_chars: int = 120,
) -> str:
    """
    Compact a verbose database schema string for fast CPU prompt evaluation on small local Ollama models.

    Truncates long `Sample row:` lines to `max_sample_chars` and limits the number of full table
    blocks to `max_tables` while preserving the `-- Relevant Schema Context:` header.

    Args:
    ----
        schema: Raw or RAG-enriched database schema string.
        max_tables: Maximum number of detailed `Table:` blocks to include.
        max_sample_chars: Maximum character length per `Sample row:` line.

    Returns:
    -------
        str: Compacted schema string.

    """
    if not schema.strip():
        return schema
    nl = chr(10)
    blocks = schema.split(f"{nl}{nl}")
    out_blocks: List[str] = []
    table_count = 0
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("Table:"):
            if table_count >= max_tables:
                continue
            table_count += 1
        lines: List[str] = []
        for line in stripped.splitlines():
            if line.strip().startswith("Sample row:") and len(line) > max_sample_chars:
                lines.append(line[:max_sample_chars] + "...")
            else:
                lines.append(line)
        out_blocks.append(nl.join(lines))
    return f"{nl}{nl}".join(out_blocks)


def build_clinical_sql_prompt(
    schema: str,
    nl_query: str,
    history_context: str = "",
    require_markdown_fence: bool = True,
    compact_schema: bool = False,
) -> str:
    """
    Construct the clinical Text-to-SQL system prompt for an LLM.

    Args:
    ----
        schema: The database schema text.
        nl_query: The clinician or user query.
        history_context: Optional formatted prior conversation history block.
        require_markdown_fence: Whether to instruct the LLM to wrap SQL in markdown fences.
        compact_schema: Whether to compact sample rows and table count for fast local CPU inference.

    Returns:
    -------
        str: Formatted system and user prompt.

    """
    effective_schema = compact_schema_for_prompt(schema) if compact_schema else schema
    formatting_instruction = (
        """Write a SQL query that answers the user's request.
You may provide an explanation of your thought process before or after the query.
Ensure the SQL query is enclosed in a markdown code block (e.g. ```sql ... ```).
The query should be a valid DuckDB SQL SELECT statement."""
        if require_markdown_fence
        else """Write a SQL query that answers the user's request. 
You may provide an explanation of your thought process as SQL comments (`--`) before the query.
Return ONLY the raw SQL query, with no markdown formatting and no code blocks.
The query should be a valid DuckDB SQL SELECT statement."""
    )

    return f"""You are a DuckDB SQL expert. Given the following database schema for Type 1 Diabetes (T1D) clinical trial datasets:

{effective_schema}

Context & Rules for T1D Analytics:
1. Terminology & Aliases:
   - "Lows" / "Hypoglycemia": Look for Time Below Range (TBR, <70 or <54 mg/dL), `pct_time_under_70`, `hypo`.
   - "Highs" / "Hyperglycemia": Look for Time Above Range (TAR, >180 or >250 mg/dL), `pct_time_over_180`, `hyper`.
   - "In-range" / "TIR": Look for Time in Range (70-180 mg/dL), `pct_time_in_range_70_180`, `tir`, `cgm_in_range`.
   - "A1C" / "HbA1c": Look for glycated hemoglobin, `a1c`, `hba1c`.
   - "Variability": Look for Coefficient of Variation (`cv`), Standard Deviation (`sd`).
   - "Average sugar": Look for Mean Glucose, `mean_cgm`, Glucose Management Indicator (`gmi`).
   - "Adverse Events": "DKA" (Diabetic Ketoacidosis), "SH" (Severe Hypoglycemia).
   - Demographics: "Duration" (years with diabetes), "Age", "BMI", "Gender/Sex", "Weight".
   - Treatment: "Pump" (CSII), "Injections/Shots" (MDI), "Loop" (AID/Closed-Loop), "Insulin" (TDD, Basal, Bolus).
   - Study Timepoints: "Baseline" (visit 0 or baseline flag), "End of trial/Follow-up" (max visit month).
2. SQL Generation Rules for Multi-Trial Data:
   - Trial Names: Acronyms (DCLP3, DCLP5, Pedap, etc.) usually correspond directly to table names.
   - Aggregating Combined Trials: To calculate metrics for multiple trials "together", "combined", or "overall", you MUST use a CTE with `UNION ALL` to combine raw rows from all requested tables *before* applying aggregate functions like `AVG()`.
   - Comparing Individual Trials: To calculate metrics for multiple trials "individually" or "each", use `UNION ALL` with an identifying literal string column (e.g., `SELECT 'DCLP3' as trial, AVG(...) ... UNION ALL SELECT 'DCLP5'...`).
   - Aligning UNION Columns: Clinical trial tables often have different schemas. When using `UNION ALL`, you MUST explicitly `SELECT` and alias only the specific columns needed (e.g. `age`, `tir`) to ensure they align perfectly in number, order, and type across all blocks. NEVER use `SELECT *` with `UNION ALL`.
   - Linking Data: If questions span patient demographics and CGM outcomes, use `JOIN` on the patient/subject ID columns (e.g., `pt_id`, `subject_id`).
3. Formatting & Logic:
   - Handle colloquial/informal language intuitively (e.g., "betwixt" -> `BETWEEN`, "kids/pediatric" -> `age < 18`, "adults" -> `age >= 18`).
   - Account for string vs. numeric data types; use `CAST(col AS FLOAT)` if computing averages on textual columns that contain numbers.
   - Ignore NULL values appropriately when calculating averages or sums.
4. Strict Schema Enforcement:
   - CRITICAL: You must ONLY use the exact table names and column names provided in the schema above.
   - Do NOT invent, guess, or hallucinate table names (e.g., do not use "Visit1" or "demographics" if they are not in the schema).
   - If a requested concept (like "demographics") is not a table, find the relevant columns within the existing tables (e.g., `visits` or `patients`).
5. Academic "Table 1" (Baseline Participant Characteristics & Stratified Outcomes) Rules:
   - Trigger Phrases: When asked for `"Table 1"`, `"baseline characteristics"`, `"cohort summary table"`, `"CONSORT baseline table"`, or `"stratified demographic and glycemic table"`, produce a publication-ready DuckDB summary query.
   - Parametric Continuous Metrics (Mean ± SD): Use `ROUND(AVG(col), 1) || ' ± ' || ROUND(STDDEV_SAMP(col), 1)`.
   - Non-Parametric Skewed Metrics (Median (IQR)): Use `ROUND(MEDIAN(col), 2) || ' (' || ROUND(QUANTILE_CONT(col, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(col, 0.75), 2) || ')'`.
   - Categorical Proportions (n (%)): Use `SUM(CASE WHEN cond THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN cond THEN 1.0 ELSE 0.0 END), 1) || '%)'`.
   - Stratified Arms + Overall Rollup: Use `GROUP BY ROLLUP(TxGroup)` and `COALESCE(TxGroup, 'Overall Cohort') AS "Cohort Arm"` to include both treatment arms and the overall cohort.
   - Transposed Publication Table 1: When asked for a transposed NEJM/Lancet-style Table 1 where rows are clinical characteristics and columns are study arms (`"Characteristic"`, `"Closed-Loop Arm"`, `"Control Arm"`, `"Overall Cohort"`, `"Arm Difference"`), use a `UNION ALL` of characteristic rows with explicit `CAST(... AS VARCHAR)` alignment and `ORDER BY sort_ord`.

{formatting_instruction}

{history_context}User request: {nl_query}"""


build_system_prompt = build_clinical_sql_prompt


def probe_ollama_tags_and_version(
    ollama_url: Optional[str] = None,
    timeout: float = 2.0,
    urlopen_fn: Callable[..., _UrlResponseProtocol] = urllib.request.urlopen,
    include_version: bool = True,
) -> Tuple[List[ModelInfo], Optional[str]]:
    """
    Probe a local Ollama service for available models (`/api/tags`) and optional version (`/api/version`).

    Args:
    ----
        ollama_url: Optional explicit URL to the Ollama `/api/tags` endpoint.
        timeout: HTTP request timeout in seconds.
        urlopen_fn: HTTP opener callable (defaults to `urllib.request.urlopen`).
        include_version: Whether to also query `/api/version` for the Ollama daemon version.

    Returns:
    -------
        Tuple[List[ModelInfo], Optional[str]]: Discovered Ollama `ModelInfo` records and version string.

    """
    host = resolve_ollama_base_url()
    tags_url = ollama_url if ollama_url is not None else f"{host}/api/tags"
    req = urllib.request.Request(tags_url)
    with urlopen_fn(req, timeout=timeout) as response:
        if (
            isinstance(response, _HttpStatusProtocol)
            and isinstance(response.status, int)
            and not isinstance(response.status, bool)
            and response.status != 200
        ):
            raise urllib.error.URLError(f"HTTP {response.status}")
        raw_bytes = response.read()
        data = json.loads(raw_bytes.decode("utf-8"))

    version: Optional[str] = None
    if include_version:
        try:
            v_req = urllib.request.Request(f"{host}/api/version")
            with urlopen_fn(v_req, timeout=timeout) as v_resp:
                v_raw = v_resp.read()
                v_data = json.loads(v_raw.decode("utf-8"))
                version = str(v_data.get("version", "")) or None
        except Exception:
            version = None

    raw_models: List[ModelInfo] = [
        ModelInfo(
            name=str(model.get("name")),
            size=model.get("size"),
            provider="ollama",
            configured=True,
            requires_key=False,
            available=True,
            reachable=True,
            error_code=None,
        )
        for model in data.get("models", [])
        if model.get("name")
    ]
    by_name = {m.name: m for m in raw_models}
    sorted_names = sort_ollama_models(list(by_name.keys()))
    models = [by_name[name] for name in sorted_names]
    return models, version


def generate_ollama_json_completion(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: float = 30.0,
    urlopen_fn: Callable[..., _UrlResponseProtocol] = urllib.request.urlopen,
) -> str:
    """
    Execute a synchronous JSON-mode generation request against the local Ollama `/api/generate` endpoint.

    Args:
    ----
        model: Local Ollama model name (e.g. 'gemma4').
        prompt: Prompt string to send.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to predict (`num_predict`).
        timeout: HTTP timeout in seconds.
        urlopen_fn: HTTP opener callable (defaults to `urllib.request.urlopen`).

    Returns:
    -------
        str: Raw `"response"` text from the Ollama JSON response payload.

    """
    base_url = resolve_ollama_base_url()
    api_url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen_fn(req, timeout=timeout) as response:
        raw_body = response.read()
        result = json.loads(raw_body.decode("utf-8"))
        return str(result.get("response", "[]"))


def probe_providers_status(
    urlopen_fn: Callable[..., _UrlResponseProtocol] = urllib.request.urlopen,
) -> ProvidersStatusResponse:
    """
    Report connection health and credential status for all supported LLM providers.

    Args:
    ----
        urlopen_fn: HTTP opener callable (defaults to `urllib.request.urlopen`).

    Returns:
    -------
        ProvidersStatusResponse: Status records for ollama, openai, anthropic, google, and configured any-llm providers.

    """
    statuses: List[ProviderStatus] = []

    ollama_healthy = False
    ollama_details = "Ollama service unavailable"
    try:
        probe_ollama_tags_and_version(
            timeout=2,
            urlopen_fn=urlopen_fn,
            include_version=False,
        )
        ollama_healthy = True
        ollama_details = "Ollama service online"
    except Exception as e:
        ollama_details = f"Ollama unreachable: {e}"

    statuses.append(
        ProviderStatus(
            provider="ollama",
            configured=True,
            healthy=ollama_healthy,
            details=ollama_details,
            reachable=ollama_healthy,
            error_code="PROVIDER_ONLINE" if ollama_healthy else "PROVIDER_OFFLINE",
        )
    )

    openai_conf = is_provider_configured("openai")
    statuses.append(
        ProviderStatus(
            provider="openai",
            configured=openai_conf,
            healthy=openai_conf,
            details="Configured via OPENAI_API_KEY"
            if openai_conf
            else "Missing OPENAI_API_KEY",
            reachable=openai_conf,
            error_code="PROVIDER_ONLINE" if openai_conf else "MISSING_API_KEY",
        )
    )

    anthropic_conf = is_provider_configured("anthropic")
    statuses.append(
        ProviderStatus(
            provider="anthropic",
            configured=anthropic_conf,
            healthy=anthropic_conf,
            details="Configured via ANTHROPIC_API_KEY"
            if anthropic_conf
            else "Missing ANTHROPIC_API_KEY",
            reachable=anthropic_conf,
            error_code="PROVIDER_ONLINE" if anthropic_conf else "MISSING_API_KEY",
        )
    )

    google_conf = is_provider_configured("google")
    statuses.append(
        ProviderStatus(
            provider="google",
            configured=google_conf,
            healthy=google_conf,
            details="Configured via GEMINI_API_KEY/GOOGLE_API_KEY"
            if google_conf
            else "Missing GEMINI_API_KEY",
            reachable=google_conf,
            error_code="PROVIDER_ONLINE" if google_conf else "MISSING_API_KEY",
        )
    )

    for prov in ANY_LLM_PROVIDER_CATALOG:
        if prov in ("openai", "anthropic", "google", "gemini", "ollama"):
            continue
        if is_provider_configured(prov):
            env_name = PROVIDER_ENV_KEYS.get(prov, f"{prov.upper()}_API_KEY")
            statuses.append(
                ProviderStatus(
                    provider=prov,
                    configured=True,
                    healthy=True,
                    details=f"Configured via {env_name}",
                    reachable=True,
                    error_code="PROVIDER_ONLINE",
                )
            )

    return ProvidersStatusResponse(providers=statuses)


def fetch_available_models(
    include_cloud: bool = True,
    configured_models: Optional[str] = None,
    cloud_catalog: Optional[Mapping[str, Sequence[str]]] = None,
    urlopen_fn: Callable[..., _UrlResponseProtocol] = urllib.request.urlopen,
) -> ModelsResponse:
    """
    List available local Ollama models and configured any-llm cloud providers.

    Args:
    ----
        include_cloud: Whether to include cloud and configured any-llm provider models.
        configured_models: Optional comma-separated list of custom configured provider/model IDs.
        cloud_catalog: Optional cloud model catalog mapping.
        urlopen_fn: HTTP opener callable (defaults to `urllib.request.urlopen`).

    Returns:
    -------
        ModelsResponse: List of available local and cloud models.

    """
    models: List[ModelInfo] = []
    ollama_online = False
    installed_ollama_names: List[str] = []
    try:
        ollama_models, _ = probe_ollama_tags_and_version(
            timeout=5,
            urlopen_fn=urlopen_fn,
            include_version=False,
        )
        ollama_online = True
        installed_ollama_names = [m.name for m in ollama_models]
        models.extend(ollama_models)
    except Exception as e:
        logger.error(f"Error fetching models from local Ollama instance: {e}")

    if include_cloud:
        for prov, m, conf in discover_configured_any_llm_models(
            extra_configured_models=configured_models,
            cloud_catalog=cloud_catalog,
        ):
            if prov == "ollama":
                if ollama_online and installed_ollama_names:
                    continue
                models.append(
                    ModelInfo(
                        name=m,
                        provider="ollama",
                        configured=True,
                        requires_key=False,
                        available=False,
                        reachable=False,
                        error_code="MODEL_NOT_PULLED"
                        if ollama_online
                        else "PROVIDER_OFFLINE",
                    )
                )
            else:
                models.append(
                    ModelInfo(
                        name=m,
                        provider=prov,
                        configured=conf,
                        requires_key=True,
                        available=conf,
                        reachable=conf,
                        error_code=None if conf else "MISSING_API_KEY",
                    )
                )

    return ModelsResponse(models=models)
