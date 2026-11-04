"""Centralized LLM provider configuration, prompt construction, and streaming helpers."""

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

from t1d_analytics.api_schemas import (
    ModelInfo,
    ModelsResponse,
    ProvidersStatusResponse,
    ProviderStatus,
)
from t1d_analytics.clinical_prompts import (
    MULTI_TRIAL_UNION_SECTION,
    STRICT_SCHEMA_SECTION,
    TABLE_1_RULES_SECTION,
    TERMINOLOGY_RULES_SECTION,
    build_clinical_sql_prompt,
    build_system_prompt,
    compact_schema_for_prompt,
)
from t1d_analytics.errors import MissingApiKeyError, MissingSdkError
from t1d_analytics.llm_stream_protocols import (
    AnthropicChunkProtocol,
    AnthropicDeltaChunkProtocol,
    AnthropicStreamChunkProtocol,
    DirectTextChunkProtocol,
    GeminiCandidateChunkProtocol,
    GeminiStreamChunkProtocol,
    GoogleChunkProtocol,
    LlmCompletionChoiceProtocol,
    LlmCompletionMessageProtocol,
    LlmCompletionResponseProtocol,
    LlmDeltaChoicesContainerProtocol,
    LlmMessageChoicesContainerProtocol,
    LlmStreamChunkProtocol,
    ObjectChunkInput,
    OpenAiChunkProtocol,
    OpenAiStreamChunkProtocol,
    StreamChoiceElement,
    StreamChoiceMappingDict,
    StreamChunkInput,
    StreamDeltaMappingDict,
    StreamMappingChunk,
    StreamScalar,
    _CandidateContentProtocol,
    _CandidateProtocol,
    _ChoiceWithDeltaProtocol,
    _ChoiceWithMessageProtocol,
    _DeltaContentProtocol,
    _DeltaOrMessageProtocol,
    _DeltaTextProtocol,
    _extract_anthropic_chunk_delta,
    _extract_candidate_text,
    _extract_delta_or_message_str,
    _extract_direct_text,
    _extract_google_chunk_delta,
    _extract_mapping_chunk_delta,
    _extract_openai_chunk_delta,
    _extract_single_openai_choice_delta,
    _TextPartProtocol,
    extract_stream_delta,
    is_mapping_chunk,
    is_sequence_of_choices,
)
from t1d_analytics.ollama_discovery import (
    _FAST_SMALL_OLLAMA_ORDER,
    _HEAVY_OLLAMA_SUBSTRINGS,
    _LIGHTWEIGHT_OLLAMA_RANK,
    DEFAULT_CHOICE_HF_REPOS,
    DEFAULT_CHOICE_OLLAMA_MODELS,
    DEFAULT_OLLAMA_HOST,
    _HttpStatusProtocol,
    _ollama_model_sort_key,
    _UrlResponseProtocol,
    configure_ollama_env,
    generate_ollama_json_completion,
    is_filesystem_path_token,
    normalize_ollama_host,
    probe_ollama_tags_and_version,
    read_configured_ollama_model_list,
    resolve_installed_ollama_model_tag,
    resolve_ollama_base_url,
    sort_ollama_models,
)

logger = logging.getLogger(__name__)


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
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
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
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as e:
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
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as e:
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


__all__ = [
    "ANY_LLM_PROVIDER_CATALOG",
    "CLOUD_MODEL_CATALOG",
    "DEFAULT_CHOICE_HF_REPOS",
    "DEFAULT_CHOICE_OLLAMA_MODELS",
    "DEFAULT_OLLAMA_HOST",
    "MULTI_TRIAL_UNION_SECTION",
    "PROVIDER_ALT_ENV_KEYS",
    "PROVIDER_ENV_KEYS",
    "STRICT_SCHEMA_SECTION",
    "TABLE_1_RULES_SECTION",
    "TERMINOLOGY_RULES_SECTION",
    "_FAST_SMALL_OLLAMA_ORDER",
    "_HEAVY_OLLAMA_SUBSTRINGS",
    "_LIGHTWEIGHT_OLLAMA_RANK",
    "AnthropicChunkProtocol",
    "AnthropicDeltaChunkProtocol",
    "AnthropicStreamChunkProtocol",
    "AnyLlmFactoryProtocol",
    "DirectTextChunkProtocol",
    "GeminiCandidateChunkProtocol",
    "GeminiStreamChunkProtocol",
    "GoogleChunkProtocol",
    "LlmCompletionChoiceProtocol",
    "LlmCompletionMessageProtocol",
    "LlmCompletionResponseProtocol",
    "LlmDeltaChoicesContainerProtocol",
    "LlmMessageChoicesContainerProtocol",
    "LlmStreamChunkProtocol",
    "LlmTransportProtocol",
    "ObjectChunkInput",
    "OpenAiChunkProtocol",
    "OpenAiStreamChunkProtocol",
    "StreamChoiceElement",
    "StreamChoiceMappingDict",
    "StreamChunkInput",
    "StreamDeltaMappingDict",
    "StreamMappingChunk",
    "StreamScalar",
    "_AnyLlmListModelsFn",
    "_AnyLlmModelProtocol",
    "_AnyLlmTransportAdapter",
    "_CandidateContentProtocol",
    "_CandidateProtocol",
    "_ChoiceWithDeltaProtocol",
    "_ChoiceWithMessageProtocol",
    "_DeltaContentProtocol",
    "_DeltaOrMessageProtocol",
    "_DeltaTextProtocol",
    "_HttpStatusProtocol",
    "_TextPartProtocol",
    "_UrlResponseProtocol",
    "_discover_additional_provider_models",
    "_extract_anthropic_chunk_delta",
    "_extract_candidate_text",
    "_extract_delta_or_message_str",
    "_extract_direct_text",
    "_extract_google_chunk_delta",
    "_extract_mapping_chunk_delta",
    "_extract_openai_chunk_delta",
    "_extract_single_openai_choice_delta",
    "_ollama_model_sort_key",
    "build_clinical_sql_prompt",
    "build_system_prompt",
    "compact_schema_for_prompt",
    "complete_llm_prompt",
    "configure_ollama_env",
    "create_any_llm_transport",
    "discover_configured_any_llm_models",
    "extract_stream_delta",
    "fetch_available_models",
    "generate_ollama_json_completion",
    "get_effective_api_key",
    "is_filesystem_path_token",
    "is_mapping_chunk",
    "is_provider_configured",
    "is_sequence_of_choices",
    "normalize_ollama_host",
    "parse_configured_models_string",
    "probe_ollama_tags_and_version",
    "probe_providers_status",
    "read_configured_ollama_model_list",
    "resolve_installed_ollama_model_tag",
    "resolve_ollama_base_url",
    "resolve_provider_and_model",
    "sort_ollama_models",
    "validate_provider_readiness",
]
