"""Centralized LLM provider configuration, prompt construction, and streaming helpers."""

import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Protocol,
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
    build_compact_clinical_sql_prompt,
    build_system_prompt,
    compact_schema_for_prompt,
    is_compact_local_model,
)
from t1d_analytics.errors import (
    LlmAuthenticationError,
    LlmConnectionError,
    LlmProviderError,
    LlmQuotaExceededError,
    LlmRateLimitError,
    LlmTranslationError,
    MissingSdkError,
    ProviderNotConfiguredError,
)
from t1d_analytics.llm_stream_protocols import (
    LlmCompletionResponseProtocol,
    LlmMessageChoicesContainerProtocol,
    StreamChunkInput,
    extract_stream_delta,
)
from t1d_analytics.ollama_discovery import (
    DEFAULT_CHOICE_HF_REPOS,
    DEFAULT_CHOICE_OLLAMA_MODELS,
    DEFAULT_OLLAMA_GENERATE_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_HOST,
    OllamaRunnerOptionsDict,
    UrlResponseProtocol,
    build_safe_ollama_options,
    call_local_ollama_generate,
    configure_ollama_env,
    generate_ollama_json_completion,
    generate_ollama_text_completion,
    is_filesystem_path_token,
    normalize_ollama_host,
    probe_ollama_tags_and_version,
    read_configured_ollama_model_list,
    resolve_installed_ollama_model_tag,
    resolve_local_ollama_fallback_candidates,
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
        self, provider: str, api_key: str | None = None
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
    ) -> LlmCompletionResponseProtocol | StreamChunkInput:
        """Execute a chat completion call."""
        ...


class AnyLlmFactoryProtocol(Protocol):
    """Callable protocol for creating an `LlmTransportProtocol` for a given provider."""

    def __call__(
        self, provider: str, api_key: str | None = None
    ) -> LlmTransportProtocol:
        """Create and return an LLM transport instance."""
        ...


PROVIDER_ENV_KEYS: dict[str, str] = {
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

PROVIDER_ALT_ENV_KEYS: dict[str, tuple[str, ...]] = {
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

CLOUD_MODEL_CATALOG: dict[str, list[str]] = {
    "openai": ["openai/gpt-4o", "openai/gpt-4o-mini"],
    "anthropic": ["anthropic/claude-3-5-sonnet", "anthropic/claude-3-5-haiku"],
    "google": ["google/gemini-1.5-pro", "google/gemini-1.5-flash"],
}

ANY_LLM_PROVIDER_CATALOG: dict[str, list[str]] = {
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
    model_name: str, provider: str | None = None
) -> tuple[str, str]:
    """
    Resolve the LLM provider identifier and model name from a composite model string.

    Supports both 'provider/model' and Mozilla any-llm 'provider:model' syntax when the
    prefix matches a known any-llm provider, as well as 'sql' and 'hf.co/<org>/<repo>'
    Ollama Hugging Face model tags.

    Args:
        model_name: Model identifier or 'provider/model' string (e.g., 'openai/gpt-4o').
        provider: Optional explicit provider override.

    Returns:
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
        provider: Provider identifier string (e.g., 'ollama', 'openai', 'anthropic', 'google', 'groq').

    Returns:
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


def get_effective_api_key(provider: str, api_key: str | None = None) -> str | None:
    """
    Resolve the effective API key for a cloud provider from explicit argument or environment.

    Args:
        provider: Provider identifier string.
        api_key: Optional explicit API key override.

    Returns:
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
        provider: Provider identifier (e.g. 'ollama', 'openai', 'anthropic', 'google').

    Raises:
        MissingSdkError: If `any-llm-sdk` is not installed for a non-Ollama provider.
        ProviderNotConfiguredError: If the required provider API key environment variable is not set.

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
            raise ProviderNotConfiguredError(
                f"Provider '{provider}' requires environment variable '{env_var}' to be set.",
                params={"provider": provider, "env_var": env_var},
            )


def parse_configured_models_string(raw_models: str) -> list[tuple[str, str]]:
    """
    Parse a comma-separated list of configured any-llm models into (provider, full_model_name) pairs.

    Accepts 'provider/model' or 'provider:model' entries. Bare model names default to 'ollama'.

    Args:
        raw_models: Comma-separated model list string.

    Returns:
        List[Tuple[str, str]]: List of (provider, canonical 'provider/model' or bare model) tuples.

    """
    results: list[tuple[str, str]] = []
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
    list_models_fn: _AnyLlmListModelsFn | None,
) -> list[tuple[str, str, bool]]:
    """
    Discover models for non-primary any-llm providers that have configured credentials.

    Args:
        active_catalog: Primary cloud catalog mapping already processed.
        seen: Mutable set of discovered model names.
        list_models_fn: Optional callable to query live SDK models.

    Returns:
        List[Tuple[str, str, bool]]: Additional `(provider, model_name, is_configured)` tuples.

    """
    extra_discovered: list[tuple[str, str, bool]] = []
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
    extra_configured_models: str | None = None,
    list_models_fn: _AnyLlmListModelsFn | None = None,
    cloud_catalog: Mapping[str, Sequence[str]] | None = None,
) -> list[tuple[str, str, bool]]:
    """
    Discover all models configured for Mozilla any-llm across environment variables and catalogs.

    Combines standard cloud catalog models (OpenAI, Anthropic, Google), any additional
    providers configured via environment variables (e.g., GROQ_API_KEY, MISTRAL_API_KEY),
    explicitly configured model strings in T1D_ANY_LLM_MODELS / ANY_LLM_CONFIGURED_MODELS,
    and optional live model discovery via `list_models_fn`.

    Args:
        extra_configured_models: Optional comma-separated model string from client headers.
        list_models_fn: Optional callable to query live models (e.g., `any_llm.list_models`).
        cloud_catalog: Optional primary cloud catalog mapping overriding `CLOUD_MODEL_CATALOG`.

    Returns:
        List[Tuple[str, str, bool]]: List of (provider, model_name, is_configured) tuples.

    """
    discovered: list[tuple[str, str, bool]] = []
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

    def __init__(self, provider: str, api_key: str | None = None) -> None:
        """
        Initialize the underlying `AnyLLM` client instance.

        Args:
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
    ) -> LlmCompletionResponseProtocol | StreamChunkInput:
        """
        Delegate chat completion to the underlying `AnyLLM` instance.

        Args:
            model: Model identifier string.
            messages: Sequence of chat message mappings.
            stream: Whether to request streaming token chunks.

        Returns:
            Union[LlmCompletionResponseProtocol, StreamChunkInput]: Completion response or stream iterator.

        """
        completion_fn: Callable[
            ...,
            LlmCompletionResponseProtocol | StreamChunkInput,
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
    provider: str, api_key: str | None = None
) -> LlmTransportProtocol:
    """
    Create an `LlmTransportProtocol` backed by `any_llm.AnyLLM`.

    Args:
        provider: Resolved LLM provider identifier.
        api_key: Optional provider API key.

    Returns:
        LlmTransportProtocol: Configured transport adapter.

    """
    return _AnyLlmTransportAdapter(provider, api_key=api_key)


def classify_llm_transport_error(
    exc: BaseException,
    provider: str,
    model_name: str,
) -> LlmProviderError:
    """
    Classify a raw SDK or HTTP transport exception into a typed `LlmProviderError` subclass.

    Args:
        exc: Raw transport exception raised by `AnyLLM` or `urllib`.
        provider: Resolved LLM provider identifier.
        model_name: Target model name.

    Returns:
        LlmProviderError: Specific `LlmRateLimitError`, `LlmAuthenticationError`,
            `LlmQuotaExceededError`, `LlmConnectionError`, or `LlmTranslationError`.

    """
    if isinstance(exc, LlmProviderError):
        return exc

    msg = str(exc).strip()
    lower_msg = msg.lower()
    cls_name = type(exc).__name__.lower()
    http_code: int | None = None
    if isinstance(exc, urllib.error.HTTPError):
        http_code = int(exc.code)
    else:
        raw_status = getattr(exc, "status_code", None)
        if isinstance(raw_status, int):
            http_code = raw_status

    params: dict[str, str | int | float | bool | None] = {
        "provider": provider,
        "model": model_name,
        "error": msg,
    }

    if (
        http_code == 402
        or "insufficient_quota" in lower_msg
        or "quota exceeded" in lower_msg
        or "billing" in lower_msg
    ):
        return LlmQuotaExceededError(msg, params=params)

    if (
        http_code == 429
        or "ratelimit" in cls_name
        or "rate limit" in lower_msg
        or "429" in lower_msg
    ):
        return LlmRateLimitError(msg, params=params)

    if (
        http_code in (401, 403)
        or "authentication" in cls_name
        or "permissiondenied" in cls_name
        or "invalid api key" in lower_msg
        or "incorrect api key" in lower_msg
        or "unauthorized" in lower_msg
    ):
        return LlmAuthenticationError(msg, params=params)

    if (
        isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError, OSError))
        and not isinstance(exc, urllib.error.HTTPError)
    ) or any(
        marker in lower_msg
        for marker in (
            "connection refused",
            "failed to connect",
            "connecterror",
            "timed out",
            "timeout",
            "llama-server process has terminated",
            "llama runner process has terminated",
            "server disconnected",
        )
    ):
        return LlmConnectionError(msg, params=params)

    return LlmTranslationError(msg, params=params)


def complete_llm_prompt(
    prompt: str,
    model_name: str,
    api_key: str | None = None,
    provider: str | None = None,
    llm_factory: AnyLlmFactoryProtocol | None = None,
    use_direct_ollama: bool = False,
) -> str:
    """
    Execute a synchronous LLM chat completion and return the stripped text.

    When `resolved_provider == "ollama"` and `use_direct_ollama=True` (or when
    `any-llm` fails and `llm_factory is None`), calls `call_local_ollama_generate`
    directly with CPU-safe runner options. Transport failures are normalized into
    typed `LlmProviderError` subclasses (`LlmRateLimitError`, `LlmAuthenticationError`,
    `LlmQuotaExceededError`, `LlmConnectionError`, or `LlmTranslationError`).

    Args:
        prompt: The user prompt string to send to the model.
        model_name: Model identifier or `provider/model` string.
        api_key: Optional explicit API key for cloud providers.
        provider: Optional explicit provider override.
        llm_factory: Optional injectable factory `(provider, api_key) -> LlmTransportProtocol`.
        use_direct_ollama: When True and targeting `ollama` without a custom `llm_factory`, call `call_local_ollama_generate` directly.

    Returns:
        str: Stripped completion text from the first response choice.

    """
    resolved_provider, actual_model = resolve_provider_and_model(model_name, provider)
    if resolved_provider == "ollama":
        normalize_ollama_host()
        if use_direct_ollama and llm_factory is None:
            try:
                return call_local_ollama_generate(
                    model=actual_model,
                    prompt=prompt,
                    num_ctx=2048,
                    num_gpu=0,
                )
            except Exception as direct_exc:
                raise classify_llm_transport_error(
                    direct_exc, resolved_provider, actual_model
                ) from direct_exc

    effective_api_key = get_effective_api_key(resolved_provider, api_key)

    active_factory: AnyLlmFactoryProtocol = (
        llm_factory if llm_factory is not None else create_any_llm_transport
    )
    llm = (
        active_factory(resolved_provider, api_key=effective_api_key)
        if effective_api_key
        else active_factory(resolved_provider)
    )

    try:
        response = llm.completion(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_stream_delta(response, resolved_provider).strip()
    except Exception as primary_exc:
        if resolved_provider == "ollama" and llm_factory is None:
            try:
                return generate_ollama_text_completion(
                    model=actual_model,
                    prompt=prompt,
                    num_ctx=2048,
                    num_gpu=0,
                )
            except Exception:
                raise classify_llm_transport_error(
                    primary_exc, resolved_provider, actual_model
                ) from primary_exc
        raise classify_llm_transport_error(
            primary_exc, resolved_provider, actual_model
        ) from primary_exc


def probe_providers_status(
    urlopen_fn: Callable[..., UrlResponseProtocol] = urllib.request.urlopen,
) -> ProvidersStatusResponse:
    """
    Report connection health and credential status for all supported LLM providers.

    Args:
        urlopen_fn: HTTP opener callable (defaults to `urllib.request.urlopen`).

    Returns:
        ProvidersStatusResponse: Status records for ollama, openai, anthropic, google, and configured any-llm providers.

    """
    statuses: list[ProviderStatus] = []

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
    configured_models: str | None = None,
    cloud_catalog: Mapping[str, Sequence[str]] | None = None,
    urlopen_fn: Callable[..., UrlResponseProtocol] = urllib.request.urlopen,
) -> ModelsResponse:
    """
    List available local Ollama models and configured any-llm cloud providers.

    Args:
        include_cloud: Whether to include cloud and configured any-llm provider models.
        configured_models: Optional comma-separated list of custom configured provider/model IDs.
        cloud_catalog: Optional cloud model catalog mapping.
        urlopen_fn: HTTP opener callable (defaults to `urllib.request.urlopen`).

    Returns:
        ModelsResponse: List of available local and cloud models.

    """
    models: list[ModelInfo] = []
    ollama_online = False
    installed_ollama_names: list[str] = []
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


def request_sql_completion(
    schema_context: str,
    user_query: str,
    model_name: str = "gemma:2b",
    provider: str | None = None,
    api_key: str | None = None,
    history_context: str = "",
    llm_factory: AnyLlmFactoryProtocol | None = None,
) -> str:
    """
    Request a raw LLM SQL completion from a natural-language question and schema context.

    Uses compact prompts automatically for local CPU Ollama models.

    Args:
        schema_context: Database schema string.
        user_query: Natural language clinical question.
        model_name: Target model identifier.
        provider: Optional explicit provider override.
        api_key: Optional API key for cloud providers.
        history_context: Optional prior conversation history text.
        llm_factory: Optional injectable LLM transport factory.

    Returns:
        str: Raw LLM completion response containing the generated SQL query.

    """
    resolved_provider, _ = resolve_provider_and_model(model_name, provider)
    if is_compact_local_model(model_name, resolved_provider):
        prompt = build_compact_clinical_sql_prompt(
            schema_context=schema_context,
            user_query=user_query,
            history_context=history_context,
        )
    else:
        prompt = build_clinical_sql_prompt(
            schema=schema_context,
            nl_query=user_query,
            history_context=history_context,
            require_markdown_fence=True,
        )
    return complete_llm_prompt(
        prompt=prompt,
        model_name=model_name,
        api_key=api_key,
        provider=resolved_provider,
        llm_factory=llm_factory,
        use_direct_ollama=(llm_factory is None and resolved_provider == "ollama"),
    )


__all__ = [
    "ANY_LLM_PROVIDER_CATALOG",
    "CLOUD_MODEL_CATALOG",
    "DEFAULT_CHOICE_HF_REPOS",
    "DEFAULT_CHOICE_OLLAMA_MODELS",
    "DEFAULT_OLLAMA_GENERATE_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_HOST",
    "MULTI_TRIAL_UNION_SECTION",
    "PROVIDER_ALT_ENV_KEYS",
    "PROVIDER_ENV_KEYS",
    "STRICT_SCHEMA_SECTION",
    "TABLE_1_RULES_SECTION",
    "TERMINOLOGY_RULES_SECTION",
    "AnyLlmFactoryProtocol",
    "LlmCompletionResponseProtocol",
    "LlmMessageChoicesContainerProtocol",
    "LlmTransportProtocol",
    "OllamaRunnerOptionsDict",
    "StreamChunkInput",
    "UrlResponseProtocol",
    "build_clinical_sql_prompt",
    "build_compact_clinical_sql_prompt",
    "build_safe_ollama_options",
    "build_system_prompt",
    "call_local_ollama_generate",
    "classify_llm_transport_error",
    "compact_schema_for_prompt",
    "complete_llm_prompt",
    "configure_ollama_env",
    "create_any_llm_transport",
    "discover_configured_any_llm_models",
    "extract_stream_delta",
    "fetch_available_models",
    "generate_ollama_json_completion",
    "generate_ollama_text_completion",
    "get_effective_api_key",
    "is_compact_local_model",
    "is_filesystem_path_token",
    "is_provider_configured",
    "normalize_ollama_host",
    "parse_configured_models_string",
    "probe_ollama_tags_and_version",
    "probe_providers_status",
    "read_configured_ollama_model_list",
    "request_sql_completion",
    "resolve_installed_ollama_model_tag",
    "resolve_local_ollama_fallback_candidates",
    "resolve_ollama_base_url",
    "resolve_provider_and_model",
    "sort_ollama_models",
    "validate_provider_readiness",
]
