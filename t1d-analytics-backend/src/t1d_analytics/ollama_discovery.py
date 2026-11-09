"""Ollama daemon discovery, model sorting, tag probing, and JSON generation helpers."""

import json
import os
import urllib.error
import urllib.request
from typing import (
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from typing_extensions import TypedDict

from t1d_analytics.api_schemas import ModelInfo
from t1d_analytics.errors import LlmTranslationError

DEFAULT_OLLAMA_HOST: str = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_GENERATE_TIMEOUT_SECONDS: float = float(
    os.environ.get("T1D_OLLAMA_TIMEOUT_SECONDS", "25.0")
)


class OllamaRunnerOptionsDict(TypedDict):
    """Typed dictionary for safe CPU-friendly Ollama `/api/generate` runner options."""

    temperature: float
    num_predict: int
    num_ctx: int
    num_gpu: int
    num_thread: int
    use_mmap: bool


def _parse_env_int(env_key: str, default_val: int) -> int:
    """
    Read an integer from `os.environ[env_key]` with a fallback default on missing or invalid values.

    Args:
    ----
        env_key: Environment variable name to inspect.
        default_val: Fallback integer value when unset or unparseable.

    Returns:
    -------
        int: Parsed integer or `default_val`.

    """
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return default_val
    try:
        return int(raw)
    except ValueError:
        return default_val


def build_safe_ollama_options(
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_ctx: Optional[int] = None,
    num_gpu: Optional[int] = None,
    num_thread: Optional[int] = None,
    use_mmap: bool = False,
) -> OllamaRunnerOptionsDict:
    """
    Build safe CPU-bounded Ollama `/api/generate` runner options preventing `llama-server` SIGSEGV crashes.

    Reads optional overrides from `OLLAMA_NUM_CTX` (default 2048), `OLLAMA_NUM_GPU`
    (default 0), and `OLLAMA_NUM_THREAD` (default 8).

    Args:
    ----
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to predict (`num_predict`).
        num_ctx: Optional explicit context length override.
        num_gpu: Optional explicit GPU layer count override (`0` forces CPU-only).
        num_thread: Optional explicit CPU thread count override.
        use_mmap: Whether to memory-map model weights (defaults to `False` for CPU stability).

    Returns:
    -------
        OllamaRunnerOptionsDict: Bounded options dictionary for Ollama `/api/generate`.

    """
    resolved_ctx = (
        num_ctx if num_ctx is not None else _parse_env_int("OLLAMA_NUM_CTX", 2048)
    )
    resolved_gpu = (
        num_gpu if num_gpu is not None else _parse_env_int("OLLAMA_NUM_GPU", 0)
    )
    resolved_thread = (
        num_thread if num_thread is not None else _parse_env_int("OLLAMA_NUM_THREAD", 8)
    )
    return {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": resolved_ctx,
        "num_gpu": resolved_gpu,
        "num_thread": resolved_thread,
        "use_mmap": use_mmap,
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

_LIGHTWEIGHT_OLLAMA_RANK: Tuple[str, ...] = _FAST_SMALL_OLLAMA_ORDER

_HEAVY_OLLAMA_SUBSTRINGS: Tuple[str, ...] = (
    "26b",
    "27b",
    "32b",
    "70b",
)


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
    bs = chr(92)
    stripped = token.strip()
    if not stripped:
        return True
    if stripped.startswith(("/", bs, ".")):
        return True
    if (
        len(stripped) >= 2
        and stripped[0].isalpha()
        and stripped[1] == ":"
        and (len(stripped) == 2 or stripped[2] in (bs, "/"))
    ):
        return True
    return bs in stripped or (
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


def resolve_local_ollama_fallback_candidates(
    requested_model: str,
    installed_models: Sequence[str],
) -> List[str]:
    """
    Return distinct installed lightweight Ollama fallback models in priority order excluding `requested_model`.

    Args:
    ----
        requested_model: Primary requested Ollama model identifier (e.g., `'gemma:2b'`).
        installed_models: Sequence of model names reported by Ollama `/api/tags`.

    Returns:
    -------
        List[str]: Ordered list of distinct lightweight fallback model tags (e.g., `['llama3.2:1b', 'phi4-mini']`).

    """
    if not installed_models:
        return []
    ranked = sort_ollama_models(installed_models)
    matched = resolve_installed_ollama_model_tag(requested_model, ranked)
    req_clean = requested_model.strip()
    req_bare = (
        req_clean[7:].strip() if req_clean.lower().startswith("ollama/") else req_clean
    )
    excluded = {req_clean.lower(), req_bare.lower()}
    if matched is not None:
        excluded.add(matched.strip().lower())

    candidates: List[str] = []
    seen_lower = set(excluded)
    for cand in ranked:
        clean_cand = cand.strip()
        if not clean_cand:
            continue
        lower_cand = clean_cand.lower()
        bare_cand = (
            lower_cand[7:].strip() if lower_cand.startswith("ollama/") else lower_cand
        )
        if lower_cand in seen_lower or bare_cand in seen_lower:
            continue
        if any(heavy in lower_cand for heavy in _HEAVY_OLLAMA_SUBSTRINGS):
            continue
        seen_lower.add(lower_cand)
        seen_lower.add(bare_cand)
        candidates.append(clean_cand)
    return candidates


def resolve_ollama_base_url(
    raw_host: Optional[str] = None,
    default_host: str = DEFAULT_OLLAMA_HOST,
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
    default_host: str = DEFAULT_OLLAMA_HOST,
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
        except (urllib.error.URLError, OSError, ValueError, RuntimeError):
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
        "options": build_safe_ollama_options(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
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


def generate_ollama_text_completion(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_ctx: Optional[int] = None,
    num_gpu: Optional[int] = None,
    num_thread: Optional[int] = None,
    timeout: float = DEFAULT_OLLAMA_GENERATE_TIMEOUT_SECONDS,
    urlopen_fn: Callable[..., _UrlResponseProtocol] = urllib.request.urlopen,
) -> str:
    """
    Execute a synchronous text generation request against the local Ollama `/api/generate` endpoint with CPU-safe options.

    Args:
    ----
        model: Local Ollama model name (e.g. `'gemma:2b'`).
        prompt: Prompt string to send.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to predict (`num_predict`).
        num_ctx: Optional context length override.
        num_gpu: Optional GPU layer override (`0` forces CPU-only execution).
        num_thread: Optional CPU thread count override.
        timeout: HTTP timeout in seconds.
        urlopen_fn: HTTP opener callable (defaults to `urllib.request.urlopen`).

    Returns:
    -------
        str: Stripped `"response"` text from the Ollama JSON response payload.

    Raises:
    ------
        LlmTranslationError: If Ollama returns an HTTP error status or a JSON `{"error": ...}` payload.

    """
    base_url = resolve_ollama_base_url()
    api_url = f"{base_url}/api/generate"
    effective_ctx = (
        num_ctx
        if num_ctx is not None
        else _parse_env_int(
            "OLLAMA_CONTEXT_LENGTH", _parse_env_int("OLLAMA_NUM_CTX", 2048)
        )
    )
    effective_predict = _parse_env_int("OLLAMA_NUM_PREDICT", min(max_tokens, 768))
    options = build_safe_ollama_options(
        temperature=temperature,
        max_tokens=effective_predict,
        num_ctx=effective_ctx,
        num_gpu=num_gpu,
        num_thread=num_thread,
    )
    options["num_predict"] = effective_predict
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "24h"),
        "options": options,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen_fn(req, timeout=timeout) as response:
        if (
            isinstance(response, _HttpStatusProtocol)
            and isinstance(response.status, int)
            and not isinstance(response.status, bool)
            and response.status >= 400
        ):
            raise LlmTranslationError(
                params={"error": f"Ollama HTTP {response.status} for model '{model}'"}
            )
        raw_body = response.read()
        result = json.loads(raw_body.decode("utf-8"))
        err_field = result.get("error")
        if err_field:
            raise LlmTranslationError(params={"error": str(err_field)})
        return str(result.get("response", "")).strip()


call_local_ollama_generate = generate_ollama_text_completion


__all__ = [
    "DEFAULT_CHOICE_HF_REPOS",
    "DEFAULT_CHOICE_OLLAMA_MODELS",
    "DEFAULT_OLLAMA_GENERATE_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_HOST",
    "_FAST_SMALL_OLLAMA_ORDER",
    "_HEAVY_OLLAMA_SUBSTRINGS",
    "_LIGHTWEIGHT_OLLAMA_RANK",
    "OllamaRunnerOptionsDict",
    "_HttpStatusProtocol",
    "_UrlResponseProtocol",
    "_ollama_model_sort_key",
    "_parse_env_int",
    "build_safe_ollama_options",
    "call_local_ollama_generate",
    "configure_ollama_env",
    "generate_ollama_json_completion",
    "generate_ollama_text_completion",
    "is_filesystem_path_token",
    "normalize_ollama_host",
    "probe_ollama_tags_and_version",
    "read_configured_ollama_model_list",
    "resolve_installed_ollama_model_tag",
    "resolve_local_ollama_fallback_candidates",
    "resolve_ollama_base_url",
    "sort_ollama_models",
]
