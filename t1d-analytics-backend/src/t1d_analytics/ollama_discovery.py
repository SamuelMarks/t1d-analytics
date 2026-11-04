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

from t1d_analytics.api_schemas import ModelInfo

DEFAULT_OLLAMA_HOST: str = "http://127.0.0.1:11434"

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


__all__ = [
    "DEFAULT_CHOICE_HF_REPOS",
    "DEFAULT_CHOICE_OLLAMA_MODELS",
    "DEFAULT_OLLAMA_HOST",
    "_FAST_SMALL_OLLAMA_ORDER",
    "_HEAVY_OLLAMA_SUBSTRINGS",
    "_LIGHTWEIGHT_OLLAMA_RANK",
    "_HttpStatusProtocol",
    "_UrlResponseProtocol",
    "_ollama_model_sort_key",
    "configure_ollama_env",
    "generate_ollama_json_completion",
    "is_filesystem_path_token",
    "normalize_ollama_host",
    "probe_ollama_tags_and_version",
    "read_configured_ollama_model_list",
    "resolve_installed_ollama_model_tag",
    "resolve_ollama_base_url",
    "sort_ollama_models",
]
