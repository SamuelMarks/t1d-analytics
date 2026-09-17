"""Downloader module for saving files and links."""

import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

import requests

from t1d_analytics.i18n import get_translator
from t1d_analytics.models import DatasetInfo

_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _get_file_lock(target_path: Path) -> threading.Lock:
    """
    Get or create a thread lock for a given destination file path to prevent concurrent writes.

    Args:
    ----
        target_path: Destination file path.

    Returns:
    -------
        threading.Lock: Mutex for synchronizing operations on this file path.

    """
    resolved = str(target_path.resolve())
    with _file_locks_guard:
        if resolved not in _file_locks:
            _file_locks[resolved] = threading.Lock()
        return _file_locks[resolved]


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a safe filename or directory name.

    Args:
    ----
        name: The original string.

    Returns:
    -------
        A sanitized string safe for file systems.

    """
    sanitized = re.sub(r"[^a-zA-Z0-9\-_ ]", "", name)
    sanitized = sanitized.strip().replace(" ", "_")
    return sanitized if sanitized else "dataset"


def mask_credential(val: str, visible_chars: int = 4) -> str:
    """
    Securely mask credentials, bearer tokens, and session secrets for logs and display.

    Args:
    ----
        val: The raw secret string to mask.
        visible_chars: Number of prefix and suffix characters to leave unmasked.

    Returns:
    -------
        str: Masked string with asterisks replacing sensitive characters.

    """
    if not val:
        return ""
    if len(val) <= visible_chars * 2:
        return "*" * len(val)
    return f"{val[:visible_chars]}{'*' * (len(val) - visible_chars * 2)}{val[-visible_chars:]}"


def parse_content_disposition(header: str) -> Optional[str]:
    """
    Parse filename from Content-Disposition header according to RFC 5987 and RFC 6266.

    Args:
    ----
        header: The Content-Disposition header value.

    Returns:
    -------
        Optional[str]: Extracted filename if found, otherwise None.

    """
    if not header or not isinstance(header, str):
        return None

    # RFC 5987 filename* takes precedence
    match_star = re.search(
        r"filename\*\s*=\s*(?:[a-zA-Z0-9_\-]+)?''([^;\r\n]+)",
        header,
        re.IGNORECASE,
    )
    if match_star:
        raw_val = match_star.group(1).strip().strip("\"'")
        parsed_name = Path(unquote(raw_val)).name
        if parsed_name:
            return parsed_name

    # Fallback to standard filename parameter
    match_std = re.search(
        r'filename\s*=\s*["\']?([^"\';\r\n]+)["\']?',
        header,
        re.IGNORECASE,
    )
    if match_std:
        raw_val = match_std.group(1).strip().strip("\"'")
        parsed_name = Path(unquote(raw_val)).name
        if parsed_name:
            return parsed_name

    return None


def parse_headers_file(file_path: Path) -> dict[str, str]:
    """
    Parse a custom headers file in JSON or colon-separated key-value format.

    Args:
    ----
        file_path: Path to the headers file.

    Returns:
    -------
        dict[str, str]: Dictionary of parsed header key-value pairs.

    """
    if not file_path.exists():
        return {}
    content = file_path.read_text().strip()
    if not content:
        return {}
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass

    headers: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers


def download_file(
    url: str,
    dest_dir: Path,
    expected_sha256: Optional[str] = None,
    token: Optional[str] = None,
    auth: Optional[tuple[str, str]] = None,
    headers: Optional[dict[str, str]] = None,
    cookies: Optional[dict[str, str]] = None,
    resume: bool = True,
    keep_partial: bool = False,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    token_refresh_callback: Optional[Callable[[], str]] = None,
) -> None:
    """
    Download a file from a URL or save a DOI link safely, resiliently, and atomically.

    Supports authentication, custom headers, HTTP Range resumption, SHA-256 integrity verification,
    token refresh callback, and progress tracking.

    Args:
    ----
        url: The URL to download.
        dest_dir: The directory to save the file.
        expected_sha256: Optional expected SHA-256 hex digest for integrity verification.
        token: Optional Bearer token for authorization.
        auth: Optional HTTP Basic auth tuple (username, password).
        headers: Optional custom HTTP request headers.
        cookies: Optional session cookies.
        resume: Whether to resume partial downloads using HTTP Range requests.
        keep_partial: Whether to preserve partial .tmp file on download error.
        progress_callback: Optional callback invoked with (downloaded_bytes, total_bytes).
        token_refresh_callback: Optional callback invoked on 401 to refresh Bearer token.

    """
    _ = get_translator()
    is_doi = (
        url.startswith("https://doi.org/")
        or url.startswith("http://doi.org/")
        or url.startswith("https://dx.doi.org/")
        or url.startswith("http://dx.doi.org/")
    )
    if is_doi:
        link_file = dest_dir / "dataset_link.txt"
        if not link_file.exists():
            link_file.write_text(url)
            print(_("Saved DOI link: {}", url))
        else:
            print(_("DOI link already exists, skipping: {}", url))
        return

    parsed = urlparse(url)
    unquoted_path = unquote(parsed.path) if parsed.path else ""
    raw_filename = unquoted_path.split("/")[-1] if unquoted_path else ""
    filename = Path(raw_filename).name if raw_filename else ""

    if (
        not filename
        or filename == "/"
        or filename.lower().endswith((".aspx", ".php", ".jsp", ".cgi"))
    ):
        query_filename = None
        if parsed.query:
            params = urllib.parse.parse_qs(parsed.query)
            for k in ("file", "filename", "name", "dataset"):
                if k in params and params[k] and Path(unquote(params[k][0])).name:
                    query_filename = Path(unquote(params[k][0])).name
                    break
        filename = query_filename or filename or "downloaded_file"

    dest_path = dest_dir / filename
    with _get_file_lock(dest_path):
        if dest_path.exists():
            print(_("File already exists, skipping: {}", filename))
            return

        tmp_path = dest_path.with_name(f"{filename}.tmp")
        print(_("Downloading {}...", filename))

        req_headers = dict(headers or {})
        effective_token = token or os.environ.get("T1D_AUTH_TOKEN")
        if effective_token and "Authorization" not in req_headers:
            req_headers["Authorization"] = f"Bearer {effective_token}"

        existing_bytes = 0
        if resume and tmp_path.exists():
            existing_bytes = tmp_path.stat().st_size
            if existing_bytes > 0:
                req_headers["Range"] = f"bytes={existing_bytes}-"

        try:
            response = requests.get(
                url,
                stream=True,
                headers=req_headers,
                auth=auth,
                cookies=cookies,
                timeout=30,
            )
            try:
                # Handle 401 Unauthorized via token refresh callback if configured
                if response.status_code == 401 and token_refresh_callback:
                    response.close()
                    effective_token = token_refresh_callback()
                    req_headers["Authorization"] = f"Bearer {effective_token}"
                    response = requests.get(
                        url,
                        stream=True,
                        headers=req_headers,
                        auth=auth,
                        cookies=cookies,
                        timeout=30,
                    )

                # Handle Range Not Satisfiable by re-requesting from byte 0
                if existing_bytes > 0 and response.status_code == 416:
                    response.close()
                    tmp_path.unlink(missing_ok=True)
                    existing_bytes = 0
                    req_headers.pop("Range", None)
                    response = requests.get(
                        url,
                        stream=True,
                        headers=req_headers,
                        auth=auth,
                        cookies=cookies,
                        timeout=30,
                    )

                response.raise_for_status()

                cd = response.headers.get("content-disposition")
                parsed_cd_name = parse_content_disposition(cd or "")
                if parsed_cd_name:
                    filename = parsed_cd_name
                    dest_path = dest_dir / filename
                    if dest_path.exists():
                        tmp_path.unlink(missing_ok=True)
                        print(_("File already exists, skipping: {}", filename))
                        return
                    new_tmp_path = dest_path.with_name(f"{filename}.tmp")
                    if new_tmp_path != tmp_path and tmp_path.exists():
                        if new_tmp_path.exists():
                            new_tmp_path.unlink(missing_ok=True)
                        tmp_path.rename(new_tmp_path)
                    tmp_path = new_tmp_path

                is_partial = response.status_code == 206
                mode = "ab" if (is_partial and existing_bytes > 0) else "wb"
                if not is_partial:
                    existing_bytes = 0

                total_size_header = response.headers.get("content-length")
                total_size: Optional[int] = (
                    int(total_size_header)
                    if total_size_header and total_size_header.isdigit()
                    else None
                )
                if total_size is not None and is_partial:
                    total_size += existing_bytes

                downloaded = existing_bytes
                start_time = time.time()

                with open(tmp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                            elif sys.stdout.isatty():
                                elapsed = max(time.time() - start_time, 1e-4)
                                rate_mb = (
                                    (downloaded - existing_bytes)
                                    / elapsed
                                    / (1024 * 1024)
                                )
                                pct = (
                                    (downloaded / total_size * 100)
                                    if total_size
                                    else 0.0
                                )
                                msg = f"{filename}: {downloaded / (1024 * 1024):.1f} MB ({pct:.1f}%) @ {rate_mb:.2f} MB/s"
                                sys.stdout.write("\r" + msg)
                                sys.stdout.flush()

                if sys.stdout.isatty() and progress_callback is None:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                if expected_sha256:
                    hasher = hashlib.sha256()
                    with open(tmp_path, "rb") as f:
                        while True:
                            block = f.read(65536)
                            if not block:
                                break
                            hasher.update(block)
                    digest = hasher.hexdigest()
                    if digest.lower() != expected_sha256.lower():
                        tmp_path.unlink(missing_ok=True)
                        raise ValueError(
                            f"SHA-256 mismatch for {filename}: expected {expected_sha256}, got {digest}"
                        )

                tmp_path.replace(dest_path)
            finally:
                response.close()
        except Exception as e:
            if not keep_partial:
                tmp_path.unlink(missing_ok=True)
            print(_("Failed to download {}: {}", filename, e))
            raise


def download_aspnet_postback_file(
    page_url: str,
    event_target: str,
    event_argument: str,
    dest_dir: Path,
    session: Optional[requests.Session] = None,
    form_data: Optional[dict[str, str]] = None,
    extra_headers: Optional[dict[str, str]] = None,
    expected_sha256: Optional[str] = None,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Path:
    """
    Simulate ASP.NET __doPostBack submission to download exported dataset files.

    Maintains session state and cookies across simulated POST submissions.

    Args:
    ----
        page_url: The URL of the ASP.NET page containing the form and GridView.
        event_target: Target control identifier (__EVENTTARGET, e.g. ctl00$CphMain$GridView).
        event_argument: Argument passed to event (__EVENTARGUMENT, e.g. Select$0).
        dest_dir: Directory to save the exported file.
        session: Optional requests.Session to persist cookies; creates new Session if None.
        form_data: Optional initial hidden form state fields. If None, GETs page_url to extract.
        extra_headers: Optional HTTP headers for the postback request.
        expected_sha256: Optional expected SHA-256 checksum to verify download integrity.
        progress_callback: Optional progress callback function.

    Returns:
    -------
        Path: The path to the downloaded file.

    Raises:
    ------
        ValueError: If SHA-256 checksum verification fails.
        RuntimeError: If download fails or HTTP status is erroneous.

    """
    from t1d_analytics.parser import extract_aspnet_form_data

    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()

    current_form_data = form_data
    if current_form_data is None:
        init_resp = sess.get(page_url, headers=extra_headers, timeout=30)
        init_resp.raise_for_status()
        current_form_data = extract_aspnet_form_data(init_resp.text)

    post_payload = dict(current_form_data)
    post_payload["__EVENTTARGET"] = event_target
    post_payload["__EVENTARGUMENT"] = event_argument

    post_headers = dict(extra_headers or {})
    post_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    resp = sess.post(
        page_url,
        data=post_payload,
        headers=post_headers,
        stream=True,
        timeout=60,
    )
    try:
        resp.raise_for_status()
        cd = resp.headers.get("content-disposition")
        filename = parse_content_disposition(cd or "")
        if not filename:
            filename = f"{sanitize_filename(event_target)}_{sanitize_filename(event_argument)}.zip"

        dest_path = dest_dir / filename
        tmp_path = dest_path.with_name(f"{filename}.tmp")

        downloaded = 0
        total_size_header = resp.headers.get("content-length")
        total_size = (
            int(total_size_header)
            if total_size_header and total_size_header.isdigit()
            else None
        )

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

        if expected_sha256:
            hasher = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                while True:
                    b = f.read(65536)
                    if not b:
                        break
                    hasher.update(b)
            digest = hasher.hexdigest()
            if digest.lower() != expected_sha256.lower():
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {filename}: expected {expected_sha256}, got {digest}"
                )

        tmp_path.replace(dest_path)
        return dest_path
    finally:
        resp.close()


def process_datasets(
    datasets: list[DatasetInfo],
    output_dir: str,
    concurrency: int = 4,
    token: Optional[str] = None,
    auth: Optional[tuple[str, str]] = None,
    headers: Optional[dict[str, str]] = None,
    cookies: Optional[dict[str, str]] = None,
    page_url: Optional[str] = None,
    token_refresh_callback: Optional[Callable[[], str]] = None,
) -> dict[str, Any]:
    """
    Process and download all given datasets in parallel without halting on single file failures.

    Args:
    ----
        datasets: List of DatasetInfo objects.
        output_dir: Base directory to save downloads.
        concurrency: Maximum number of concurrent downloads (default: 4).
        token: Optional Bearer token for authentication.
        auth: Optional HTTP Basic auth tuple (user, password).
        headers: Optional custom HTTP headers.
        cookies: Optional session cookies.
        page_url: Optional page URL for resolving simulated ASP.NET postbacks.
        token_refresh_callback: Optional callback to refresh Bearer token on 401.

    Returns:
    -------
        dict[str, Any]: Metrics containing total, succeeded, failed counts, and error list.

    """
    _ = get_translator()
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    print_lock = threading.Lock()
    failed_items: list[dict[str, str]] = []
    succeeded_items: list[str] = []

    def download_item(dataset: DatasetInfo, is_doc: bool) -> None:
        """
        Download a single dataset or document item.

        Args:
        ----
            dataset: DatasetInfo metadata object.
            is_doc: Whether this download task represents a document link.

        """
        folder_name = sanitize_filename(dataset.protocol)
        dest_dir = base_path / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        url = dataset.document_url if is_doc else dataset.dataset_url
        expected_hash = None if is_doc else dataset.expected_sha256
        kind = "document" if is_doc else "dataset"

        try:
            if not is_doc and not url and dataset.postback_target:
                target_page = page_url or "https://public.t1d.org/datasets/diabetes"
                download_aspnet_postback_file(
                    target_page,
                    dataset.postback_target,
                    dataset.postback_argument or "",
                    dest_dir,
                    extra_headers=headers,
                    expected_sha256=expected_hash,
                )
                with print_lock:
                    succeeded_items.append(f"{dataset.protocol}:{kind}")
                return

            assert url is not None
            download_file(
                url,
                dest_dir,
                expected_sha256=expected_hash,
                token=token,
                auth=auth,
                headers=headers,
                cookies=cookies,
                token_refresh_callback=token_refresh_callback,
            )
            with print_lock:
                succeeded_items.append(f"{dataset.protocol}:{kind}")
        except Exception as e:
            with print_lock:
                failed_items.append(
                    {
                        "protocol": dataset.protocol,
                        "kind": kind,
                        "error": str(e),
                    }
                )
                print(
                    _(
                        f"Error downloading {kind} for protocol {{}}: {{}}",
                        dataset.protocol,
                        e,
                    )
                )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, concurrency)
    ) as executor:
        futures = []
        for dataset in datasets:
            folder_name = sanitize_filename(dataset.protocol)
            dest_dir = base_path / folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            print(_("Processing protocol: {}", dataset.protocol))

            if dataset.dataset_url or dataset.postback_target:
                futures.append(executor.submit(download_item, dataset, False))
            if dataset.document_url:
                futures.append(executor.submit(download_item, dataset, True))

        concurrent.futures.wait(futures)

    return {
        "total": len(futures),
        "succeeded": len(succeeded_items),
        "failed": len(failed_items),
        "errors": failed_items,
    }
