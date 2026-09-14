"""Downloader module for saving files and links."""

import hashlib
import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import requests

from t1d_analytics.i18n import get_translator
from t1d_analytics.models import DatasetInfo


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


def download_file(
    url: str, dest_dir: Path, expected_sha256: Optional[str] = None
) -> None:
    """
    Download a file from a URL or save a DOI link safely and atomically.

    Args:
    ----
        url: The URL to download.
        dest_dir: The directory to save the file.
        expected_sha256: Optional expected SHA-256 hex digest for integrity verification.

    """
    _ = get_translator()
    if url.startswith("https://doi.org/"):
        link_file = dest_dir / "dataset_link.txt"
        if not link_file.exists():
            link_file.write_text(url)
            print(_("Saved DOI link: {}", url))
        else:
            print(_("DOI link already exists, skipping: {}", url))
        return

    parsed_path = urlparse(url).path
    filename = unquote(parsed_path.split("/")[-1]) if parsed_path else ""
    if not filename or filename == "/":
        filename = "downloaded_file"

    dest_path = dest_dir / filename
    if dest_path.exists():
        print(_("File already exists, skipping: {}", filename))
        return

    tmp_path = dest_path.with_name(f"{filename}.tmp")
    print(_("Downloading {}...", filename))
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        hasher = hashlib.sha256() if expected_sha256 else None
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)

        if expected_sha256 and hasher:
            digest = hasher.hexdigest()
            if digest.lower() != expected_sha256.lower():
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {filename}: expected {expected_sha256}, got {digest}"
                )

        tmp_path.replace(dest_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        print(_("Failed to download {}: {}", filename, e))
        raise


def process_datasets(datasets: list[DatasetInfo], output_dir: str) -> None:
    """
    Process and download all given datasets without halting on single file failures.

    Args:
    ----
        datasets: List of DatasetInfo objects.
        output_dir: Base directory to save downloads.

    """
    _ = get_translator()
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        folder_name = sanitize_filename(dataset.protocol)
        dest_dir = base_path / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(_("Processing protocol: {}", dataset.protocol))
        if dataset.dataset_url:
            try:
                download_file(dataset.dataset_url, dest_dir)
            except Exception as e:
                print(
                    _(
                        "Error downloading dataset for protocol {}: {}",
                        dataset.protocol,
                        e,
                    )
                )
        if dataset.document_url:
            try:
                download_file(dataset.document_url, dest_dir)
            except Exception as e:
                print(
                    _(
                        "Error downloading document for protocol {}: {}",
                        dataset.protocol,
                        e,
                    )
                )
