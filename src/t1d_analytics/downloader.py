"""Downloader module for saving files and links."""

import re
from pathlib import Path
from urllib.parse import unquote

import requests

from t1d_analytics.models import DatasetInfo


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a safe filename or directory name.

    Args:
        name: The original string.

    Returns:
        A sanitized string.

    """
    name = re.sub(r"[^a-zA-Z0-9\-_ ]", "", name)
    name = name.strip().replace(" ", "_")
    return name


def download_file(url: str, dest_dir: Path) -> None:
    """
    Download a file from a URL or save a DOI link.

    Args:
        url: The URL to download.
        dest_dir: The directory to save the file.

    Raises:
        requests.RequestException: If the HTTP request fails.

    """
    if url.startswith("https://doi.org/"):
        link_file = dest_dir / "dataset_link.txt"
        if not link_file.exists():
            link_file.write_text(url)
            print(f"Saved DOI link: {url}")
        else:
            print(f"DOI link already exists, skipping: {url}")
        return

    filename = unquote(url.split("/")[-1])
    if not filename:
        filename = "downloaded_file"

    dest_path = dest_dir / filename
    if dest_path.exists():
        print(f"File already exists, skipping: {filename}")
        return

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def process_datasets(datasets: list[DatasetInfo], output_dir: str) -> None:
    """
    Process and download all given datasets.

    Args:
        datasets: List of DatasetInfo objects.
        output_dir: Base directory to save downloads.

    """
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        folder_name = sanitize_filename(dataset.protocol)
        dest_dir = base_path / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing protocol: {dataset.protocol}")
        if dataset.dataset_url:
            download_file(dataset.dataset_url, dest_dir)
        if dataset.document_url:
            download_file(dataset.document_url, dest_dir)
