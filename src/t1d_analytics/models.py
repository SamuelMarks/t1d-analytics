"""Data models for T1D downloader."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetInfo:
    """Information about a dataset parsed from the website."""

    protocol: str
    dataset_url: Optional[str]
    document_url: Optional[str]
