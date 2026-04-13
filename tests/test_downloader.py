"""Tests for downloader module."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest import CaptureFixture
from requests import HTTPError

from t1d_analytics.downloader import (
    download_file,
    process_datasets,
    sanitize_filename,
)
from t1d_analytics.models import DatasetInfo


def test_sanitize_filename() -> None:
    """Test filename sanitization."""
    assert sanitize_filename("Test (Project) 1!") == "Test_Project_1"
    assert sanitize_filename("  spaces  ") == "spaces"


def test_download_file_doi(tmp_path: Path) -> None:
    """Test downloading a DOI link."""
    download_file("https://doi.org/10.123/456", tmp_path)
    link_file = tmp_path / "dataset_link.txt"
    assert link_file.exists()
    assert link_file.read_text() == "https://doi.org/10.123/456"


def test_download_file_s3(tmp_path: Path, requests_mock: Mock) -> None:
    """Test downloading an S3 file."""
    requests_mock.get("http://test/file%20name.zip", content=b"data")
    download_file("http://test/file%20name.zip", tmp_path)
    target = tmp_path / "file name.zip"
    assert target.exists()
    assert target.read_bytes() == b"data"


def test_download_file_no_filename(tmp_path: Path, requests_mock: Mock) -> None:
    """Test downloading with empty filename."""
    requests_mock.get("http://test/", content=b"data")
    download_file("http://test/", tmp_path)
    target = tmp_path / "downloaded_file"
    assert target.exists()
    assert target.read_bytes() == b"data"


def test_download_file_exists(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Test skipping existing file."""
    target = tmp_path / "file.zip"
    target.write_bytes(b"old")
    download_file("http://test/file.zip", tmp_path)
    assert target.read_bytes() == b"old"
    assert "skipping" in capsys.readouterr().out


def test_download_file_error(tmp_path: Path, requests_mock: Mock) -> None:
    """Test handling HTTP errors during download."""
    requests_mock.get("http://test/file.zip", status_code=500)
    with pytest.raises(HTTPError):
        download_file("http://test/file.zip", tmp_path)


def test_process_datasets(tmp_path: Path, requests_mock: Mock) -> None:
    """Test processing multiple datasets."""
    requests_mock.get("http://test/d1.zip", content=b"d1")
    requests_mock.get("http://test/doc.pdf", content=b"doc")

    datasets = [
        DatasetInfo("Proto 1", "http://test/d1.zip", "http://test/doc.pdf"),
        DatasetInfo("Proto 2", None, None),
    ]

    process_datasets(datasets, str(tmp_path))

    assert (tmp_path / "Proto_1" / "d1.zip").exists()
    assert (tmp_path / "Proto_1" / "doc.pdf").exists()
    assert (tmp_path / "Proto_2").exists()
