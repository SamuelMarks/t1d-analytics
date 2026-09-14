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
    assert sanitize_filename("???!!!") == "dataset"
    assert sanitize_filename("") == "dataset"


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


def test_download_file_doi_exists(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Test skipping existing DOI link."""
    link_file = tmp_path / "dataset_link.txt"
    link_file.write_text("https://doi.org/10.123/456")
    download_file("https://doi.org/10.123/456", tmp_path)
    assert "DOI link already exists" in capsys.readouterr().out


def test_download_file_with_query_params(tmp_path: Path, requests_mock: Mock) -> None:
    """Test downloading a file whose URL has query parameters."""
    requests_mock.get("http://test/data.zip?auth=secret&version=2", content=b"content")
    download_file("http://test/data.zip?auth=secret&version=2", tmp_path)
    target = tmp_path / "data.zip"
    assert target.exists()
    assert target.read_bytes() == b"content"


def test_download_file_sha256_verification(tmp_path: Path, requests_mock: Mock) -> None:
    """Test downloading a file with matching and mismatching SHA-256."""
    import hashlib

    data = b"verified_content"
    correct_hash = hashlib.sha256(data).hexdigest()
    requests_mock.get("http://test/hashed.csv", content=data)

    # Success case
    download_file("http://test/hashed.csv", tmp_path, expected_sha256=correct_hash)
    assert (tmp_path / "hashed.csv").exists()

    # Mismatch case
    tmp_path2 = tmp_path / "sub"
    tmp_path2.mkdir()
    requests_mock.get("http://test/bad_hash.csv", content=data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_file(
            "http://test/bad_hash.csv",
            tmp_path2,
            expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
    assert not (tmp_path2 / "bad_hash.csv").exists()


def test_process_datasets_with_errors(
    tmp_path: Path, requests_mock: Mock, capsys: CaptureFixture[str]
) -> None:
    """Test process_datasets continues when an individual file fails."""
    requests_mock.get("http://test/broken_dataset.zip", status_code=500)
    requests_mock.get("http://test/broken_doc.pdf", status_code=404)

    datasets = [
        DatasetInfo(
            "FailedProtocol",
            "http://test/broken_dataset.zip",
            "http://test/broken_doc.pdf",
        )
    ]
    process_datasets(datasets, str(tmp_path))
    out = capsys.readouterr().out
    assert "Error downloading dataset" in out
    assert "Error downloading document" in out


def test_download_file_empty_chunks(tmp_path: Path) -> None:
    """Test downloading a file with empty chunks yielded by iter_content."""
    from unittest.mock import MagicMock, patch

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"chunk1", b"", b"chunk2"]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        download_file("http://test/chunks.csv", tmp_path)
        target = tmp_path / "chunks.csv"
        assert target.exists()
        assert target.read_bytes() == b"chunk1chunk2"
