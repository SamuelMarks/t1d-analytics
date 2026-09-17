"""Tests for downloader module."""

import typing
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock

import pytest
import requests
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


def test_download_file_content_disposition(tmp_path: Path, requests_mock: Mock) -> None:
    """Test downloading a file using Content-Disposition header filename."""
    requests_mock.get(
        "http://test/download.aspx?id=99",
        content=b"server_data",
        headers={"Content-Disposition": 'attachment; filename="clinical_trials.csv"'},
    )
    download_file("http://test/download.aspx?id=99", tmp_path)
    target = tmp_path / "clinical_trials.csv"
    assert target.exists()
    assert target.read_bytes() == b"server_data"

    # Test skipping when content-disposition file already exists
    download_file("http://test/download.aspx?id=99", tmp_path)


def test_download_file_query_filename_fallback(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test downloading with query parameter filename fallback for server scripts."""
    requests_mock.get(
        "http://test/get.php?dataset=study_records.zip",
        content=b"zip_bytes",
    )
    download_file("http://test/get.php?dataset=study_records.zip", tmp_path)
    target = tmp_path / "study_records.zip"
    assert target.exists()
    assert target.read_bytes() == b"zip_bytes"


def test_download_file_doi_schemes(tmp_path: Path) -> None:
    """Test downloading DOI links with different URL schemes."""
    dir1 = tmp_path / "doi1"
    dir1.mkdir()
    download_file("http://doi.org/10.1000/1", dir1)
    assert (dir1 / "dataset_link.txt").read_text() == "http://doi.org/10.1000/1"

    dir2 = tmp_path / "doi2"
    dir2.mkdir()
    download_file("https://dx.doi.org/10.1000/2", dir2)
    assert (dir2 / "dataset_link.txt").read_text() == "https://dx.doi.org/10.1000/2"

    dir3 = tmp_path / "doi3"
    dir3.mkdir()
    download_file("http://dx.doi.org/10.1000/3", dir3)
    assert (dir3 / "dataset_link.txt").read_text() == "http://dx.doi.org/10.1000/3"


def test_download_file_query_without_matching_param(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test downloading a file with query parameters that do not contain a recognized filename key."""
    requests_mock.get(
        "http://test/export.cgi?report=1&format=raw", content=b"report_content"
    )
    download_file("http://test/export.cgi?report=1&format=raw", tmp_path)
    target = tmp_path / "export.cgi"
    assert target.exists()
    assert target.read_bytes() == b"report_content"


def test_download_file_content_disposition_unmatched(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test downloading when Content-Disposition has filename token but no valid value."""
    requests_mock.get(
        "http://test/doc.bin",
        content=b"bytes",
        headers={"Content-Disposition": "attachment; filename"},
    )
    download_file("http://test/doc.bin", tmp_path)
    target = tmp_path / "doc.bin"
    assert target.exists()
    assert target.read_bytes() == b"bytes"


def test_parse_content_disposition_rfc5987_and_rfc6266() -> None:
    """Test RFC 5987 and RFC 6266 filename extraction from Content-Disposition headers."""
    from t1d_analytics.downloader import parse_content_disposition

    assert parse_content_disposition("") is None
    assert (
        parse_content_disposition(
            "attachment; filename*=UTF-8''clinical%20trial%20data.csv"
        )
        == "clinical trial data.csv"
    )
    assert (
        parse_content_disposition("attachment; filename*=iso-8859-1''test%20study.csv")
        == "test study.csv"
    )
    assert (
        parse_content_disposition('attachment; filename="standard_dataset.csv"')
        == "standard_dataset.csv"
    )
    assert (
        parse_content_disposition("attachment; filename='single_quote_dataset.csv'")
        == "single_quote_dataset.csv"
    )
    assert (
        parse_content_disposition("attachment; filename=unquoted.csv; size=1024")
        == "unquoted.csv"
    )
    assert (
        parse_content_disposition(
            "attachment; filename=\"fallback.csv\"; filename*=UTF-8''prefer_star%20dataset.csv"
        )
        == "prefer_star dataset.csv"
    )
    # Malformed filename* with empty parsed name
    assert parse_content_disposition("attachment; filename*=UTF-8''/") is None
    # Malformed filename= with empty parsed name
    assert parse_content_disposition('attachment; filename="/"') is None


def test_parse_headers_file(tmp_path: Path) -> None:
    """Test parsing headers from JSON and key-value text files."""
    from t1d_analytics.downloader import parse_headers_file

    # Nonexistent
    assert parse_headers_file(tmp_path / "missing.txt") == {}

    # Empty
    empty_f = tmp_path / "empty.txt"
    empty_f.write_text("")
    assert parse_headers_file(empty_f) == {}

    # JSON non-dict
    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2, 3]")
    assert parse_headers_file(list_json) == {}

    # JSON format
    json_f = tmp_path / "headers.json"
    json_f.write_text('{"Authorization": "Bearer token123", "X-Key": "val456"}')
    assert parse_headers_file(json_f) == {
        "Authorization": "Bearer token123",
        "X-Key": "val456",
    }

    # Colon format with comments, invalid lines, and whitespace
    text_f = tmp_path / "headers.txt"
    text_f.write_text(
        "# Custom header config\n"
        "X-Api-Key: secret_abc\n"
        "\n"
        "line_without_colon\n"
        "Accept: text/csv\n"
    )
    assert parse_headers_file(text_f) == {
        "X-Api-Key": "secret_abc",
        "Accept": "text/csv",
    }


def test_download_file_auth_and_headers(
    tmp_path: Path, requests_mock: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test token, basic auth, and custom headers propagation in requests."""
    # 1. Bearer token via argument
    requests_mock.get(
        "http://test/auth.csv",
        request_headers={"Authorization": "Bearer token_arg"},
        content=b"auth_data",
    )
    download_file("http://test/auth.csv", tmp_path, token="token_arg")
    assert (tmp_path / "auth.csv").read_bytes() == b"auth_data"

    # 2. Bearer token via environment variable
    monkeypatch.setenv("T1D_AUTH_TOKEN", "env_token_xyz")
    dir_env = tmp_path / "env"
    dir_env.mkdir()
    requests_mock.get(
        "http://test/env_auth.csv",
        request_headers={"Authorization": "Bearer env_token_xyz"},
        content=b"env_data",
    )
    download_file("http://test/env_auth.csv", dir_env)
    assert (dir_env / "env_auth.csv").read_bytes() == b"env_data"

    # 3. Basic Auth and custom headers
    dir_basic = tmp_path / "basic"
    dir_basic.mkdir()
    requests_mock.get(
        "http://test/basic.csv",
        content=b"basic_data",
    )
    download_file(
        "http://test/basic.csv",
        dir_basic,
        auth=("alice", "secret"),
        headers={"X-Custom": "enabled"},
        cookies={"session": "active"},
    )
    assert (dir_basic / "basic.csv").read_bytes() == b"basic_data"
    last_req = requests_mock.last_request
    assert last_req is not None
    assert last_req.headers["X-Custom"] == "enabled"


def test_download_file_range_resumption(tmp_path: Path, requests_mock: Mock) -> None:
    """Test resuming an interrupted download with HTTP Range requests."""
    dest_path = tmp_path / "resume.dat"
    tmp_path_file = tmp_path / "resume.dat.tmp"

    # Write initial partial data
    tmp_path_file.write_bytes(b"first_half_")

    def matcher(request: requests.PreparedRequest) -> bool:
        return request.headers.get("Range") == "bytes=11-"

    requests_mock.get(
        "http://test/resume.dat",
        additional_matcher=matcher,
        status_code=206,
        content=b"second_half",
        headers={"Content-Length": "11"},
    )

    download_file("http://test/resume.dat", tmp_path, resume=True)
    assert dest_path.exists()
    assert dest_path.read_bytes() == b"first_half_second_half"


def test_download_file_range_416_recovery(tmp_path: Path, requests_mock: Mock) -> None:
    """Test recovery from HTTP 416 Range Not Satisfiable by re-requesting from byte 0."""
    dest_path = tmp_path / "reset.dat"
    tmp_file = tmp_path / "reset.dat.tmp"
    tmp_file.write_bytes(b"stale_bytes")

    # First request with Range returns 416, subsequent returns 200 with full content
    requests_mock.register_uri(
        "GET",
        "http://test/reset.dat",
        [
            {"status_code": 416, "content": b""},
            {"status_code": 200, "content": b"full_clean_content"},
        ],
    )

    download_file("http://test/reset.dat", tmp_path, resume=True)
    assert dest_path.exists()
    assert dest_path.read_bytes() == b"full_clean_content"


def test_download_file_socket_dropout_cleanup(
    tmp_path: Path, mocker: MagicMock
) -> None:
    """Test partial .tmp file cleanup on network error unless keep_partial is set."""
    from unittest.mock import MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Length": "100"}
    mock_resp.raise_for_status.return_value = None

    def fail_chunks(chunk_size: int = 8192) -> typing.Iterator[bytes]:
        yield b"initial_chunk"
        raise ConnectionResetError("Socket reset by peer")

    mock_resp.iter_content.side_effect = fail_chunks

    with patch("requests.get", return_value=mock_resp):
        # 1. keep_partial=False (default): .tmp file should be cleaned up
        with pytest.raises(ConnectionResetError):
            download_file("http://test/dropout.bin", tmp_path, keep_partial=False)
        assert not (tmp_path / "dropout.bin.tmp").exists()
        assert not (tmp_path / "dropout.bin").exists()

        # 2. keep_partial=True: .tmp file is kept for future resumption
        with pytest.raises(ConnectionResetError):
            download_file("http://test/dropout.bin", tmp_path, keep_partial=True)
        assert (tmp_path / "dropout.bin.tmp").exists()
        assert (tmp_path / "dropout.bin.tmp").read_bytes() == b"initial_chunk"


def test_download_file_progress_callback_and_terminal(
    tmp_path: Path, requests_mock: Mock, mocker: MagicMock, capsys: CaptureFixture[str]
) -> None:
    """Test progress callback and terminal animation execution."""
    progress_updates: list[tuple[int, Optional[int]]] = []

    def on_progress(downloaded: int, total: Optional[int]) -> None:
        progress_updates.append((downloaded, total))

    requests_mock.get(
        "http://test/prog.dat",
        content=b"abcdefghij",
        headers={"Content-Length": "10"},
    )

    # 1. Custom progress callback
    download_file(
        "http://test/prog.dat",
        tmp_path,
        progress_callback=on_progress,
    )
    assert len(progress_updates) > 0
    assert progress_updates[-1] == (10, 10)

    # 2. Terminal isatty simulation
    dir_tty = tmp_path / "tty"
    dir_tty.mkdir()
    requests_mock.get(
        "http://test/prog_tty.dat",
        content=b"0123456789",
        headers={"Content-Length": "10"},
    )
    mocker.patch("sys.stdout.isatty", return_value=True)
    download_file("http://test/prog_tty.dat", dir_tty)
    assert (dir_tty / "prog_tty.dat").exists()


def test_process_datasets_concurrency(tmp_path: Path, requests_mock: Mock) -> None:
    """Test process_datasets with concurrent ThreadPoolExecutor execution."""
    for i in range(1, 5):
        requests_mock.get(f"http://test/d{i}.zip", content=f"content_{i}".encode())
        requests_mock.get(f"http://test/doc{i}.pdf", content=f"pdf_{i}".encode())

    datasets = [
        DatasetInfo(f"Study_{i}", f"http://test/d{i}.zip", f"http://test/doc{i}.pdf")
        for i in range(1, 5)
    ]

    process_datasets(datasets, str(tmp_path), concurrency=4)

    for i in range(1, 5):
        assert (tmp_path / f"Study_{i}" / f"d{i}.zip").exists()
        assert (tmp_path / f"Study_{i}" / f"doc{i}.pdf").exists()


def test_download_file_empty_tmp_and_content_disposition_rename(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test handling of 0-byte existing tmp file and renaming tmp when Content-Disposition specifies filename."""
    # Pre-create empty .tmp file
    tmp_file = tmp_path / "stream_data.tmp"
    tmp_file.write_bytes(b"")

    requests_mock.get(
        "http://test/stream_data",
        content=b"content_disposition_renamed",
        headers={"Content-Disposition": 'attachment; filename="target_renamed.csv"'},
    )

    download_file("http://test/stream_data", tmp_path, resume=True)
    target = tmp_path / "target_renamed.csv"
    assert target.exists()
    assert target.read_bytes() == b"content_disposition_renamed"
    assert not tmp_file.exists()


def test_download_file_new_tmp_path_already_exists(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test unlinking existing new_tmp_path when Content-Disposition target temp exists."""
    url_tmp = tmp_path / "data.tmp"
    url_tmp.write_bytes(b"partial url data")
    target_tmp = tmp_path / "actual_name.csv.tmp"
    target_tmp.write_bytes(b"old target data")

    requests_mock.get(
        "http://test/data",
        content=b"fresh full content",
        headers={"Content-Disposition": 'attachment; filename="actual_name.csv"'},
    )

    download_file("http://test/data", tmp_path, resume=True)
    target = tmp_path / "actual_name.csv"
    assert target.exists()
    assert target.read_bytes() == b"fresh full content"


def test_mask_credential() -> None:
    """Test mask_credential for various secret lengths."""
    from t1d_analytics.downloader import mask_credential

    assert mask_credential("") == ""
    assert mask_credential("short") == "*****"
    assert mask_credential("12345678") == "********"
    masked = mask_credential("secret_api_key_value_123456789")
    assert masked.startswith("secr")
    assert masked.endswith("6789")
    assert "*" in masked
    assert "api_key" not in masked


def test_download_file_token_refresh_callback(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test download_file recovers from 401 Unauthorized via token_refresh_callback."""
    # First request returns 401 Unauthorized
    requests_mock.get(
        "http://test/secure_data.csv",
        [
            {"status_code": 401, "text": "Unauthorized"},
            {"status_code": 200, "content": b"authorized_data"},
        ],
    )

    refresh_cb = Mock(return_value="refreshed_token_xyz")
    download_file(
        "http://test/secure_data.csv",
        tmp_path,
        token="expired_token",
        token_refresh_callback=refresh_cb,
    )

    target = tmp_path / "secure_data.csv"
    assert target.exists()
    assert target.read_bytes() == b"authorized_data"
    refresh_cb.assert_called_once()


def test_download_aspnet_postback_file_with_form_data(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test download_aspnet_postback_file with pre-parsed form data."""
    from t1d_analytics.downloader import download_aspnet_postback_file

    requests_mock.post(
        "http://test/portal.aspx",
        content=b"exported_csv_data",
        headers={"Content-Disposition": 'attachment; filename="export_study1.csv"'},
    )

    progress_records: list[tuple[int, Optional[int]]] = []

    def on_progress(downloaded: int, total: Optional[int]) -> None:
        """Track download progress."""
        progress_records.append((downloaded, total))

    saved_path = download_aspnet_postback_file(
        page_url="http://test/portal.aspx",
        event_target="ctl00$CphMain$Grid",
        event_argument="Export$0",
        dest_dir=tmp_path,
        form_data={"__VIEWSTATE": "viewstate_data", "__EVENTVALIDATION": "val_data"},
        progress_callback=on_progress,
    )

    assert saved_path.name == "export_study1.csv"
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"exported_csv_data"
    assert len(progress_records) > 0


def test_download_aspnet_postback_file_fetch_initial_form_and_fallback_filename(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test download_aspnet_postback_file GETs page when form_data is None, and handles fallback filename."""
    from t1d_analytics.downloader import download_aspnet_postback_file

    initial_html = """
    <html>
        <input type="hidden" name="__VIEWSTATE" value="initial_vs" />
        <input type="hidden" name="__EVENTVALIDATION" value="initial_ev" />
    </html>
    """
    requests_mock.get("http://test/datasets.aspx", text=initial_html)
    requests_mock.post(
        "http://test/datasets.aspx",
        content=b"zip_bytes_content",
    )

    saved_path = download_aspnet_postback_file(
        page_url="http://test/datasets.aspx",
        event_target="ctl00$ExportBtn",
        event_argument="Study$A",
        dest_dir=tmp_path,
    )

    assert saved_path.exists()
    assert saved_path.read_bytes() == b"zip_bytes_content"
    assert "ctl00ExportBtn_StudyA.zip" == saved_path.name


def test_download_aspnet_postback_file_sha256_verification(
    tmp_path: Path, requests_mock: Mock
) -> None:
    """Test download_aspnet_postback_file verifies sha256 checksums."""
    import hashlib

    from t1d_analytics.downloader import download_aspnet_postback_file

    content = b"sample_clinical_payload"
    correct_hash = hashlib.sha256(content).hexdigest()

    requests_mock.post(
        "http://test/export.aspx",
        content=content,
        headers={"Content-Disposition": 'attachment; filename="data.csv"'},
    )

    # Valid checksum passes
    p = download_aspnet_postback_file(
        page_url="http://test/export.aspx",
        event_target="ctl",
        event_argument="arg",
        dest_dir=tmp_path / "valid",
        form_data={"__VIEWSTATE": "vs"},
        expected_sha256=correct_hash,
    )
    assert p.exists()

    # Invalid checksum raises ValueError
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_aspnet_postback_file(
            page_url="http://test/export.aspx",
            event_target="ctl",
            event_argument="arg",
            dest_dir=tmp_path / "invalid",
            form_data={"__VIEWSTATE": "vs"},
            expected_sha256="wrong_hash_xyz",
        )


def test_process_datasets_postback_flow(tmp_path: Path, requests_mock: Mock) -> None:
    """Test process_datasets handling DatasetInfo with postback instead of direct URL."""
    requests_mock.get(
        "http://test/portal.aspx",
        text='<input type="hidden" name="__VIEWSTATE" value="vs" />',
    )
    requests_mock.post(
        "http://test/portal.aspx",
        content=b"postback_dataset_bytes",
        headers={"Content-Disposition": 'attachment; filename="postback_data.csv"'},
    )

    ds = DatasetInfo(
        protocol="Protocol_PB",
        dataset_url=None,
        document_url=None,
        postback_target="ctl00$CphMain$Grid",
        postback_argument="Export$0",
    )

    res = process_datasets(
        [ds],
        str(tmp_path),
        page_url="http://test/portal.aspx",
    )

    assert res["total"] == 1
    assert res["succeeded"] == 1
    assert res["failed"] == 0
    saved = tmp_path / "Protocol_PB" / "postback_data.csv"
    assert saved.exists()
    assert saved.read_bytes() == b"postback_dataset_bytes"


def test_download_aspnet_postback_file_with_empty_chunk(
    tmp_path: Path,
) -> None:
    """Test download_aspnet_postback_file handles empty chunks during streaming."""
    from unittest.mock import MagicMock

    from t1d_analytics.downloader import download_aspnet_postback_file

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-disposition": 'attachment; filename="test_chunk.csv"'}
    mock_resp.iter_content.return_value = iter([b"header,", b"", b"value\n"])
    mock_session.post.return_value = mock_resp

    p = download_aspnet_postback_file(
        page_url="http://test/chunk.aspx",
        event_target="target",
        event_argument="arg",
        dest_dir=tmp_path,
        session=mock_session,
        form_data={"__VIEWSTATE": "vs"},
    )
    assert p.exists()
    assert p.read_bytes() == b"header,value\n"
