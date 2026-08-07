"""Integration tests for the CLI."""

import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import duckdb
import pytest
from t1d_analytics.cli import main


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """
    Create a temporary workspace directory for integration tests.

    Yields
    ------
        Path: The temporary directory path.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_cli_integration_flow(requests_mock: Any, temp_workspace: Path) -> None:
    """Test the complete CLI integration flow: download, load, and query."""
    # Setup mock for parser
    requests_mock.get(
        "http://fake.url",
        text="""
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr>
            <td>TestProtocol</td><td></td><td></td><td></td>
            <td><a data-url="https://doi.org/10.1234/test_dataset">Dataset</a></td>
            <td><a data-url="http://example.com/doc.txt">Doc</a></td>
        </tr>
    </table>
    """,
    )

    # Setup mock for downloader
    requests_mock.get("http://example.com/doc.txt", text="dummy doc content")

    data_dir = temp_workspace / "data"
    db_path = temp_workspace / "integration.duckdb"

    # Test Download
    test_args_download = [
        "t1d-analytics",
        "download",
        "-u",
        "http://fake.url",
        "-o",
        str(data_dir),
    ]
    with patch.object(sys, "argv", test_args_download):
        main()

    # Verify download created folders and files
    protocol_dir = data_dir / "TestProtocol"
    assert protocol_dir.exists()
    assert (protocol_dir / "dataset_link.txt").exists()
    assert (protocol_dir / "doc.txt").exists()

    # Provide a dummy CSV to simulate extraction/download of actual data
    with open(protocol_dir / "test_data.csv", "w", encoding="utf-8") as f:
        f.write("id,value\n1,100\n2,200\n")

    # Test Load
    test_args_load = [
        "t1d-analytics",
        "load",
        "-d",
        str(data_dir),
        "--db",
        str(db_path),
    ]
    with patch.object(sys, "argv", test_args_load):
        main()

    # Verify DuckDB database
    conn = duckdb.connect(str(db_path), read_only=True)
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    assert "test_data" in tables
    rows = conn.execute("SELECT * FROM test_data").fetchall()
    assert len(rows) == 2
    assert rows[0][0] == 1
    conn.close()

    # Test Query (Mocking input)
    test_args_query = ["t1d-analytics", "query", "--db", str(db_path)]
    with (
        patch.object(sys, "argv", test_args_query),
        patch("builtins.input", side_effect=["SELECT count(*) FROM test_data", "exit"]),
    ):
        main()
