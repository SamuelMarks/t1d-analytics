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


def test_cli_extract_integration(temp_workspace: Path) -> None:
    """Test CLI extract subcommand integration."""
    import zipfile

    zip_path = temp_workspace / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner.csv", "colA,colB\n1,2\n")

    test_args = ["t1d-analytics", "extract", "-d", str(temp_workspace)]
    with patch.object(sys, "argv", test_args):
        main()

    extracted_file = temp_workspace / "sample" / "inner.csv"
    assert extracted_file.exists()
    assert extracted_file.read_text().startswith("colA,colB")


def test_cli_manifest_integration(temp_workspace: Path) -> None:
    """Test CLI manifest subcommand integration."""
    db_path = temp_workspace / "manifest.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE _t1d_ingestion_manifest (
            source_file TEXT PRIMARY KEY,
            table_name TEXT,
            file_hash TEXT,
            row_count BIGINT,
            loaded_at TIMESTAMP,
            file_size BIGINT
        )
    """
    )
    conn.execute(
        "INSERT INTO _t1d_ingestion_manifest VALUES ('data/file.csv', 'file', 'hash123', 100, CURRENT_TIMESTAMP, 2048)"
    )
    conn.close()

    test_args = ["t1d-analytics", "manifest", "--db", str(db_path)]
    with patch.object(sys, "argv", test_args):
        main()


def test_cli_generate_training_data_integration(
    temp_workspace: Path, mocker: Any
) -> None:
    """Test CLI generate-training-data subcommand integration."""
    db_path = temp_workspace / "gen.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE patients (patient_id INTEGER, age INTEGER)")
    conn.execute("INSERT INTO patients VALUES (1, 25), (2, 30)")
    conn.close()

    mocker.patch(
        "t1d_analytics.training_data.TrainingDataGenerator._generate_pairs",
        return_value=[
            (
                "How many patients?",
                "SELECT count(*) FROM patients",
                "SELECT * FROM patients",
            )
        ],
    )

    test_args = [
        "t1d-analytics",
        "generate-training-data",
        "--db",
        str(db_path),
        "--num-pairs",
        "1",
        "--split-ratios",
        "0.8,0.1,0.1",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    conn = duckdb.connect(str(db_path))
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    assert "sft_data" in tables
    assert "dpo_data" in tables
    conn.close()


def test_cli_export_training_data_jsonl_and_parquet_integration(
    temp_workspace: Path,
) -> None:
    """Test CLI export-training-data subcommand integration for JSONL and Parquet."""
    db_path = temp_workspace / "export.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE sft_data (prompt TEXT, completion TEXT, split TEXT DEFAULT 'train')"
    )
    conn.execute(
        "INSERT INTO sft_data VALUES ('What is avg?', 'SELECT avg(val) FROM tbl', 'train')"
    )
    conn.close()

    # 1. Export JSONL
    jsonl_out = temp_workspace / "exported.jsonl"
    test_args_jsonl = [
        "t1d-analytics",
        "export-training-data",
        "--db",
        str(db_path),
        "--table",
        "sft_data",
        "--format",
        "jsonl",
        "--output-file",
        str(jsonl_out),
    ]
    with patch.object(sys, "argv", test_args_jsonl):
        main()
    assert jsonl_out.exists()
    assert "What is avg?" in jsonl_out.read_text()

    # 2. Export Parquet
    parquet_out = temp_workspace / "exported.parquet"
    test_args_parquet = [
        "t1d-analytics",
        "export-training-data",
        "--db",
        str(db_path),
        "--table",
        "sft_data",
        "--format",
        "parquet",
        "--output-file",
        str(parquet_out),
    ]
    with patch.object(sys, "argv", test_args_parquet):
        main()
    assert parquet_out.exists()


def test_cli_evaluate_integration(temp_workspace: Path, mocker: Any) -> None:
    """Test CLI evaluate subcommand integration."""
    import json

    db_path = temp_workspace / "eval.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE tbl (id INT)")
    conn.execute("INSERT INTO tbl VALUES (1)")
    conn.close()

    test_set_path = temp_workspace / "test_set.json"
    with open(test_set_path, "w") as f:
        json.dump([{"prompt": "Get count", "gold_sql": "SELECT count(*) FROM tbl"}], f)

    mocker.patch(
        "t1d_analytics.api.generate_sql_from_nl",
        return_value=("ok", "SELECT count(*) FROM tbl"),
    )

    report_out = temp_workspace / "report.md"
    test_args = [
        "t1d-analytics",
        "evaluate",
        "--db",
        str(db_path),
        "--test-set",
        str(test_set_path),
        "--output",
        str(report_out),
    ]
    with patch.object(sys, "argv", test_args):
        main()

    assert report_out.exists()
    assert "Text-to-SQL Benchmark Report" in report_out.read_text()


def test_cli_doctor_integration(temp_workspace: Path) -> None:
    """Test CLI doctor subcommand integration."""
    db_path = temp_workspace / "doctor.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE patients (id INT)")
    conn.close()

    test_args = ["t1d-analytics", "doctor", "--db", str(db_path)]
    with patch.object(sys, "argv", test_args):
        main()


def test_cli_profile_integration(temp_workspace: Path) -> None:
    """Test CLI profile subcommand integration."""
    db_path = temp_workspace / "profile.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute('CREATE TABLE "clinical_data" ("id" INT, "reading" FLOAT)')
    conn.execute("INSERT INTO clinical_data VALUES (1, 105.5), (2, 120.0)")
    conn.close()

    test_args = [
        "t1d-analytics",
        "profile",
        "--db",
        str(db_path),
        "--table",
        "clinical_data",
    ]
    with patch.object(sys, "argv", test_args):
        main()
