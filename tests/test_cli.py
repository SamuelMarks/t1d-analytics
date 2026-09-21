"""Tests for the CLI module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from pytest import CaptureFixture

from t1d_analytics.cli import main
from t1d_analytics.diagnostics import OllamaStatus


@patch("t1d_analytics.cli.fetch_html")
@patch("t1d_analytics.cli.parse_datasets")
@patch("t1d_analytics.cli.process_datasets")
def test_main_success(
    mock_process: MagicMock,
    mock_parse: MagicMock,
    mock_fetch: MagicMock,
    capsys: CaptureFixture[str],
) -> None:
    """Test successful CLI execution for download."""
    mock_fetch.return_value = "<html></html>"
    mock_parse.return_value = ["mock_dataset"]

    with patch("sys.argv", ["t1d-analytics", "download"]):
        main()

    mock_fetch.assert_called_once()
    mock_parse.assert_called_once_with(
        "<html></html>", base_url="https://public.t1d.org/datasets/diabetes"
    )
    mock_process.assert_called_once_with(
        ["mock_dataset"],
        "./data",
        concurrency=4,
        token=None,
        auth=None,
        headers=None,
        page_url="https://public.t1d.org/datasets/diabetes",
        token_refresh_callback=None,
    )
    assert "Done!" in capsys.readouterr().out


@patch("t1d_analytics.cli.fetch_html")
@patch("t1d_analytics.cli.parse_datasets")
@patch("t1d_analytics.cli.process_datasets")
def test_main_download_with_auth_and_concurrency(
    mock_process: MagicMock,
    mock_parse: MagicMock,
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    """Test CLI download with token, basic auth, concurrency, and custom headers."""
    mock_fetch.return_value = "<html></html>"
    mock_parse.return_value = ["mock_dataset"]

    headers_file = tmp_path / "headers.json"
    headers_file.write_text('{"X-Custom-Auth": "secret123"}')

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "download",
            "--token",
            "bearer_abc",
            "--user",
            "admin",
            "--password",
            "pass123",
            "--concurrency",
            "8",
            "--headers",
            str(headers_file),
        ],
    ):
        main()

    mock_process.assert_called_once_with(
        ["mock_dataset"],
        "./data",
        concurrency=8,
        token="bearer_abc",
        auth=("admin", "pass123"),
        headers={"X-Custom-Auth": "secret123"},
        page_url="https://public.t1d.org/datasets/diabetes",
        token_refresh_callback=None,
    )


@patch("t1d_analytics.cli.fetch_html")
@patch("t1d_analytics.cli.parse_datasets")
def test_main_no_datasets(
    mock_parse: MagicMock, mock_fetch: MagicMock, capsys: CaptureFixture[str]
) -> None:
    """Test CLI execution with no datasets found."""
    mock_fetch.return_value = "<html></html>"
    mock_parse.return_value = []

    with patch("sys.argv", ["t1d-analytics", "download"]):
        main()

    assert "No datasets found. Exiting." in capsys.readouterr().out


@patch("t1d_analytics.cli.fetch_html")
def test_main_exception(mock_fetch: MagicMock, capsys: CaptureFixture[str]) -> None:
    """Test CLI error handling."""
    mock_fetch.side_effect = Exception("Test Error")

    with patch("sys.argv", ["t1d-analytics", "download"]):
        with pytest.raises(SystemExit) as e:
            main()

    assert e.value.code == 1
    assert "Test Error" in capsys.readouterr().err


@patch("t1d_analytics.analytics.extract_zips")
def test_extract_subcommand(mock_extract: MagicMock) -> None:
    """Test successful CLI execution for extract."""
    with patch("sys.argv", ["t1d-analytics", "extract", "--data-dir", "./test_data"]):
        main()

    mock_extract.assert_called_once_with("./test_data")


@patch("t1d_analytics.analytics.load_data_to_duckdb")
def test_load_subcommand(mock_load: MagicMock) -> None:
    """Test successful CLI execution for load."""
    with patch(
        "sys.argv",
        ["t1d-analytics", "load", "--data-dir", "./test_data", "--db", "test.db"],
    ):
        main()

    mock_load.assert_called_once_with(
        "./test_data",
        "test.db",
        prefix_subdirs=False,
        include_sas=False,
        include_spss=False,
    )


@patch("t1d_analytics.analytics.load_data_to_duckdb")
def test_load_subcommand_with_flags(mock_load: MagicMock) -> None:
    """Test load CLI execution with clinical and prefix flags."""
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "load",
            "--data-dir",
            "./test_data",
            "--db",
            "test.db",
            "--prefix-subdirs",
            "--include-sas",
            "--include-spss",
        ],
    ):
        main()

    mock_load.assert_called_once_with(
        "./test_data",
        "test.db",
        prefix_subdirs=True,
        include_sas=True,
        include_spss=True,
    )


@patch("t1d_analytics.analytics.get_ingestion_manifest")
def test_manifest_subcommand_empty(
    mock_manifest: MagicMock, capsys: CaptureFixture[str]
) -> None:
    """Test manifest subcommand with empty results."""
    mock_manifest.return_value = []
    with patch("sys.argv", ["t1d-analytics", "manifest", "--db", "test.db"]):
        main()

    mock_manifest.assert_called_once_with("test.db")
    assert "No ingestion manifest records found." in capsys.readouterr().out


@patch("t1d_analytics.analytics.get_ingestion_manifest")
def test_manifest_subcommand_with_records(
    mock_manifest: MagicMock, capsys: CaptureFixture[str]
) -> None:
    """Test manifest subcommand displaying records."""
    mock_manifest.return_value = [
        {
            "source_file": "/path/to/data.csv",
            "table_name": "clinical_trial",
            "file_hash": "abc123hash",
            "row_count": 1500,
            "loaded_at": "2026-03-01 12:00:00",
            "file_size": 20480,
        }
    ]
    with patch("sys.argv", ["t1d-analytics", "manifest", "--db", "test.db"]):
        main()

    mock_manifest.assert_called_once_with("test.db")
    out = capsys.readouterr().out
    assert "clinical_trial" in out
    assert "1500" in out
    assert "/path/to/data.csv" in out


@patch("t1d_analytics.analytics.run_query_repl")
def test_query_subcommand(mock_query: MagicMock) -> None:
    """Test successful CLI execution for query."""
    with patch("sys.argv", ["t1d-analytics", "query", "--db", "test.db"]):
        main()

    mock_query.assert_called_once_with("test.db", model="gemma4", provider="ollama")


@patch("t1d_analytics.analytics.run_query_repl")
def test_query_subcommand_with_model(mock_query: MagicMock) -> None:
    """Test query CLI execution with custom model flag."""
    with patch(
        "sys.argv",
        ["t1d-analytics", "query", "--db", "test.db", "--model", "custom-gemma"],
    ):
        main()

    mock_query.assert_called_once_with(
        "test.db", model="custom-gemma", provider="ollama"
    )


@patch("t1d_analytics.analytics.run_query_repl")
def test_query_subcommand_with_provider(mock_query: MagicMock) -> None:
    """Test query CLI execution with provider and custom model flags."""
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "query",
            "--db",
            "test.db",
            "--model",
            "openai/gpt-4o",
            "--provider",
            "openai",
        ],
    ):
        main()

    mock_query.assert_called_once_with(
        "test.db", model="openai/gpt-4o", provider="openai"
    )


@patch("duckdb.connect")
@patch("t1d_analytics.training_data.TrainingDataGenerator")
def test_generate_training_data_subcommand(
    mock_generator_class: MagicMock, mock_connect: MagicMock
) -> None:
    """Test successful CLI execution for generate-training-data."""
    mock_generator = MagicMock()
    mock_generator_class.return_value = mock_generator
    mock_generator._extract_schema.return_value = {"table1": "schema1"}
    mock_generator._generate_pairs.return_value = [("prompt", "chosen", "rejected")]

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "generate-training-data",
            "--db",
            "test.db",
            "--num-pairs",
            "5",
            "--model",
            "test_model",
        ],
    ):
        main()

    mock_connect.assert_called_once_with("test.db")
    mock_generator_class.assert_called_once_with(
        mock_connect.return_value, "test_model", provider="ollama"
    )
    mock_generator._extract_schema.assert_called_once()
    mock_generator._generate_pairs.assert_called_once_with("schema1", 5)
    mock_generator.write_to_db.assert_called_once_with(
        [("prompt", "chosen", "rejected")], split_ratios=None
    )


@patch("duckdb.connect")
@patch("t1d_analytics.training_data.TrainingDataGenerator")
def test_generate_training_data_with_provider_and_ratios(
    mock_generator_class: MagicMock, mock_connect: MagicMock
) -> None:
    """Test generate-training-data with custom provider and split ratios."""
    mock_generator = MagicMock()
    mock_generator_class.return_value = mock_generator
    mock_generator._extract_schema.return_value = {"table1": "schema1"}
    mock_generator._generate_pairs.return_value = [("prompt", "chosen", "rejected")]

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "generate-training-data",
            "--db",
            "test.db",
            "--num-pairs",
            "2",
            "--model",
            "gpt-4o",
            "--provider",
            "openai",
            "--split-ratios",
            "0.8,0.1,0.1",
        ],
    ):
        main()

    mock_generator_class.assert_called_once_with(
        mock_connect.return_value, "gpt-4o", provider="openai"
    )
    mock_generator.write_to_db.assert_called_once_with(
        [("prompt", "chosen", "rejected")], split_ratios=(0.8, 0.1, 0.1)
    )


@patch("t1d_analytics.training_data.evaluate_text_to_sql")
def test_evaluate_subcommand(
    mock_eval: MagicMock, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Test CLI evaluate subcommand with JSON array, JSONL, and output export."""
    # 1. Missing test file
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "evaluate",
            "--db",
            "test.db",
            "--test-set",
            str(tmp_path / "nonexistent.json"),
        ],
    ):
        main()
    assert "does not exist" in capsys.readouterr().out

    # 2. JSON array test set with CSV output
    test_file_json = tmp_path / "test_set.json"
    test_file_json.write_text('[{"prompt": "Get users", "gold_sql": "SELECT 1"}]')
    csv_out = tmp_path / "report.csv"

    mock_eval.return_value = {
        "markdown_report": "# Evaluation Report",
        "csv_report": "question,gold_sql\nGet users,SELECT 1\n",
    }

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "evaluate",
            "--db",
            "test.db",
            "--test-set",
            str(test_file_json),
            "--model",
            "custom_model",
            "--output",
            str(csv_out),
        ],
    ):
        main()

    mock_eval.assert_called_once_with(
        "test.db",
        [{"prompt": "Get users", "gold_sql": "SELECT 1"}],
        model="custom_model",
    )
    assert csv_out.exists()
    assert "Report saved to" in capsys.readouterr().out

    # 3. JSONL test set with Markdown output and blank lines, plus run without --output
    test_file_jsonl = tmp_path / "test_set.jsonl"
    test_file_jsonl.write_text(
        '{"prompt": "Q1", "gold_sql": "SELECT 1"}\n\n{"prompt": "Q2", "gold_sql": "SELECT 2"}\n'
    )
    md_out = tmp_path / "report.md"

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "evaluate",
            "--db",
            "test.db",
            "--test-set",
            str(test_file_jsonl),
            "--output",
            str(md_out),
        ],
    ):
        main()

    assert md_out.exists()
    assert md_out.read_text() == "# Evaluation Report"

    # 4. Evaluate run without --output flag
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "evaluate",
            "--db",
            "test.db",
            "--test-set",
            str(test_file_jsonl),
        ],
    ):
        main()


@patch("duckdb.connect")
@patch("t1d_analytics.training_data.TrainingDataGenerator")
def test_generate_training_data_invalid_ratios_fallback(
    mock_generator_class: MagicMock, mock_connect: MagicMock
) -> None:
    """Test generate-training-data ignores split ratios that do not contain 3 parts."""
    mock_generator = MagicMock()
    mock_generator_class.return_value = mock_generator
    mock_generator._extract_schema.return_value = {"table1": "schema1"}
    mock_generator._generate_pairs.return_value = []

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "generate-training-data",
            "--db",
            "test.db",
            "--num-pairs",
            "1",
            "--split-ratios",
            "0.8,0.2",
        ],
    ):
        main()

    mock_generator.write_to_db.assert_called_once_with([], split_ratios=None)


def test_cli_main_invalid_command(mocker: MagicMock) -> None:
    """Test CLI invalid command."""
    from t1d_analytics.cli import main

    # Passing an invalid command. argparse might catch it if it's restricted by subparsers,
    # but we can mock args to bypass argparse validation and reach the branch.
    mock_args = mocker.MagicMock()
    mock_args.command = "invalid_cmd"
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=mock_args)

    # It should fall through the if-elif chain and exit cleanly (or raise no exception)
    main()


def test_doctor_subcommand(capsys: CaptureFixture[str]) -> None:
    """Test doctor subcommand runs diagnostics and outputs report."""
    with patch("sys.argv", ["t1d-analytics", "doctor", "--db", "missing.duckdb"]):
        main()

    out = capsys.readouterr().out
    assert "T1D Analytics System Health Report" in out
    assert "Database (missing.duckdb):" in out
    assert "Action:" in out


def test_doctor_subcommand_healthy(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Test doctor subcommand when healthy and models are present."""
    db_path = str(tmp_path / "healthy.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE patients (id INT)")
    conn.close()

    with patch(
        "t1d_analytics.diagnostics.check_ollama_health",
        return_value=OllamaStatus(
            accessible=True,
            available_models=["gemma4"],
            message="Online",
            remediation=None,
        ),
    ):
        with patch("sys.argv", ["t1d-analytics", "doctor", "--db", db_path]):
            main()

    out = capsys.readouterr().out
    assert "Overall Status: HEALTHY" in out
    assert "Models:    gemma4" in out


@patch("duckdb.connect")
@patch("t1d_analytics.training_data.TrainingDataGenerator")
def test_export_training_data_subcommand_jsonl(
    mock_generator_class: MagicMock,
    mock_connect: MagicMock,
    capsys: CaptureFixture[str],
) -> None:
    """Test successful CLI execution for export-training-data in JSONL format."""
    mock_gen = MagicMock()
    mock_generator_class.return_value = mock_gen

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "export-training-data",
            "--db",
            "test.db",
            "--table",
            "sft_data",
            "--format",
            "jsonl",
            "--output-file",
            "out.jsonl",
        ],
    ):
        main()

    mock_connect.assert_called_once_with("test.db", read_only=True)
    mock_gen.export_to_jsonl.assert_called_once_with("sft_data", Path("out.jsonl"))
    assert "Export complete!" in capsys.readouterr().out


@patch("duckdb.connect")
@patch("t1d_analytics.training_data.TrainingDataGenerator")
def test_export_training_data_subcommand_parquet(
    mock_generator_class: MagicMock,
    mock_connect: MagicMock,
    capsys: CaptureFixture[str],
) -> None:
    """Test successful CLI execution for export-training-data in Parquet format."""
    mock_gen = MagicMock()
    mock_generator_class.return_value = mock_gen

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "export-training-data",
            "--db",
            "test.db",
            "--table",
            "dpo_data",
            "--format",
            "parquet",
            "--output-file",
            "out.parquet",
        ],
    ):
        main()

    mock_connect.assert_called_once_with("test.db", read_only=True)
    mock_gen.export_to_parquet.assert_called_once_with("dpo_data", Path("out.parquet"))
    assert "Export complete!" in capsys.readouterr().out


@patch("t1d_analytics.cli.fetch_html")
@patch("t1d_analytics.cli.parse_datasets")
@patch("t1d_analytics.cli.process_datasets")
def test_main_download_with_failures(
    mock_process: MagicMock,
    mock_parse: MagicMock,
    mock_fetch: MagicMock,
    capsys: CaptureFixture[str],
) -> None:
    """Test CLI exits with status 1 when download encounters failures."""
    mock_fetch.return_value = "<html></html>"
    mock_parse.return_value = ["mock_dataset"]
    mock_process.return_value = {
        "total": 2,
        "succeeded": 1,
        "failed": 1,
        "errors": [{"protocol": "p1", "kind": "dataset", "error": "timeout"}],
    }

    with patch("sys.argv", ["t1d-analytics", "download"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    err_output = capsys.readouterr().err
    assert "Download encountered errors: 1 file(s) failed." in err_output


@patch("t1d_analytics.analytics.profile_table")
def test_main_profile_subcommand(
    mock_profile: MagicMock, capsys: CaptureFixture[str]
) -> None:
    """Test successful CLI execution for profile subcommand."""
    mock_profile.return_value = {
        "table_name": "demographics",
        "total_rows": 100,
        "column_count": 2,
        "columns": [
            {
                "column_name": "age",
                "data_type": "INTEGER",
                "null_count": 5,
                "null_percentage": 5.0,
                "distinct_count": 50,
                "min": 1,
                "max": 80,
                "mean": 35.5,
            },
            {
                "column_name": "gender",
                "data_type": "VARCHAR",
                "null_count": 0,
                "null_percentage": 0.0,
                "distinct_count": 2,
            },
        ],
    }

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "profile",
            "--db",
            "test.duckdb",
            "--table",
            "demographics",
        ],
    ):
        main()

    out = capsys.readouterr().out
    assert "Table Profile: demographics" in out
    assert "Total Rows: 100" in out
    assert "age (INTEGER)" in out
    assert "gender (VARCHAR)" in out


def test_main_watch(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Test watch subcommand with finite iterations and change detection."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_file = data_dir / "sample.csv"
    sample_file.write_text("id,val\n1,10\n")
    db_file = tmp_path / "watch.duckdb"

    # Run with 2 iterations: iteration 1 sees new file, iteration 2 sees no change
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "watch",
            "--data-dir",
            str(data_dir),
            "--db",
            str(db_file),
            "--interval",
            "0.001",
            "--iterations",
            "2",
        ],
    ):
        main()

    out = capsys.readouterr().out
    assert "Watching directory" in out
    assert "Ingestion complete" in out

    # Run when data-dir does not exist
    non_existent = tmp_path / "does_not_exist"
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "watch",
            "--data-dir",
            str(non_existent),
            "--db",
            str(db_file),
            "--interval",
            "0.001",
            "--iterations",
            "1",
        ],
    ):
        main()


@patch("t1d_analytics.training_data.TrainingDataGenerator._generate_pairs")
def test_main_train(
    mock_generate: MagicMock, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Test train subcommand synthetic orchestrator."""
    db_file = tmp_path / "train.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE subjects (id INT, age INT)")
    conn.execute("INSERT INTO subjects VALUES (1, 25), (2, 30)")
    conn.close()

    mock_generate.return_value = [
        ("How many subjects?", "SELECT COUNT(*) FROM subjects", "SELECT 1")
    ]
    out_dir = tmp_path / "train_out"

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--samples-per-table",
            "1",
            "--export-formats",
            "jsonl,parquet",
        ],
    ):
        main()

    out = capsys.readouterr().out
    assert "T1D Synthetic Training Orchestrator" in out
    assert "Training orchestration completed successfully" in out
    assert (out_dir / "training_data.jsonl").exists()
    assert (out_dir / "training_data.parquet").exists()

    # Test with no formats requested
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--samples-per-table",
            "1",
            "--export-formats",
            "none",
        ],
    ):
        main()


@patch("t1d_analytics.training_data.TrainingDataGenerator._generate_pairs")
def test_train_subcommand_advanced_flags(
    mock_generate: MagicMock, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Test train subcommand with split-validation, execute-training, and manifest creation."""
    db_file = tmp_path / "train_adv.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE subjects (id INT)")
    conn.execute("INSERT INTO subjects VALUES (1)")
    conn.close()

    mock_generate.return_value = [
        (f"Prompt {i}", f"SELECT {i}", f"SELECT {i + 1}") for i in range(10)
    ]
    out_dir = tmp_path / "train_adv_out"

    # Run with execute-training and split-validation (splits created by write_to_db)
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--samples-per-table",
            "10",
            "--backend",
            "remote-tpu-maxtext",
            "--execute-training",
            "--split-validation",
        ],
    ):
        main()

    out = capsys.readouterr().out
    assert "Initiating training runner with backend 'remote-tpu-maxtext'..." in out
    assert "Split validation passed: 0 prompt overlap detected." in out
    manifest_file = out_dir / "training_manifest.json"
    assert manifest_file.exists()

    # Test split-validation warning when tables have overlapping prompts
    mock_generate.return_value = [("Overlap", "SELECT 1", "SELECT 2")] * 10
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--samples-per-table",
            "10",
            "--split-validation",
        ],
    ):
        main()

    out_leak = capsys.readouterr().out
    assert "overlapping prompts between train and val splits" in out_leak

    # Test split-validation when tables are missing
    empty_db = tmp_path / "empty_splits.duckdb"
    conn = duckdb.connect(str(empty_db))
    conn.close()
    mock_generate.return_value = []

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(empty_db),
            "--output-dir",
            str(out_dir),
            "--samples-per-table",
            "0",
            "--export-formats",
            "none",
            "--split-validation",
        ],
    ):
        main()

    out_missing = capsys.readouterr().out
    assert "Split validation skipped: split tables not found." in out_missing


def test_cli_lang_options(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test top-level --lang parameter for all supported languages."""
    missing_db = str(tmp_path / "missing.duckdb")
    expected_snippets = {
        "en": "does not exist",
        "ja": "存在しません",
        "ar": "غير موجود",
        "he": "אינו קיים",
    }
    orig_lang = os.environ.get("LANG")
    try:
        for lang, snippet in expected_snippets.items():
            with patch(
                "sys.argv",
                ["t1d-analytics", "--lang", lang, "doctor", "--db", missing_db],
            ):
                main()
            out = capsys.readouterr().out
            assert (
                snippet in out
            ), f"Expected '{snippet}' in output for lang='{lang}', got: {out}"
    finally:
        if orig_lang is not None:
            os.environ["LANG"] = orig_lang
        else:
            os.environ.pop("LANG", None)
        from t1d_analytics.i18n import get_translator

        get_translator(orig_lang)


@patch("t1d_analytics.training_data.TrainingDataGenerator._generate_pairs")
def test_train_subcommand_hyperparameters_and_validation(
    mock_generate: MagicMock, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Test train subcommand hyperparameter parsing, validation constraints, and dry run."""
    db_file = tmp_path / "train_hp.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE subjects (id INT)")
    conn.execute("INSERT INTO subjects VALUES (1)")
    conn.close()

    mock_generate.return_value = [("Prompt", "SELECT 1", "SELECT 2")]
    out_dir = tmp_path / "hp_out"

    # Successful run with custom hyperparameters and dry-run
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--samples-per-table",
            "1",
            "--learning-rate",
            "1e-4",
            "--batch-size",
            "8",
            "--num-epochs",
            "2",
            "--warmup-steps",
            "20",
            "--weight-decay",
            "0.05",
            "--max-seq-length",
            "1024",
            "--checkpoint-interval",
            "200",
            "--eval-steps",
            "50",
            "--dry-run",
            "--execute-training",
        ],
    ):
        main()

    out = capsys.readouterr().out
    assert "Training run completed for model 'gemma4'" in out
    results_file = out_dir / "training_results.json"
    assert results_file.exists()

    # Invalid learning rate
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--learning-rate",
            "0.0",
        ],
    ):
        with pytest.raises(SystemExit):
            main()

    # Invalid batch size
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--batch-size",
            "0",
        ],
    ):
        with pytest.raises(SystemExit):
            main()

    # Invalid num epochs
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--num-epochs",
            "0",
        ],
    ):
        with pytest.raises(SystemExit):
            main()

    # Invalid warmup steps
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--warmup-steps",
            "-1",
        ],
    ):
        with pytest.raises(SystemExit):
            main()

    # Invalid weight decay
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--weight-decay",
            "-0.01",
        ],
    ):
        with pytest.raises(SystemExit):
            main()

    # Invalid max sequence length
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "train",
            "--db",
            str(db_file),
            "--output-dir",
            str(out_dir),
            "--max-seq-length",
            "32",
        ],
    ):
        with pytest.raises(SystemExit):
            main()


@patch("t1d_analytics.training_data.TrainingDataGenerator._generate_pairs")
def test_train_subcommand_tpu_execution(
    mock_generate: MagicMock, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Test train subcommand remote TPU execution dispatch and config generation."""
    db_file = tmp_path / "train_tpu.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE subjects (id INT)")
    conn.execute("INSERT INTO subjects VALUES (1)")
    conn.close()

    mock_generate.return_value = [("Prompt", "SELECT 1", "SELECT 2")]
    out_dir = tmp_path / "tpu_cli_out"

    with patch.dict(os.environ, {"T1D_MOCK_TPU": "1"}):
        with patch(
            "sys.argv",
            [
                "t1d-analytics",
                "train",
                "--db",
                str(db_file),
                "--output-dir",
                str(out_dir),
                "--samples-per-table",
                "1",
                "--backend",
                "remote-tpu-maxtext",
                "--tpu-zone",
                "us-central2-b",
                "--tpu-type",
                "v4-8",
                "--tpu-slices",
                "2",
                "--gcs-bucket",
                "gs://t1d-test-bucket",
                "--execute-training",
            ],
        ):
            main()

    out = capsys.readouterr().out
    assert "Initiating training runner with backend 'remote-tpu-maxtext'..." in out
    assert (out_dir / "checkpoints" / "maxtext_config.yml").exists()
    assert (out_dir / "training_results.json").exists()


def test_load_auth_plugin_success() -> None:
    """Test load_auth_plugin successfully loads a callable function."""
    from t1d_analytics.cli import load_auth_plugin

    cb = load_auth_plugin("os:getcwd")
    assert callable(cb)
    assert isinstance(cb(), str)


def test_load_auth_plugin_errors() -> None:
    """Test load_auth_plugin error handling for invalid plugin specifications."""
    from t1d_analytics.cli import load_auth_plugin

    # Format error
    with pytest.raises(ValueError, match="Invalid auth plugin specifier"):
        load_auth_plugin("invalid_no_colon")

    # Module import error
    with pytest.raises(ValueError, match="Failed to import auth plugin module"):
        load_auth_plugin("nonexistent_module_xyz:func")

    # Attribute error
    with pytest.raises(ValueError, match="has no attribute"):
        load_auth_plugin("os.path:nonexistent_func_xyz")

    # Not callable error
    with pytest.raises(ValueError, match="is not callable"):
        load_auth_plugin("sys:version")


@patch("t1d_analytics.cli.fetch_html")
@patch("t1d_analytics.cli.parse_datasets")
@patch("t1d_analytics.cli.process_datasets")
def test_main_download_with_auth_plugin(
    mock_process: MagicMock,
    mock_parse: MagicMock,
    mock_fetch: MagicMock,
) -> None:
    """Test CLI download with --auth-plugin dynamically providing Bearer token."""
    mock_fetch.return_value = "<html></html>"
    mock_parse.return_value = ["mock_dataset"]

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "download",
            "--auth-plugin",
            "os.path:basename",
        ],
    ):
        with patch("t1d_analytics.cli.load_auth_plugin") as mock_load:
            mock_plugin = MagicMock(return_value="dyn_token_123")
            mock_load.return_value = mock_plugin
            main()

    mock_process.assert_called_once_with(
        ["mock_dataset"],
        "./data",
        concurrency=4,
        token="dyn_token_123",
        auth=None,
        headers=None,
        page_url="https://public.t1d.org/datasets/diabetes",
        token_refresh_callback=mock_plugin,
    )


@patch("t1d_analytics.cli.fetch_html")
@patch("t1d_analytics.cli.parse_datasets")
@patch("t1d_analytics.cli.process_datasets")
def test_main_download_with_token_and_auth_plugin(
    mock_process: MagicMock,
    mock_parse: MagicMock,
    mock_fetch: MagicMock,
) -> None:
    """Test CLI download preserves explicit --token when --auth-plugin is also provided."""
    mock_fetch.return_value = "<html></html>"
    mock_parse.return_value = ["mock_dataset"]

    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "download",
            "--token",
            "explicit_token_abc",
            "--auth-plugin",
            "os.path:basename",
        ],
    ):
        with patch("t1d_analytics.cli.load_auth_plugin") as mock_load:
            mock_plugin = MagicMock(return_value="dyn_token_123")
            mock_load.return_value = mock_plugin
            main()

    mock_process.assert_called_once_with(
        ["mock_dataset"],
        "./data",
        concurrency=4,
        token="explicit_token_abc",
        auth=None,
        headers=None,
        page_url="https://public.t1d.org/datasets/diabetes",
        token_refresh_callback=mock_plugin,
    )


@patch("t1d_analytics.analytics.export_table")
def test_cli_generic_export_success(
    mock_export: MagicMock,
    capsys: CaptureFixture[str],
) -> None:
    """Test CLI generic export subcommand dispatches to export_table with provided arguments."""
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "export",
            "--db",
            "my_test.duckdb",
            "--table",
            "patients",
            "--format",
            "parquet",
            "--output-file",
            "/tmp/patients.parquet",
        ],
    ):
        main()

    mock_export.assert_called_once_with(
        db_path="my_test.duckdb",
        table_name="patients",
        output_path="/tmp/patients.parquet",
        export_format="parquet",
    )
    out = capsys.readouterr().out
    assert "Exporting table 'patients' to /tmp/patients.parquet..." in out
    assert "Export complete!" in out


def test_cli_doctor_cloud_providers_display(
    capsys: CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Test CLI doctor displays Cloud LLM Providers status and remediation advice."""
    from t1d_analytics.diagnostics import (
        CloudProviderStatus,
        DatabaseStatus,
        DatabaseStatusCode,
        OllamaStatus,
        SystemHealth,
    )

    mock_health = SystemHealth(
        status="healthy",
        backend_accessible=True,
        version="0.1.0",
        database=DatabaseStatus(
            configured_path="test.duckdb",
            exists=True,
            connected=True,
            status_code=DatabaseStatusCode.HEALTHY,
            message="All good",
        ),
        ollama=OllamaStatus(
            accessible=True,
            available_models=["gemma4"],
            message="Online",
        ),
        providers={
            "openai": CloudProviderStatus(
                provider="openai",
                configured=True,
                sdk_installed=True,
                api_key_set=True,
                message="Ready",
            ),
            "anthropic": CloudProviderStatus(
                provider="anthropic",
                configured=False,
                sdk_installed=True,
                api_key_set=False,
                message="Missing key",
                remediation="Set ANTHROPIC_API_KEY",
            ),
        },
    )

    with patch("t1d_analytics.diagnostics.get_system_health", return_value=mock_health):
        with patch("sys.argv", ["t1d-analytics", "doctor"]):
            main()

    out = capsys.readouterr().out
    assert "Cloud LLM Providers:" in out
    assert "Openai:" in out
    assert "Configured:    True" in out
    assert "Anthropic:" in out
    assert "Configured:    False" in out
    assert "Action:        Set ANTHROPIC_API_KEY" in out


def test_cli_bridge_gemma_sql_check(capsys: CaptureFixture[str]) -> None:
    """Test bridge-gemma-sql check action when installed and when missing."""
    with patch(
        "t1d_analytics.gemma_bridge.check_gemma_sql_installed", return_value=True
    ):
        with patch("sys.argv", ["t1d-analytics", "bridge-gemma-sql", "check"]):
            main()
    out = capsys.readouterr().out
    assert "gemma-4-sql installed: True" in out

    with patch(
        "t1d_analytics.gemma_bridge.check_gemma_sql_installed", return_value=False
    ):
        with patch("sys.argv", ["t1d-analytics", "bridge-gemma-sql", "check"]):
            main()
    out2 = capsys.readouterr().out
    assert "gemma-4-sql installed: False" in out2
    assert "Action: Install" in out2


def test_cli_bridge_gemma_sql_etl(capsys: CaptureFixture[str]) -> None:
    """Test bridge-gemma-sql etl action dispatches with correct arguments."""
    mock_proc = MagicMock(returncode=0, stdout="Mock ETL Done")
    with patch(
        "t1d_analytics.gemma_bridge.run_gemma_sql_etl", return_value=mock_proc
    ) as mock_etl:
        with patch(
            "sys.argv",
            [
                "t1d-analytics",
                "bridge-gemma-sql",
                "etl",
                "--stage",
                "sft",
                "--db",
                "my.duckdb",
                "--table",
                "sft_data",
                "--output-dir",
                "/tmp/out",
            ],
        ):
            main()

    mock_etl.assert_called_once_with(
        stage="sft",
        duckdb_path="my.duckdb",
        duckdb_table="sft_data",
        output_dir="/tmp/out",
    )
    out = capsys.readouterr().out
    assert "Mock ETL Done" in out
    assert "gemma-4-sql ETL completed successfully." in out


def test_cli_bridge_gemma_sql_train(capsys: CaptureFixture[str]) -> None:
    """Test bridge-gemma-sql train action with valid config and missing config."""
    # 1. Success with --config
    mock_proc = MagicMock(returncode=0, stdout="Mock Train Done")
    with patch(
        "t1d_analytics.gemma_bridge.run_gemma_sql_train", return_value=mock_proc
    ) as mock_train:
        with patch(
            "sys.argv",
            [
                "t1d-analytics",
                "bridge-gemma-sql",
                "train",
                "--stage",
                "pretrain",
                "--config",
                "maxtext.yml",
            ],
        ):
            main()

    mock_train.assert_called_once_with(
        stage="pretrain",
        config_path="maxtext.yml",
    )
    out = capsys.readouterr().out
    assert "Mock Train Done" in out
    assert "gemma-4-sql training completed successfully." in out

    # 2. Missing --config
    with patch(
        "sys.argv",
        [
            "t1d-analytics",
            "bridge-gemma-sql",
            "train",
            "--stage",
            "pretrain",
        ],
    ):
        with pytest.raises(SystemExit):
            main()
    err = capsys.readouterr().err
    assert "--config path is required" in err
