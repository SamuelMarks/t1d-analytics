"""Tests for the CLI module."""

from unittest.mock import MagicMock, patch

import pytest
from pytest import CaptureFixture
from t1d_analytics.cli import main


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
    mock_parse.assert_called_once_with("<html></html>")
    mock_process.assert_called_once_with(["mock_dataset"], "./data")
    assert "Done!" in capsys.readouterr().out


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

    mock_load.assert_called_once_with("./test_data", "test.db")


@patch("t1d_analytics.analytics.run_query_repl")
def test_query_subcommand(mock_query: MagicMock) -> None:
    """Test successful CLI execution for query."""
    with patch("sys.argv", ["t1d-analytics", "query", "--db", "test.db"]):
        main()

    mock_query.assert_called_once_with("test.db")


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
        mock_connect.return_value, "test_model"
    )
    mock_generator._extract_schema.assert_called_once()
    mock_generator._generate_pairs.assert_called_once_with("schema1", 5)
    mock_generator.write_to_db.assert_called_once_with(
        [("prompt", "chosen", "rejected")]
    )


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
