"""Tests for the training_data module."""

import json
from typing import Generator
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from t1d_analytics.training_data import TrainingDataGenerator


@pytest.fixture
def mock_db() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Provide an in-memory DuckDB connection pre-populated with a dummy schema.

    Yields:
        A DuckDB connection with a dummy `demographics` table.

    """
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE demographics (
            patient_id INTEGER,
            age INTEGER,
            hba1c DOUBLE
        )
        """
    )
    yield conn
    conn.close()


def test_extract_schema(mock_db: duckdb.DuckDBPyConnection) -> None:
    """Test extracting the schema from the DuckDB connection."""
    generator = TrainingDataGenerator(mock_db, "gemma4")
    schema = generator._extract_schema()
    
    assert "demographics" in schema
    desc = schema["demographics"]
    assert "Table: demographics" in desc
    assert "- patient_id (INTEGER)" in desc
    assert "- age (INTEGER)" in desc
    assert "- hba1c (DOUBLE)" in desc


@patch("urllib.request.urlopen")
def test_generate_pairs_success(mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection) -> None:
    """Test generating pairs with a successful LLM response."""
    # Create a mock response
    mock_response = MagicMock()
    # The JSON string response the LLM would return
    mock_json_response = json.dumps([
        "What is the average age?",
        "SELECT AVG(age) FROM demographics;",
        "SELECT SUM(age) FROM demographics;"
    ])
    mock_response.read.return_value = json.dumps({"response": mock_json_response}).encode("utf-8")
    
    # Enter context manager for urlopen
    mock_urlopen.return_value.__enter__.return_value = mock_response

    generator = TrainingDataGenerator(mock_db, "gemma4")
    pairs = generator._generate_pairs("dummy_schema", 1)
    
    assert len(pairs) == 1
    prompt, chosen, rejected = pairs[0]
    assert prompt == "What is the average age?"
    assert chosen == "SELECT AVG(age) FROM demographics;"
    assert rejected == "SELECT SUM(age) FROM demographics;"


@patch("urllib.request.urlopen")
def test_generate_pairs_failure_malformed(mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection) -> None:
    """Test generating pairs when the LLM returns malformed data (not a list of 3 strings)."""
    mock_response = MagicMock()
    # Return a list of 2 instead of 3
    mock_json_response = json.dumps(["Only two", "strings here"])
    mock_response.read.return_value = json.dumps({"response": mock_json_response}).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    generator = TrainingDataGenerator(mock_db, "gemma4")
    pairs = generator._generate_pairs("dummy_schema", 1)
    
    # Since it failed validation, no pairs should be returned
    assert len(pairs) == 0


@patch("urllib.request.urlopen")
def test_generate_pairs_http_error(mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection) -> None:
    """Test generating pairs when the HTTP request fails."""
    mock_urlopen.side_effect = Exception("HTTP Error")

    generator = TrainingDataGenerator(mock_db, "gemma4")
    pairs = generator._generate_pairs("dummy_schema", 1)
    
    # Exception should be caught and empty list returned
    assert len(pairs) == 0


def test_write_to_db(mock_db: duckdb.DuckDBPyConnection) -> None:
    """Test writing generated pairs to the database."""
    generator = TrainingDataGenerator(mock_db, "gemma4")
    pairs = [
        (
            "What is the average age?",
            "SELECT AVG(age) FROM demographics;",
            "SELECT SUM(age) FROM demographics;"
        )
    ]
    generator.write_to_db(pairs)
    
    # Verify pretrain_data
    pretrain = mock_db.execute("SELECT * FROM pretrain_data").fetchall()
    assert len(pretrain) == 1
    assert "What is the average age?" in pretrain[0][0]
    assert "SELECT AVG(age)" in pretrain[0][0]
    
    # Verify sft_data
    sft = mock_db.execute("SELECT * FROM sft_data").fetchall()
    assert len(sft) == 1
    assert sft[0][0] == "What is the average age?"
    assert sft[0][1] == "SELECT AVG(age) FROM demographics;"
    
    # Verify dpo_data
    dpo = mock_db.execute("SELECT * FROM dpo_data").fetchall()
    assert len(dpo) == 1
    assert dpo[0][0] == "What is the average age?"
    assert dpo[0][1] == "SELECT AVG(age) FROM demographics;"
    assert dpo[0][2] == "SELECT SUM(age) FROM demographics;"
