"""Tests for the training_data module."""

import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from t1d_analytics.training_data import TrainingDataGenerator


@pytest.fixture
def mock_db() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Provide an in-memory DuckDB connection pre-populated with a dummy schema.

    Yields
    ------
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


def test_extract_schema_ignores_internal_tables(
    mock_db: duckdb.DuckDBPyConnection,
) -> None:
    """Test extracting schema ignores manifest, sessions, and training tables."""
    mock_db.execute("CREATE TABLE _t1d_ingestion_manifest (source_file VARCHAR)")
    mock_db.execute("CREATE TABLE chat_sessions (session_id VARCHAR)")
    mock_db.execute("CREATE TABLE sft_data (prompt VARCHAR)")
    mock_db.execute("CREATE TABLE dpo_data (prompt VARCHAR)")
    mock_db.execute("CREATE TABLE pretrain_data (text VARCHAR)")
    mock_db.execute("CREATE TABLE pretrain_train (text VARCHAR)")

    generator = TrainingDataGenerator(mock_db, "gemma4")
    schema = generator._extract_schema()

    assert "demographics" in schema
    assert "_t1d_ingestion_manifest" not in schema
    assert "chat_sessions" not in schema
    assert "sft_data" not in schema
    assert "dpo_data" not in schema
    assert "pretrain_data" not in schema
    assert "pretrain_train" not in schema


@patch("urllib.request.urlopen")
def test_generate_pairs_success(
    mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
    """Test generating pairs with a successful LLM response."""
    # Create a mock response
    mock_response = MagicMock()
    # The JSON string response the LLM would return
    mock_json_response = json.dumps(
        [
            "What is the average age?",
            "SELECT AVG(age) FROM demographics;",
            "SELECT SUM(age) FROM demographics;",
        ]
    )
    mock_response.read.return_value = json.dumps(
        {"response": mock_json_response}
    ).encode("utf-8")

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
def test_generate_pairs_failure_malformed(
    mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
    """Test generating pairs when the LLM returns malformed data (not a list of 3 strings)."""
    mock_response = MagicMock()
    # Return a list of 2 instead of 3
    mock_json_response = json.dumps(["Only two", "strings here"])
    mock_response.read.return_value = json.dumps(
        {"response": mock_json_response}
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    generator = TrainingDataGenerator(mock_db, "gemma4")
    pairs = generator._generate_pairs("dummy_schema", 1)

    # Since it failed validation, no pairs should be returned
    assert len(pairs) == 0


@patch("urllib.request.urlopen")
def test_generate_pairs_http_error(
    mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
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
            "SELECT SUM(age) FROM demographics;",
        )
    ]
    generator.write_to_db(pairs)

    # Verify pretrain_data
    pretrain = mock_db.execute("SELECT * FROM pretrain_data").fetchall()
    assert len(pretrain) == 1
    assert "What is the average age?" in pretrain[0][0]
    assert "\nSQL: SELECT AVG(age)" in pretrain[0][0]

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


def test_is_valid_sql(mock_db: duckdb.DuckDBPyConnection) -> None:
    """Test _is_valid_sql with valid and invalid queries."""
    generator = TrainingDataGenerator(mock_db, "gemma4")
    assert generator._is_valid_sql("SELECT * FROM demographics")
    assert not generator._is_valid_sql("SELECT * FROM nonexistent_table")
    assert not generator._is_valid_sql("MALFORMED SQL QUERY !!!")


@patch("urllib.request.urlopen")
def test_generate_pairs_sql_validation_discard(
    mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
    """Test generating pairs discards invalid SQL when validate_sql is enabled."""
    mock_response = MagicMock()
    mock_json_response = json.dumps(
        [
            "What is the average age?",
            "SELECT * FROM nonexistent_table;",
            "SELECT * FROM demographics;",
        ]
    )
    mock_response.read.return_value = json.dumps(
        {"response": mock_json_response}
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    generator = TrainingDataGenerator(mock_db, "gemma4")
    pairs = generator._generate_pairs("dummy_schema", 1, validate_sql=True)
    assert len(pairs) == 0


def test_export_to_jsonl_and_parquet(
    mock_db: duckdb.DuckDBPyConnection, tmp_path: pytest.TempPathFactory
) -> None:
    """Test exporting training tables to JSONL and Parquet."""
    from pathlib import Path

    out_dir = Path(str(tmp_path))
    generator = TrainingDataGenerator(mock_db, "gemma4")
    pairs = [
        (
            "What is the average age?",
            "SELECT AVG(age) FROM demographics;",
            "SELECT SUM(age) FROM demographics;",
        )
    ]
    generator.write_to_db(pairs)

    jsonl_path = out_dir / "sft.jsonl"
    generator.export_to_jsonl("sft_data", jsonl_path)
    assert jsonl_path.exists()
    assert len(jsonl_path.read_text()) > 0

    parquet_path = out_dir / "dpo.parquet"
    generator.export_to_parquet("dpo_data", parquet_path)
    assert parquet_path.exists()
    assert parquet_path.stat().st_size > 0


@patch("urllib.request.urlopen")
def test_generate_pairs_ollama_host_without_http(
    mock_urlopen: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
    """Test _generate_pairs formats OLLAMA_HOST without http prefix."""
    import os

    mock_response = MagicMock()
    mock_json = json.dumps(["Q?", "SELECT 1;", "SELECT 2;"])
    mock_response.read.return_value = json.dumps({"response": mock_json}).encode(
        "utf-8"
    )
    mock_urlopen.return_value.__enter__.return_value = mock_response

    with patch.dict(os.environ, {"OLLAMA_HOST": "localhost:11434"}):
        generator = TrainingDataGenerator(mock_db, "gemma4")
        pairs = generator._generate_pairs("dummy", 1, validate_sql=False)
        assert len(pairs) == 1


def test_training_data_generator_provider_parsing(
    mock_db: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test provider and model name parsing in TrainingDataGenerator."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    gen1 = TrainingDataGenerator(mock_db, "openai/gpt-4o")
    assert gen1.provider == "openai"
    assert gen1.model == "gpt-4o"

    gen2 = TrainingDataGenerator(mock_db, "claude-3-5-sonnet", provider="anthropic")
    assert gen2.provider == "anthropic"
    assert gen2.model == "claude-3-5-sonnet"


def test_validate_provider_readiness_errors(
    mock_db: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test error handling in validate_provider_readiness."""
    from t1d_analytics.training_data import validate_provider_readiness

    # 1. Missing API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(
        RuntimeError, match="requires environment variable 'OPENAI_API_KEY'"
    ):
        validate_provider_readiness("openai")

    # 2. Missing any-llm SDK
    with patch.dict("sys.modules", {"any_llm": None}):
        with pytest.raises(RuntimeError, match="requires any-llm-sdk"):
            validate_provider_readiness("openai")

    # 3. Custom provider not in key map
    validate_provider_readiness("custom-local")

    # 4. Non-ollama provider failure when any-llm create fails
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("any_llm.AnyLLM.create", side_effect=Exception("API failure")):
        gen = TrainingDataGenerator(mock_db, "openai/gpt-4o")
        with pytest.raises(RuntimeError, match="Failed to initialize any-llm"):
            gen._generate_pairs("schema", 1)


@patch("any_llm.AnyLLM.create")
def test_generate_pairs_any_llm_success(
    mock_create: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
    """Test generating training pairs via any-llm."""
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        [
            "What is patient age?",
            "SELECT age FROM demographics;",
            "SELECT invalid_col FROM demographics;",
        ]
    )
    mock_llm.completion.return_value = mock_resp
    mock_create.return_value = mock_llm

    gen = TrainingDataGenerator(mock_db, "gemma4", provider="ollama")
    pairs = gen._generate_pairs("schema_text", 1, validate_sql=True)
    assert len(pairs) == 1
    assert pairs[0][0] == "What is patient age?"
    assert pairs[0][1] == "SELECT age FROM demographics;"


@patch("any_llm.AnyLLM.create")
@patch("urllib.request.urlopen")
def test_generate_pairs_any_llm_fallback_to_urllib(
    mock_urlopen: MagicMock, mock_create: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
    """Test any-llm failure falling back to urllib request in _generate_pairs."""
    # any-llm throws an exception
    mock_llm = MagicMock()
    mock_llm.completion.side_effect = RuntimeError("any-llm failure")
    mock_create.return_value = mock_llm

    # urllib responds with valid JSON
    mock_response = MagicMock()
    mock_json_response = json.dumps(
        [
            "Fallback question?",
            "SELECT age FROM demographics;",
            "SELECT 1 FROM demographics;",
        ]
    )
    mock_response.read.return_value = json.dumps(
        {"response": mock_json_response}
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    gen = TrainingDataGenerator(mock_db, "gemma4")
    pairs = gen._generate_pairs("schema_text", 1, validate_sql=True)
    assert len(pairs) == 1
    assert pairs[0][0] == "Fallback question?"


def test_write_to_db_with_stratified_split_ratios(
    mock_db: duckdb.DuckDBPyConnection,
) -> None:
    """Test write_to_db creates and populates stratified train/val/test tables."""
    gen = TrainingDataGenerator(mock_db, "gemma4")
    pairs = [
        (
            f"Question {i}?",
            f"SELECT {i} FROM demographics;",
            f"SELECT {i + 100} FROM demographics;",
        )
        for i in range(10)
    ]
    gen.write_to_db(pairs, split_ratios=(0.8, 0.1, 0.1))

    def get_count(query: str) -> int:
        row = mock_db.execute(query).fetchone()
        assert row is not None
        return int(row[0])

    # Check split table counts
    assert get_count("SELECT COUNT(*) FROM pretrain_train") == 8
    assert get_count("SELECT COUNT(*) FROM pretrain_val") == 1
    assert get_count("SELECT COUNT(*) FROM pretrain_test") == 1

    assert get_count("SELECT COUNT(*) FROM sft_train") == 8
    assert get_count("SELECT COUNT(*) FROM sft_val") == 1
    assert get_count("SELECT COUNT(*) FROM sft_test") == 1

    assert get_count("SELECT COUNT(*) FROM dpo_train") == 8
    assert get_count("SELECT COUNT(*) FROM dpo_val") == 1
    assert get_count("SELECT COUNT(*) FROM dpo_test") == 1

    # Base tables also contain all 10 pairs
    assert get_count("SELECT COUNT(*) FROM pretrain_data") == 10


def test_normalize_sql() -> None:
    """Test SQL query normalization utility."""
    from t1d_analytics.training_data import normalize_sql

    assert normalize_sql("  SELECT  *  FROM  users ; ") == "select * from users"


def test_evaluate_text_to_sql_benchmark(tmp_path: Path, mocker: MagicMock) -> None:
    """Test evaluate_text_to_sql computing EM, EX, syntax errors, and reports."""
    from t1d_analytics.training_data import evaluate_text_to_sql

    db_file = tmp_path / "benchmark.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE users (id INT, age INT)")
    conn.execute("INSERT INTO users VALUES (1, 25), (2, 35)")
    conn.close()

    test_data = [
        # Case 1: Exact Match and Execution Accuracy
        {"prompt": "Get all users", "gold_sql": "SELECT * FROM users"},
        # Case 2: Execution Match but different formatting (EX=True, EM=False)
        {"prompt": "Average age", "gold_sql": "SELECT avg(age) FROM users"},
        # Case 3: Syntax error in predicted query
        {"prompt": "Invalid query", "gold_sql": "SELECT count(*) FROM users"},
        # Case 4: Exception in model generation
        {"prompt": "Error model query", "gold_sql": "SELECT * FROM users"},
        # Case 5: Bad gold SQL
        {"prompt": "Bad gold query", "gold_sql": "INVALID SQL QUERY !!!"},
    ]

    def mock_generate(db_p: str, q: str, model_name: str) -> tuple[str, str]:
        if "Get all users" in q:
            return "resp1", "SELECT * FROM users"
        elif "Average age" in q:
            return "resp2", "SELECT   AVG(age) AS avg_age  FROM users"
        elif "Error model query" in q:
            raise RuntimeError("Model crash")
        else:
            return "resp3", "SELECT FROM users WHERE"

    mocker.patch("t1d_analytics.api.generate_sql_from_nl", side_effect=mock_generate)

    metrics = evaluate_text_to_sql(str(db_file), test_data, model="gemma4")
    assert metrics["total_samples"] == 5
    assert metrics["exact_match_rate"] > 0
    assert metrics["execution_accuracy"] > 0
    assert metrics["syntax_failure_rate"] > 0
    assert "Text-to-SQL Benchmark Report" in metrics["markdown_report"]
    assert "gold_sql,pred_sql" in metrics["csv_report"]


@patch("any_llm.AnyLLM.create")
@patch("urllib.request.urlopen")
def test_generate_pairs_any_llm_create_failure_fallback(
    mock_urlopen: MagicMock, mock_create: MagicMock, mock_db: duckdb.DuckDBPyConnection
) -> None:
    """Test any-llm create failure falling back to urllib."""
    mock_create.side_effect = ImportError("any-llm create failure")

    mock_response = MagicMock()
    mock_json_response = json.dumps(
        [
            "Fallback Q?",
            "SELECT age FROM demographics;",
            "SELECT 1 FROM demographics;",
        ]
    )
    mock_response.read.return_value = json.dumps(
        {"response": mock_json_response}
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    gen = TrainingDataGenerator(mock_db, "gemma4")
    pairs = gen._generate_pairs("schema_text", 1, validate_sql=True)
    assert len(pairs) == 1
    assert pairs[0][0] == "Fallback Q?"


def test_parse_llm_json_array() -> None:
    """Test _parse_llm_json_array handles markdown code blocks and fallback parsing."""
    from t1d_analytics.training_data import _parse_llm_json_array

    # Markdown wrapped
    md_text = '```json\n["Question", "SELECT 1;", "SELECT 2;"]\n```'
    assert _parse_llm_json_array(md_text) == ["Question", "SELECT 1;", "SELECT 2;"]

    # Fallback search in conversational preamble
    convo = (
        'Sure, here is your data:\n["Q", "SELECT a;", "SELECT b;"]\nHope this helps!'
    )
    assert _parse_llm_json_array(convo) == ["Q", "SELECT a;", "SELECT b;"]

    # Non-list JSON with list inside regex match
    assert _parse_llm_json_array('{"key": ["q", "c", "r"]}') == ["q", "c", "r"]

    # Non-list JSON with no regex match
    assert _parse_llm_json_array('{"key": "value"}') is None

    # Invalid JSON in brackets
    assert _parse_llm_json_array("Invalid [brackets that cannot be json] text") is None

    # Invalid JSON
    assert _parse_llm_json_array("This is not JSON at all.") is None


def test_evaluate_text_to_sql_order_by_and_multiset(
    tmp_path: Path, mocker: MagicMock
) -> None:
    """Test evaluate_text_to_sql distinguishes queries with and without ORDER BY."""
    from t1d_analytics.training_data import evaluate_text_to_sql

    db_file = tmp_path / "orders.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE tbl (val INT)")
    conn.execute("INSERT INTO tbl VALUES (10), (20)")
    conn.close()

    test_cases = [
        # Query without ORDER BY (multiset match)
        {
            "prompt": "Unordered",
            "gold_sql": "SELECT val FROM tbl",
        },
        # Query with ORDER BY (match)
        {
            "prompt": "Ordered Match",
            "gold_sql": "SELECT val FROM tbl ORDER BY val DESC",
        },
        # Query with ORDER BY (mismatch)
        {
            "prompt": "Ordered Mismatch",
            "gold_sql": "SELECT val FROM tbl ORDER BY val DESC",
        },
        # Query without ORDER BY (mismatch values)
        {
            "prompt": "Unordered Mismatch",
            "gold_sql": "SELECT val FROM tbl",
        },
    ]

    def mock_gen(db_p: str, q: str, model_name: str) -> tuple[str, str]:
        if "Unordered Mismatch" in q:
            return "ok", "SELECT val FROM tbl WHERE val = 10"
        elif "Ordered Mismatch" in q:
            return "ok", "SELECT val FROM tbl ORDER BY val ASC"
        elif "Ordered Match" in q:
            return "ok", "SELECT val FROM tbl ORDER BY val DESC"
        return "ok", "SELECT val FROM tbl"

    mocker.patch("t1d_analytics.api.generate_sql_from_nl", side_effect=mock_gen)
    res = evaluate_text_to_sql(str(db_file), test_cases)
    assert res["execution_accuracy"] == 50.0
