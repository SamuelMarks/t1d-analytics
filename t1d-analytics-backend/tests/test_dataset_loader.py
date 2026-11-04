"""Unit tests for dataset loading, tokenization, prompt masking, and dataset containers."""

from pathlib import Path
from typing import Generator, List
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from t1d_analytics.dataset_loader import (
    StreamingJsonlDataset,
    TokenizedPairDataset,
    prepare_torch_dataset,
)
from t1d_analytics.sql_utils import SqlValue


@pytest.fixture(autouse=True)
def mock_huggingface_tokenizer() -> Generator[None, None, None]:
    """Provide a deterministic lightweight tokenizer for dataset loader tests."""

    class MockTokenizer:
        """Lightweight mock tokenizer implementing TokenizerProtocol."""

        def __init__(self, *args: SqlValue, **kwargs: SqlValue) -> None:
            """Initialize pad_token_id."""
            self.pad_token_id = 0

        def encode(
            self, text: str, add_special_tokens: bool = False, **kwargs: SqlValue
        ) -> List[int]:
            """Encode text characters into integer IDs."""
            _ = add_special_tokens
            return [ord(c) % 50 for c in text] if text else []

        @classmethod
        def from_pretrained(
            cls, *args: SqlValue, **kwargs: SqlValue
        ) -> "MockTokenizer":
            """Return a MockTokenizer instance."""
            return cls()

    with patch(
        "transformers.AutoTokenizer.from_pretrained", MockTokenizer.from_pretrained
    ):
        yield


def test_prepare_torch_dataset_missing_and_unsupported(tmp_path: Path) -> None:
    """Test prepare_torch_dataset error conditions for missing files and unsupported formats."""
    missing = tmp_path / "nonexistent.jsonl"
    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        prepare_torch_dataset(missing)

    unsupported = tmp_path / "data.csv"
    unsupported.write_text("""a,b
1,2
""")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        prepare_torch_dataset(unsupported)


def test_prepare_torch_dataset_jsonl_parsing_and_error(tmp_path: Path) -> None:
    """Test prepare_torch_dataset JSONL format reading, malformed row skipping, and empty error."""
    jsonl_file = tmp_path / "valid.jsonl"
    jsonl_file.write_text(
        """{"prompt": "Show glucose", "sql": "SELECT glucose FROM cgm"}

not a json
{"question": "Average a1c", "completion": "SELECT AVG(a1c) FROM patient"}
"""
    )
    dataset = prepare_torch_dataset(
        jsonl_file, max_seq_length=32, tokenizer_name="mock"
    )
    assert len(dataset) == 2
    assert dataset[0][0].shape[0] == 31
    assert dataset[0][1].shape[0] == 31

    tokenized_seq = TokenizedPairDataset(dataset)
    assert len(tokenized_seq) == 2
    assert tokenized_seq[0][0].shape[0] == 31
    assert len(list(tokenized_seq)) == 2

    class _SimpleTokenizer:
        """Tokenizer implementation for streaming dataset verification."""

        pad_token_id: int = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            """Encode text characters."""
            _ = add_special_tokens
            return [ord(c) % 50 for c in text]

    streaming_ds = StreamingJsonlDataset(
        jsonl_file, tokenizer=_SimpleTokenizer(), max_seq_length=32
    )
    streamed_items = list(streaming_ds)
    assert len(streamed_items) == 2

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text(chr(10) * 2)
    with pytest.raises(ValueError, match="no valid training pairs"):
        prepare_torch_dataset(empty_file, tokenizer_name="mock")
    with pytest.raises(ValueError, match="contains no valid training pairs"):
        prepare_torch_dataset(empty_file)

    with patch("builtins.open", side_effect=OSError("Disk error")):
        with pytest.raises(ValueError, match="Failed to read JSONL dataset"):
            prepare_torch_dataset(jsonl_file)


def test_prepare_torch_dataset_parquet_parsing_and_error(tmp_path: Path) -> None:
    """Test prepare_torch_dataset Parquet format reading and error handling."""
    pq_file = tmp_path / "clinical.parquet"
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE pairs AS SELECT 'What is TBR?' AS prompt, 'SELECT tbr FROM cgm' AS sql"
    )
    conn.execute(f"COPY pairs TO '{pq_file}' (FORMAT PARQUET)")
    conn.close()

    dataset = prepare_torch_dataset(pq_file, max_seq_length=64, tokenizer_name="mock")
    assert len(dataset) == 1

    bad_pq = tmp_path / "corrupted.parquet"
    bad_pq.write_text("invalid parquet binary")
    with pytest.raises(ValueError):
        prepare_torch_dataset(bad_pq, tokenizer_name="mock")


def test_prepare_torch_dataset_edge_cases(tmp_path: Path) -> None:
    """Test prepare_torch_dataset with non-dict rows and empty text strings."""
    jsonl_file = tmp_path / "edge_cases.jsonl"
    jsonl_file.write_text(
        """123
"plain string"
{"prompt": "", "completion": ""}
"""
    )

    class _UnitTok:
        """Minimal tokenizer returning single token ID."""

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            """Return single token."""
            _ = (text, add_special_tokens)
            return [1]

    dataset = prepare_torch_dataset(
        jsonl_file,
        max_seq_length=16,
        tokenizer=_UnitTok(),
    )
    assert len(dataset) == 1


def test_prepare_torch_dataset_parquet_empty_and_no_cols(tmp_path: Path) -> None:
    """Test prepare_torch_dataset when Parquet has no rows or columns cannot be resolved."""
    pq_empty = tmp_path / "empty_rows.parquet"
    conn = duckdb.connect()
    conn.execute("CREATE TABLE empty_table (prompt VARCHAR, completion VARCHAR)")
    conn.execute(f"COPY empty_table TO '{pq_empty}' (FORMAT PARQUET)")
    conn.close()

    with pytest.raises(ValueError, match="contains no valid training pairs"):
        prepare_torch_dataset(pq_empty)

    mock_rel = MagicMock()
    mock_rel.description = None
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_rel
    with patch("duckdb.connect", return_value=mock_conn):
        with pytest.raises(ValueError, match="Parquet file contains no schema columns"):
            prepare_torch_dataset(pq_empty)


def test_prepare_torch_dataset_with_tokenizer(tmp_path: Path) -> None:
    """Test prepare_torch_dataset with custom tokenizer, prompt masking, padding, and truncation."""
    ds = tmp_path / "train_tok.jsonl"
    ds.write_text("""{"prompt": "question", "completion": "answer"}
""")

    class MockTokenizer:
        """Mock tokenizer for testing encode and pad_token_id."""

        pad_token_id: int = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            """Encode text into integer token ids."""
            _ = add_special_tokens
            return [ord(c) % 50 for c in text]

    mock_tok = MockTokenizer()
    res_pad = prepare_torch_dataset(ds, max_seq_length=128, tokenizer=mock_tok)
    assert len(res_pad) == 1
    inps, labels = res_pad[0]
    assert inps.shape[0] == 127
    assert labels.shape[0] == 127
    assert (labels == -100).any()

    res_trunc = prepare_torch_dataset(ds, max_seq_length=10, tokenizer=mock_tok)
    assert len(res_trunc) == 1
    assert res_trunc[0][0].shape[0] == 9


def test_prepare_torch_dataset_tokenizer_name_and_failure(tmp_path: Path) -> None:
    """Test prepare_torch_dataset with explicit tokenizer_name and failure handling."""
    ds = tmp_path / "train_tok_name.jsonl"
    ds.write_text("""{"prompt": "q", "completion": "a"}
""")

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.pad_token_id = 0

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        res = prepare_torch_dataset(ds, tokenizer_name="mock")
        assert len(res) == 1

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=RuntimeError("Tokenizer load error"),
    ):
        with pytest.raises(ValueError):
            prepare_torch_dataset(ds, tokenizer_name="google/gemma-4-2b")


def test_table_1_clinical_variable_mappings_and_numeric_coercion(
    tmp_path: Path,
) -> None:
    """Verify Table 1 clinical variable mappings in likely_matches.json and numeric column coercion."""
    from t1d_analytics.ingestion import (
        _CLINICAL_NUMERIC_COLUMNS,
        get_clinical_variable_mappings,
        load_data_to_duckdb,
    )
    from t1d_analytics.seed import _BASE_TABLE_NUMERIC_COLUMNS

    mappings = get_clinical_variable_mappings()
    for required_concept in (
        "race_ethnicity",
        "t1d_duration_years",
        "bmi",
        "total_daily_insulin",
        "insulin_delivery_modality",
        "cgm_active_wear_pct",
    ):
        assert required_concept in mappings
        assert len(mappings[required_concept]) >= 4

    for numeric_col in (
        "bmi",
        "t1d_duration_years",
        "total_daily_insulin",
        "cgm_active_wear_pct",
    ):
        assert numeric_col in _CLINICAL_NUMERIC_COLUMNS
        assert numeric_col in _BASE_TABLE_NUMERIC_COLUMNS

    csv_file = tmp_path / "trial_table1.csv"
    csv_file.write_text(
        "PtID,Race,Duration,BMI,TDD,InsModality,PctWear\n"
        "1,White,12.5,24.1,42.0,Pump,96.5\n",
        encoding="utf-8",
    )
    db_file = tmp_path / "table1_mapped.duckdb"
    load_data_to_duckdb(str(tmp_path), str(db_file))

    conn = duckdb.connect(str(db_file), read_only=True)
    desc = {
        str(r[0]): str(r[1]) for r in conn.execute("DESCRIBE trial_table1").fetchall()
    }
    conn.close()
    assert "race_ethnicity" in desc
    assert "t1d_duration_years" in desc
    assert "bmi" in desc
    assert "total_daily_insulin" in desc
    assert "insulin_delivery_modality" in desc
    assert "cgm_active_wear_pct" in desc
