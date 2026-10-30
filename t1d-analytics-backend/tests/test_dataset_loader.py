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

    with patch("json.loads", side_effect=Exception("Mock generic exception")):
        with pytest.raises(ValueError, match="Failed to read JSONL dataset"):
            prepare_torch_dataset(jsonl_file, tokenizer_name="mock")
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
