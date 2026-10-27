"""Dataset loading, tokenization, and prompt-masking utilities for PyTorch causal LM training."""

import importlib
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from t1d_analytics.errors import (
    DatabaseNotFoundError,
    InsufficientDataError,
)
from t1d_analytics.hf_models import (
    PaddedTokenizerProtocol,
    TensorLike,
    TokenizerProtocol,
    TransformersModuleProtocol,
)

logger = logging.getLogger(__name__)


def _load_raw_pairs_from_file(dataset_path: Path) -> List[Tuple[str, str]]:
    """
    Read (prompt, target) text pairs from a .jsonl/.json or .parquet file.

    Args:
    ----
        dataset_path: Path to the dataset file.

    Returns:
    -------
        List[Tuple[str, str]]: Extracted prompt and target completion pairs.

    Raises:
    ------
        InsufficientDataError: If the file cannot be read or contains no valid pairs.

    """
    raw_pairs: List[Tuple[str, str]] = []
    suffix = dataset_path.suffix.lower()

    if suffix in (".jsonl", ".json"):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                        if isinstance(record, dict):
                            p = str(
                                record.get("prompt") or record.get("question") or ""
                            )
                            c = str(
                                record.get("completion")
                                or record.get("sql")
                                or record.get("response")
                                or ""
                            )
                            raw_pairs.append((p, c))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping malformed JSON line in {dataset_path}"
                        )
        except Exception as e:
            raise InsufficientDataError(
                f"Failed to read JSONL dataset at '{dataset_path}': {e}"
            ) from e
    else:
        import duckdb

        escaped = str(dataset_path.resolve()).replace("'", "''")
        conn = duckdb.connect()
        try:
            rel = conn.execute(f"SELECT * FROM read_parquet('{escaped}')")
            cols = [c[0] for c in rel.description] if rel.description else []
            if not cols:
                raise InsufficientDataError("Parquet file contains no schema columns.")
            rows = rel.fetchall()
            for r in rows:
                p = str(r[0]) if len(r) > 0 and r[0] is not None else ""
                c = (
                    str(r[1])
                    if len(r) > 1 and r[1] is not None
                    else (
                        " ".join(str(x) for x in r if x is not None)
                        if len(r) <= 1
                        else ""
                    )
                )
                raw_pairs.append((p, c))
        except Exception as e:
            raise InsufficientDataError(
                f"Failed to read Parquet dataset at '{dataset_path}': {e}"
            ) from e
        finally:
            conn.close()

    if not raw_pairs:
        raise InsufficientDataError(
            f"Dataset at '{dataset_path}' contains no valid training pairs."
        )

    return raw_pairs


def _resolve_tokenizer(
    tokenizer: Optional[TokenizerProtocol],
    tokenizer_name: Optional[str],
) -> TokenizerProtocol:
    """
    Resolve an active tokenizer from an explicit instance or Hugging Face model name.

    Args:
    ----
        tokenizer: Optional explicit tokenizer object.
        tokenizer_name: Optional Hugging Face model identifier.

    Returns:
    -------
        TokenizerProtocol: Resolved tokenizer instance.

    Raises:
    ------
        InsufficientDataError: If no tokenizer is provided or loading from Hugging Face fails.

    """
    if tokenizer is not None:
        return tokenizer

    model_id = tokenizer_name or "google/gemma-4-2b"
    try:
        tf_mod: TransformersModuleProtocol = importlib.import_module("transformers")
        auto_tok_cls = tf_mod.AutoTokenizer
        return auto_tok_cls.from_pretrained(model_id)
    except Exception as e:
        raise InsufficientDataError(
            f"Failed to load Hugging Face tokenizer '{model_id}'. "
            f"Ensure 'transformers' is installed and model '{model_id}' is accessible: {e}"
        ) from e


def _tokenize_pair_with_masking(
    prompt: str,
    target: str,
    active_tok: TokenizerProtocol,
    max_seq_length: int,
) -> Tuple[TensorLike, TensorLike]:
    """
    Tokenize a single (prompt, target) pair with -100 prompt-token masking and padding.

    Args:
    ----
        prompt: Natural language question or prompt string.
        target: Target SQL completion string.
        active_tok: Tokenizer instance providing `.encode()`.
        max_seq_length: Maximum sequence length for padding/truncation.

    Returns:
    -------
        Tuple[TensorLike, TensorLike]: `(input_ids, labels)` PyTorch `long` tensors of length `max_seq_length - 1`.

    """
    import torch

    formatted_prompt = f"""<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""
    formatted_target = f"{target}<end_of_turn>"

    p_ids = list(active_tok.encode(formatted_prompt, add_special_tokens=False))
    t_ids = list(active_tok.encode(formatted_target, add_special_tokens=False))
    raw_pad = (
        active_tok.pad_token_id
        if isinstance(active_tok, PaddedTokenizerProtocol)
        else None
    )
    pad_id = int(raw_pad) if isinstance(raw_pad, int) else 0

    full_ids = p_ids + t_ids
    labels = ([-100] * len(p_ids)) + list(t_ids)

    if len(full_ids) < max_seq_length:
        pad_len = max_seq_length - len(full_ids)
        full_ids = full_ids + ([pad_id] * pad_len)
        labels = labels + ([-100] * pad_len)
    else:
        full_ids = full_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
    target_ids = torch.tensor(labels[1:], dtype=torch.long)
    return (input_ids, target_ids)


class TokenizedPairDataset(Sequence[Tuple[TensorLike, TensorLike]]):
    """Sequence dataset wrapping pre-tokenized `(input_ids, labels)` tensor pairs."""

    def __init__(self, pairs: Sequence[Tuple[TensorLike, TensorLike]]) -> None:
        """
        Initialize the dataset with tokenized tensor pairs.

        Args:
        ----
            pairs: Sequence of `(input_ids, labels)` tensor tuples.

        """
        self._pairs: List[Tuple[TensorLike, TensorLike]] = list(pairs)

    def __len__(self) -> int:
        """
        Return the number of tokenized pairs in the dataset.

        Returns
        -------
            int: Total sample count.

        """
        return len(self._pairs)

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> Tuple[TensorLike, TensorLike]:
        """
        Return the `(input_ids, labels)` tuple at the given index.

        Args:
        ----
            index: Zero-based sample index.

        Returns:
        -------
            Tuple[TensorLike, TensorLike]: Tokenized `(input_ids, labels)` pair.

        """
        return self._pairs[index]

    def __iter__(self) -> Iterator[Tuple[TensorLike, TensorLike]]:
        """
        Iterate over all `(input_ids, labels)` pairs in the dataset.

        Returns
        -------
            Iterator[Tuple[TensorLike, TensorLike]]: Iterator over tensor pairs.

        """
        return iter(self._pairs)


class StreamingJsonlDataset:
    """Streaming iterable dataset yielding tokenized `(input_ids, labels)` pairs from a JSONL file."""

    def __init__(
        self,
        dataset_path: Path,
        tokenizer: TokenizerProtocol,
        max_seq_length: int = 128,
    ) -> None:
        """
        Initialize the streaming JSONL dataset.

        Args:
        ----
            dataset_path: Path to the `.jsonl` file on disk.
            tokenizer: Pretrained tokenizer instance.
            max_seq_length: Maximum token sequence length.

        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__(self) -> Iterator[Tuple[TensorLike, TensorLike]]:
        """
        Stream and tokenize valid `(prompt, target)` pairs from the JSONL file.

        Yields
        ------
            Tuple[TensorLike, TensorLike]: Tokenized `(input_ids, labels)` tensor pair.

        """
        for prompt, target in _load_raw_pairs_from_file(self.dataset_path):
            yield _tokenize_pair_with_masking(
                prompt, target, self.tokenizer, self.max_seq_length
            )


def prepare_torch_dataset(
    dataset_path: Path,
    max_seq_length: int = 128,
    vocab_size: int = 1000,
    tokenizer: Optional[TokenizerProtocol] = None,
    tokenizer_name: Optional[str] = None,
) -> List[Tuple[TensorLike, TensorLike]]:
    """
    Load and tokenize a .jsonl or .parquet dataset into PyTorch (input_ids, labels) tensors.

    Prompt tokens are masked with -100 in labels so loss is computed exclusively on SQL completions.

    Args:
    ----
        dataset_path: Path to dataset file on disk (.jsonl or .parquet).
        max_seq_length: Fixed sequence length for padding/truncation.
        vocab_size: Vocabulary upper bound.
        tokenizer: Optional pretrained tokenizer object with `.encode()` method.
        tokenizer_name: Optional HuggingFace model/tokenizer identifier.

    Returns:
    -------
        List[Tuple[TensorLike, TensorLike]]: List of (input_ids, labels) tensor pairs.

    Raises:
    ------
        DatabaseNotFoundError: If dataset_path does not exist.
        TrainingExecutionError: If file format is unsupported or dataset is empty/corrupt.

    """
    _ = vocab_size
    if not dataset_path.exists():
        raise DatabaseNotFoundError(f"Training dataset not found at '{dataset_path}'.")

    suffix = dataset_path.suffix.lower()
    if suffix not in (".jsonl", ".json", ".parquet"):
        raise InsufficientDataError(
            f"Unsupported dataset format '{suffix}'. Expected .jsonl or .parquet."
        )

    raw_pairs = _load_raw_pairs_from_file(dataset_path)
    active_tok = _resolve_tokenizer(tokenizer, tokenizer_name)

    dataset: List[Tuple[TensorLike, TensorLike]] = []
    for prompt, target in raw_pairs:
        dataset.append(
            _tokenize_pair_with_masking(prompt, target, active_tok, max_seq_length)
        )

    return dataset
