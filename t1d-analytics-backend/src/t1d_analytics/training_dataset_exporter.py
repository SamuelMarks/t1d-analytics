"""JSONL/Parquet dataset exporters, DuckDB Pretrain/SFT/DPO split writers, and generation checkpoint I/O."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


def load_checkpoint_pairs(
    checkpoint_path: Path | None,
) -> list[tuple[str, str, str]]:
    """
    Load previously generated `(prompt, chosen, rejected)` triples from a JSON checkpoint file.

    Args:
        checkpoint_path: Optional path to checkpoint JSON file.

    Returns:
        list[tuple[str, str, str]]: Loaded triples or empty list.

    """
    pairs: list[tuple[str, str, str]] = []
    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(ckpt, list):
                for item in ckpt:
                    if isinstance(item, dict) and "prompt" in item:
                        pairs.append((item["prompt"], item["chosen"], item["rejected"]))
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
        ) as e:
            logger.warning(f"Could not load checkpoint from {checkpoint_path}: {e}")
    return pairs


def save_checkpoint_pairs(
    checkpoint_path: Path | None,
    pairs: Sequence[tuple[str, str, str]],
) -> None:
    """
    Persist generated `(prompt, chosen, rejected)` triples to a JSON checkpoint file.

    Args:
        checkpoint_path: Optional path to checkpoint JSON file.
        pairs: Generated triples to save.

    """
    if not checkpoint_path:
        return
    try:
        checkpoint_path.write_text(
            json.dumps(
                [{"prompt": p, "chosen": c, "rejected": r} for p, c, r in pairs],
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Could not write checkpoint: {e}")


def write_pairs_to_tables(
    conn: duckdb.DuckDBPyConnection,
    pairs: Sequence[tuple[str, str, str]],
    suffix: str = "",
) -> None:
    """
    Write `(prompt, chosen, rejected)` triples into DuckDB `pretrain`, `sft`, and `dpo` tables.

    Args:
        conn: Active writable DuckDB connection.
        pairs: Sequence of `(prompt, chosen, rejected)` triples.
        suffix: Table suffix (`''`, `'_train'`, `'_val'`, or `'_test'`).

    """
    pretrain_tbl = (
        f"pretrain{suffix}" if suffix.startswith("_") else f"pretrain_data{suffix}"
    )
    sft_tbl = f"sft{suffix}" if suffix.startswith("_") else f"sft_data{suffix}"
    dpo_tbl = f"dpo{suffix}" if suffix.startswith("_") else f"dpo_data{suffix}"

    conn.execute(f'CREATE TABLE IF NOT EXISTS "{pretrain_tbl}" (text TEXT)')
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{sft_tbl}" (prompt TEXT, completion TEXT)'
    )
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{dpo_tbl}" (prompt TEXT, chosen TEXT, rejected TEXT)'
    )

    if not pairs:
        return

    nl = """
"""
    pretrain_rows = [(f"Question: {p}{nl}SQL: {c}",) for p, c, _ in pairs]
    sft_rows = [(p, c) for p, c, _ in pairs]
    dpo_rows = [(p, c, r) for p, c, r in pairs]

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.executemany(f'INSERT INTO "{pretrain_tbl}" VALUES (?)', pretrain_rows)
        conn.executemany(f'INSERT INTO "{sft_tbl}" VALUES (?, ?)', sft_rows)
        conn.executemany(f'INSERT INTO "{dpo_tbl}" VALUES (?, ?, ?)', dpo_rows)
        conn.execute("COMMIT")
    except (duckdb.Error, RuntimeError, ValueError, TypeError):
        conn.execute("ROLLBACK")
        raise


def write_training_splits_to_db(
    conn: duckdb.DuckDBPyConnection,
    pairs: Sequence[tuple[str, str, str]],
    split_ratios: tuple[float, float, float] | None = None,
) -> None:
    """
    Write generated triples to base DuckDB tables and optional stratified train/val/test split tables.

    Args:
        conn: Active writable DuckDB connection.
        pairs: Sequence of `(prompt, chosen_sql, rejected_sql)` triples.
        split_ratios: Optional `(train_ratio, val_ratio, test_ratio)` tuple.

    """
    write_pairs_to_tables(conn, pairs, suffix="")

    if split_ratios:
        train_r, val_r, _ = split_ratios
        n = len(pairs)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train : n_train + n_val]
        test_pairs = pairs[n_train + n_val :]

        write_pairs_to_tables(conn, train_pairs, suffix="_train")
        write_pairs_to_tables(conn, val_pairs, suffix="_val")
        write_pairs_to_tables(conn, test_pairs, suffix="_test")


def export_table_to_jsonl(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
) -> None:
    """
    Export a DuckDB training table to newline-delimited JSONL format.

    Args:
        conn: Active DuckDB connection.
        table_name: Source table name (e.g. `'sft_data'`).
        output_path: Destination JSONL path on disk.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    escaped_path = str(output_path).replace("'", "''")
    conn.execute(
        f"""COPY "{table_name}" TO '{escaped_path}' (FORMAT JSON, ARRAY FALSE)"""
    )


def export_table_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
) -> None:
    """
    Export a DuckDB training table to Apache Parquet format.

    Args:
        conn: Active DuckDB connection.
        table_name: Source table name (e.g. `'dpo_data'`).
        output_path: Destination Parquet path on disk.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    escaped_path = str(output_path).replace("'", "''")
    conn.execute(f"""COPY "{table_name}" TO '{escaped_path}' (FORMAT PARQUET)""")


__all__ = [
    "export_table_to_jsonl",
    "export_table_to_parquet",
    "load_checkpoint_pairs",
    "save_checkpoint_pairs",
    "write_pairs_to_tables",
    "write_training_splits_to_db",
]
