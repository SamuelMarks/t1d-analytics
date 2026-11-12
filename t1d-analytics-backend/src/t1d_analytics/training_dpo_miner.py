"""DPO preference mining, SQL execution-set equivalence scoring, and Text-to-SQL benchmark evaluation."""

from __future__ import annotations

import collections
import csv
import io
import json
import logging
import re
from collections.abc import Callable, Hashable, Mapping, Sequence

import duckdb
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from t1d_analytics.api_schemas import SqlGoldComparisonModel
from t1d_analytics.errors import T1DError
from t1d_analytics.sql_fallback_synthesizer import TranslationResult
from t1d_analytics.sql_utils import SqlValue, normalize_sql

logger = logging.getLogger(__name__)

type HashableInput = (
    SqlValue
    | tuple[HashableInput, ...]
    | list[HashableInput]
    | set[HashableInput]
    | dict[str, HashableInput]
)
type HashableResult = SqlValue | tuple[Hashable, ...]

_SQL_VALUE_LIST_ADAPTER: TypeAdapter[list[SqlValue]] = TypeAdapter(list[SqlValue])


class EvalItemResultDict(TypedDict):
    """Per-question evaluation metrics for Text-to-SQL benchmarking."""

    question: str
    gold_sql: str
    pred_sql: str
    exact_match: bool
    execution_accuracy: bool
    syntax_error: bool


class TextToSqlEvalReportDict(TypedDict):
    """Aggregated evaluation summary and reports for Text-to-SQL benchmarking."""

    total_samples: int
    exact_match_rate: float
    execution_accuracy: float
    syntax_failure_rate: float
    results: list[EvalItemResultDict]
    markdown_report: str
    csv_report: str


def make_hashable(val: HashableInput) -> HashableResult:
    """
    Recursively convert unhashable types into immutable hashable equivalents.

    Args:
        val: Python value (lists, dicts, sets, tuples, primitives).

    Returns:
        HashableResult: An immutable, hashable version of the input value.

    """
    if isinstance(val, (list, tuple)):
        return tuple(make_hashable(x) for x in val)
    if isinstance(val, dict):
        items_map = [(str(k), make_hashable(v)) for k, v in val.items()]
        items_map.sort(key=lambda pair: pair[0])
        return tuple(items_map)
    if isinstance(val, set):
        items_set = [make_hashable(x) for x in val]
        items_set.sort(key=lambda item: str(item))
        return tuple(items_set)
    return val


def parse_llm_json_array(content_str: str) -> list[SqlValue] | None:
    """
    Parse a JSON array from LLM response text, stripping markdown blocks if present.

    Args:
        content_str: Raw text returned by the LLM.

    Returns:
        list[SqlValue] | None: Parsed JSON list if valid, otherwise None.

    """
    nl = "\n"
    cleaned = content_str.strip()
    if cleaned.startswith("```"):
        lines = [
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ]
        cleaned = nl.join(lines).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return _SQL_VALUE_LIST_ADAPTER.validate_python(data)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
        logger.debug(f"Failed to decode exact JSON list: {e}")
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            extracted = list(json.loads(match.group(0)))
            return _SQL_VALUE_LIST_ADAPTER.validate_python(extracted)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
            logger.debug(f"Failed to decode extracted JSON list: {e}")
    return None


def mine_dpo_rejected_sql_perturbation(chosen_sql: str) -> str:
    """
    Synthesize a hard-negative DPO rejected SQL query by introducing a realistic clinical schema/aggregate error.

    Args:
        chosen_sql: Verified chosen SQL query string.

    Returns:
        str: Perturbed negative SQL query suitable for Direct Preference Optimization.

    """
    stripped = chosen_sql.strip().rstrip(";")
    if re.search(r"\bAVG\s*\(", stripped, flags=re.IGNORECASE):
        perturbed = re.sub(
            r"\bAVG\s*\(", "SUM(", stripped, count=1, flags=re.IGNORECASE
        )
        return f"{perturbed};"
    if re.search(r"\bCOUNT\s*\(\s*\*\s*\)", stripped, flags=re.IGNORECASE):
        perturbed = re.sub(
            r"\bCOUNT\s*\(\s*\*\s*\)",
            "COUNT(DISTINCT nonexistent_col)",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        )
        return f"{perturbed};"
    return f"{stripped} WHERE 1 = 0;"


def execute_eval_sql(
    conn: duckdb.DuckDBPyConnection,
    sql_query: str,
) -> tuple[list[tuple[SqlValue, ...]] | None, bool]:
    """
    Execute a SQL query on `conn` and return `(rows, had_execution_error)`.

    Args:
        conn: Read-only DuckDB connection.
        sql_query: SQL query to execute.

    Returns:
        tuple[list[tuple[SqlValue, ...]] | None, bool]: `(fetched_rows, had_error)`.

    """
    if not sql_query:
        return None, False
    try:
        return conn.execute(sql_query).fetchall(), False
    except (duckdb.Error, RuntimeError, ValueError, TypeError) as e:
        logger.debug(f"Execution error for SQL '{sql_query}': {e}")
        return None, True


def compare_sql_result_sets(
    gold_rows: Sequence[tuple[SqlValue, ...]],
    pred_rows: Sequence[tuple[SqlValue, ...]],
    has_order_by: bool,
) -> bool:
    """
    Compare gold and predicted SQL result sets using ordered or multiset equality.

    Args:
        gold_rows: Gold query result rows.
        pred_rows: Predicted query result rows.
        has_order_by: Whether either query specifies an `ORDER BY` clause.

    Returns:
        bool: True if result sets are equivalent.

    """
    if has_order_by and list(pred_rows) == list(gold_rows):
        return True
    hashable_pred = [make_hashable(r) for r in pred_rows]
    hashable_gold = [make_hashable(r) for r in gold_rows]
    if has_order_by:
        return (
            len(pred_rows) > 1
            and collections.Counter(hashable_pred) == collections.Counter(hashable_gold)
            and len({r[-1] for r in gold_rows if r}) < len(gold_rows)
        )
    return collections.Counter(hashable_pred) == collections.Counter(hashable_gold)


def canonicalize_cell_for_overlap(val: SqlValue) -> str:
    """
    Convert a SQL cell value into a normalized string token for relaxed value-overlap scoring.

    Args:
        val: Raw SQL cell value (`str`, `int`, `float`, `bool`, or `None`).

    Returns:
        str: Canonicalized string token (rounding floats to 1 decimal place for numeric tolerance).

    """
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return f"{float(val):.1f}"
    text = str(val).strip()
    try:
        return f"{float(text):.1f}"
    except ValueError:
        return " ".join(text.lower().split())


def compute_result_value_overlap_pct(
    gold_rows: Sequence[Mapping[str, SqlValue]],
    pred_rows: Sequence[Mapping[str, SqlValue]],
) -> float:
    """
    Compute the multiset cell-value overlap percentage (0.0 to 100.0) between gold and predicted result rows.

    Args:
        gold_rows: Gold query result dictionaries.
        pred_rows: Predicted query result dictionaries.

    Returns:
        float: Overlap percentage rounded to 1 decimal place.

    """
    if not gold_rows and not pred_rows:
        return 100.0
    if not gold_rows or not pred_rows:
        return 0.0

    def _extract_tokens(rows: Sequence[Mapping[str, SqlValue]]) -> list[str]:
        """
        Extract canonical cell and numeric sub-tokens from a sequence of row dicts.

        Args:
            rows: Sequence of SQL row dictionaries.

        Returns:
            list[str]: Canonicalized token list.

        """
        tokens: list[str] = []
        for row in rows:
            for val in row.values():
                canon = canonicalize_cell_for_overlap(val)
                tokens.append(canon)
                if isinstance(val, str):
                    for num_match in re.findall(r"-?\d+(?:\.\d+)?", val):
                        tokens.append(f"{float(num_match):.1f}")
        return tokens

    gold_counter = collections.Counter(_extract_tokens(gold_rows))
    pred_counter = collections.Counter(_extract_tokens(pred_rows))
    total_gold = sum(gold_counter.values())
    if total_gold == 0:
        return 100.0
    matched = sum((gold_counter & pred_counter).values())
    return round((matched / total_gold) * 100.0, 1)


def rows_to_tuples_and_columns(
    rows: Sequence[Mapping[str, SqlValue]],
) -> tuple[list[tuple[SqlValue, ...]], list[str]]:
    """
    Convert a sequence of row dictionaries into `(row_tuples, column_names)`.

    Args:
        rows: Sequence of row dictionaries returned by `execute_fn`.

    Returns:
        tuple[list[tuple[SqlValue, ...]], list[str]]: `(list_of_row_tuples, ordered_column_names)`.

    """
    if not rows:
        return [], []
    cols = list(rows[0].keys())
    tuples = [tuple(row.get(c) for c in cols) for row in rows]
    return tuples, cols


def compare_predicted_and_gold_sql(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    execute_fn: Callable[[str, str], list[dict[str, SqlValue]]],
    precomputed_pred_rows: list[dict[str, SqlValue]] | None = None,
    precomputed_pred_error: str | None = None,
) -> SqlGoldComparisonModel:
    """
    Execute and compare a predicted SQL query against a gold-standard SQL query on DuckDB.

    Args:
        db_path: Validated path to the DuckDB database file.
        pred_sql: SQL query generated by the target LLM.
        gold_sql: Canonical gold-standard SQL query.
        execute_fn: Read-only DuckDB query execution callable (`execute_sql_read_only`).
        precomputed_pred_rows: Optional already-executed result rows for `pred_sql`.
        precomputed_pred_error: Optional error message if `pred_sql` already failed execution.

    Returns:
        SqlGoldComparisonModel: Structured gold-standard comparison metrics and preview rows.

    """
    clean_pred = pred_sql.strip()
    clean_gold = gold_sql.strip()
    em = (
        normalize_sql(clean_pred) == normalize_sql(clean_gold)
        if clean_pred and clean_gold
        else False
    )

    try:
        gold_rows = execute_fn(db_path, clean_gold) if clean_gold else []
    except (
        T1DError,
        duckdb.Error,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:
        logger.debug(f"Gold SQL execution error for '{clean_gold}': {exc}")
        gold_rows = []

    pred_rows: list[dict[str, SqlValue]] | None = None
    syntax_err = False
    err_msg: str | None = precomputed_pred_error

    if precomputed_pred_error is not None:
        syntax_err = True
    elif precomputed_pred_rows is not None:
        pred_rows = precomputed_pred_rows
    elif not clean_pred:
        syntax_err = True
        err_msg = "No SQL query was generated by the model."
    else:
        try:
            pred_rows = execute_fn(db_path, clean_pred)
        except (
            T1DError,
            duckdb.Error,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            syntax_err = True
            err_msg = str(exc)

    gold_tuples, gold_cols = rows_to_tuples_and_columns(gold_rows)
    pred_tuples, pred_cols = rows_to_tuples_and_columns(pred_rows or [])

    ex = False
    overlap_pct = 0.0
    if not syntax_err and pred_rows is not None:
        has_order_by = (
            "order by" in clean_gold.lower() or "order by" in clean_pred.lower()
        )
        ex = compare_sql_result_sets(gold_tuples, pred_tuples, has_order_by)
        overlap_pct = (
            100.0 if ex else compute_result_value_overlap_pct(gold_rows, pred_rows)
        )

    return SqlGoldComparisonModel(
        gold_sql=clean_gold,
        pred_sql=clean_pred,
        exact_match=em,
        execution_accuracy=ex,
        syntax_error=syntax_err,
        gold_row_count=len(gold_rows),
        pred_row_count=len(pred_rows) if pred_rows is not None else 0,
        gold_columns=gold_cols,
        pred_columns=pred_cols,
        column_overlap_pct=overlap_pct,
        gold_rows_preview=gold_rows[:20],
        error_message=err_msg,
    )


def evaluate_single_sql_sample(
    conn: duckdb.DuckDBPyConnection,
    db_path: str,
    sample: Mapping[str, str],
    model: str,
    translate_fn: Callable[..., TranslationResult],
) -> EvalItemResultDict:
    """
    Evaluate a single Text-to-SQL benchmark item and return its `EvalItemResultDict`.

    Args:
        conn: Read-only DuckDB connection.
        db_path: Path to the DuckDB file.
        sample: Benchmark item dictionary containing `prompt`/`question` and `gold_sql`/`chosen`.
        model: Model identifier string.
        translate_fn: NL-to-SQL translation callable returning `TranslationResult`.

    Returns:
        EvalItemResultDict: Evaluation result record for the single sample.

    """
    question = sample.get("prompt") or sample.get("question") or ""
    gold_sql = (
        sample.get("gold_sql") or sample.get("chosen") or sample.get("completion") or ""
    )

    try:
        pred_sql = translate_fn(db_path, question, model_name=model).sql_query
    except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.debug(f"SQL generation failed for question '{question}': {e}")
        pred_sql = ""

    em = (
        normalize_sql(pred_sql) == normalize_sql(gold_sql)
        if pred_sql and gold_sql
        else False
    )

    pred_rows, syntax_err = execute_eval_sql(conn, pred_sql)
    gold_rows, _ = execute_eval_sql(conn, gold_sql)

    ex = False
    if pred_rows is not None and gold_rows is not None:
        has_order_by = "order by" in gold_sql.lower() or "order by" in pred_sql.lower()
        ex = compare_sql_result_sets(gold_rows, pred_rows, has_order_by)

    return {
        "question": question,
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "exact_match": em,
        "execution_accuracy": ex,
        "syntax_error": syntax_err,
    }


def format_benchmark_reports(
    model: str,
    total: int,
    em_count: int,
    ex_count: int,
    syntax_error_count: int,
    em_rate: float,
    ex_rate: float,
    syntax_rate: float,
    results: Sequence[EvalItemResultDict],
) -> tuple[str, str]:
    """
    Format Markdown and CSV reports for `evaluate_text_to_sql`.

    Args:
        model: Evaluated model name.
        total: Total sample count.
        em_count: Exact match count.
        ex_count: Execution accuracy match count.
        syntax_error_count: Syntax failure count.
        em_rate: Exact match percentage.
        ex_rate: Execution accuracy percentage.
        syntax_rate: Syntax failure percentage.
        results: Per-sample evaluation records.

    Returns:
        tuple[str, str]: `(markdown_report, csv_report)`.

    """
    nl = """
"""
    md_lines = [
        f"# Text-to-SQL Benchmark Report: {model}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Samples | {total} |",
        f"| Exact Match (EM) | {em_rate:.1f}% ({em_count}/{total}) |",
        f"| Execution Accuracy (EX) | {ex_rate:.1f}% ({ex_count}/{total}) |",
        f"| Syntax Failure Rate | {syntax_rate:.1f}% ({syntax_error_count}/{total}) |",
        "",
        "## Detailed Results",
        "",
        "| Question | Exact Match | Execution Match | Syntax Error |",
        "|---|---|---|---|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['question'][:50]} | {'PASS' if r['exact_match'] else 'FAIL'} | {'PASS' if r['execution_accuracy'] else 'FAIL'} | {'ERROR' if r['syntax_error'] else 'OK'} |"
        )
    md_report = nl.join(md_lines)

    csv_buf = io.StringIO()
    writer = csv.DictWriter(
        csv_buf,
        fieldnames=[
            "question",
            "gold_sql",
            "pred_sql",
            "exact_match",
            "execution_accuracy",
            "syntax_error",
        ],
    )
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return md_report, csv_buf.getvalue()


def evaluate_text_to_sql(
    db_path: str,
    test_data: list[dict[str, str]],
    model: str = "gemma4",
    translate_fn: Callable[..., TranslationResult] | None = None,
) -> TextToSqlEvalReportDict:
    """
    Evaluate Text-to-SQL generation accuracy against a holdout test set.

    Computes Exact Match (EM), Execution Accuracy (EX), and Syntax Failure Rates.

    Args:
        db_path: Path to DuckDB instance.
        test_data: List of test cases with 'prompt'/'question' and 'gold_sql'/'chosen'.
        model: Model identifier.
        translate_fn: Optional injectable NL-to-SQL translation callable.

    Returns:
        TextToSqlEvalReportDict: Evaluation summary metrics and generated markdown/CSV reports.

    """
    from t1d_analytics.chat_service import translate_nl_to_sql

    active_translate = translate_fn or translate_nl_to_sql

    conn = duckdb.connect(db_path, read_only=True)
    try:
        results = [
            evaluate_single_sql_sample(conn, db_path, item, model, active_translate)
            for item in test_data
        ]
    finally:
        conn.close()

    total = len(results)
    em_count = sum(1 for r in results if r["exact_match"])
    ex_count = sum(1 for r in results if r["execution_accuracy"])
    syntax_error_count = sum(1 for r in results if r["syntax_error"])

    em_rate = (em_count / total * 100) if total else 0.0
    ex_rate = (ex_count / total * 100) if total else 0.0
    syntax_rate = (syntax_error_count / total * 100) if total else 0.0

    md_report, csv_report = format_benchmark_reports(
        model=model,
        total=total,
        em_count=em_count,
        ex_count=ex_count,
        syntax_error_count=syntax_error_count,
        em_rate=em_rate,
        ex_rate=ex_rate,
        syntax_rate=syntax_rate,
        results=results,
    )

    return {
        "total_samples": total,
        "exact_match_rate": em_rate,
        "execution_accuracy": ex_rate,
        "syntax_failure_rate": syntax_rate,
        "results": results,
        "markdown_report": md_report,
        "csv_report": csv_report,
    }


__all__ = [
    "EvalItemResultDict",
    "HashableInput",
    "HashableResult",
    "TextToSqlEvalReportDict",
    "canonicalize_cell_for_overlap",
    "compare_predicted_and_gold_sql",
    "compare_sql_result_sets",
    "compute_result_value_overlap_pct",
    "evaluate_single_sql_sample",
    "evaluate_text_to_sql",
    "execute_eval_sql",
    "format_benchmark_reports",
    "make_hashable",
    "mine_dpo_rejected_sql_perturbation",
    "parse_llm_json_array",
    "rows_to_tuples_and_columns",
]
