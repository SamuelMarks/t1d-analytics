"""Module for generating synthetic Text-to-SQL training data using an LLM."""

import csv
import io
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import duckdb

logger = logging.getLogger(__name__)


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query for exact-match comparison.

    Args:
    ----
        sql: SQL query string.

    Returns:
    -------
        str: Normalized query string.

    """
    sql = sql.strip().rstrip(";")
    return re.sub(r"\s+", " ", sql).strip().lower()


def _make_hashable(val: Any) -> Any:
    """
    Recursively convert unhashable types into immutable hashable equivalents.

    Args:
    ----
        val: Any Python value (lists, dicts, tuples, primitives).

    Returns:
    -------
        Any: An immutable, hashable version of the input value.

    """
    if isinstance(val, (list, tuple)):
        return tuple(_make_hashable(x) for x in val)
    if isinstance(val, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
    if isinstance(val, set):
        return tuple(sorted(_make_hashable(x) for x in val))
    return val


def _parse_llm_json_array(content_str: str) -> Optional[list[object]]:
    """
    Parse a JSON array from LLM response text, stripping markdown blocks if present.

    Args:
    ----
        content_str: Raw response text from the LLM.

    Returns:
    -------
        Optional[list[object]]: Parsed list of objects if successful, otherwise None.

    """
    cleaned = content_str.strip()
    if cleaned.startswith("```"):
        lines = [
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return list(json.loads(match.group(0)))
        except Exception:
            pass
    return None


PROVIDER_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def validate_provider_readiness(provider: str) -> None:
    """
    Validate that the requested LLM provider is available and properly configured.

    Args:
    ----
        provider: Provider identifier (e.g. 'ollama', 'openai', 'anthropic', 'google').

    Raises:
    ------
        RuntimeError: If dependencies or API credentials are missing.

    """
    prov = provider.lower()
    if prov != "ollama":
        try:
            import any_llm  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            raise RuntimeError(
                f"Provider '{provider}' requires any-llm-sdk. Install with: pip install any-llm-sdk[{provider}]"
            )
        if prov in PROVIDER_KEY_ENV_VARS:
            env_var = PROVIDER_KEY_ENV_VARS[prov]
            if not os.environ.get(env_var):
                raise RuntimeError(
                    f"Provider '{provider}' requires environment variable '{env_var}' to be set."
                )


class TrainingDataGenerator:
    """
    Generator for synthetic Text-to-SQL pairs based on a DuckDB database schema.

    This class extracts the schema from a DuckDB database and uses an LLM
    (via any-llm or local Ollama) to generate natural language prompts along with
    chosen and rejected SQL queries for Text-to-SQL model training.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        model: str = "gemma4",
        provider: str = "ollama",
    ) -> None:
        """
        Initialize the TrainingDataGenerator.

        Args:
        ----
            conn: An active DuckDB connection to read the schema and write data.
            model: The model name or provider/model string (e.g. 'gemma4', 'openai/gpt-4o').
            provider: The LLM provider (default: 'ollama').

        """
        self.conn = conn
        if "/" in model:
            prov, mod = model.split("/", 1)
            self.provider = prov
            self.model = mod
        else:
            self.provider = provider
            self.model = model
        validate_provider_readiness(self.provider)

    def _extract_schema(self) -> dict[str, str]:
        """
        Extract the database schema using SHOW TABLES and DESCRIBE.

        Returns
        -------
            A dictionary mapping table names to their textual schema descriptions.

        """
        schema: dict[str, str] = {}
        tables = self.conn.execute("SHOW TABLES").fetchall()
        ignored_names = {"chat_sessions"}
        ignored_prefixes = ("_t1d_", "pretrain_", "sft_", "dpo_")
        for row in tables:
            table_name = row[0]
            lower_name = table_name.lower()
            if (
                table_name.startswith("_t1d_")
                or lower_name in ignored_names
                or lower_name in ("pretrain", "sft", "dpo")
                or lower_name.startswith(ignored_prefixes)
            ):
                continue

            columns = self.conn.execute(f'DESCRIBE "{table_name}"').fetchall()
            schema_desc = f"Table: {table_name}\nColumns:\n"
            for col in columns:
                col_name = col[0]
                col_type = col[1]
                schema_desc += f"- {col_name} ({col_type})\n"
            schema[table_name] = schema_desc
        return schema

    def _is_valid_sql(self, sql_query: str) -> bool:
        """
        Validate whether a SQL statement parses and can execute against the schema.

        Args:
        ----
            sql_query: The SQL statement to validate.

        Returns:
        -------
            True if the statement is valid and executable, False otherwise.

        """
        try:
            self.conn.execute(f"EXPLAIN {sql_query}")
            return True
        except Exception:
            return False

    def _generate_pairs(
        self,
        schema: str,
        count: int,
        validate_sql: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 512,
        checkpoint_path: Optional[Path] = None,
        custom_prompt_template: Optional[str] = None,
    ) -> list[tuple[str, str, str]]:
        """
        Generate (prompt, chosen_sql, rejected_sql) pairs using the LLM.

        Args:
        ----
            schema: The textual representation of the table's schema.
            count: The number of pairs to generate.
            validate_sql: Whether to validate chosen SQL queries against DuckDB schema.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens per generation request.
            checkpoint_path: Optional path to save intermediate progress and resume.
            custom_prompt_template: Optional custom prompt template string containing {schema}.

        Returns:
        -------
            A list of tuples, each containing:
            - The natural language prompt
            - The correct (chosen) SQL query
            - The incorrect (rejected) SQL query

        """
        pairs: list[tuple[str, str, str]] = []
        if checkpoint_path and checkpoint_path.exists():
            try:
                ckpt = json.loads(checkpoint_path.read_text())
                if isinstance(ckpt, list):
                    for item in ckpt:
                        if isinstance(item, dict) and "prompt" in item:
                            pairs.append(
                                (item["prompt"], item["chosen"], item["rejected"])
                            )
                if len(pairs) >= count:
                    return pairs[:count]
            except Exception as e:
                logger.warning(f"Could not load checkpoint from {checkpoint_path}: {e}")

        base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            base_url = f"http://{base_url}"
        api_url = f"{base_url}/api/generate"

        use_any_llm = True
        llm_instance = None
        try:
            from any_llm import AnyLLM

            llm_instance = AnyLLM.create(self.provider)
        except Exception:
            use_any_llm = False

        if self.provider != "ollama" and (not use_any_llm or llm_instance is None):
            raise RuntimeError(
                f"Failed to initialize any-llm for provider '{self.provider}'"
            )

        attempts = 0
        max_attempts = count * 3

        while len(pairs) < count and attempts < max_attempts:
            attempts += 1
            if custom_prompt_template:
                prompt = custom_prompt_template.format(schema=schema)
            else:
                prompt = (
                    f"Given the following database schema:\n{schema}\n"
                    "Generate a natural language question, a correct SQL query to answer it, "
                    "and an incorrect SQL query with a subtle mistake. "
                    "Output as a JSON array with exactly three strings: "
                    '["Question", "Correct SQL", "Incorrect SQL"]. '
                    "Do not include any other text."
                )

            parsed: Optional[list[object]] = None

            if use_any_llm and llm_instance is not None:
                try:
                    resp = llm_instance.completion(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    raw_choice = resp.choices[0].message.content
                    content_str = (raw_choice or "").strip()
                    parsed = _parse_llm_json_array(content_str)
                except Exception as e:
                    logger.warning(f"any-llm generation attempt {attempts} failed: {e}")

            if parsed is None and self.provider == "ollama":
                data = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                }
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(data).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=30.0) as response:
                        result = json.loads(response.read().decode("utf-8"))
                        generated_text = result.get("response", "[]")
                        parsed = _parse_llm_json_array(generated_text)
                except Exception as e:
                    logger.warning(
                        f"Generation attempt {attempts} failed: {e}. Retrying..."
                    )

            if isinstance(parsed, list) and len(parsed) == 3:
                q, chosen, rej = (
                    str(parsed[0]),
                    str(parsed[1]),
                    str(parsed[2]),
                )
                if validate_sql and not self._is_valid_sql(chosen):
                    logger.warning(f"Discarding pair with invalid chosen SQL: {chosen}")
                    continue
                pairs.append((q, chosen, rej))
                if checkpoint_path:
                    try:
                        checkpoint_path.write_text(
                            json.dumps(
                                [
                                    {"prompt": p, "chosen": c, "rejected": r}
                                    for p, c, r in pairs
                                ],
                                indent=2,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Could not write checkpoint: {e}")

        return pairs

    def write_to_db(
        self,
        pairs: list[tuple[str, str, str]],
        split_ratios: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """
        Write generated pairs to training tables in DuckDB.

        Populates base tables (pretrain_data, sft_data, dpo_data), and if split_ratios is
        specified, outputs stratified train, val, and test tables:
        (pretrain_train, pretrain_val, pretrain_test, sft_train, sft_val, sft_test,
        dpo_train, dpo_val, dpo_test).

        Args:
        ----
            pairs: A list of tuples containing (prompt, chosen_sql, rejected_sql).
            split_ratios: Optional tuple of (train_ratio, val_ratio, test_ratio).

        """
        self._write_pairs_to_tables(pairs, suffix="")

        if split_ratios:
            train_r, val_r, _ = split_ratios
            n = len(pairs)
            n_train = int(n * train_r)
            n_val = int(n * val_r)

            train_pairs = pairs[:n_train]
            val_pairs = pairs[n_train : n_train + n_val]
            test_pairs = pairs[n_train + n_val :]

            self._write_pairs_to_tables(train_pairs, suffix="_train")
            self._write_pairs_to_tables(val_pairs, suffix="_val")
            self._write_pairs_to_tables(test_pairs, suffix="_test")

    def _write_pairs_to_tables(
        self, pairs: list[tuple[str, str, str]], suffix: str = ""
    ) -> None:
        """
        Write pairs to pretrain, sft, and dpo tables with a specific name suffix.

        Args:
        ----
            pairs: List of (prompt, chosen, rejected) pairs.
            suffix: Table suffix (e.g. '', '_train', '_val', '_test').

        """
        pretrain_tbl = (
            f"pretrain{suffix}" if suffix.startswith("_") else f"pretrain_data{suffix}"
        )
        sft_tbl = f"sft{suffix}" if suffix.startswith("_") else f"sft_data{suffix}"
        dpo_tbl = f"dpo{suffix}" if suffix.startswith("_") else f"dpo_data{suffix}"

        self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{pretrain_tbl}" (text TEXT)')
        self.conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{sft_tbl}" (prompt TEXT, completion TEXT)'
        )
        self.conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{dpo_tbl}" (prompt TEXT, chosen TEXT, rejected TEXT)'
        )

        if not pairs:
            return

        pretrain_rows = [(f"Question: {p}\nSQL: {c}",) for p, c, _ in pairs]
        sft_rows = [(p, c) for p, c, _ in pairs]
        dpo_rows = [(p, c, r) for p, c, r in pairs]

        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.executemany(
                f'INSERT INTO "{pretrain_tbl}" VALUES (?)', pretrain_rows
            )
            self.conn.executemany(f'INSERT INTO "{sft_tbl}" VALUES (?, ?)', sft_rows)
            self.conn.executemany(f'INSERT INTO "{dpo_tbl}" VALUES (?, ?, ?)', dpo_rows)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def export_to_jsonl(self, table_name: str, output_path: Path) -> None:
        """
        Export a training table to JSONL format.

        Args:
        ----
            table_name: Name of the table to export (e.g. 'sft_data').
            output_path: Path to write the JSONL file to.

        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        escaped_path = str(output_path).replace("'", "''")
        self.conn.execute(
            f"""COPY "{table_name}" TO '{escaped_path}' (FORMAT JSON, ARRAY FALSE)"""
        )

    def export_to_parquet(self, table_name: str, output_path: Path) -> None:
        """
        Export a training table to Parquet format.

        Args:
        ----
            table_name: Name of the table to export (e.g. 'dpo_data').
            output_path: Path to write the Parquet file to.

        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        escaped_path = str(output_path).replace("'", "''")
        self.conn.execute(
            f"""COPY "{table_name}" TO '{escaped_path}' (FORMAT PARQUET)"""
        )


def evaluate_text_to_sql(
    db_path: str,
    test_data: list[dict[str, str]],
    model: str = "gemma4",
) -> dict[str, Any]:
    """
    Evaluate Text-to-SQL generation accuracy against a holdout test set.

    Computes Exact Match (EM), Execution Accuracy (EX), and Syntax Failure Rates.

    Args:
    ----
        db_path: Path to DuckDB instance.
        test_data: List of test cases with 'prompt'/'question' and 'gold_sql'/'chosen'.
        model: Model identifier.

    Returns:
    -------
        dict[str, Any]: Evaluation summary metrics and generated markdown/CSV reports.

    """
    from t1d_analytics.api import generate_sql_from_nl

    conn = duckdb.connect(db_path, read_only=True)
    total = len(test_data)
    em_count = 0
    ex_count = 0
    syntax_error_count = 0
    results: list[dict[str, Any]] = []

    for item in test_data:
        question = item.get("prompt") or item.get("question") or ""
        gold_sql = (
            item.get("gold_sql") or item.get("chosen") or item.get("completion") or ""
        )

        try:
            _, pred_sql = generate_sql_from_nl(db_path, question, model_name=model)
        except Exception:
            pred_sql = ""

        em = (
            normalize_sql(pred_sql) == normalize_sql(gold_sql)
            if pred_sql and gold_sql
            else False
        )
        if em:
            em_count += 1

        ex = False
        syntax_err = False
        try:
            pred_rows = conn.execute(pred_sql).fetchall() if pred_sql else None
        except Exception:
            pred_rows = None
            syntax_err = True
            syntax_error_count += 1

        try:
            gold_rows = conn.execute(gold_sql).fetchall() if gold_sql else None
        except Exception:
            gold_rows = None

        if pred_rows is not None and gold_rows is not None:
            if "order by" in gold_sql.lower() or "order by" in pred_sql.lower():
                ex = pred_rows == gold_rows
            else:
                import collections

                hashable_pred = [_make_hashable(r) for r in pred_rows]
                hashable_gold = [_make_hashable(r) for r in gold_rows]
                ex = collections.Counter(hashable_pred) == collections.Counter(
                    hashable_gold
                )
            if ex:
                ex_count += 1

        results.append(
            {
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "exact_match": em,
                "execution_accuracy": ex,
                "syntax_error": syntax_err,
            }
        )

    conn.close()

    em_rate = (em_count / total * 100) if total else 0.0
    ex_rate = (ex_count / total * 100) if total else 0.0
    syntax_rate = (syntax_error_count / total * 100) if total else 0.0

    # Markdown report
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
    md_report = "\n".join(md_lines)

    # CSV report
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
    csv_report = csv_buf.getvalue()

    return {
        "total_samples": total,
        "exact_match_rate": em_rate,
        "execution_accuracy": ex_rate,
        "syntax_failure_rate": syntax_rate,
        "results": results,
        "markdown_report": md_report,
        "csv_report": csv_report,
    }
