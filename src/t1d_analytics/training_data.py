"""Module for generating synthetic Text-to-SQL training data using an LLM."""

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """
    Generator for synthetic Text-to-SQL pairs based on a DuckDB database schema.

    This class extracts the schema from a DuckDB database and uses a local LLM
    (e.g., Ollama) to generate natural language prompts along with a chosen
    and rejected SQL query for Text-to-SQL model training.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, model: str) -> None:
        """
        Initialize the TrainingDataGenerator.

        Args:
        ----
            conn: An active DuckDB connection to read the schema and write data.
            model: The local Ollama model to use for generation (e.g., 'gemma4').

        """
        self.conn = conn
        self.model = model

    def _extract_schema(self) -> dict[str, str]:
        """
        Extract the database schema using SHOW TABLES and DESCRIBE.

        Returns
        -------
            A dictionary mapping table names to their textual schema descriptions.

        """
        schema: dict[str, str] = {}
        tables = self.conn.execute("SHOW TABLES").fetchall()
        for row in tables:
            table_name = row[0]
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
        self, schema: str, count: int, validate_sql: bool = True
    ) -> list[tuple[str, str, str]]:
        """
        Generate (prompt, chosen_sql, rejected_sql) pairs using the LLM.

        Args:
        ----
            schema: The textual representation of the table's schema.
            count: The number of pairs to generate.
            validate_sql: Whether to validate chosen SQL queries against DuckDB schema.

        Returns:
        -------
            A list of tuples, each containing:
            - The natural language prompt
            - The correct (chosen) SQL query
            - The incorrect (rejected) SQL query

        """
        pairs: list[tuple[str, str, str]] = []
        base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        if not base_url.startswith("http://") and not base_url.startswith("https://"):
            base_url = f"http://{base_url}"
        api_url = f"{base_url}/api/generate"

        attempts = 0
        max_attempts = count * 3

        while len(pairs) < count and attempts < max_attempts:
            attempts += 1
            prompt = (
                f"Given the following database schema:\n{schema}\n"
                "Generate a natural language question, a correct SQL query to answer it, "
                "and an incorrect SQL query with a subtle mistake. "
                "Output as a JSON array with exactly three strings: "
                '["Question", "Correct SQL", "Incorrect SQL"]. '
                "Do not include any other text."
            )
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
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
                    parsed = json.loads(generated_text)
                    if isinstance(parsed, list) and len(parsed) == 3:
                        q, chosen, rej = (
                            str(parsed[0]),
                            str(parsed[1]),
                            str(parsed[2]),
                        )
                        if validate_sql and not self._is_valid_sql(chosen):
                            logger.warning(
                                f"Discarding pair with invalid chosen SQL: {chosen}"
                            )
                            continue
                        pairs.append((q, chosen, rej))
            except Exception as e:
                logger.warning(
                    f"Generation attempt {attempts} failed: {e}. Retrying..."
                )

        return pairs

    def write_to_db(self, pairs: list[tuple[str, str, str]]) -> None:
        """
        Write generated pairs to `pretrain_data`, `sft_data`, and `dpo_data` tables.

        Args:
        ----
            pairs: A list of tuples containing (prompt, chosen_sql, rejected_sql).

        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pretrain_data (
                text TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sft_data (
                prompt TEXT,
                completion TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dpo_data (
                prompt TEXT,
                chosen TEXT,
                rejected TEXT
            )
            """
        )

        for prompt, chosen, rejected in pairs:
            # Pretrain data contains question and SQL with actual newline
            pretrain_text = f"Question: {prompt}\nSQL: {chosen}"
            self.conn.execute("INSERT INTO pretrain_data VALUES (?)", (pretrain_text,))

            # SFT data pairs prompt with chosen
            self.conn.execute("INSERT INTO sft_data VALUES (?, ?)", (prompt, chosen))

            # DPO data includes prompt, chosen, and rejected
            self.conn.execute(
                "INSERT INTO dpo_data VALUES (?, ?, ?)", (prompt, chosen, rejected)
            )

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
