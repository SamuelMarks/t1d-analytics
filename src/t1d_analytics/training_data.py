"""Module for generating synthetic Text-to-SQL training data using an LLM."""

import json
import urllib.request

import duckdb


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
            columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            schema_desc = f"Table: {table_name}\nColumns:\n"
            for col in columns:
                col_name = col[0]
                col_type = col[1]
                schema_desc += f"- {col_name} ({col_type})\n"
            schema[table_name] = schema_desc
        return schema

    def _generate_pairs(self, schema: str, count: int) -> list[tuple[str, str, str]]:
        """
        Generate (prompt, chosen_sql, rejected_sql) pairs using the LLM.

        Args:
        ----
            schema: The textual representation of the table's schema.
            count: The number of pairs to generate.

        Returns:
        -------
            A list of tuples, each containing:
            - The natural language prompt
            - The correct (chosen) SQL query
            - The incorrect (rejected) SQL query

        """
        pairs: list[tuple[str, str, str]] = []
        for _ in range(count):
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
                "http://localhost:11434/api/generate",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    generated_text = result.get("response", "[]")
                    parsed = json.loads(generated_text)
                    if isinstance(parsed, list) and len(parsed) == 3:
                        pairs.append((str(parsed[0]), str(parsed[1]), str(parsed[2])))
            except Exception:
                # In case of API failure or bad JSON, skip or retry
                pass
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
            # Pretrain data is just the raw text of the correct query and prompt
            pretrain_text = f"Question: {prompt}\\nSQL: {chosen}"
            self.conn.execute("INSERT INTO pretrain_data VALUES (?)", (pretrain_text,))

            # SFT data pairs prompt with chosen
            self.conn.execute("INSERT INTO sft_data VALUES (?, ?)", (prompt, chosen))

            # DPO data includes prompt, chosen, and rejected
            self.conn.execute(
                "INSERT INTO dpo_data VALUES (?, ?, ?)", (prompt, chosen, rejected)
            )
