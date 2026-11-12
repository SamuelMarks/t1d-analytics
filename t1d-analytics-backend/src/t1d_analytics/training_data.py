"""Schema-driven synthetic Text-to-SQL pair generator coordinating DPO mining and dataset serialization."""

import logging
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType

import duckdb

from t1d_analytics.clinical_query_catalog import (
    ALL_GOLD_TEMPLATE_SUITE as ALL_GOLD_TEMPLATE_SUITE,
)
from t1d_analytics.clinical_query_catalog import (
    BECK_2023_TABLE_1_SQL as BECK_2023_TABLE_1_SQL,
)
from t1d_analytics.clinical_query_catalog import (
    STARTER_TEMPLATE_SUITE as STARTER_TEMPLATE_SUITE,
)
from t1d_analytics.clinical_query_catalog import (
    TABLE_1_BENCHMARK_SUITE as TABLE_1_BENCHMARK_SUITE,
)
from t1d_analytics.clinical_query_catalog import (
    Table1BenchmarkSpecDict as Table1BenchmarkSpecDict,
)
from t1d_analytics.clinical_query_catalog import (
    get_all_gold_templates as get_all_gold_templates,
)
from t1d_analytics.clinical_query_catalog import (
    get_table_1_benchmark_suite as get_table_1_benchmark_suite,
)
from t1d_analytics.errors import LlmProviderError, T1DError
from t1d_analytics.llm_client import (
    PROVIDER_ENV_KEYS,
    create_any_llm_transport,
    extract_stream_delta,
    generate_ollama_json_completion,
    validate_provider_readiness,
)
from t1d_analytics.sql_utils import SqlValue, extract_table_schemas
from t1d_analytics.training_dataset_exporter import (
    export_table_to_jsonl as export_table_to_jsonl,
)
from t1d_analytics.training_dataset_exporter import (
    export_table_to_parquet as export_table_to_parquet,
)
from t1d_analytics.training_dataset_exporter import (
    load_checkpoint_pairs as load_checkpoint_pairs,
)
from t1d_analytics.training_dataset_exporter import (
    save_checkpoint_pairs as save_checkpoint_pairs,
)
from t1d_analytics.training_dataset_exporter import (
    write_pairs_to_tables as write_pairs_to_tables,
)
from t1d_analytics.training_dataset_exporter import (
    write_training_splits_to_db as write_training_splits_to_db,
)
from t1d_analytics.training_dpo_miner import (
    EvalItemResultDict as EvalItemResultDict,
)
from t1d_analytics.training_dpo_miner import (
    HashableInput as HashableInput,
)
from t1d_analytics.training_dpo_miner import (
    HashableResult as HashableResult,
)
from t1d_analytics.training_dpo_miner import (
    TextToSqlEvalReportDict as TextToSqlEvalReportDict,
)
from t1d_analytics.training_dpo_miner import (
    canonicalize_cell_for_overlap as canonicalize_cell_for_overlap,
)
from t1d_analytics.training_dpo_miner import (
    compare_predicted_and_gold_sql as compare_predicted_and_gold_sql,
)
from t1d_analytics.training_dpo_miner import (
    compute_result_value_overlap_pct as compute_result_value_overlap_pct,
)
from t1d_analytics.training_dpo_miner import (
    evaluate_text_to_sql as evaluate_text_to_sql,
)
from t1d_analytics.training_dpo_miner import (
    make_hashable as make_hashable,
)
from t1d_analytics.training_dpo_miner import (
    mine_dpo_rejected_sql_perturbation as mine_dpo_rejected_sql_perturbation,
)
from t1d_analytics.training_dpo_miner import (
    parse_llm_json_array as parse_llm_json_array,
)
from t1d_analytics.training_dpo_miner import (
    rows_to_tuples_and_columns as rows_to_tuples_and_columns,
)

logger = logging.getLogger(__name__)

PROVIDER_KEY_ENV_VARS: dict[str, str] = PROVIDER_ENV_KEYS


class TrainingDataGenerator:
    """
    Generator for synthetic Text-to-SQL pairs based on a DuckDB database schema.

    Extracts schema metadata from DuckDB, synthesizes `(question, chosen_sql, rejected_sql)`
    triples via an LLM transport, validates SQL against DuckDB `EXPLAIN`, and delegates
    table split persistence and JSONL/Parquet serialization to `training_dataset_exporter`.
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

    def __enter__(self) -> "TrainingDataGenerator":
        """
        Enter the runtime context related to this generator.

        Returns:
            TrainingDataGenerator: The generator instance itself.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit the runtime context and close the underlying DuckDB connection.

        Args:
            exc_type: Exception type if raised inside the context block.
            exc_val: Exception instance if raised inside the context block.
            exc_tb: Exception traceback if raised inside the context block.

        """
        self.conn.close()

    def extract_schema(self) -> dict[str, str]:
        """
        Extract the database schema using SHOW TABLES and DESCRIBE.

        Returns:
            dict[str, str]: Mapping from table names to textual schema descriptions.

        """
        return extract_table_schemas(self.conn, exclude_training_tables=True)

    def build_generation_prompt(
        self,
        schema: str,
        custom_prompt_template: str | None = None,
    ) -> str:
        """
        Build the prompt for generating synthetic `(Question, Chosen SQL, Rejected SQL)` triples.

        Args:
            schema: Textual representation of the table's schema.
            custom_prompt_template: Optional custom template containing `{schema}`.

        Returns:
            str: Formatted generation prompt string.

        """
        if custom_prompt_template:
            return custom_prompt_template.format(schema=schema)
        nl = """
"""
        return (
            f"Given the following database schema:{nl}{schema}{nl}"
            "Generate a natural language question, a correct SQL query to answer it, "
            "and an incorrect SQL query with a subtle mistake. "
            "Output as a JSON array with exactly three strings: "
            '["Question", "Correct SQL", "Incorrect SQL"]. '
            "Do not include any other text."
        )

    def is_valid_sql(self, sql_query: str) -> bool:
        """
        Validate whether a SQL statement parses and can execute against the DuckDB schema.

        Args:
            sql_query: The SQL statement to validate.

        Returns:
            bool: True if the statement is valid and executable, False otherwise.

        """
        try:
            self.conn.execute(f"EXPLAIN {sql_query}")
            return True
        except (duckdb.Error, RuntimeError, ValueError) as e:
            logger.debug(f"SQL validation failed: {e}")
            return False

    def generate_pairs(
        self,
        schema: str,
        count: int,
        validate_sql: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 512,
        checkpoint_path: Path | None = None,
        custom_prompt_template: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        Generate `(prompt, chosen_sql, rejected_sql)` pairs using the configured LLM transport.

        Args:
            schema: The textual representation of the table's schema.
            count: The number of pairs to generate.
            validate_sql: Whether to validate chosen SQL queries against DuckDB schema.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens per generation request.
            checkpoint_path: Optional path to save intermediate progress and resume.
            custom_prompt_template: Optional custom prompt template string containing `{schema}`.

        Returns:
            list[tuple[str, str, str]]: Generated `(prompt, chosen_sql, rejected_sql)` triples.

        Raises:
            LlmProviderError: If initializing the cloud LLM transport fails for a non-Ollama provider.

        """
        pairs = load_checkpoint_pairs(checkpoint_path)
        if len(pairs) >= count:
            return pairs[:count]

        use_any_llm = True
        llm_transport = None
        try:
            llm_transport = create_any_llm_transport(self.provider)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as e:
            logger.debug(f"Failed to create AnyLLM transport: {e}")
            use_any_llm = False

        if self.provider != "ollama" and (not use_any_llm or llm_transport is None):
            raise LlmProviderError(
                f"Failed to initialize any-llm for provider '{self.provider}'",
                params={"provider": self.provider},
            )

        attempts = 0
        max_attempts = count * 3

        while len(pairs) < count and attempts < max_attempts:
            attempts += 1
            prompt = self.build_generation_prompt(
                schema=schema,
                custom_prompt_template=custom_prompt_template,
            )

            parsed: list[SqlValue] | None = None

            if use_any_llm and llm_transport is not None:
                try:
                    resp = llm_transport.completion(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content_str = extract_stream_delta(resp, self.provider).strip()
                    parsed = parse_llm_json_array(content_str)
                except (T1DError, OSError, RuntimeError, ValueError, TypeError) as e:
                    logger.warning(f"any-llm generation attempt {attempts} failed: {e}")

            if parsed is None and self.provider == "ollama":
                try:
                    generated_text = generate_ollama_json_completion(
                        model=self.model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        urlopen_fn=urllib.request.urlopen,
                    )
                    parsed = parse_llm_json_array(generated_text)
                except (
                    urllib.error.URLError,
                    OSError,
                    RuntimeError,
                    ValueError,
                    TypeError,
                ) as e:
                    logger.warning(
                        f"Generation attempt {attempts} failed: {e}. Retrying..."
                    )

            if isinstance(parsed, list) and len(parsed) == 3:
                q, chosen, rej = (
                    str(parsed[0]),
                    str(parsed[1]),
                    str(parsed[2])
                    or mine_dpo_rejected_sql_perturbation(str(parsed[1])),
                )
                if validate_sql and not self.is_valid_sql(chosen):
                    logger.warning(f"Discarding pair with invalid chosen SQL: {chosen}")
                    continue
                pairs.append((q, chosen, rej))
                save_checkpoint_pairs(checkpoint_path, pairs)

        return pairs

    def write_pairs_to_tables(
        self,
        pairs: list[tuple[str, str, str]],
        suffix: str = "",
    ) -> None:
        """
        Write pairs to `pretrain`, `sft`, and `dpo` DuckDB tables with a specific name suffix.

        Args:
            pairs: List of `(prompt, chosen, rejected)` pairs.
            suffix: Table suffix (`''`, `'_train'`, `'_val'`, `'_test'`).

        """
        write_pairs_to_tables(self.conn, pairs, suffix=suffix)

    def write_to_db(
        self,
        pairs: list[tuple[str, str, str]],
        split_ratios: tuple[float, float, float] | None = None,
    ) -> None:
        """
        Write generated pairs to DuckDB training tables and optional stratified splits.

        Args:
            pairs: List of `(prompt, chosen_sql, rejected_sql)` tuples.
            split_ratios: Optional tuple of `(train_ratio, val_ratio, test_ratio)`.

        """
        write_training_splits_to_db(self.conn, pairs, split_ratios=split_ratios)

    def export_to_jsonl(self, table_name: str, output_path: Path) -> None:
        """
        Export a training table to JSONL format.

        Args:
            table_name: Name of the table to export (e.g. `'sft_data'`).
            output_path: Path to write the JSONL file to.

        """
        export_table_to_jsonl(self.conn, table_name, output_path)

    def export_to_parquet(self, table_name: str, output_path: Path) -> None:
        """
        Export a training table to Parquet format.

        Args:
            table_name: Name of the table to export (e.g. `'dpo_data'`).
            output_path: Path to write the Parquet file to.

        """
        export_table_to_parquet(self.conn, table_name, output_path)
