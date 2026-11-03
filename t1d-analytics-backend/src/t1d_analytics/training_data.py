"""Module for generating synthetic Text-to-SQL training data using an LLM."""

import csv
import io
import json
import logging
import re
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import (
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import duckdb
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from t1d_analytics.errors import LlmProviderError
from t1d_analytics.llm_client import (
    PROVIDER_ENV_KEYS,
    create_any_llm_transport,
    extract_stream_delta,
    generate_ollama_json_completion,
)
from t1d_analytics.llm_client import (
    validate_provider_readiness as validate_provider_readiness,
)
from t1d_analytics.sql_utils import (
    SqlValue,
    extract_table_schemas,
)
from t1d_analytics.sql_utils import (
    normalize_sql as normalize_sql,
)

logger = logging.getLogger(__name__)

HashableInput = Union[
    SqlValue,
    Tuple["HashableInput", ...],
    List["HashableInput"],
    Set["HashableInput"],
    Dict[str, "HashableInput"],
]
HashableResult = Union[SqlValue, Tuple[Hashable, ...]]


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
    results: List[EvalItemResultDict]
    markdown_report: str
    csv_report: str


class Table1BenchmarkSpecDict(TypedDict):
    """Specification for an Academic Table 1 Text-to-SQL benchmark question and gold SQL."""

    id: str
    complexity: str
    question: str
    gold_sql: str


TABLE_1_BENCHMARK_SUITE: Tuple[Table1BenchmarkSpecDict, ...] = (
    {
        "id": "table1-overall",
        "complexity": "Low",
        "question": (
            "Generate a Table 1 summary of overall participant count, mean age, "
            "percent female, mean HbA1c, and mean Time in Range from the patients table."
        ),
        "gold_sql": (
            "SELECT COUNT(*) AS n_participants, ROUND(AVG(age), 1) AS mean_age_yr, "
            "ROUND(STDDEV_SAMP(age), 1) AS sd_age_yr, "
            "SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS female_n, "
            "ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) AS female_pct, "
            "ROUND(AVG(hba1c), 2) AS mean_hba1c_pct, ROUND(AVG(tir), 1) AS mean_tir_pct, "
            "ROUND(AVG(tbr), 2) AS mean_tbr_pct, ROUND(AVG(tar), 1) AS mean_tar_pct, "
            "ROUND(AVG(mean_glucose), 1) AS mean_cgm_mgdl FROM patients;"
        ),
    },
    {
        "id": "table1-by-arm",
        "complexity": "Low-Medium",
        "question": (
            "Create a Table 1 comparing Closed-Loop vs Control treatment arms in patients "
            "with N, Age (Mean ± SD), Female n (%), Baseline HbA1c (Mean ± SD), TIR, TBR, TAR, and total DKA events."
        ),
        "gold_sql": (
            "SELECT TxGroup AS treatment_arm, COUNT(*) AS n, "
            "ROUND(AVG(age), 1) || ' ± ' || ROUND(STDDEV_SAMP(age), 1) AS age_mean_sd, "
            "SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) || '%)' AS female_n_pct, "
            "ROUND(AVG(hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(hba1c), 2) AS hba1c_mean_sd, "
            "ROUND(AVG(tir), 1) || ' ± ' || ROUND(STDDEV_SAMP(tir), 1) AS tir_mean_sd, "
            "ROUND(AVG(tbr), 2) || ' ± ' || ROUND(STDDEV_SAMP(tbr), 2) AS tbr_mean_sd, "
            "ROUND(AVG(tar), 1) || ' ± ' || ROUND(STDDEV_SAMP(tar), 1) AS tar_mean_sd, "
            "SUM(numhospdka) AS total_dka_events FROM patients GROUP BY TxGroup ORDER BY TxGroup;"
        ),
    },
    {
        "id": "table1-by-age",
        "complexity": "Medium",
        "question": (
            "Build an age-stratified Table 1 across Pediatric (<18 yr), Adult (18-49 yr), "
            "and Older Adult (>=50 yr) cohorts showing sample size, % Closed-Loop, mean and median HbA1c, "
            "mean TIR, and % achieving ADA targets (TIR >= 70% and HbA1c < 7.0%)."
        ),
        "gold_sql": (
            "SELECT CASE WHEN age < 18 THEN '1. Pediatric (<18 yr)' "
            "WHEN age < 50 THEN '2. Adult (18-49 yr)' ELSE '3. Older Adult (>=50 yr)' END AS age_stratum, "
            "COUNT(*) AS n, ROUND(AVG(age), 1) AS mean_age, "
            "ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN 1.0 ELSE 0.0 END), 1) AS closed_loop_pct, "
            "ROUND(AVG(hba1c), 2) AS mean_hba1c, ROUND(MEDIAN(hba1c), 2) AS median_hba1c, "
            "ROUND(AVG(tir), 1) AS mean_tir, "
            "ROUND(100.0 * AVG(CASE WHEN tir >= 70.0 AND hba1c < 7.0 THEN 1.0 ELSE 0.0 END), 1) AS ada_goal_achieved_pct "
            "FROM patients GROUP BY 1 ORDER BY 1;"
        ),
    },
    {
        "id": "table1-by-pump",
        "complexity": "Medium",
        "question": (
            "Produce a device-stratified Table 1 joining subjects and patients by pump manufacturer "
            "(Insulet, Medtronic, Tandem) with N, Age (Mean ± SD), % Female, mean HbA1c, mean TIR, mean CGM glucose, and DKA count."
        ),
        "gold_sql": (
            "SELECT s.Pt_PumpManuf AS pump_manufacturer, COUNT(*) AS n, "
            "ROUND(AVG(p.age), 1) || ' ± ' || ROUND(STDDEV_SAMP(p.age), 1) AS age_mean_sd, "
            "ROUND(100.0 * AVG(CASE WHEN p.gender = 'F' THEN 1.0 ELSE 0.0 END), 1) AS female_pct, "
            "ROUND(AVG(p.hba1c), 2) AS mean_hba1c, ROUND(AVG(p.tir), 1) AS mean_tir, "
            "ROUND(AVG(p.mean_glucose), 1) AS mean_glucose_mgdl, SUM(p.numhospdka) AS dka_events "
            "FROM subjects s JOIN patients p ON s.PtID = p.subject_id GROUP BY s.Pt_PumpManuf ORDER BY s.Pt_PumpManuf;"
        ),
    },
    {
        "id": "table1-longitudinal-lab",
        "complexity": "Medium-High",
        "question": (
            "Generate a longitudinal laboratory Table 1 joining patients and hba1c_labs by treatment arm "
            "showing randomized N, lab completers N, baseline vs endpoint HbA1c, HbA1c change (delta), and baseline vs endpoint C-Peptide."
        ),
        "gold_sql": (
            "WITH lab_bounds AS (SELECT subject_id, FIRST(hba1c ORDER BY VisitDate ASC) AS baseline_hba1c, "
            "LAST(hba1c ORDER BY VisitDate ASC) AS endpoint_hba1c, "
            "FIRST(C_Peptide ORDER BY VisitDate ASC) AS baseline_c_peptide, "
            "LAST(C_Peptide ORDER BY VisitDate ASC) AS endpoint_c_peptide "
            "FROM hba1c_labs GROUP BY subject_id) "
            "SELECT p.TxGroup AS treatment_arm, COUNT(p.subject_id) AS randomized_n, "
            "COUNT(l.subject_id) AS lab_completers_n, ROUND(AVG(l.baseline_hba1c), 2) AS baseline_a1c, "
            "ROUND(AVG(l.endpoint_hba1c), 2) AS endpoint_a1c, "
            "ROUND(AVG(l.endpoint_hba1c - l.baseline_hba1c), 2) AS delta_a1c, "
            "ROUND(AVG(l.baseline_c_peptide), 3) AS baseline_c_pep, "
            "ROUND(AVG(l.endpoint_c_peptide), 3) AS endpoint_c_pep "
            "FROM patients p LEFT JOIN lab_bounds l ON p.subject_id = l.subject_id GROUP BY p.TxGroup ORDER BY p.TxGroup;"
        ),
    },
    {
        "id": "table1-multi-domain-rollup",
        "complexity": "High",
        "question": (
            "Construct a multi-domain Table 1 with an Overall Cohort rollup joining patients, "
            "raw cgm_readings, and dka_adverse_events by treatment arm, reporting N, Age, HbA1c, "
            "raw sensor CGM Mean ± SD, sensor CV%, TIR%, DKA events, severe hypoglycemia events, and hospitalizations."
        ),
        "gold_sql": (
            "WITH cgm_agg AS (SELECT subject_id, AVG(glucose_value) AS sensor_mean_mgdl, "
            "STDDEV_SAMP(glucose_value) AS sensor_sd_mgdl, "
            "100.0 * STDDEV_SAMP(glucose_value) / NULLIF(AVG(glucose_value), 0) AS sensor_cv_pct "
            "FROM cgm_readings GROUP BY subject_id), "
            "ae_agg AS (SELECT subject_id, SUM(CASE WHEN EventType = 'DKA' THEN 1 ELSE 0 END) AS dka_ae, "
            "SUM(CASE WHEN EventType = 'Severe Hypoglycemia' THEN 1 ELSE 0 END) AS sev_hypo_ae, "
            "SUM(CASE WHEN Hospitalized THEN 1 ELSE 0 END) AS hosp_ae FROM dka_adverse_events GROUP BY subject_id) "
            "SELECT COALESCE(p.TxGroup, 'Overall Cohort') AS cohort_column, COUNT(*) AS n_subjects, "
            "ROUND(AVG(p.age), 1) || ' ± ' || ROUND(STDDEV_SAMP(p.age), 1) AS age_yr_mean_sd, "
            "ROUND(AVG(p.hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(p.hba1c), 2) AS hba1c_pct_mean_sd, "
            "ROUND(AVG(c.sensor_mean_mgdl), 1) || ' ± ' || ROUND(AVG(c.sensor_sd_mgdl), 1) AS cgm_mgdl_mean_sd, "
            "ROUND(AVG(c.sensor_cv_pct), 1) AS cgm_cv_pct, ROUND(AVG(p.tir), 1) AS tir_70_180_pct, "
            "COALESCE(SUM(a.dka_ae), 0) AS dka_events, COALESCE(SUM(a.sev_hypo_ae), 0) AS sev_hypo_events, "
            "COALESCE(SUM(a.hosp_ae), 0) AS hospitalizations "
            "FROM patients p LEFT JOIN cgm_agg c ON p.subject_id = c.subject_id "
            "LEFT JOIN ae_agg a ON p.subject_id = a.subject_id "
            "GROUP BY ROLLUP(p.TxGroup) ORDER BY CASE WHEN p.TxGroup IS NULL THEN 1 ELSE 0 END, p.TxGroup;"
        ),
    },
    {
        "id": "table1-transposed-nejm",
        "complexity": "Very High",
        "question": (
            "Generate a transposed NEJM/Lancet-style Table 1 where each row is a clinical characteristic "
            "(Sample Size N, Age Mean ± SD, Female Sex n (%), Baseline HbA1c Mean ± SD, Time in Range Mean ± SD, "
            "Time Below Range Median (IQR)) and columns are Closed-Loop Arm, Control Arm, Overall Cohort, and Arm Difference."
        ),
        "gold_sql": (
            "WITH base AS (SELECT p.*, s.Pt_PumpManuf FROM patients p LEFT JOIN subjects s ON p.subject_id = s.PtID) "
            "SELECT 1 AS sort_ord, 'Sample Size (N)' AS characteristic, "
            "CAST(SUM(CASE WHEN TxGroup = 'Closed-Loop' THEN 1 ELSE 0 END) AS VARCHAR) AS closed_loop_arm, "
            "CAST(SUM(CASE WHEN TxGroup = 'Control' THEN 1 ELSE 0 END) AS VARCHAR) AS control_arm, "
            "CAST(COUNT(*) AS VARCHAR) AS overall_cohort, '—' AS arm_difference FROM base "
            "UNION ALL SELECT 2, 'Age (years), Mean ± SD', "
            "ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN age END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN age END), 1), "
            "ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN age END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN age END), 1), "
            "ROUND(AVG(age), 1) || ' ± ' || ROUND(STDDEV_SAMP(age), 1), "
            "CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN age END) - AVG(CASE WHEN TxGroup = 'Control' THEN age END), 2) AS VARCHAR) FROM base "
            "UNION ALL SELECT 3, 'Female Sex, n (%)', "
            "SUM(CASE WHEN TxGroup = 'Closed-Loop' AND gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END END), 1) || '%)', "
            "SUM(CASE WHEN TxGroup = 'Control' AND gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN TxGroup = 'Control' THEN CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END END), 1) || '%)', "
            "SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) || ' (' || ROUND(100.0 * AVG(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END), 1) || '%)', '—' FROM base "
            "UNION ALL SELECT 4, 'Baseline HbA1c (%), Mean ± SD', "
            "ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END), 2) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END), 2), "
            "ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2), "
            "ROUND(AVG(hba1c), 2) || ' ± ' || ROUND(STDDEV_SAMP(hba1c), 2), "
            "CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN hba1c END) - AVG(CASE WHEN TxGroup = 'Control' THEN hba1c END), 2) AS VARCHAR) FROM base "
            "UNION ALL SELECT 5, 'Time in Range 70-180 mg/dL (%), Mean ± SD', "
            "ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END), 1), "
            "ROUND(AVG(CASE WHEN TxGroup = 'Control' THEN tir END), 1) || ' ± ' || ROUND(STDDEV_SAMP(CASE WHEN TxGroup = 'Control' THEN tir END), 1), "
            "ROUND(AVG(tir), 1) || ' ± ' || ROUND(STDDEV_SAMP(tir), 1), "
            "CAST(ROUND(AVG(CASE WHEN TxGroup = 'Closed-Loop' THEN tir END) - AVG(CASE WHEN TxGroup = 'Control' THEN tir END), 2) AS VARCHAR) FROM base "
            "UNION ALL SELECT 6, 'Time Below Range <70 mg/dL (%), Median (IQR)', "
            "ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Closed-Loop' THEN tbr END, 0.75), 2) || ')', "
            "ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(CASE WHEN TxGroup = 'Control' THEN tbr END, 0.75), 2) || ')', "
            "ROUND(QUANTILE_CONT(tbr, 0.5), 2) || ' (' || ROUND(QUANTILE_CONT(tbr, 0.25), 2) || '-' || ROUND(QUANTILE_CONT(tbr, 0.75), 2) || ')', '—' FROM base ORDER BY sort_ord;"
        ),
    },
)


def get_table_1_benchmark_suite() -> List[Dict[str, str]]:
    """
    Return the 7-tier Academic Table 1 Text-to-SQL benchmark dataset for `evaluate_text_to_sql`.

    Returns
    -------
        List[Dict[str, str]]: List of benchmark dicts with `id`, `complexity`, `question`, and `gold_sql`.

    """
    return [
        {
            "id": item["id"],
            "complexity": item["complexity"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
        }
        for item in TABLE_1_BENCHMARK_SUITE
    ]


def _make_hashable(val: HashableInput) -> HashableResult:
    """
    Recursively convert unhashable types into immutable hashable equivalents.

    Args:
    ----
        val: Python value (lists, dicts, tuples, primitives).

    Returns:
    -------
        HashableResult: An immutable, hashable version of the input value.

    """
    if isinstance(val, (list, tuple)):
        return tuple(_make_hashable(x) for x in val)
    if isinstance(val, dict):
        items_map = [(str(k), _make_hashable(v)) for k, v in val.items()]
        items_map.sort(key=lambda pair: pair[0])
        return tuple(items_map)
    if isinstance(val, set):
        items_set = [_make_hashable(x) for x in val]
        items_set.sort(key=lambda item: str(item))
        return tuple(items_set)
    return val


_SQL_VALUE_LIST_ADAPTER: TypeAdapter[list[SqlValue]] = TypeAdapter(list[SqlValue])


def _parse_llm_json_array(content_str: str) -> Optional[list[SqlValue]]:
    """
    Parse a JSON array from LLM response text, stripping markdown blocks if present.

    Args:
    ----
        content_str: Raw response text from the LLM.

    Returns:
    -------
        Optional[list[SqlValue]]: Parsed list of values if successful, otherwise None.

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
            return _SQL_VALUE_LIST_ADAPTER.validate_python(data)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to decode exact JSON list: {e}")
    except Exception as e:
        logger.debug(f"Unexpected error decoding exact JSON list: {e}")
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            extracted = list(json.loads(match.group(0)))
            return _SQL_VALUE_LIST_ADAPTER.validate_python(extracted)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to decode extracted JSON list: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error decoding extracted JSON list: {e}")
    return None


PROVIDER_KEY_ENV_VARS: dict[str, str] = PROVIDER_ENV_KEYS


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

    def __enter__(self) -> "TrainingDataGenerator":
        """
        Enter the runtime context related to this generator.

        Returns
        -------
            TrainingDataGenerator: The generator instance itself.

        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the runtime context and close the underlying DuckDB connection.

        Args:
        ----
            exc_type: Exception type if raised inside the context block.
            exc_val: Exception instance if raised inside the context block.
            exc_tb: Exception traceback if raised inside the context block.

        """
        self.conn.close()

    def extract_schema(self) -> dict[str, str]:
        """
        Extract the database schema using SHOW TABLES and DESCRIBE.

        Returns
        -------
            dict[str, str]: A dictionary mapping table names to their textual schema descriptions.

        """
        return extract_table_schemas(self.conn, exclude_training_tables=True)

    _extract_schema = extract_schema

    def build_generation_prompt(
        self,
        schema: str,
        custom_prompt_template: Optional[str] = None,
    ) -> str:
        """
        Build the prompt for generating synthetic (Question, Chosen SQL, Rejected SQL) triples.

        Args:
        ----
            schema: Textual representation of the table's schema.
            custom_prompt_template: Optional custom template containing `{schema}`.

        Returns:
        -------
            str: Formatted generation prompt string.

        """
        if custom_prompt_template:
            return custom_prompt_template.format(schema=schema)
        return (
            f"Given the following database schema:\n{schema}\n"
            "Generate a natural language question, a correct SQL query to answer it, "
            "and an incorrect SQL query with a subtle mistake. "
            "Output as a JSON array with exactly three strings: "
            '["Question", "Correct SQL", "Incorrect SQL"]. '
            "Do not include any other text."
        )

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
        except duckdb.Error as e:
            logger.debug(f"SQL validation failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error in SQL validation: {e}")
            return False

    def generate_pairs(
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
        Generate (prompt, chosen_sql, rejected_sql) pairs using the configured LLM transport.

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
            list[tuple[str, str, str]]: Generated (prompt, chosen_sql, rejected_sql) triples.

        """
        return self._generate_pairs(
            schema=schema,
            count=count,
            validate_sql=validate_sql,
            temperature=temperature,
            max_tokens=max_tokens,
            checkpoint_path=checkpoint_path,
            custom_prompt_template=custom_prompt_template,
        )

    def _load_checkpoint_pairs(
        self,
        checkpoint_path: Optional[Path],
    ) -> list[tuple[str, str, str]]:
        """
        Load previously generated `(prompt, chosen, rejected)` triples from a JSON checkpoint file.

        Args:
        ----
            checkpoint_path: Optional path to checkpoint JSON file.

        Returns:
        -------
            list[tuple[str, str, str]]: Loaded triples or empty list.

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
            except Exception as e:
                logger.warning(f"Could not load checkpoint from {checkpoint_path}: {e}")
        return pairs

    def _save_checkpoint_pairs(
        self,
        checkpoint_path: Optional[Path],
        pairs: list[tuple[str, str, str]],
    ) -> None:
        """
        Persist generated `(prompt, chosen, rejected)` triples to a JSON checkpoint file.

        Args:
        ----
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
                )
            )
        except Exception as e:
            logger.warning(f"Could not write checkpoint: {e}")

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
        Generate (prompt, chosen_sql, rejected_sql) pairs using the configured LLM transport.

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
            list[tuple[str, str, str]]: Generated (prompt, chosen_sql, rejected_sql) triples.

        Raises:
        ------
            LlmProviderError: If initializing the cloud LLM transport fails for a non-Ollama provider.

        """
        pairs = self._load_checkpoint_pairs(checkpoint_path)
        if len(pairs) >= count:
            return pairs[:count]

        use_any_llm = True
        llm_transport = None
        try:
            llm_transport = create_any_llm_transport(self.provider)
        except Exception as e:
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

            parsed: Optional[list[SqlValue]] = None

            if use_any_llm and llm_transport is not None:
                try:
                    resp = llm_transport.completion(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content_str = extract_stream_delta(resp, self.provider).strip()
                    parsed = _parse_llm_json_array(content_str)
                except Exception as e:
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
                self._save_checkpoint_pairs(checkpoint_path, pairs)

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


def _execute_eval_sql(
    conn: duckdb.DuckDBPyConnection,
    sql_query: str,
) -> tuple[Optional[list[tuple[SqlValue, ...]]], bool]:
    """
    Execute a SQL query on `conn` and return `(rows, had_execution_error)`.

    Args:
    ----
        conn: Read-only DuckDB connection.
        sql_query: SQL query to execute.

    Returns:
    -------
        tuple[Optional[list[tuple[SqlValue, ...]]], bool]: `(fetched_rows, had_error)`.

    """
    if not sql_query:
        return None, False
    try:
        return conn.execute(sql_query).fetchall(), False
    except duckdb.Error as e:
        logger.debug(f"Execution error for SQL '{sql_query}': {e}")
        return None, True
    except Exception as e:
        logger.debug(f"Unexpected error for SQL '{sql_query}': {e}")
        return None, True


def evaluate_text_to_sql(
    db_path: str,
    test_data: list[dict[str, str]],
    model: str = "gemma4",
    translate_fn: Optional[Callable[..., tuple[str, str]]] = None,
) -> TextToSqlEvalReportDict:
    """
    Evaluate Text-to-SQL generation accuracy against a holdout test set.

    Computes Exact Match (EM), Execution Accuracy (EX), and Syntax Failure Rates.

    Args:
    ----
        db_path: Path to DuckDB instance.
        test_data: List of test cases with 'prompt'/'question' and 'gold_sql'/'chosen'.
        model: Model identifier.
        translate_fn: Optional injectable NL-to-SQL translation callable.

    Returns:
    -------
        TextToSqlEvalReportDict: Evaluation summary metrics and generated markdown/CSV reports.

    """
    from t1d_analytics.chat_service import generate_sql_from_nl

    active_translate = translate_fn or generate_sql_from_nl

    conn = duckdb.connect(db_path, read_only=True)
    total = len(test_data)
    em_count = 0
    ex_count = 0
    syntax_error_count = 0
    results: list[EvalItemResultDict] = []

    for item in test_data:
        question = item.get("prompt") or item.get("question") or ""
        gold_sql = (
            item.get("gold_sql") or item.get("chosen") or item.get("completion") or ""
        )

        try:
            _, pred_sql = active_translate(db_path, question, model_name=model)
        except Exception as e:
            logger.debug(f"SQL generation failed for question '{question}': {e}")
            pred_sql = ""

        em = (
            normalize_sql(pred_sql) == normalize_sql(gold_sql)
            if pred_sql and gold_sql
            else False
        )
        if em:
            em_count += 1

        pred_rows, syntax_err = _execute_eval_sql(conn, pred_sql)
        if syntax_err:
            syntax_error_count += 1
        gold_rows, _ = _execute_eval_sql(conn, gold_sql)

        ex = False
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
