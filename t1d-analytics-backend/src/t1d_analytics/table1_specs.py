"""Declarative clinical variable metadata specifications and schema capability probes for Table 1 generation."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import duckdb
from pydantic.dataclasses import dataclass

from t1d_analytics.jaeb_cohort_sql import BECK2023_REQUIRED_TRIAL_TABLES
from t1d_analytics.sql_utils import escape_sql_identifier

ContinuousSummaryFormat = Literal["mean_sd", "median_iqr", "range"]
RowVariableKind = Literal["continuous", "categorical", "hybrid"]

EXTENDED_JAEB_REQUIRED_TABLES: frozenset[str] = frozenset(
    {
        "tblaptsummary",
        "tblalabhba1c",
        "hptroster",
        "hscreening",
        "hlocalhba1c",
    }
)


@dataclass(frozen=True)
class CategoryBinSpec:
    """
    Specification for a single categorical or cut-point bin row inside a Table 1 variable block.

    Attributes:
        label: Indented display label in the `Characteristic` column (e.g. `' <6 years, n (%)'`).
        sql_condition: Boolean SQL predicate evaluated over `base` (e.g. `'age < 6'`).

    """

    label: str
    sql_condition: str


@dataclass(frozen=True)
class Table1RowSpec:
    """
    Specification for a clinical characteristic row section in a publication Table 1.

    Attributes:
        key: Canonical identifier for the variable (e.g. `'age'`, `'hba1c'`, `'tir'`).
        header_label: Section header text in the `Characteristic` column.
        value_expr: SQL expression for continuous aggregation (`AVG`, `STDDEV_SAMP`, `MEDIAN`).
        range_expr: SQL expression for range (`MIN`, `MAX`) evaluation.
        null_check_expr: SQL column/expression checked for `IS NULL` missingness.
        var_type: One of `'continuous'`, `'categorical'`, or `'hybrid'`.
        decimals: Number of decimal places for `Mean ± SD`, `Median (IQR)`, and `Range`.
        unit_suffix: Optional unit suffix appended to numbers (e.g. `'%'` for TIR/TBR).
        summary_formats: Tuple of continuous summary sub-rows to emit (`'mean_sd'`, `'median_iqr'`, `'range'`).
        bins: Tuple of `CategoryBinSpec` cut-point or category sub-rows.
        include_missing_row: Whether to append a `' Missing, n (%)'` sub-row.
        dynamic_category_expr: Optional SQL column expression for data-driven `GROUP BY` category discovery.

    """

    key: str
    header_label: str
    value_expr: str
    range_expr: str
    null_check_expr: str
    var_type: RowVariableKind = "hybrid"
    decimals: int = 1
    unit_suffix: str = ""
    summary_formats: tuple[ContinuousSummaryFormat, ...] = ("mean_sd", "range")
    bins: tuple[CategoryBinSpec, ...] = ()
    include_missing_row: bool = False
    dynamic_category_expr: str | None = None


@dataclass(frozen=True)
class Table1StratumSpec:
    """
    Specification for a stratification column in a generated Table 1 query.

    Attributes:
        key_id: Safe internal SQL identifier suffix (`'all'`, `'ciq'`, `'ctrl'`, `'s1'`, ...).
        label: Base display name for the column header (e.g. `'Overall'`, `'Control-IQ'`, `'DCLP3'`).
        predicate_sql: Boolean SQL condition over `base` selecting rows in this stratum (`'1=1'` for Overall).

    """

    key_id: str
    label: str
    predicate_sql: str

    @property
    def column_alias(self) -> str:
        """
        Return the clean SQL column alias without hardcoded sample sizes.

        Returns:
            str: Column alias string (e.g. `'Overall'`, `'Control-IQ'`, `'Control'`).

        """
        return self.label


@dataclass(frozen=True)
class Table1QuerySpec:
    """
    Complete compositional specification for compiling a DuckDB Table 1 SQL query.

    Attributes:
        target_table: Source table or view name (default `'patients'`).
        where_clauses: Tuple of SQL filter predicates defining the participant subset.
        strata: Tuple of `Table1StratumSpec` column definitions.
        rows: Tuple of `Table1RowSpec` characteristic row sections.
        include_n_row: Whether to include a top `'Participants, N'` row in the result set.
        limit_rows: Optional participant row limit (`LIMIT N`) applied inside the `base` CTE.
        raw_source_cte_sql: Optional leading CTE block deriving `raw_jaeb_cohort` from raw study tables.

    """

    target_table: str
    where_clauses: tuple[str, ...]
    strata: tuple[Table1StratumSpec, ...]
    rows: tuple[Table1RowSpec, ...]
    include_n_row: bool = True
    limit_rows: int | None = None
    raw_source_cte_sql: str | None = None


def build_beck_2023_row_specs(
    summary_formats: tuple[ContinuousSummaryFormat, ...] = ("mean_sd", "range"),
) -> dict[str, Table1RowSpec]:
    """
    Build the dictionary of canonical clinical characteristic row specifications.

    Includes all 10 baseline sections from Beck et al. (2023) Table 1 plus extended
    glycemic, metabolic, and safety characteristics (`tbr_54`, `tar`, `tar_250`,
    `mean_glucose`, `cv`, `dka_events`, `severe_hypo_events`, `study`).

    Args:
        summary_formats: Continuous summary formats to emit for continuous/hybrid variables.

    Returns:
        dict[str, Table1RowSpec]: Mapping from variable key to `Table1RowSpec`.

    """
    return {
        "age": Table1RowSpec(
            key="age",
            header_label="Age at enrollment",
            value_expr="age_ex",
            range_expr="age",
            null_check_expr="age",
            var_type="hybrid",
            decimals=0,
            unit_suffix="",
            summary_formats=summary_formats,
            bins=(
                CategoryBinSpec(" <6 years, n (%)", "age < 6"),
                CategoryBinSpec(" 6 to <14 years, n (%)", "age >= 6 AND age < 14"),
                CategoryBinSpec(" 14 to <25 years, n (%)", "age >= 14 AND age < 25"),
                CategoryBinSpec(" 25 to <50 years, n (%)", "age >= 25 AND age < 50"),
                CategoryBinSpec(" ≥50 years, n (%)", "age >= 50"),
            ),
            include_missing_row=False,
        ),
        "gender": Table1RowSpec(
            key="gender",
            header_label="Sex, n (%)",
            value_expr="NULL",
            range_expr="NULL",
            null_check_expr="gender",
            var_type="categorical",
            bins=(
                CategoryBinSpec(" Female", "gender IN ('Female', 'F')"),
                CategoryBinSpec(" Male", "gender IN ('Male', 'M')"),
            ),
        ),
        "race_ethnicity": Table1RowSpec(
            key="race_ethnicity",
            header_label="Race/Ethnicity, n (%)",
            value_expr="NULL",
            range_expr="NULL",
            null_check_expr="race_ethnicity",
            var_type="categorical",
            bins=(
                CategoryBinSpec(
                    " White non-Hispanic", "race_ethnicity = 'White non-Hispanic'"
                ),
                CategoryBinSpec(
                    " Black non-Hispanic", "race_ethnicity = 'Black non-Hispanic'"
                ),
                CategoryBinSpec(
                    " Hispanic or Latino", "race_ethnicity = 'Hispanic or Latino'"
                ),
                CategoryBinSpec(" Asian", "race_ethnicity = 'Asian'"),
                CategoryBinSpec(
                    " Native Hawaiian/other Pacific Islander",
                    "race_ethnicity = 'Native Hawaiian/other Pacific Islander'",
                ),
                CategoryBinSpec(
                    " More than one race", "race_ethnicity = 'More than one race'"
                ),
            ),
        ),
        "annual_income": Table1RowSpec(
            key="annual_income",
            header_label="Annual household income, n (%)",
            value_expr="NULL",
            range_expr="NULL",
            null_check_expr="annual_income",
            var_type="categorical",
            bins=(
                CategoryBinSpec(" < $75,000", "annual_income = '< $75,000'"),
                CategoryBinSpec(
                    " ≥ $75,000", "annual_income IN ('>= $75,000', '≥ $75,000')"
                ),
                CategoryBinSpec(
                    " Missing", "annual_income = 'Missing' OR annual_income IS NULL"
                ),
            ),
        ),
        "t1d_duration": Table1RowSpec(
            key="t1d_duration",
            header_label="Duration of T1D at enrollment",
            value_expr="dur",
            range_expr="dur",
            null_check_expr="dur",
            var_type="hybrid",
            decimals=1,
            unit_suffix="",
            summary_formats=summary_formats,
            bins=(
                CategoryBinSpec(" <1 years, n (%)", "dur < 1.0"),
                CategoryBinSpec(" 1 to <5 years, n (%)", "dur >= 1.0 AND dur < 5.0"),
                CategoryBinSpec(" 5 to <10 years, n (%)", "dur >= 5.0 AND dur < 10.0"),
                CategoryBinSpec(" ≥10 years, n (%)", "dur >= 10.0"),
            ),
        ),
        "prestudy_insulin_delivery": Table1RowSpec(
            key="prestudy_insulin_delivery",
            header_label="Prestudy insulin delivery, n (%)",
            value_expr="NULL",
            range_expr="NULL",
            null_check_expr="ins_deliv",
            var_type="categorical",
            bins=(
                CategoryBinSpec(" Pump", "ins_deliv = 'Pump'"),
                CategoryBinSpec(" MDI", "ins_deliv = 'MDI'"),
            ),
        ),
        "prestudy_cgm_use": Table1RowSpec(
            key="prestudy_cgm_use",
            header_label="Prestudy CGM use, n (%)",
            value_expr="NULL",
            range_expr="NULL",
            null_check_expr="cgm_use",
            var_type="categorical",
            bins=(
                CategoryBinSpec(" Yes", "cgm_use = 'Yes'"),
                CategoryBinSpec(" No", "cgm_use = 'No'"),
            ),
        ),
        "hba1c": Table1RowSpec(
            key="hba1c",
            header_label="Baseline central lab HbA1c, %",
            value_expr="hba1c",
            range_expr="hba1c",
            null_check_expr="hba1c",
            var_type="hybrid",
            decimals=1,
            unit_suffix="",
            summary_formats=summary_formats,
            bins=(
                CategoryBinSpec(" <7.0%, n (%)", "hba1c < 7.0"),
                CategoryBinSpec(" 7.0%–7.4%, n (%)", "hba1c >= 7.0 AND hba1c < 7.5"),
                CategoryBinSpec(" 7.5%–7.9%, n (%)", "hba1c >= 7.5 AND hba1c < 8.0"),
                CategoryBinSpec(" 8.0%–8.4%, n (%)", "hba1c >= 8.0 AND hba1c < 8.5"),
                CategoryBinSpec(" ≥8.5%, n (%)", "hba1c >= 8.5"),
            ),
            include_missing_row=True,
        ),
        "tir": Table1RowSpec(
            key="tir",
            header_label="Baseline time in range 70–180\u2009mg/dL",
            value_expr="tir",
            range_expr="tir",
            null_check_expr="tir",
            var_type="hybrid",
            decimals=0,
            unit_suffix="%",
            summary_formats=summary_formats,
            bins=(
                CategoryBinSpec(" ≥70%, n (%)", "tir >= 70.0"),
                CategoryBinSpec(" 60% to <70%, n (%)", "tir >= 60.0 AND tir < 70.0"),
                CategoryBinSpec(" 50% to <60%, n (%)", "tir >= 50.0 AND tir < 60.0"),
                CategoryBinSpec(" 40% to <50%, n (%)", "tir >= 40.0 AND tir < 50.0"),
                CategoryBinSpec(" <40%, n (%)", "tir < 40.0"),
            ),
            include_missing_row=True,
        ),
        "tbr": Table1RowSpec(
            key="tbr",
            header_label="Baseline time <70\u2009mg/dL",
            value_expr="tbr",
            range_expr="tbr",
            null_check_expr="tbr",
            var_type="hybrid",
            decimals=1,
            unit_suffix="%",
            summary_formats=summary_formats,
            bins=(
                CategoryBinSpec(" <1%, n (%)", "tbr < 1.0"),
                CategoryBinSpec(" 1% to <4% n (%)", "tbr >= 1.0 AND tbr < 4.0"),
                CategoryBinSpec(" ≥4%, n (%)", "tbr >= 4.0"),
            ),
            include_missing_row=True,
        ),
        "tbr_54": Table1RowSpec(
            key="tbr_54",
            header_label="Baseline time <54\u2009mg/dL",
            value_expr="tbr_54",
            range_expr="tbr_54",
            null_check_expr="tbr_54",
            var_type="hybrid",
            decimals=2,
            unit_suffix="%",
            summary_formats=summary_formats,
            bins=(
                CategoryBinSpec(" <1%, n (%)", "tbr_54 < 1.0"),
                CategoryBinSpec(" ≥1%, n (%)", "tbr_54 >= 1.0"),
            ),
            include_missing_row=True,
        ),
        "tar": Table1RowSpec(
            key="tar",
            header_label="Baseline time >180\u2009mg/dL",
            value_expr="tar",
            range_expr="tar",
            null_check_expr="tar",
            var_type="continuous",
            decimals=0,
            unit_suffix="%",
            summary_formats=summary_formats,
        ),
        "tar_250": Table1RowSpec(
            key="tar_250",
            header_label="Baseline time >250\u2009mg/dL",
            value_expr="tar_250",
            range_expr="tar_250",
            null_check_expr="tar_250",
            var_type="continuous",
            decimals=1,
            unit_suffix="%",
            summary_formats=summary_formats,
        ),
        "mean_glucose": Table1RowSpec(
            key="mean_glucose",
            header_label="Baseline mean glucose, mg/dL",
            value_expr="mean_glucose",
            range_expr="mean_glucose",
            null_check_expr="mean_glucose",
            var_type="continuous",
            decimals=0,
            unit_suffix="",
            summary_formats=summary_formats,
        ),
        "cv": Table1RowSpec(
            key="cv",
            header_label="Baseline coefficient of variation",
            value_expr="cv",
            range_expr="cv",
            null_check_expr="cv",
            var_type="continuous",
            decimals=0,
            unit_suffix="%",
            summary_formats=summary_formats,
        ),
        "dka_events": Table1RowSpec(
            key="dka_events",
            header_label="DKA events, n (%)",
            value_expr="dka_events",
            range_expr="dka_events",
            null_check_expr="dka_events",
            var_type="categorical",
            bins=(
                CategoryBinSpec(" 0 events", "COALESCE(dka_events, 0) = 0"),
                CategoryBinSpec(" ≥1 event", "COALESCE(dka_events, 0) >= 1"),
            ),
        ),
        "severe_hypo_events": Table1RowSpec(
            key="severe_hypo_events",
            header_label="Severe hypoglycemia events, n (%)",
            value_expr="severe_hypo_events",
            range_expr="severe_hypo_events",
            null_check_expr="severe_hypo_events",
            var_type="categorical",
            bins=(
                CategoryBinSpec(" 0 events", "COALESCE(severe_hypo_events, 0) = 0"),
                CategoryBinSpec(" ≥1 event", "COALESCE(severe_hypo_events, 0) >= 1"),
            ),
        ),
        "study": Table1RowSpec(
            key="study",
            header_label="Clinical trial, n (%)",
            value_expr="NULL",
            range_expr="NULL",
            null_check_expr="study",
            var_type="categorical",
            bins=(
                CategoryBinSpec(" DCLP3", "UPPER(study) = 'DCLP3'"),
                CategoryBinSpec(" DCLP5", "UPPER(study) = 'DCLP5'"),
                CategoryBinSpec(" PEDAP", "UPPER(study) = 'PEDAP'"),
            ),
        ),
        "consensus_targets": Table1RowSpec(
            key="consensus_targets",
            header_label="Glycemic consensus targets, n (%)",
            value_expr="NULL",
            range_expr="NULL",
            null_check_expr="tir",
            var_type="categorical",
            bins=(
                CategoryBinSpec(
                    " Time in range 70–180\u2009mg/dL >70%",
                    "tir > 70.0",
                ),
                CategoryBinSpec(
                    " Time <70\u2009mg/dL <4%",
                    "tbr < 4.0",
                ),
                CategoryBinSpec(
                    " Time in range 70–180\u2009mg/dL >70% and time <70\u2009mg/dL <4%",
                    "tir > 70.0 AND tbr < 4.0",
                ),
            ),
        ),
    }


BECK_2023_DEFAULT_ROW_KEYS: tuple[str, ...] = (
    "age",
    "gender",
    "race_ethnicity",
    "annual_income",
    "t1d_duration",
    "prestudy_insulin_delivery",
    "prestudy_cgm_use",
    "hba1c",
    "tir",
    "tbr",
)


def is_categorical_column_in_db(
    db_path: str | None,
    target_table: str,
    col_name: str,
) -> bool:
    """
    Check whether `col_name` on `target_table` in `db_path` contains non-numeric categorical string values.

    Args:
        db_path: Optional path to the active DuckDB database file.
        target_table: Target table name.
        col_name: Column name on `target_table`.

    Returns:
        bool: True when `col_name` has non-null, non-numeric string values, else False.

    """
    if not db_path or not Path(db_path).exists():
        return False
    safe_tbl = escape_sql_identifier(target_table)
    safe_col = escape_sql_identifier(col_name)
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            num_chk = conn.execute(
                f'SELECT COUNT(TRY_CAST("{safe_col}" AS DOUBLE)), COUNT("{safe_col}") FROM "{safe_tbl}"'
            ).fetchone()
        finally:
            conn.close()
        return bool(num_chk and int(num_chk[1] or 0) > 0 and int(num_chk[0] or 0) == 0)
    except (duckdb.Error, OSError, ValueError, TypeError):
        return False


def db_has_raw_jaeb_tables(
    db_path: str | None,
    schema_map: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """
    Check whether the active database or `schema_map` contains all raw JAEB study tables (`dclp3_*`, `dclp5_*`, `pedap_*`).

    Args:
        db_path: Optional path to the active DuckDB database file.
        schema_map: Optional mapping from table names to column names.

    Returns:
        bool: True if all raw JAEB study tables required for `build_raw_jaeb_cohort_cte_sql` are present.

    """
    if schema_map:
        lower_tables = {t.lower() for t in schema_map}
        if BECK2023_REQUIRED_TRIAL_TABLES.issubset(lower_tables):
            return True
    if not db_path or not Path(db_path).exists():
        return False
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            tables = {str(r[0]).lower() for r in conn.execute("SHOW TABLES").fetchall()}
        finally:
            conn.close()
        return BECK2023_REQUIRED_TRIAL_TABLES.issubset(tables)
    except (duckdb.Error, OSError):
        return False


def db_has_extended_jaeb_tables(
    db_path: str | None,
    schema_map: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """
    Check whether the active database or `schema_map` contains the extended JAEB study tables (`tblaptsummary`, `hptroster`).

    Args:
        db_path: Optional path to the active DuckDB database file.
        schema_map: Optional mapping from table names to column names.

    Returns:
        bool: True if extended JAEB study tables are present alongside raw JAEB tables.

    """
    if schema_map:
        lower_tables = {t.lower() for t in schema_map}
        if EXTENDED_JAEB_REQUIRED_TABLES.issubset(lower_tables):
            return True
    if not db_path or not Path(db_path).exists():
        return False
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            tables = {str(r[0]).lower() for r in conn.execute("SHOW TABLES").fetchall()}
        finally:
            conn.close()
        return EXTENDED_JAEB_REQUIRED_TABLES.issubset(tables)
    except (duckdb.Error, OSError):
        return False


__all__ = [
    "BECK_2023_DEFAULT_ROW_KEYS",
    "EXTENDED_JAEB_REQUIRED_TABLES",
    "CategoryBinSpec",
    "ContinuousSummaryFormat",
    "RowVariableKind",
    "Table1QuerySpec",
    "Table1RowSpec",
    "Table1StratumSpec",
    "build_beck_2023_row_specs",
    "db_has_extended_jaeb_tables",
    "db_has_raw_jaeb_tables",
    "is_categorical_column_in_db",
]
