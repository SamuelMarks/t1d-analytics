"""Statistical SQL cell expression formatters and Markdown table formatters for Table 1 results."""

from collections.abc import Mapping, Sequence

from t1d_analytics.sql_utils import SqlValue, escape_sql_identifier
from t1d_analytics.table1_specs import (
    ContinuousSummaryFormat,
    Table1RowSpec,
    Table1StratumSpec,
)


def format_mean_sd_sql(
    val_expr: str,
    cond_expr: str | None,
    decimals: int,
    unit_suffix: str,
) -> str:
    """
    Build a DuckDB SQL expression formatting `Mean ± SD` for `val_expr`.

    Args:
        val_expr: Column or expression in `base` CTE.
        cond_expr: Optional stratum filter condition (e.g. `'is_ciq = 1'`).
        decimals: Number of decimal places (`0` uses `BIGINT` cast, `>0` uses `printf`).
        unit_suffix: Unit string appended after mean and SD (e.g. `'%'`).

    Returns:
        str: SQL expression producing `'Mean ± SD'`.

    """
    scoped = f"CASE WHEN {cond_expr} THEN {val_expr} END" if cond_expr else val_expr
    sep = f"{unit_suffix}\u2009±\u2009"
    tail = f" || '{unit_suffix}'" if unit_suffix else ""
    if decimals == 0:
        return (
            f"COALESCE(CAST(ROUND(AVG({scoped}), 0) AS BIGINT) || '{sep}' || "
            f"CAST(ROUND(STDDEV_SAMP({scoped}), 0) AS BIGINT){tail}, '—')"
        )
    fmt = f"%.{decimals}f"
    return (
        f"COALESCE(printf('{fmt}', ROUND(AVG({scoped}), {decimals})) || '{sep}' || "
        f"printf('{fmt}', ROUND(STDDEV_SAMP({scoped}), {decimals})){tail}, '—')"
    )


def format_median_iqr_sql(
    val_expr: str,
    cond_expr: str | None,
    decimals: int,
    unit_suffix: str,
) -> str:
    """
    Build a DuckDB SQL expression formatting `Median (IQR)` for `val_expr`.

    Args:
        val_expr: Column or expression in `base` CTE.
        cond_expr: Optional stratum filter condition.
        decimals: Number of decimal places.
        unit_suffix: Unit string appended to values.

    Returns:
        str: SQL expression producing `'Median (Q25–Q75)'`.

    """
    scoped = f"CASE WHEN {cond_expr} THEN {val_expr} END" if cond_expr else val_expr
    fmt = f"%.{decimals}f" if decimals > 0 else "%.0f"
    return (
        f"COALESCE(printf('{fmt}', ROUND(QUANTILE_CONT({scoped}, 0.5), {decimals})) || '{unit_suffix} (' || "
        f"printf('{fmt}', ROUND(QUANTILE_CONT({scoped}, 0.25), {decimals})) || '{unit_suffix}–' || "
        f"printf('{fmt}', ROUND(QUANTILE_CONT({scoped}, 0.75), {decimals})) || '{unit_suffix})', '—')"
    )


def format_range_sql(
    range_expr: str,
    cond_expr: str | None,
    decimals: int,
    unit_suffix: str,
) -> str:
    """
    Build a DuckDB SQL expression formatting `Min–Max` range for `range_expr`.

    Args:
        range_expr: Column or expression in `base` CTE.
        cond_expr: Optional stratum filter condition.
        decimals: Number of decimal places.
        unit_suffix: Unit string appended to min and max.

    Returns:
        str: SQL expression producing `'Min–Max'`.

    """
    scoped = f"CASE WHEN {cond_expr} THEN {range_expr} END" if cond_expr else range_expr
    sep = f"{unit_suffix}–"
    tail = f" || '{unit_suffix}'" if unit_suffix else ""
    if decimals == 0:
        if unit_suffix:
            return (
                f"COALESCE(CAST(ROUND(MIN({scoped}), 0) AS BIGINT) || '{sep}' || "
                f"CAST(ROUND(MAX({scoped}), 0) AS BIGINT){tail}, '—')"
            )
        return (
            f"COALESCE(CAST(MIN({scoped}) AS VARCHAR) || '–' || "
            f"CAST(MAX({scoped}) AS VARCHAR), '—')"
        )
    fmt = f"%.{decimals}f"
    return (
        f"COALESCE(printf('{fmt}', ROUND(MIN({scoped}), {decimals})) || '{sep}' || "
        f"printf('{fmt}', ROUND(MAX({scoped}), {decimals})){tail}, '—')"
    )


def format_continuous_summary_for_stratum(
    row_spec: Table1RowSpec,
    fmt: ContinuousSummaryFormat,
    cond: str | None,
) -> str:
    """
    Format a continuous summary SQL expression (`mean_sd`, `median_iqr`, or `range`) for a single stratum.

    Args:
        row_spec: Characteristic row section specification.
        fmt: Summary format identifier.
        cond: Optional SQL stratum filter condition.

    Returns:
        str: Formatted SQL summary expression.

    """
    if fmt == "mean_sd":
        return format_mean_sd_sql(
            row_spec.value_expr, cond, row_spec.decimals, row_spec.unit_suffix
        )
    if fmt == "median_iqr":
        return format_median_iqr_sql(
            row_spec.value_expr, cond, row_spec.decimals, row_spec.unit_suffix
        )
    return format_range_sql(
        row_spec.range_expr, cond, row_spec.decimals, row_spec.unit_suffix
    )


def format_categorical_pct_case_sql(stratum: Table1StratumSpec) -> str:
    """
    Build the outer `CASE` SQL projection formatting `c_<k> (<pct>)` or `s_<k>` for `stratum`.

    Args:
        stratum: Column stratification specification.

    Returns:
        str: Outer `SELECT` column expression aliased to `stratum.column_alias`.

    """
    k = stratum.key_id
    safe_alias = escape_sql_identifier(stratum.column_alias)
    return (
        f"    CASE\n"
        f"        WHEN r.is_cat = 1 THEN CAST(r.c_{k} AS VARCHAR) || ' (' || "
        f"CASE WHEN r.c_{k} > 0 AND 100.0 * r.c_{k} / NULLIF(t.n_{k}, 0) < 1.0 THEN '<1' "
        f"ELSE COALESCE(printf('%.0f', 100.0 * r.c_{k} / NULLIF(t.n_{k}, 0)), '0') END || ')'\n"
        f"        ELSE r.s_{k}\n"
        f'    END AS "{safe_alias}"'
    )


def format_table1_rows_markdown(
    rows: Sequence[Mapping[str, SqlValue]],
) -> str:
    """
    Render executed Table 1 result rows into a GitHub-flavored Markdown table.

    Args:
        rows: Ordered sequence of row dictionaries returned by `compile_table1_sql`.

    Returns:
        str: Markdown table string (or empty string if `rows` is empty).

    """
    if not rows:
        return ""
    headers = list(rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| "
        + " | ".join(
            str(row.get(col, "") if row.get(col) is not None else "").replace(
                "|", "\\|"
            )
            for col in headers
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *body_lines])


__all__ = [
    "format_categorical_pct_case_sql",
    "format_continuous_summary_for_stratum",
    "format_mean_sd_sql",
    "format_median_iqr_sql",
    "format_range_sql",
    "format_table1_rows_markdown",
]
