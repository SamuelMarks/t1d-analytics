"""DuckDB statistical SQL CTE and UNION ALL query compiler for compositional Table 1 specifications."""

from collections.abc import Sequence

from t1d_analytics.sql_utils import escape_sql_identifier, escape_sql_literal
from t1d_analytics.table1_formatter import (
    format_categorical_pct_case_sql,
    format_continuous_summary_for_stratum,
)
from t1d_analytics.table1_specs import (
    Table1QuerySpec,
    Table1RowSpec,
    Table1StratumSpec,
)


def build_base_cte_sql(
    spec: Table1QuerySpec,
    available_cols: set[str] | None = None,
) -> str:
    """
    Build the `base` and `totals` CTE blocks for a `Table1QuerySpec`.

    Args:
        spec: Table 1 query specification.
        available_cols: Optional set of lowercase column names present in `spec.target_table`.

    Returns:
        str: SQL fragment defining `WITH base AS (...), totals AS (...)`.

    """
    safe_tbl = escape_sql_identifier(spec.target_table)
    table_ref = (
        safe_tbl if safe_tbl in ("patients", "raw_jaeb_cohort") else f'"{safe_tbl}"'
    )
    has_all = available_cols is None or spec.raw_source_cte_sql is not None
    cols_lower = available_cols or set()

    def _col_or_default(col_name: str, fallback_sql: str) -> str:
        """
        Return `col_name` if present in `available_cols` (or unrestricted), else `fallback_sql`.

        Args:
            col_name: Candidate column name.
            fallback_sql: SQL literal fallback expression.

        Returns:
            str: Resolved SQL column or literal fallback.

        """
        if has_all or col_name.lower() in cols_lower:
            return col_name
        return fallback_sql

    age_exact_src = _col_or_default("age_exact", "NULL")
    dur_src = _col_or_default("t1d_duration_years", "NULL")
    ins_src = _col_or_default("prestudy_insulin_delivery", "NULL")
    cgm_src = _col_or_default("prestudy_cgm_use", "NULL")

    stratum_flags: list[str] = []
    totals_exprs: list[str] = []
    for st in spec.strata:
        if st.key_id == "all":
            totals_exprs.append("COUNT(*) AS n_all")
        else:
            stratum_flags.append(
                f"        CASE WHEN {st.predicate_sql} THEN 1 ELSE 0 END AS is_{st.key_id}"
            )
            totals_exprs.append(f"COALESCE(SUM(is_{st.key_id}), 0) AS n_{st.key_id}")

    base_projections = [
        "        *",
        f"        COALESCE(TRY_CAST({age_exact_src} AS DOUBLE), TRY_CAST(age AS DOUBLE) + 0.5) AS age_ex",
        f"        {dur_src} AS dur",
        f"        {ins_src} AS ins_deliv",
        f"        {cgm_src} AS cgm_use",
        *stratum_flags,
    ]

    where_sql = (
        f"\n    WHERE {' AND '.join(spec.where_clauses)}" if spec.where_clauses else ""
    )
    limit_sql = (
        f"\n    LIMIT {int(spec.limit_rows)}"
        if spec.limit_rows is not None and spec.limit_rows > 0
        else ""
    )
    proj_joined = ",\n".join(base_projections)
    totals_joined = ",\n        ".join(totals_exprs)
    lead_prefix = (
        f"WITH {spec.raw_source_cte_sql},\nbase AS ("
        if spec.raw_source_cte_sql
        else "WITH base AS ("
    )
    return (
        f"{lead_prefix}\n"
        f"    SELECT\n"
        f"{proj_joined}\n"
        f"    FROM {table_ref}{where_sql}{limit_sql}\n"
        f"),\n"
        f"totals AS (\n"
        f"    SELECT\n"
        f"        {totals_joined}\n"
        f"    FROM base\n"
        f")"
    )


def build_single_row_union_branch(
    ord_idx: int,
    label: str,
    is_cat: int,
    c_exprs: Sequence[str],
    s_exprs: Sequence[str],
    from_clause: str,
) -> str:
    """
    Build a single `SELECT ...` branch inside `raw_rows` CTE.

    Args:
        ord_idx: 1-based row ordering integer.
        label: Characteristic display string.
        is_cat: `1` for categorical `n (%)` rows, `0` for header or string-formatted rows.
        c_exprs: Count SQL expressions for each stratum.
        s_exprs: Formatted string SQL expressions for each stratum.
        from_clause: Optional `' FROM base'` or `' FROM totals'` suffix.

    Returns:
        str: SQL `SELECT` branch string.

    """
    esc_label = escape_sql_literal(label)
    c_part = ", ".join(c_exprs)
    s_part = ", ".join(s_exprs)
    if ord_idx == 1:
        return (
            f"    SELECT {ord_idx} AS ord, '{esc_label}' AS Characteristic, "
            f"{is_cat} AS is_cat, {c_part}, {s_part}{from_clause}"
        )
    return (
        f"    SELECT {ord_idx}, '{esc_label}', {is_cat}, "
        f"{c_part}, {s_part}{from_clause}"
    )


def build_dynamic_category_group_by_branch(
    cat_expr: str,
    strata: Sequence[Table1StratumSpec],
    ord_idx: int,
) -> str:
    """
    Build a `GROUP BY` union branch that discovers distinct categorical values dynamically from `base`.

    Args:
        cat_expr: SQL column expression for the categorical variable.
        strata: Active stratification columns.
        ord_idx: Base row ordering integer.

    Returns:
        str: SQL `SELECT ... GROUP BY` branch string.

    """
    c_dyn: list[str] = [
        "COUNT(*)" if st.key_id == "all" else f"COALESCE(SUM(is_{st.key_id}), 0)"
        for st in strata
    ]
    c_part = ", ".join(c_dyn)
    s_part = ", ".join(["''"] * len(strata))
    return (
        f"    SELECT {ord_idx} + (DENSE_RANK() OVER (ORDER BY CAST({cat_expr} AS VARCHAR))) * 0.001, "
        f"' ' || CAST({cat_expr} AS VARCHAR), 1, {c_part}, {s_part} "
        f"FROM base WHERE {cat_expr} IS NOT NULL GROUP BY CAST({cat_expr} AS VARCHAR)"
    )


def append_row_spec_branches(
    row_spec: Table1RowSpec,
    strata: Sequence[Table1StratumSpec],
    start_ord: int,
) -> tuple[list[str], int]:
    """
    Compile all `UNION ALL` branches for a single `Table1RowSpec`.

    Args:
        row_spec: Characteristic row section specification.
        strata: Active stratification columns.
        start_ord: Starting `ord` index.

    Returns:
        tuple[list[str], int]: `(branches, next_ord_index)`.

    """
    branches: list[str] = []
    ord_idx = start_ord
    n_strata = len(strata)

    c_zeros = [f"0 AS c_{st.key_id}" if ord_idx == 1 else "0" for st in strata]
    s_empties = [f"'' AS s_{st.key_id}" if ord_idx == 1 else "''" for st in strata]
    branches.append(
        build_single_row_union_branch(
            ord_idx, row_spec.header_label, 0, c_zeros, s_empties, ""
        )
    )
    ord_idx += 1

    if row_spec.var_type in ("continuous", "hybrid"):
        for fmt in row_spec.summary_formats:
            c_z = ["0"] * n_strata
            s_vals = [
                format_continuous_summary_for_stratum(
                    row_spec,
                    fmt,
                    None if st.key_id == "all" else f"is_{st.key_id} = 1",
                )
                for st in strata
            ]
            sub_label = (
                " Mean\u2009±\u2009SD"
                if fmt == "mean_sd"
                else (" Median (IQR)" if fmt == "median_iqr" else " Range")
            )
            branches.append(
                build_single_row_union_branch(
                    ord_idx, sub_label, 0, c_z, s_vals, " FROM base"
                )
            )
            ord_idx += 1

    for bin_spec in row_spec.bins:
        c_counts: list[str] = []
        for st in strata:
            if st.key_id == "all":
                c_counts.append(
                    f"COALESCE(SUM(CASE WHEN {bin_spec.sql_condition} THEN 1 ELSE 0 END), 0)"
                )
            else:
                c_counts.append(
                    f"COALESCE(SUM(CASE WHEN is_{st.key_id} = 1 AND ({bin_spec.sql_condition}) THEN 1 ELSE 0 END), 0)"
                )
        branches.append(
            build_single_row_union_branch(
                ord_idx,
                bin_spec.label,
                1,
                c_counts,
                ["''"] * n_strata,
                " FROM base",
            )
        )
        ord_idx += 1

    if not row_spec.bins and row_spec.dynamic_category_expr:
        branches.append(
            build_dynamic_category_group_by_branch(
                row_spec.dynamic_category_expr, strata, ord_idx
            )
        )
        ord_idx += 1

    if row_spec.include_missing_row:
        c_miss: list[str] = []
        for st in strata:
            if st.key_id == "all":
                c_miss.append(
                    f"COALESCE(SUM(CASE WHEN {row_spec.null_check_expr} IS NULL THEN 1 ELSE 0 END), 0)"
                )
            else:
                c_miss.append(
                    f"COALESCE(SUM(CASE WHEN is_{st.key_id} = 1 AND {row_spec.null_check_expr} IS NULL THEN 1 ELSE 0 END), 0)"
                )
        branches.append(
            build_single_row_union_branch(
                ord_idx,
                " Missing, n (%)",
                1,
                c_miss,
                ["''"] * n_strata,
                " FROM base",
            )
        )
        ord_idx += 1

    return branches, ord_idx


def compile_table1_sql(
    spec: Table1QuerySpec,
    available_cols: set[str] | None = None,
) -> str:
    """
    Compile a `Table1QuerySpec` into an executable, self-contained DuckDB SQL statement.

    Computes all summary statistics (`Mean ± SD`, `Median (IQR)`, `Range`, and
    categorical `n (%)` counts/percentages) dynamically from the underlying database
    without hardcoded cell values.

    Args:
        spec: Complete Table 1 query specification.
        available_cols: Optional set of lowercase column names on `spec.target_table`.

    Returns:
        str: Executable DuckDB SQL query producing the requested Table 1.

    """
    cte_header = build_base_cte_sql(spec, available_cols=available_cols)
    all_branches: list[str] = []
    ord_idx = 1

    if spec.include_n_row:
        c_zeros = [f"0 AS c_{st.key_id}" for st in spec.strata]
        s_totals = [
            f"CAST(n_{st.key_id} AS VARCHAR) AS s_{st.key_id}" for st in spec.strata
        ]
        all_branches.append(
            build_single_row_union_branch(
                ord_idx,
                "Participants, N",
                0,
                c_zeros,
                s_totals,
                " FROM totals",
            )
        )
        ord_idx += 1

    for row_spec in spec.rows:
        row_branches, ord_idx = append_row_spec_branches(row_spec, spec.strata, ord_idx)
        all_branches.extend(row_branches)

    select_cols: list[str] = [
        '    r.Characteristic AS "Characteristic"',
        *(format_categorical_pct_case_sql(st) for st in spec.strata),
    ]

    union_body = "\n    UNION ALL\n".join(all_branches)
    outer_select = ",\n".join(select_cols)
    return (
        f"{cte_header},\n"
        f"raw_rows AS (\n"
        f"{union_body}\n"
        f")\n"
        f"SELECT\n"
        f"{outer_select}\n"
        f"FROM raw_rows r\n"
        f"CROSS JOIN totals t\n"
        f"ORDER BY r.ord;"
    )


__all__ = [
    "append_row_spec_branches",
    "build_base_cte_sql",
    "build_dynamic_category_group_by_branch",
    "build_single_row_union_branch",
    "compile_table1_sql",
]
