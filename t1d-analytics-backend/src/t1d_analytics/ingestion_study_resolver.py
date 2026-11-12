"""Clinical study prefix resolution, ontology mapping, and column projection builders for ETL ingestion."""

import json
import re
from pathlib import Path

from t1d_analytics.sql_utils import escape_sql_identifier

CLINICAL_NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "age",
        "ageasofranddt",
        "agebaseline",
        "hba1c",
        "a1c",
        "a1cval",
        "c_peptide",
        "glucose",
        "glucose_value",
        "sensorglucose",
        "historicglucose",
        "meancgm",
        "meanglucose",
        "mean_glucose",
        "tir",
        "tbr",
        "tar",
        "bmi",
        "bmibaseline",
        "bodymassindex",
        "bmi_kg_m2",
        "t1d_duration_years",
        "duration",
        "diagduration",
        "yrsdiag",
        "t1dduration",
        "pt_duration",
        "total_daily_insulin",
        "tdd",
        "totaldailydose",
        "unitsperkg",
        "basalbolustotal",
        "cgm_active_wear_pct",
        "pctwear",
        "activewear",
        "sensorwearpct",
        "validcgmdays",
        "weight_kg",
        "height_cm",
        "insulin_total_daily",
        "insulin_total_daily_units",
        "insulin_basal",
        "insulin_bolus",
        "numsevhypo",
        "pt_numsevhypo",
        "numhospdka",
        "pt_numhospdka",
    }
)

JAEB_STUDY_ACRONYMS: tuple[str, ...] = (
    "dclp3",
    "dclp5",
    "pedap",
    "flair",
    "loop",
    "aide",
    "city",
    "clver",
    "wisdm",
    "tight",
    "hyclo",
    "pso4",
    "pso3",
    "ctr3",
    "dan05",
    "t1dexip",
    "t1dexi",
)


def get_supported_tabular_globs(
    include_sas: bool = False,
    include_spss: bool = False,
) -> list[str]:
    """
    Return the ordered list of supported tabular file glob patterns for dataset discovery.

    Args:
        include_sas: Whether to include SAS dataset patterns (`*.sas7bdat`, `*.xpt`).
        include_spss: Whether to include SPSS dataset patterns (`*.sav`).

    Returns:
        list[str]: Glob pattern strings suitable for `Path.rglob`.

    """
    base_exts = ["csv", "txt", "parquet", "csv.gz"]
    if include_sas:
        base_exts.extend(["sas7bdat", "xpt"])
    if include_spss:
        base_exts.append("sav")

    patterns: list[str] = []
    for ext in base_exts:
        patterns.append(f"*.{ext}")
        patterns.append(f"*.{ext.upper()}")
    return patterns


def get_clinical_variable_mappings() -> dict[str, list[str]]:
    """
    Load the clinical variable ontology mapping standard concepts to trial-specific column names.

    Returns:
        dict[str, list[str]]: Mapping from standard concept names to list of original column names.

    """
    matches_file = Path(__file__).parent / "likely_matches.json"
    if matches_file.exists():
        try:
            with open(matches_file, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {
                        str(k): [str(x) for x in v]
                        for k, v in data.items()
                        if isinstance(v, list)
                    }
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
    return {}


def resolve_subdir_prefix(rel_parent: Path) -> str:
    """
    Resolve a sanitized table-name prefix from a relative parent directory path.

    Compacts verbose multi-trial JAEB study directory paths containing known study
    codes (`dclp3`, `dclp5`, `pedap`, etc.) into their canonical study prefix.

    Args:
        rel_parent: Relative parent directory path under the root ingestion directory.

    Returns:
        str: Sanitized lowercase prefix string.

    """
    raw_prefix = re.sub(r"[^a-zA-Z0-9_]", "_", str(rel_parent)).strip("_").lower()
    if len(raw_prefix) > 24:
        for study_code in JAEB_STUDY_ACRONYMS:
            if study_code in raw_prefix:
                return study_code
    return raw_prefix


def resolve_target_table_stem(
    data_file: Path,
    data_dir: Path,
    prefix_subdirs: bool,
    stem_counts: dict[str, int],
) -> str:
    """
    Derive a sanitized SQL table name stem for `data_file`.

    Args:
        data_file: Source tabular file path.
        data_dir: Root ingestion directory path.
        prefix_subdirs: Whether to prefix table names with relative subdirectory names.
        stem_counts: Frequency map of normalized file stems.

    Returns:
        str: Valid SQL table identifier starting with an alphabetic character.

    """
    s_name = (
        Path(data_file.stem).stem
        if data_file.name.endswith(".csv.gz")
        else data_file.stem
    )
    raw_stem = re.sub(r"[^a-zA-Z0-9_]", "_", s_name).lower().strip("_") or "dataset"
    safe_stem = raw_stem
    if prefix_subdirs or stem_counts.get(raw_stem, 0) > 1:
        rel_parent = data_file.parent.relative_to(data_dir)
        if rel_parent != Path():
            prefix = resolve_subdir_prefix(rel_parent)
            if prefix and not raw_stem.startswith(f"{prefix}_"):
                safe_stem = f"{prefix}_{raw_stem}"

    table_name = safe_stem.strip("_") or "dataset"
    return table_name if table_name[0].isalpha() else f"t_{table_name}"


def discover_tabular_files(
    data_dir: Path,
    include_sas: bool,
    include_spss: bool,
) -> tuple[list[Path], dict[str, int]]:
    """
    Scan `data_dir` for supported tabular files and compute stem collision counts.

    Args:
        data_dir: Root directory containing tabular datasets.
        include_sas: Whether to include SAS files.
        include_spss: Whether to include SPSS files.

    Returns:
        tuple[list[Path], dict[str, int]]: `(discovered_files, stem_frequency_map)`.

    """
    patterns = get_supported_tabular_globs(
        include_sas=include_sas,
        include_spss=include_spss,
    )
    raw_files: list[Path] = []
    for pat in patterns:
        raw_files.extend(data_dir.rglob(pat))
    data_files = [
        f
        for f in dict.fromkeys(raw_files)
        if not f.name.startswith("._")
        and "Glossary" not in f.name
        and "Protocol" not in f.name
    ]
    stem_counts: dict[str, int] = {}
    for df in data_files:
        s_name = Path(df.stem).stem if df.name.endswith(".csv.gz") else df.stem
        raw_stem = re.sub(r"[^a-zA-Z0-9_]", "_", s_name).lower().strip("_") or "dataset"
        stem_counts[raw_stem] = stem_counts.get(raw_stem, 0) + 1
    return data_files, stem_counts


def build_column_select_expressions(
    cols: list[tuple[str, ...]],
    reverse_map: dict[str, str],
) -> list[str]:
    """
    Build standardized DuckDB `SELECT` column expressions for `temp_view`.

    Args:
        cols: Output rows from `DESCRIBE temp_view`.
        reverse_map: Lowercase original column name to standardized clinical column name mapping.

    Returns:
        list[str]: SQL column projection expressions.

    """
    select_exprs: list[str] = []
    seen_std_names: set[str] = set()
    for c in cols:
        orig_col = str(c[0])
        col_type = str(c[1]).upper() if len(c) > 1 else ""
        std_name = reverse_map.get(orig_col.lower(), orig_col)

        if std_name.lower() in seen_std_names:
            std_name = orig_col

        counter = 1
        base_std_name = std_name
        while std_name.lower() in seen_std_names:
            counter += 1
            std_name = f"{base_std_name}_{counter}"

        seen_std_names.add(std_name.lower())

        escaped_orig = escape_sql_identifier(orig_col)
        escaped_std = escape_sql_identifier(std_name)
        quoted_orig = f'"{escaped_orig}"'
        col_expr = quoted_orig
        if std_name.lower() in CLINICAL_NUMERIC_COLUMNS and "CHAR" in col_type:
            col_expr = f"TRY_CAST(NULLIF(TRIM({quoted_orig}), '') AS DOUBLE)"
        if std_name != orig_col or col_expr != quoted_orig:
            select_exprs.append(f'{col_expr} AS "{escaped_std}"')
        else:
            select_exprs.append(quoted_orig)
    return select_exprs


__all__ = [
    "CLINICAL_NUMERIC_COLUMNS",
    "JAEB_STUDY_ACRONYMS",
    "build_column_select_expressions",
    "discover_tabular_files",
    "get_clinical_variable_mappings",
    "get_supported_tabular_globs",
    "resolve_subdir_prefix",
    "resolve_target_table_stem",
]
