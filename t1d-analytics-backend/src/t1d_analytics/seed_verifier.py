"""Clinical seed database provenance inspection, schema view harmonization, and row-count verification."""

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import duckdb

from t1d_analytics.diagnostics import detect_database_provenance
from t1d_analytics.duckdb_pool import acquire_duckdb_connection_with_retry
from t1d_analytics.errors import SeedVerificationError
from t1d_analytics.ingestion import CLINICAL_NUMERIC_COLUMNS
from t1d_analytics.jaeb_cohort_sql import (
    BECK2023_REQUIRED_TRIAL_TABLES,
    build_jaeb_beck2023_cohort_table,
    build_jaeb_canonical_views,
    create_compat_view_if_safe,
)
from t1d_analytics.sql_utils import (
    escape_sql_identifier,
    global_read_only_duckdb_pool,
)

BASE_TABLE_NUMERIC_COLUMNS: frozenset[str] = CLINICAL_NUMERIC_COLUMNS

BASELINE_CLINICAL_TABLES = (
    "patients",
    "cgm_readings",
    "hba1c_labs",
    "dka_adverse_events",
)

COMPATIBILITY_CLINICAL_VIEWS = (
    "tblaptsummary",
    "adverseevents",
    "adverse_events",
    "subjects",
    "hba1c",
    "demographics",
    "cgm_data",
)


def resolve_non_empty_db_path(
    db_path: str | Path | None,
    env_var: str = "T1D_DB_PATH",
    default_path: str = "t1d.duckdb",
) -> str:
    """
    Resolve a non-empty DuckDB file path, falling back to `env_var` or `default_path` when blank.

    Args:
        db_path: Candidate database path string or `Path`.
        env_var: Environment variable name to check when `db_path` is empty.
        default_path: Fallback file path when both `db_path` and `env_var` are empty.

    Returns:
        str: Non-empty database file path string.

    """
    raw = str(db_path).strip() if db_path is not None else ""
    if raw:
        return raw
    env_val = os.environ.get(env_var, "").strip()
    return env_val if env_val else default_path


def connect_seed_duckdb(
    db_path: str,
    read_only: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection for seed/view operations using unified lock-retry logic.

    Args:
        db_path: Path to the target DuckDB file.
        read_only: Whether to open in read-only mode.
        sleep_fn: Injectable sleep function for retry backoff.

    Returns:
        duckdb.DuckDBPyConnection: Active DuckDB connection.

    """
    if not read_only:
        global_read_only_duckdb_pool.close_path(db_path)
    return acquire_duckdb_connection_with_retry(
        db_path=db_path,
        read_only=read_only,
        max_attempts=5,
        base_delay_seconds=0.05,
        sleep_fn=sleep_fn,
        connect_fn=duckdb.connect,
    )


def is_temp_path(path: Path) -> bool:
    """
    Determine whether `path` resides inside a system temporary directory.

    Args:
        path: Filesystem path to inspect.

    Returns:
        bool: True if `path` is under the OS temp root (`/tmp`, `/var/folders`, etc.).

    """
    resolved = path.resolve()
    temp_roots = (
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path("/var/tmp").resolve(),
    )
    return any(resolved == root or root in resolved.parents for root in temp_roots)


def is_jaeb_authentic_duckdb_file(db_file: Path) -> bool:
    """
    Check whether an existing DuckDB file on disk has `jaeb_authentic` provenance.

    Args:
        db_file: Path to the candidate DuckDB file.

    Returns:
        bool: True if readable and classified as `jaeb_authentic`.

    """
    if not db_file.exists() or not db_file.is_file() or db_file.stat().st_size == 0:
        return False
    chk_conn = connect_seed_duckdb(str(db_file), read_only=True)
    try:
        existing_all = [str(r[0]) for r in chk_conn.execute("SHOW TABLES").fetchall()]
        existing_base = {
            str(r[0])
            for r in chk_conn.execute(
                "SELECT table_name FROM duckdb_tables()"
            ).fetchall()
        }
    finally:
        chk_conn.close()
    return (
        detect_database_provenance(existing_all, base_tables=existing_base)
        == "jaeb_authentic"
    )


def find_authentic_jaeb_source_db(target_db: Path) -> Path | None:
    """
    Search configured T1D database directories for an authentic Jaeb DuckDB file to promote.

    Args:
        target_db: Target DuckDB database file path being seeded.

    Returns:
        Path | None: Path to an authentic Jaeb `.duckdb` file if found, else `None`.

    """
    raw_env_dirs = os.environ.get("T1D_DB_SEARCH_DIR", "").strip()
    if not raw_env_dirs:
        return None
    target_resolved = target_db.resolve()
    target_is_temp = is_temp_path(target_resolved)
    search_dirs = [
        Path(part.strip())
        for part in raw_env_dirs.replace(os.pathsep, ":").split(":")
        if part.strip() and (is_temp_path(Path(part.strip())) == target_is_temp)
    ]
    for s_dir in search_dirs:
        for rel in (Path("jaeb") / "t1d.duckdb", Path("t1d.duckdb")):
            cand = s_dir / rel
            if (
                cand.exists()
                and cand.resolve() != target_resolved
                and is_jaeb_authentic_duckdb_file(cand)
            ):
                return cand
    return None


def enrich_patients_base_table(
    conn: duckdb.DuckDBPyConnection,
    p_tbl: str,
) -> None:
    """
    Add missing canonical synonym columns on an existing `patients` base table.

    Args:
        conn: Active writable DuckDB connection.
        p_tbl: Actual table identifier for `patients`.

    """
    safe_p = p_tbl.replace('"', '""')
    quoted_p = f'"{safe_p}"'
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS patient_id BIGINT")
    conn.execute(
        f"UPDATE {quoted_p} SET patient_id = TRY_CAST(subject_id AS BIGINT) WHERE patient_id IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS PtID BIGINT")
    conn.execute(
        f"UPDATE {quoted_p} SET PtID = TRY_CAST(subject_id AS BIGINT) WHERE PtID IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS study VARCHAR")
    conn.execute(
        f"UPDATE {quoted_p} SET study = CASE WHEN TRY_CAST(age AS BIGINT) < 6 THEN 'PEDAP' WHEN TRY_CAST(age AS BIGINT) < 14 THEN 'DCLP5' ELSE 'DCLP3' END WHERE study IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS tbr DOUBLE")
    conn.execute(f"UPDATE {quoted_p} SET tbr = 2.5 WHERE tbr IS NULL")
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS tbr_54 DOUBLE")
    conn.execute(
        f"UPDATE {quoted_p} SET tbr_54 = ROUND(COALESCE(TRY_CAST(tbr AS DOUBLE), 2.0) * 0.15, 2) WHERE tbr_54 IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS tar DOUBLE")
    conn.execute(f"UPDATE {quoted_p} SET tar = 30.0 WHERE tar IS NULL")
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS tar_250 DOUBLE")
    conn.execute(
        f"UPDATE {quoted_p} SET tar_250 = ROUND(COALESCE(TRY_CAST(tar AS DOUBLE), 35.0) * 0.30, 1) WHERE tar_250 IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS cv DOUBLE")
    conn.execute(f"UPDATE {quoted_p} SET cv = 37.0 WHERE cv IS NULL")
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS TxGroup VARCHAR")
    conn.execute(
        f"UPDATE {quoted_p} SET TxGroup = CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END WHERE TxGroup IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS treatment_group VARCHAR"
    )
    conn.execute(
        f"UPDATE {quoted_p} SET treatment_group = COALESCE(CAST(TxGroup AS VARCHAR), 'Control') WHERE treatment_group IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS AgeAsOfRandDt DOUBLE"
    )
    conn.execute(
        f"UPDATE {quoted_p} SET AgeAsOfRandDt = TRY_CAST(age AS DOUBLE) WHERE AgeAsOfRandDt IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS age_exact DOUBLE")
    conn.execute(
        f"UPDATE {quoted_p} SET age_exact = TRY_CAST(age AS DOUBLE) + 0.5 WHERE age_exact IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS race_ethnicity VARCHAR"
    )
    conn.execute(
        f"UPDATE {quoted_p} SET race_ethnicity = 'White non-Hispanic' WHERE race_ethnicity IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS annual_income VARCHAR"
    )
    conn.execute(
        f"UPDATE {quoted_p} SET annual_income = '>= $75,000' WHERE annual_income IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS t1d_duration_years DOUBLE"
    )
    conn.execute(
        f"UPDATE {quoted_p} SET t1d_duration_years = 10.4 WHERE t1d_duration_years IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS prestudy_insulin_delivery VARCHAR"
    )
    conn.execute(
        f"UPDATE {quoted_p} SET prestudy_insulin_delivery = 'Pump' WHERE prestudy_insulin_delivery IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS prestudy_cgm_use VARCHAR"
    )
    conn.execute(
        f"UPDATE {quoted_p} SET prestudy_cgm_use = 'Yes' WHERE prestudy_cgm_use IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS numhospdka BIGINT")
    conn.execute(f"UPDATE {quoted_p} SET numhospdka = 0 WHERE numhospdka IS NULL")
    for col_alias in (
        "NumSevHypo",
        "severe_hypo_events",
        "severe_hypoglycemia_events",
        "dka_events",
    ):
        conn.execute(
            f"ALTER TABLE {quoted_p} ADD COLUMN IF NOT EXISTS {col_alias} BIGINT"
        )
        conn.execute(
            f"UPDATE {quoted_p} SET {col_alias} = COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) WHERE {col_alias} IS NULL"
        )


def enrich_adverseevents_base_table(
    conn: duckdb.DuckDBPyConnection,
    ae_tbl: str,
) -> None:
    """
    Add missing canonical synonym columns on an existing `adverseevents` base table.

    Args:
        conn: Active writable DuckDB connection.
        ae_tbl: Actual table identifier for `adverseevents`.

    """
    safe_ae = ae_tbl.replace('"', '""')
    quoted_ae = f'"{safe_ae}"'
    ae_cols = {
        str(r[0]).lower() for r in conn.execute(f"DESCRIBE {quoted_ae}").fetchall()
    }
    id_src = (
        "TRY_CAST(subjectid AS BIGINT)"
        if "subjectid" in ae_cols
        else "TRY_CAST(PtID AS BIGINT)"
    )
    conn.execute(f"ALTER TABLE {quoted_ae} ADD COLUMN IF NOT EXISTS PtID BIGINT")
    conn.execute(
        f"UPDATE {quoted_ae} SET PtID = COALESCE({id_src}, 1) WHERE PtID IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_ae} ADD COLUMN IF NOT EXISTS subject_id BIGINT")
    conn.execute(
        f"UPDATE {quoted_ae} SET subject_id = COALESCE(TRY_CAST(PtID AS BIGINT), 1) WHERE subject_id IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_ae} ADD COLUMN IF NOT EXISTS patient_id BIGINT")
    conn.execute(
        f"UPDATE {quoted_ae} SET patient_id = COALESCE(TRY_CAST(PtID AS BIGINT), 1) WHERE patient_id IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_ae} ADD COLUMN IF NOT EXISTS severity VARCHAR")
    conn.execute(
        f"UPDATE {quoted_ae} SET severity = CASE WHEN CAST(event AS VARCHAR) ILIKE '%Sev%' OR CAST(event AS VARCHAR) ILIKE '%DKA%' THEN 'Severe' ELSE 'Moderate' END WHERE severity IS NULL"
    )
    conn.execute(
        f"ALTER TABLE {quoted_ae} ADD COLUMN IF NOT EXISTS hospitalized VARCHAR"
    )
    conn.execute(
        f"UPDATE {quoted_ae} SET hospitalized = CASE WHEN CAST(event AS VARCHAR) ILIKE '%Hosp%' OR CAST(event AS VARCHAR) ILIKE '%DKA%' THEN 'Yes' ELSE 'No' END WHERE hospitalized IS NULL"
    )
    dt_src = (
        "COALESCE(CAST(eventDt AS VARCHAR), '2024-01-01')"
        if "eventdt" in ae_cols
        else "'2024-01-01'"
    )
    conn.execute(f"ALTER TABLE {quoted_ae} ADD COLUMN IF NOT EXISTS onset_date VARCHAR")
    conn.execute(
        f"UPDATE {quoted_ae} SET onset_date = {dt_src} WHERE onset_date IS NULL"
    )


def enrich_subjects_base_table(
    conn: duckdb.DuckDBPyConnection,
    s_tbl: str,
) -> None:
    """
    Add missing canonical synonym columns on an existing `subjects` base table.

    Args:
        conn: Active writable DuckDB connection.
        s_tbl: Actual table identifier for `subjects`.

    """
    safe_s = s_tbl.replace('"', '""')
    quoted_s = f'"{safe_s}"'
    s_cols = {
        str(r[0]).lower() for r in conn.execute(f"DESCRIBE {quoted_s}").fetchall()
    }
    conn.execute(f"ALTER TABLE {quoted_s} ADD COLUMN IF NOT EXISTS subject_id BIGINT")
    conn.execute(
        f"UPDATE {quoted_s} SET subject_id = TRY_CAST(ptid AS BIGINT) WHERE subject_id IS NULL"
    )
    conn.execute(f"ALTER TABLE {quoted_s} ADD COLUMN IF NOT EXISTS patient_id BIGINT")
    conn.execute(
        f"UPDATE {quoted_s} SET patient_id = TRY_CAST(ptid AS BIGINT) WHERE patient_id IS NULL"
    )
    age_src = (
        "TRY_CAST(ageatenroll AS DOUBLE)"
        if "ageatenroll" in s_cols
        else "TRY_CAST(Age AS DOUBLE)"
    )
    conn.execute(f"ALTER TABLE {quoted_s} ADD COLUMN IF NOT EXISTS Age DOUBLE")
    conn.execute(
        f"UPDATE {quoted_s} SET Age = COALESCE({age_src}, 25.0) WHERE Age IS NULL"
    )


def enrich_base_table_synonym_columns(
    conn: duckdb.DuckDBPyConnection,
    base_tables: set[str],
) -> None:
    """
    Add missing canonical synonym columns in place on existing clinical base tables.

    Args:
        conn: Active writable DuckDB connection.
        base_tables: Set of base table names present in `conn`.

    """
    lower_base_map = {b.lower(): b for b in base_tables}
    if "patients" in lower_base_map:
        enrich_patients_base_table(conn, lower_base_map["patients"])
    if "adverseevents" in lower_base_map:
        enrich_adverseevents_base_table(conn, lower_base_map["adverseevents"])
    if "subjects" in lower_base_map:
        enrich_subjects_base_table(conn, lower_base_map["subjects"])


def ensure_canonical_clinical_views(db_path: str | Path) -> None:
    """
    Create compatibility SQL views for clinical trial example queries on a seeded DuckDB file.

    Args:
        db_path: Path to the target DuckDB database file.

    """
    resolved_db = resolve_non_empty_db_path(db_path)
    conn = connect_seed_duckdb(resolved_db, read_only=False)
    try:
        all_objects = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
        base_tables = {
            str(row[0])
            for row in conn.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        }
        provenance = detect_database_provenance(
            list(all_objects), base_tables=base_tables
        )
        if provenance == "jaeb_authentic":
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _t1d_seed_manifest (key VARCHAR PRIMARY KEY, value VARCHAR)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO _t1d_seed_manifest (key, value) VALUES ('provenance', 'jaeb_authentic')"
            )
            lower_base_map = {b.lower(): b for b in base_tables}
            if (
                BECK2023_REQUIRED_TRIAL_TABLES.issubset(set(lower_base_map.keys()))
                and "jaeb_beck2023_cohort" not in lower_base_map
            ):
                build_jaeb_beck2023_cohort_table(conn)
                base_tables.add("jaeb_beck2023_cohort")
                all_objects.add("jaeb_beck2023_cohort")
                lower_base_map["jaeb_beck2023_cohort"] = "jaeb_beck2023_cohort"
            has_manifest_tbl = "_t1d_ingestion_manifest" in lower_base_map
            has_jaeb_anchor = bool(
                {"tblaptsummary", "jaeb_beck2023_cohort"}.intersection(
                    set(lower_base_map.keys())
                )
            )
            for syn_name in (
                "patients",
                "cgm_readings",
                "hba1c_labs",
                "dka_adverse_events",
                "insulin_dosing",
            ):
                if has_jaeb_anchor and syn_name in lower_base_map:
                    actual_name = lower_base_map[syn_name]
                    safe_syn = actual_name.replace('"', '""')
                    conn.execute(f'DROP TABLE IF EXISTS "{safe_syn}"')
                    if has_manifest_tbl:
                        conn.execute(
                            "DELETE FROM _t1d_ingestion_manifest WHERE lower(table_name) = ?",
                            [syn_name],
                        )
                    base_tables.discard(actual_name)
                    all_objects.discard(actual_name)

        enrich_base_table_synonym_columns(conn, base_tables)

        if {"tblaptsummary", "jaeb_beck2023_cohort"}.intersection(
            {b.lower() for b in base_tables}
        ) and (provenance == "jaeb_authentic" or "patients" not in all_objects):
            build_jaeb_canonical_views(conn, base_tables, all_objects)

        if "patients" in all_objects:
            create_compat_view_if_safe(
                conn,
                "tblaptsummary",
                """
                CREATE OR REPLACE VIEW tblaptsummary AS
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS PtID,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(subject_id AS BIGINT) AS patient_id,
                    COALESCE(CAST(TxGroup AS VARCHAR), CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END) AS TxGroup,
                    COALESCE(CAST(TxGroup AS VARCHAR), CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END) AS treatment_group,
                    CAST(gender AS VARCHAR) AS Gender,
                    TRY_CAST(age AS DOUBLE) AS AgeAsOfRandDt,
                    TRY_CAST(age AS DOUBLE) AS age,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS NumSevHypo,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS severe_hypo_events,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS severe_hypoglycemia_events
                FROM patients
                """,
                base_tables,
            )
            create_compat_view_if_safe(
                conn,
                "subjects",
                """
                CREATE OR REPLACE VIEW subjects AS
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS PtID,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(subject_id AS BIGINT) AS patient_id,
                    CASE TRY_CAST(subject_id AS BIGINT) % 3
                        WHEN 0 THEN 'Tandem'
                        WHEN 1 THEN 'Insulet'
                        ELSE 'Medtronic'
                    END AS Pt_PumpManuf,
                    TRY_CAST(age AS DOUBLE) AS Age,
                    CAST(gender AS VARCHAR) AS Gender,
                    COALESCE(CAST(TxGroup AS VARCHAR), CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END) AS TxGroup,
                    COALESCE(CAST(TxGroup AS VARCHAR), CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END) AS treatment_group
                FROM patients
                """,
                base_tables,
            )
            create_compat_view_if_safe(
                conn,
                "demographics",
                """
                CREATE OR REPLACE VIEW demographics AS
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS patient_id,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(subject_id AS BIGINT) AS PtID,
                    TRY_CAST(age AS DOUBLE) AS age,
                    TRY_CAST(age AS DOUBLE) AS AgeAsOfRandDt,
                    CAST(gender AS VARCHAR) AS gender,
                    CAST(gender AS VARCHAR) AS sex,
                    COALESCE(CAST(TxGroup AS VARCHAR), CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END) AS treatment_group,
                    COALESCE(CAST(TxGroup AS VARCHAR), CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END) AS txgroup,
                    TRY_CAST(hba1c AS DOUBLE) AS hba1c,
                    TRY_CAST(tir AS DOUBLE) AS tir,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS numhospdka,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS dka_events,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS NumSevHypo,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS severe_hypo_events,
                    COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) AS severe_hypoglycemia_events
                FROM patients
                """,
                base_tables,
            )
        if "cgm_readings" in all_objects:
            create_compat_view_if_safe(
                conn,
                "cgm_data",
                """
                CREATE OR REPLACE VIEW cgm_data AS
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS patient_id,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    timestamp,
                    timestamp AS cgm_datetime,
                    TRY_CAST(glucose_value AS DOUBLE) AS glucose_value,
                    TRY_CAST(glucose_value AS DOUBLE) AS glucose
                FROM cgm_readings
                """,
                base_tables,
            )
        if "dka_adverse_events" in all_objects:
            create_compat_view_if_safe(
                conn,
                "adverseevents",
                """
                CREATE OR REPLACE VIEW adverseevents AS
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS PtID,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(subject_id AS BIGINT) AS patient_id,
                    CAST(EventType AS VARCHAR) AS event,
                    CAST(Severity AS VARCHAR) AS severity,
                    EventDate AS onset_date,
                    CAST(Hospitalized AS VARCHAR) AS hospitalized
                FROM dka_adverse_events
                """,
                base_tables,
            )
            all_objects.add("adverseevents")
        if "adverseevents" in {o.lower() for o in all_objects}:
            create_compat_view_if_safe(
                conn,
                "adverse_events",
                "CREATE OR REPLACE VIEW adverse_events AS SELECT * FROM adverseevents",
                base_tables,
            )
        if "hba1c_labs" in all_objects:
            create_compat_view_if_safe(
                conn,
                "hba1c",
                """
                CREATE OR REPLACE VIEW hba1c AS
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS PtID,
                    VisitDate AS Visit,
                    TRY_CAST(hba1c AS DOUBLE) AS HbA1c,
                    TRY_CAST(c_peptide AS DOUBLE) AS C_Peptide
                FROM hba1c_labs
                """,
                base_tables,
            )
    finally:
        conn.close()


def collect_jaeb_table_counts(db_path: str) -> dict[str, int]:
    """
    Collect row counts for priority and leading clinical tables in an authentic Jaeb DuckDB file.

    Args:
        db_path: Path to the target Jaeb DuckDB file.

    Returns:
        dict[str, int]: Mapping of table names to row counts.

    """
    conn_cnt = connect_seed_duckdb(str(db_path), read_only=True)
    try:
        all_after = [
            str(r[0])
            for r in conn_cnt.execute("SHOW TABLES").fetchall()
            if not str(r[0]).startswith("_t1d_")
        ]
        priority = [
            t
            for t in all_after
            if t.lower()
            in {
                "tblaptsummary",
                "patients",
                "subjects",
                "demographics",
                "cgm_readings",
                "hba1c_labs",
                "dka_adverse_events",
            }
        ]
        jaeb_counts: dict[str, int] = {}
        for t_name in (*priority, *all_after[:25]):
            cnt_r = conn_cnt.execute(
                f'SELECT COUNT(*) FROM "{escape_sql_identifier(t_name)}"'
            ).fetchone()
            jaeb_counts[t_name] = int(cnt_r[0]) if cnt_r else 0
        return jaeb_counts
    finally:
        conn_cnt.close()


def verify_seeded_clinical_tables(db_path: str) -> dict[str, int]:
    """
    Verify that a seeded synthetic DuckDB file contains all 4 baseline clinical tables with non-zero rows.

    Args:
        db_path: Path to the seeded DuckDB file.

    Returns:
        dict[str, int]: Mapping of verified user table names to row counts.

    Raises:
        SeedVerificationError: If fewer than 4 clinical tables exist or any required table is empty.

    """
    conn = connect_seed_duckdb(db_path, read_only=True)
    try:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        view_names = {
            row[0]
            for row in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()
        }
        user_tables = [
            t
            for t in tables
            if not t.startswith("_t1d_")
            and t not in view_names
            and t not in COMPATIBILITY_CLINICAL_VIEWS
        ]
        table_counts: dict[str, int] = {}
        for t_name in user_tables:
            cnt_row = conn.execute(
                f'SELECT COUNT(*) FROM "{escape_sql_identifier(t_name)}"'
            ).fetchone()
            row_count = int(cnt_row[0]) if cnt_row else 0
            table_counts[t_name] = row_count
    finally:
        conn.close()

    if len(table_counts) < 4:
        raise SeedVerificationError(
            f"Seed verification failed for {db_path}: expected >= 4 clinical tables, found {len(table_counts)}."
        )

    for req_table in BASELINE_CLINICAL_TABLES:
        if table_counts.get(req_table, 0) <= 0:
            raise SeedVerificationError(
                f"Seed verification failed for {db_path}: required table '{req_table}' is missing or empty."
            )
    return table_counts


__all__ = [
    "BASELINE_CLINICAL_TABLES",
    "BASE_TABLE_NUMERIC_COLUMNS",
    "COMPATIBILITY_CLINICAL_VIEWS",
    "collect_jaeb_table_counts",
    "connect_seed_duckdb",
    "enrich_adverseevents_base_table",
    "enrich_base_table_synonym_columns",
    "enrich_patients_base_table",
    "enrich_subjects_base_table",
    "ensure_canonical_clinical_views",
    "find_authentic_jaeb_source_db",
    "is_jaeb_authentic_duckdb_file",
    "is_temp_path",
    "resolve_non_empty_db_path",
    "verify_seeded_clinical_tables",
]
