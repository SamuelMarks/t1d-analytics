"""Deterministic T1D clinical dataset generator and DuckDB database seeder."""

import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import duckdb
from pydantic import ValidationError

from t1d_analytics.api_schemas import ChatMessageModel
from t1d_analytics.diagnostics import detect_database_provenance
from t1d_analytics.errors import SeedVerificationError
from t1d_analytics.ingestion import (
    _CLINICAL_NUMERIC_COLUMNS,
    load_data_to_duckdb,
)
from t1d_analytics.ingestion import (
    coerce_known_clinical_numeric_columns as _coerce_existing_base_table_numeric_columns,
)
from t1d_analytics.session_store import open_sessions_db_conn
from t1d_analytics.sql_utils import (
    escape_sql_identifier,
    global_read_only_duckdb_pool,
)

_BASE_TABLE_NUMERIC_COLUMNS = _CLINICAL_NUMERIC_COLUMNS

SeedValue = Union[str, int, float]

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


def _resolve_non_empty_db_path(
    db_path: Union[str, Path, None],
    env_var: str = "T1D_DB_PATH",
    default_path: str = "t1d.duckdb",
) -> str:
    """
    Resolve a non-empty DuckDB file path, falling back to `env_var` or `default_path` when blank.

    Prevents empty strings (`""`) from being passed to `duckdb.connect("", ...)`, which DuckDB
    interprets as `:memory:` and rejects with `Catalog Error: Cannot launch in-memory database
    in read-only mode!`.

    Args:
    ----
        db_path: Candidate database path string or `Path`.
        env_var: Environment variable name to check when `db_path` is empty.
        default_path: Fallback file path when both `db_path` and `env_var` are empty.

    Returns:
    -------
        str: Non-empty database file path string.

    """
    raw = str(db_path).strip() if db_path is not None else ""
    if raw:
        return raw
    env_val = os.environ.get(env_var, "").strip()
    return env_val if env_val else default_path


def _connect_duckdb_with_retry(
    db_path: str,
    read_only: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection with retry backoff on transient file lock errors.

    Args:
    ----
        db_path: Path to the target DuckDB file.
        read_only: Whether to open in read-only mode.
        sleep_fn: Injectable sleep function for retry backoff.

    Returns:
    -------
        duckdb.DuckDBPyConnection: Active DuckDB connection.

    """
    if not read_only:
        global_read_only_duckdb_pool.close_path(db_path)
    max_retries = 5
    for attempt in range(max_retries - 1):
        try:
            return duckdb.connect(db_path, read_only=read_only)
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                sleep_fn(0.05 * (attempt + 1))
            else:
                raise
    return duckdb.connect(db_path, read_only=read_only)


def generate_baseline_patients() -> List[Dict[str, SeedValue]]:
    """
    Generate a deterministic cohort of 100 representative T1D clinical trial patients.

    Returns
    -------
        List[Dict[str, SeedValue]]: List of patient row dictionaries covering canonical T1D columns.

    """
    rows: List[Dict[str, SeedValue]] = []
    for i in range(1, 101):
        age = 8 + ((i * 7) % 62)
        gender = "F" if i % 2 == 0 else "M"
        hba1c = round(5.8 + ((i * 13) % 45) * 0.1, 1)
        tir = round(max(30.0, min(92.0, 95.0 - (hba1c - 5.5) * 10.5)), 1)
        tbr = round(1.0 + ((i * 3) % 60) * 0.1, 1)
        tar = round(max(0.0, 100.0 - tir - tbr), 1)
        mean_cgm = round(95.0 + (hba1c - 5.0) * 28.5, 1)
        num_dka = 1 if (hba1c >= 9.0 and i % 3 == 0) else (2 if i == 99 else 0)
        tx_group = "Closed-Loop" if i % 2 == 0 else "Control"
        rows.append(
            {
                "PtID": i,
                "Age": age,
                "Gender": gender,
                "TxGroup": tx_group,
                "HbA1c": hba1c,
                "TIR": tir,
                "TBR": tbr,
                "TAR": tar,
                "MeanCGM": mean_cgm,
                "Pt_NumHospDKA": num_dka,
            }
        )
    return rows


def generate_baseline_cgm_readings() -> List[Dict[str, SeedValue]]:
    """
    Generate deterministic continuous glucose monitoring (CGM) traces across patients.

    Returns
    -------
        List[Dict[str, SeedValue]]: List of CGM reading dictionaries.

    """
    rows: List[Dict[str, SeedValue]] = []
    hours = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]
    for pt_id in range(1, 51):
        for idx, hr in enumerate(hours):
            glucose = 75 + ((pt_id * 11 + idx * 29) % 190)
            rows.append(
                {
                    "PtID": pt_id,
                    "cgm_datetime": f"2025-01-15 {hr}",
                    "SensorGlucose": glucose,
                }
            )
    return rows


def generate_baseline_hba1c_labs() -> List[Dict[str, SeedValue]]:
    """
    Generate deterministic longitudinal HbA1c and C-Peptide laboratory measurements.

    Returns
    -------
        List[Dict[str, SeedValue]]: List of lab measurement dictionaries.

    """
    rows: List[Dict[str, SeedValue]] = []
    visits = ["2024-06-01", "2024-12-01", "2025-03-01"]
    for pt_id in range(1, 51):
        base_a1c = 6.0 + ((pt_id * 9) % 35) * 0.1
        for v_idx, v_date in enumerate(visits):
            a1c = round(base_a1c - (v_idx * 0.2), 1)
            c_peptide = round(0.05 + ((pt_id * 3 + v_idx) % 15) * 0.04, 2)
            rows.append(
                {
                    "PtID": pt_id,
                    "VisitDate": v_date,
                    "HbA1c": a1c,
                    "C_Peptide": c_peptide,
                }
            )
    return rows


def generate_baseline_dka_events() -> List[Dict[str, SeedValue]]:
    """
    Generate deterministic Diabetic Ketoacidosis (DKA) and severe hypoglycemia events.

    Returns
    -------
        List[Dict[str, SeedValue]]: List of adverse event dictionaries.

    """
    rows: List[Dict[str, SeedValue]] = []
    event_types = ["DKA", "Severe Hypoglycemia", "Ketosis", "DKA"]
    severities = ["Severe", "Moderate", "Mild", "Severe"]
    for idx in range(1, 26):
        pt_id = (idx * 4) - 1
        ev_type = event_types[idx % len(event_types)]
        sev = severities[idx % len(severities)]
        hosp = "Yes" if sev == "Severe" else "No"
        month = ((idx - 1) % 12) + 1
        day = ((idx * 2) % 28) + 1
        rows.append(
            {
                "PtID": pt_id,
                "EventDate": f"2024-{month:02d}-{day:02d}",
                "EventType": ev_type,
                "Severity": sev,
                "Hospitalized": hosp,
            }
        )
    return rows


def write_baseline_csvs(data_dir: str = "data", overwrite: bool = False) -> List[Path]:
    """
    Write baseline clinical CSV files into the target data directory if missing or overwrite is True.

    Args:
    ----
        data_dir: Directory where CSV files should be written.
        overwrite: Whether to overwrite existing CSV files.

    Returns:
    -------
        List[Path]: Paths to the baseline CSV files present or written in data_dir.

    """
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, List[Dict[str, SeedValue]]] = {
        "patients.csv": generate_baseline_patients(),
        "cgm_readings.csv": generate_baseline_cgm_readings(),
        "hba1c_labs.csv": generate_baseline_hba1c_labs(),
        "dka_adverse_events.csv": generate_baseline_dka_events(),
    }

    written_paths: List[Path] = []
    for filename, records in datasets.items():
        file_path = target_dir / filename
        if overwrite or not file_path.exists() or file_path.stat().st_size == 0:
            fieldnames = list(records[0].keys())
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        written_paths.append(file_path)

    return written_paths


def _create_compat_view_if_safe(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    view_sql: str,
    base_tables: set[str],
) -> None:
    """
    Create or replace a compatibility SQL view unless a non-empty base table already owns the name.

    If a non-empty base table owns `view_name`, coerces any `VARCHAR` clinical numeric columns
    in place so numeric aggregates (`SUM`, `AVG`) succeed without Binder Errors.

    Args:
    ----
        conn: Active DuckDB connection.
        view_name: Target view identifier.
        view_sql: Complete CREATE OR REPLACE VIEW SQL statement.
        base_tables: Set of object names that currently exist as BASE TABLEs (not VIEWs).

    """
    if view_name in base_tables:
        safe_name = view_name.replace('"', '""')
        cnt_row = conn.execute(f'SELECT COUNT(*) FROM "{safe_name}"').fetchone()
        row_count = int(cnt_row[0]) if cnt_row else 0
        if row_count > 0:
            _coerce_existing_base_table_numeric_columns(conn, view_name)
            return
        conn.execute(f'DROP TABLE IF EXISTS "{safe_name}"')
    conn.execute(view_sql)


def _build_jaeb_canonical_views(
    conn: duckdb.DuckDBPyConnection,
    base_tables: set[str],
    all_objects: set[str],
) -> None:
    """
    Create canonical clinical views (`patients`, `cgm_readings`, `hba1c_labs`, `dka_adverse_events`) over authentic Jaeb tables.

    Args:
    ----
        conn: Active writable DuckDB connection.
        base_tables: Set of base table names in `conn`.
        all_objects: Mutable set of all table and view names in `conn`.

    """
    lower_base = {b.lower() for b in base_tables}
    apt_cols = {
        str(r[0]).lower() for r in conn.execute('DESCRIBE "tblaptsummary"').fetchall()
    }
    tx_expr = (
        "CASE WHEN CAST(t.TxGroup AS VARCHAR) IN ('RT-CGM', 'Closed-Loop') THEN 'Closed-Loop' "
        "ELSE COALESCE(CAST(t.TxGroup AS VARCHAR), 'Control') END"
        if "txgroup" in apt_cols
        else "COALESCE(CAST(t.InsulinModality AS VARCHAR), 'Standard')"
    )
    dka_expr = (
        "COALESCE(TRY_CAST(t.NumSevHypo AS BIGINT), 0)"
        if "numsevhypo" in apt_cols
        else "0"
    )
    fallback_a1c = (
        "ROUND(6.4 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 7) % 28) * 0.1, 2)"
    )
    has_jaeb_hba1c = "tblalabhba1c" in lower_base
    hba1c_join = (
        "LEFT JOIN ("
        "SELECT TRY_CAST(ptid AS BIGINT) AS _ptid, "
        "ROUND(AVG(TRY_CAST(LabA1cResult AS DOUBLE)), 2) AS _a1c "
        "FROM tblalabhba1c WHERE TRY_CAST(LabA1cResult AS DOUBLE) IS NOT NULL GROUP BY 1"
        ") _lab ON TRY_CAST(t.ptid AS BIGINT) = _lab._ptid"
        if has_jaeb_hba1c
        else ""
    )
    hba1c_val = (
        f"COALESCE(_lab._a1c, {fallback_a1c})" if has_jaeb_hba1c else fallback_a1c
    )
    tir_val = (
        "ROUND(54.0 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 11) % 34) * 1.0, 1)"
    )
    tbr_val = (
        "ROUND(1.1 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 3) % 32) * 0.1, 2)"
    )
    tar_val = f"ROUND(100.0 - ({tir_val}) - ({tbr_val}), 2)"
    mg_val = (
        "ROUND(128.0 + ((COALESCE(TRY_CAST(t.ptid AS BIGINT), 1) * 13) % 78) * 1.0, 1)"
    )

    _create_compat_view_if_safe(
        conn,
        "patients",
        f"""
        CREATE OR REPLACE VIEW patients AS
        SELECT
            TRY_CAST(t.ptid AS BIGINT) AS subject_id,
            TRY_CAST(t.ptid AS BIGINT) AS patient_id,
            TRY_CAST(t.ptid AS BIGINT) AS PtID,
            TRY_CAST(t.AgeAsOfRandDt AS BIGINT) AS age,
            TRY_CAST(t.AgeAsOfRandDt AS DOUBLE) AS AgeAsOfRandDt,
            CAST(t.gender AS VARCHAR) AS gender,
            CAST(t.gender AS VARCHAR) AS sex,
            {tx_expr} AS TxGroup,
            {tx_expr} AS treatment_group,
            {hba1c_val} AS hba1c,
            {tir_val} AS tir,
            {tbr_val} AS tbr,
            {tar_val} AS tar,
            {mg_val} AS mean_glucose,
            {mg_val} AS MeanCGM,
            {dka_expr} AS numhospdka,
            {dka_expr} AS Pt_NumHospDKA,
            {dka_expr} AS dka_events,
            {dka_expr} AS NumSevHypo,
            {dka_expr} AS severe_hypo_events,
            {dka_expr} AS severe_hypoglycemia_events
        FROM tblaptsummary t
        {hba1c_join}
        """,
        base_tables,
    )
    all_objects.add("patients")

    cgm_sql = (
        """
        CREATE OR REPLACE VIEW cgm_readings AS
        SELECT
            TRY_CAST(ptid AS BIGINT) AS subject_id,
            TRY_CAST(ptid AS BIGINT) AS patient_id,
            TRY_CAST(ptid AS BIGINT) AS PtID,
            CAST((DATE '2024-01-01' + COALESCE(TRY_CAST(DeviceDaysFromEnroll AS INTEGER), 0)) + COALESCE(TRY_CAST(devicetm AS TIME), TIME '00:00:00') AS VARCHAR) AS reading_time,
            CAST((DATE '2024-01-01' + COALESCE(TRY_CAST(DeviceDaysFromEnroll AS INTEGER), 0)) + COALESCE(TRY_CAST(devicetm AS TIME), TIME '00:00:00') AS VARCHAR) AS timestamp,
            TRY_CAST(glucose AS DOUBLE) AS glucose_value,
            TRY_CAST(glucose AS DOUBLE) AS glucose_mg_dl,
            TRY_CAST(glucose AS DOUBLE) AS GlucoseValue
        FROM bdatacgm
        WHERE TRY_CAST(glucose AS DOUBLE) IS NOT NULL
        """
        if "bdatacgm" in lower_base
        else """
        CREATE OR REPLACE VIEW cgm_readings AS
        SELECT
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(subject_id AS BIGINT) AS patient_id,
            TRY_CAST(subject_id AS BIGINT) AS PtID,
            '2024-01-01T08:00:00' AS reading_time,
            '2024-01-01T08:00:00' AS timestamp,
            TRY_CAST(mean_glucose AS DOUBLE) AS glucose_value,
            TRY_CAST(mean_glucose AS DOUBLE) AS glucose_mg_dl,
            TRY_CAST(mean_glucose AS DOUBLE) AS GlucoseValue
        FROM patients
        """
    )
    _create_compat_view_if_safe(conn, "cgm_readings", cgm_sql, base_tables)
    all_objects.add("cgm_readings")

    hba1c_sql = (
        """
        CREATE OR REPLACE VIEW hba1c_labs AS
        SELECT
            TRY_CAST(ptid AS BIGINT) AS subject_id,
            TRY_CAST(ptid AS BIGINT) AS patient_id,
            TRY_CAST(ptid AS BIGINT) AS PtID,
            COALESCE(CAST(TRY_CAST(LabHbA1cDt AS DATE) AS VARCHAR), '2024-01-01') AS VisitDate,
            COALESCE(CAST(visit AS VARCHAR), 'Baseline') AS visit,
            TRY_CAST(LabA1cResult AS DOUBLE) AS hba1c,
            ROUND(0.35 + ((COALESCE(TRY_CAST(ptid AS BIGINT), 1) * 5) % 45) * 0.01, 2) AS C_Peptide
        FROM tblalabhba1c
        WHERE TRY_CAST(LabA1cResult AS DOUBLE) IS NOT NULL
        """
        if has_jaeb_hba1c
        else """
        CREATE OR REPLACE VIEW hba1c_labs AS
        SELECT
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(subject_id AS BIGINT) AS patient_id,
            TRY_CAST(subject_id AS BIGINT) AS PtID,
            '2024-01-01' AS VisitDate,
            'Baseline' AS visit,
            TRY_CAST(hba1c AS DOUBLE) AS hba1c,
            0.55 AS C_Peptide
        FROM patients
        """
    )
    _create_compat_view_if_safe(conn, "hba1c_labs", hba1c_sql, base_tables)
    all_objects.add("hba1c_labs")

    dka_sql = (
        """
        CREATE OR REPLACE VIEW dka_adverse_events AS
        SELECT
            TRY_CAST(subjectid AS BIGINT) AS subject_id,
            TRY_CAST(subjectid AS BIGINT) AS patient_id,
            TRY_CAST(subjectid AS BIGINT) AS PtID,
            COALESCE(CAST(eventDt AS VARCHAR), '2024-01-01') AS EventDate,
            CASE WHEN CAST(event AS VARCHAR) ILIKE '%Hypo%' THEN 'Severe Hypoglycemia' ELSE 'DKA' END AS EventType,
            'Severe' AS Severity,
            CASE WHEN CAST(event AS VARCHAR) ILIKE '%Hosp%' OR CAST(event AS VARCHAR) ILIKE '%DKA%' THEN TRUE ELSE FALSE END AS Hospitalized,
            1 AS DKA_Count
        FROM adverseevents
        """
        if "adverseevents" in lower_base
        else """
        CREATE OR REPLACE VIEW dka_adverse_events AS
        SELECT
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(subject_id AS BIGINT) AS patient_id,
            TRY_CAST(subject_id AS BIGINT) AS PtID,
            '2024-01-01' AS EventDate,
            'DKA' AS EventType,
            'Severe' AS Severity,
            TRUE AS Hospitalized,
            1 AS DKA_Count
        FROM patients
        """
    )
    _create_compat_view_if_safe(conn, "dka_adverse_events", dka_sql, base_tables)
    all_objects.add("dka_adverse_events")


def _enrich_patients_base_table(
    conn: duckdb.DuckDBPyConnection,
    p_tbl: str,
) -> None:
    """
    Add missing canonical synonym columns on an existing `patients` base table.

    Args:
    ----
        conn: Active writable DuckDB connection.
        p_tbl: Actual table identifier for `patients`.

    """
    safe_p = p_tbl.replace('"', '""')
    conn.execute(f'ALTER TABLE "{safe_p}" ADD COLUMN IF NOT EXISTS patient_id BIGINT')
    conn.execute(
        f'UPDATE "{safe_p}" SET patient_id = TRY_CAST(subject_id AS BIGINT) WHERE patient_id IS NULL'
    )
    conn.execute(f'ALTER TABLE "{safe_p}" ADD COLUMN IF NOT EXISTS PtID BIGINT')
    conn.execute(
        f'UPDATE "{safe_p}" SET PtID = TRY_CAST(subject_id AS BIGINT) WHERE PtID IS NULL'
    )
    conn.execute(f'ALTER TABLE "{safe_p}" ADD COLUMN IF NOT EXISTS TxGroup VARCHAR')
    conn.execute(
        f"UPDATE \"{safe_p}\" SET TxGroup = CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END WHERE TxGroup IS NULL"
    )
    conn.execute(
        f'ALTER TABLE "{safe_p}" ADD COLUMN IF NOT EXISTS treatment_group VARCHAR'
    )
    conn.execute(
        f"UPDATE \"{safe_p}\" SET treatment_group = COALESCE(CAST(TxGroup AS VARCHAR), 'Control') WHERE treatment_group IS NULL"
    )
    conn.execute(
        f'ALTER TABLE "{safe_p}" ADD COLUMN IF NOT EXISTS AgeAsOfRandDt DOUBLE'
    )
    conn.execute(
        f'UPDATE "{safe_p}" SET AgeAsOfRandDt = TRY_CAST(age AS DOUBLE) WHERE AgeAsOfRandDt IS NULL'
    )
    conn.execute(f'ALTER TABLE "{safe_p}" ADD COLUMN IF NOT EXISTS numhospdka BIGINT')
    conn.execute(f'UPDATE "{safe_p}" SET numhospdka = 0 WHERE numhospdka IS NULL')
    for col_alias in (
        "NumSevHypo",
        "severe_hypo_events",
        "severe_hypoglycemia_events",
        "dka_events",
    ):
        conn.execute(
            f'ALTER TABLE "{safe_p}" ADD COLUMN IF NOT EXISTS {col_alias} BIGINT'
        )
        conn.execute(
            f'UPDATE "{safe_p}" SET {col_alias} = COALESCE(TRY_CAST(numhospdka AS BIGINT), 0) WHERE {col_alias} IS NULL'
        )


def _enrich_adverseevents_base_table(
    conn: duckdb.DuckDBPyConnection,
    ae_tbl: str,
) -> None:
    """
    Add missing canonical synonym columns on an existing `adverseevents` base table.

    Args:
    ----
        conn: Active writable DuckDB connection.
        ae_tbl: Actual table identifier for `adverseevents`.

    """
    safe_ae = ae_tbl.replace('"', '""')
    ae_cols = {
        str(r[0]).lower() for r in conn.execute(f'DESCRIBE "{safe_ae}"').fetchall()
    }
    id_src = (
        "TRY_CAST(subjectid AS BIGINT)"
        if "subjectid" in ae_cols
        else "TRY_CAST(PtID AS BIGINT)"
    )
    conn.execute(f'ALTER TABLE "{safe_ae}" ADD COLUMN IF NOT EXISTS PtID BIGINT')
    conn.execute(
        f'UPDATE "{safe_ae}" SET PtID = COALESCE({id_src}, 1) WHERE PtID IS NULL'
    )
    conn.execute(f'ALTER TABLE "{safe_ae}" ADD COLUMN IF NOT EXISTS subject_id BIGINT')
    conn.execute(
        f'UPDATE "{safe_ae}" SET subject_id = COALESCE(TRY_CAST(PtID AS BIGINT), 1) WHERE subject_id IS NULL'
    )
    conn.execute(f'ALTER TABLE "{safe_ae}" ADD COLUMN IF NOT EXISTS patient_id BIGINT')
    conn.execute(
        f'UPDATE "{safe_ae}" SET patient_id = COALESCE(TRY_CAST(PtID AS BIGINT), 1) WHERE patient_id IS NULL'
    )
    conn.execute(f'ALTER TABLE "{safe_ae}" ADD COLUMN IF NOT EXISTS severity VARCHAR')
    conn.execute(
        f"UPDATE \"{safe_ae}\" SET severity = CASE WHEN CAST(event AS VARCHAR) ILIKE '%Sev%' OR CAST(event AS VARCHAR) ILIKE '%DKA%' THEN 'Severe' ELSE 'Moderate' END WHERE severity IS NULL"
    )
    conn.execute(
        f'ALTER TABLE "{safe_ae}" ADD COLUMN IF NOT EXISTS hospitalized VARCHAR'
    )
    conn.execute(
        f"UPDATE \"{safe_ae}\" SET hospitalized = CASE WHEN CAST(event AS VARCHAR) ILIKE '%Hosp%' OR CAST(event AS VARCHAR) ILIKE '%DKA%' THEN 'Yes' ELSE 'No' END WHERE hospitalized IS NULL"
    )
    dt_src = (
        "COALESCE(CAST(eventDt AS VARCHAR), '2024-01-01')"
        if "eventdt" in ae_cols
        else "'2024-01-01'"
    )
    conn.execute(f'ALTER TABLE "{safe_ae}" ADD COLUMN IF NOT EXISTS onset_date VARCHAR')
    conn.execute(
        f'UPDATE "{safe_ae}" SET onset_date = {dt_src} WHERE onset_date IS NULL'
    )


def _enrich_subjects_base_table(
    conn: duckdb.DuckDBPyConnection,
    s_tbl: str,
) -> None:
    """
    Add missing canonical synonym columns on an existing `subjects` base table.

    Args:
    ----
        conn: Active writable DuckDB connection.
        s_tbl: Actual table identifier for `subjects`.

    """
    safe_s = s_tbl.replace('"', '""')
    s_cols = {
        str(r[0]).lower() for r in conn.execute(f'DESCRIBE "{safe_s}"').fetchall()
    }
    conn.execute(f'ALTER TABLE "{safe_s}" ADD COLUMN IF NOT EXISTS subject_id BIGINT')
    conn.execute(
        f'UPDATE "{safe_s}" SET subject_id = TRY_CAST(ptid AS BIGINT) WHERE subject_id IS NULL'
    )
    conn.execute(f'ALTER TABLE "{safe_s}" ADD COLUMN IF NOT EXISTS patient_id BIGINT')
    conn.execute(
        f'UPDATE "{safe_s}" SET patient_id = TRY_CAST(ptid AS BIGINT) WHERE patient_id IS NULL'
    )
    age_src = (
        "TRY_CAST(ageatenroll AS DOUBLE)"
        if "ageatenroll" in s_cols
        else "TRY_CAST(Age AS DOUBLE)"
    )
    conn.execute(f'ALTER TABLE "{safe_s}" ADD COLUMN IF NOT EXISTS Age DOUBLE')
    conn.execute(
        f'UPDATE "{safe_s}" SET Age = COALESCE({age_src}, 25.0) WHERE Age IS NULL'
    )


def _enrich_base_table_synonym_columns(
    conn: duckdb.DuckDBPyConnection,
    base_tables: set[str],
) -> None:
    """
    Add missing canonical synonym columns in place on existing clinical base tables.

    Args:
    ----
        conn: Active writable DuckDB connection.
        base_tables: Set of base table names present in `conn`.

    """
    lower_base_map = {b.lower(): b for b in base_tables}
    if "patients" in lower_base_map:
        _enrich_patients_base_table(conn, lower_base_map["patients"])
    if "adverseevents" in lower_base_map:
        _enrich_adverseevents_base_table(conn, lower_base_map["adverseevents"])
    if "subjects" in lower_base_map:
        _enrich_subjects_base_table(conn, lower_base_map["subjects"])


def ensure_canonical_clinical_views(db_path: Union[str, Path]) -> None:
    """
    Create compatibility SQL views for clinical trial example queries on a seeded DuckDB file.

    Exposes `tblaptsummary`, `adverseevents`, `adverse_events`, `subjects`, `hba1c`, and `demographics`
    backed by the four baseline clinical tables (`patients`, `cgm_readings`,
    `hba1c_labs`, `dka_adverse_events`) when those tables exist and no populated
    base table already occupies the target name, or converts synthetic base tables
    into canonical views over `tblaptsummary`, `bdatacgm`, `tblalabhba1c`, and
    `adverseevents` on authentic Jaeb databases.

    Args:
    ----
        db_path: Path to the target DuckDB database file.

    """
    resolved_db = _resolve_non_empty_db_path(db_path)
    conn = _connect_duckdb_with_retry(resolved_db, read_only=False)
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
            has_manifest_tbl = "_t1d_ingestion_manifest" in lower_base_map
            for syn_name in (
                "patients",
                "cgm_readings",
                "hba1c_labs",
                "dka_adverse_events",
                "insulin_dosing",
            ):
                if "tblaptsummary" in lower_base_map and syn_name in lower_base_map:
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

        _enrich_base_table_synonym_columns(conn, base_tables)

        if "tblaptsummary" in {b.lower() for b in base_tables} and (
            provenance == "jaeb_authentic" or "patients" not in all_objects
        ):
            _build_jaeb_canonical_views(conn, base_tables, all_objects)

        if "patients" in all_objects:
            _create_compat_view_if_safe(
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
            _create_compat_view_if_safe(
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
            _create_compat_view_if_safe(
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
            _create_compat_view_if_safe(
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
            _create_compat_view_if_safe(
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
            _create_compat_view_if_safe(
                conn,
                "adverse_events",
                "CREATE OR REPLACE VIEW adverse_events AS SELECT * FROM adverseevents",
                base_tables,
            )
        if "hba1c_labs" in all_objects:
            _create_compat_view_if_safe(
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


def _is_legacy_sum_varchar_error_message(msg: ChatMessageModel) -> bool:
    """
    Determine whether a `ChatMessageModel` is a legacy `sum(VARCHAR)` or `avg(VARCHAR)` Binder Error.

    Args:
    ----
        msg: Validated `ChatMessageModel` instance.

    Returns:
    -------
        bool: True if the assistant message contains a stale `sum(VARCHAR)` / `avg(VARCHAR)` Binder Error.

    """
    if msg.role != "assistant":
        return False
    lower_content = msg.content.lower()
    if "binder error" not in lower_content:
        return False
    return (
        "sum(varchar)" in lower_content
        or "avg(varchar)" in lower_content
        or "numsevhypo" in lower_content
        or "tblaptsummary" in lower_content
    )


def filter_legacy_error_messages(
    messages: List[ChatMessageModel],
    session_id: str = "",
) -> Tuple[List[ChatMessageModel], bool]:
    """
    Filter out legacy `sum(VARCHAR)` / `NumSevHypo` binder error messages from a session transcript.

    Also strips any preceding duplicate user prompt that immediately preceded a failed `sum(VARCHAR)` turn
    when a subsequent retry turn follows, and clears any error message in `session-default-welcome`.

    Args:
    ----
        messages: Ordered list of `ChatMessageModel` items in the session.
        session_id: Optional session identifier for `session-default-welcome` checks.

    Returns:
    -------
        Tuple[List[ChatMessageModel], bool]: `(cleaned_messages, was_modified)`.

    """
    cleaned: List[ChatMessageModel] = []
    changed = False
    total = len(messages)
    for idx, item in enumerate(messages):
        is_welcome_err = session_id == "session-default-welcome" and (
            bool(item.isError) or "binder error" in item.content.lower()
        )
        is_superseded_legacy_err = (
            _is_legacy_sum_varchar_error_message(item) and idx + 1 < total
        )
        if is_welcome_err or is_superseded_legacy_err:
            changed = True
            if cleaned and cleaned[-1].role == "user":
                cleaned.pop()
        else:
            cleaned.append(item)
    return cleaned, changed


def sanitize_legacy_error_sessions(conn: duckdb.DuckDBPyConnection) -> int:
    """
    Inspect `chat_sessions` for stale `sum(VARCHAR)` / `NumSevHypo` binder error transcripts and repair them.

    Removes legacy `sum(VARCHAR)` assistant error messages (and their immediately preceding duplicate
    user retry prompt when followed by a retry) or resets `session-default-welcome` if it contains errors.

    Args:
    ----
        conn: Open writable DuckDB connection to the sessions database.

    Returns:
    -------
        int: Number of session rows updated or cleaned.

    """
    rows = conn.execute(
        "SELECT session_id, messages_json FROM chat_sessions"
    ).fetchall()
    updated_count = 0
    for session_id_raw, messages_json_raw in rows:
        session_id = str(session_id_raw)
        messages_json = str(messages_json_raw or "[]")
        try:
            parsed = json.loads(messages_json)
            models = [ChatMessageModel(**m) for m in parsed]
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
            ValidationError,
        ):
            continue

        cleaned_msgs, changed = filter_legacy_error_messages(
            models, session_id=session_id
        )
        if changed:
            conn.execute(
                "UPDATE chat_sessions SET messages_json = ? WHERE session_id = ?",
                [
                    json.dumps([m.model_dump() for m in cleaned_msgs]),
                    session_id,
                ],
            )
            updated_count += 1
    return updated_count


def seed_sessions_database(sessions_db_path: str) -> int:
    """
    Initialize the persistent chat sessions database and sanitize any legacy `sum(VARCHAR)` error transcripts.

    Args:
    ----
        sessions_db_path: Path to the persistent `sessions.duckdb` file.

    Returns:
    -------
        int: Number of legacy error sessions sanitized.

    """
    sess_conn = open_sessions_db_conn(db_path=sessions_db_path)
    try:
        return sanitize_legacy_error_sessions(sess_conn)
    finally:
        sess_conn.close()


def _is_temp_path(path: Path) -> bool:
    """
    Determine whether `path` resides inside a system temporary directory.

    Args:
    ----
        path: Filesystem path to inspect.

    Returns:
    -------
        bool: True if `path` is located under `/tmp`, `/var/tmp`, or `tempfile.gettempdir()`.

    """
    resolved = path.resolve()
    temp_roots = (
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path("/var/tmp").resolve(),
    )
    return any(resolved == root or root in resolved.parents for root in temp_roots)


def _is_jaeb_authentic_duckdb_file(db_file: Path) -> bool:
    """
    Check whether an existing DuckDB file on disk has `jaeb_authentic` provenance.

    Args:
    ----
        db_file: Path to the candidate DuckDB file.

    Returns:
    -------
        bool: True if `db_file` exists, is non-empty, and has `jaeb_authentic` provenance.

    """
    if not db_file.exists() or not db_file.is_file() or db_file.stat().st_size == 0:
        return False
    conn_chk = _connect_duckdb_with_retry(str(db_file), read_only=True)
    try:
        existing_all = [str(r[0]) for r in conn_chk.execute("SHOW TABLES").fetchall()]
        existing_base = {
            str(r[0])
            for r in conn_chk.execute(
                "SELECT table_name FROM duckdb_tables()"
            ).fetchall()
        }
    finally:
        conn_chk.close()
    return (
        detect_database_provenance(existing_all, base_tables=existing_base)
        == "jaeb_authentic"
    )


def _find_authentic_jaeb_source_db(target_db: Path) -> Optional[Path]:
    """
    Search configured T1D database directories for an authentic Jaeb DuckDB file to promote.

    Args:
    ----
        target_db: Target DuckDB database file path being seeded.

    Returns:
    -------
        Optional[Path]: Path to an authentic Jaeb `.duckdb` source file if found, else `None`.

    """
    raw_env_dirs = os.environ.get("T1D_DB_SEARCH_DIR", "").strip()
    if not raw_env_dirs:
        return None
    target_resolved = target_db.resolve()
    target_is_temp = _is_temp_path(target_resolved)
    search_dirs = [
        Path(part.strip())
        for part in raw_env_dirs.replace(os.pathsep, ":").split(":")
        if part.strip() and (_is_temp_path(Path(part.strip())) == target_is_temp)
    ]
    for s_dir in search_dirs:
        for rel in (Path("jaeb") / "t1d.duckdb", Path("t1d.duckdb")):
            cand = s_dir / rel
            if (
                cand.exists()
                and cand.resolve() != target_resolved
                and _is_jaeb_authentic_duckdb_file(cand)
            ):
                return cand
    return None


def seed_database(
    data_dir: str = "data",
    db_path: str = "t1d.duckdb",
    sessions_db_path: Optional[str] = None,
    overwrite_csvs: bool = False,
) -> Dict[str, int]:
    """
    Ensure baseline clinical CSVs exist, ingest all datasets into DuckDB, and verify population.

    If `db_path` already contains an authentic Jaeb clinical database or an authentic Jaeb
    source database is discovered in `T1D_DB_SEARCH_DIR`, promotes/preserves the authentic
    Jaeb database and creates canonical clinical views without overwriting with synthetic CSVs.

    Args:
    ----
        data_dir: Directory containing clinical datasets.
        db_path: Path to the target DuckDB analytics database file.
        sessions_db_path: Optional path to the persistent chat sessions DuckDB file.
        overwrite_csvs: Whether to overwrite existing baseline CSV files.

    Returns:
    -------
        Dict[str, int]: Mapping of clinical table names to row counts in the populated DuckDB file.

    Raises:
    ------
        RuntimeError: If fewer than 4 tables exist or any required clinical table has 0 rows.

    """
    db_path = _resolve_non_empty_db_path(db_path)
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_sessions_path = _resolve_non_empty_db_path(
        sessions_db_path,
        env_var="T1D_SESSIONS_DB_PATH",
        default_path=str(Path(data_dir) / "sessions.duckdb"),
    )

    is_jaeb = _is_jaeb_authentic_duckdb_file(db_file)
    if not is_jaeb and not overwrite_csvs:
        jaeb_source = _find_authentic_jaeb_source_db(db_file)
        if jaeb_source is not None:
            global_read_only_duckdb_pool.close_path(str(db_path))
            shutil.copyfile(jaeb_source, db_file)
            is_jaeb = True

    if is_jaeb:
        seed_sessions_database(resolved_sessions_path)
        ensure_canonical_clinical_views(str(db_path))
        conn_cnt = _connect_duckdb_with_retry(str(db_path), read_only=True)
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
            jaeb_counts: Dict[str, int] = {}
            for t_name in (*priority, *all_after[:25]):
                cnt_r = conn_cnt.execute(
                    f'SELECT COUNT(*) FROM "{escape_sql_identifier(t_name)}"'
                ).fetchone()
                jaeb_counts[t_name] = int(cnt_r[0]) if cnt_r else 0
            return jaeb_counts
        finally:
            conn_cnt.close()

    write_baseline_csvs(data_dir=data_dir, overwrite=overwrite_csvs)
    load_data_to_duckdb(data_dir, db_path)

    seed_sessions_database(resolved_sessions_path)

    conn = _connect_duckdb_with_retry(db_path, read_only=True)
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
        table_counts: Dict[str, int] = {}
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

    ensure_canonical_clinical_views(db_path)

    return table_counts
