"""Deterministic T1D clinical dataset generator and DuckDB database seeder."""

import csv
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import duckdb

from t1d_analytics.api_schemas import ChatMessageModel
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
    "subjects",
    "hba1c",
    "demographics",
    "cgm_data",
)


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


def ensure_canonical_clinical_views(db_path: str) -> None:
    """
    Create compatibility SQL views for clinical trial example queries on a seeded DuckDB file.

    Exposes `tblaptsummary`, `adverseevents`, `subjects`, `hba1c`, and `demographics`
    backed by the four baseline clinical tables (`patients`, `cgm_readings`,
    `hba1c_labs`, `dka_adverse_events`) when those tables exist and no populated
    base table already occupies the target name.

    Args:
    ----
        db_path: Path to the target DuckDB database file.

    """
    conn = _connect_duckdb_with_retry(db_path, read_only=False)
    try:
        all_objects = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        base_tables = {
            row[0]
            for row in conn.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        }
        if "patients" in all_objects:
            _create_compat_view_if_safe(
                conn,
                "tblaptsummary",
                """
                CREATE OR REPLACE VIEW tblaptsummary AS
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS PtID,
                    CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END AS TxGroup,
                    CAST(gender AS VARCHAR) AS Gender,
                    TRY_CAST(age AS DOUBLE) AS AgeAsOfRandDt,
                    TRY_CAST(numhospdka AS BIGINT) AS NumSevHypo
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
                    CASE TRY_CAST(subject_id AS BIGINT) % 3
                        WHEN 0 THEN 'Tandem'
                        WHEN 1 THEN 'Insulet'
                        ELSE 'Medtronic'
                    END AS Pt_PumpManuf,
                    TRY_CAST(age AS DOUBLE) AS Age,
                    CAST(gender AS VARCHAR) AS Gender
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
                    TRY_CAST(age AS DOUBLE) AS age,
                    CAST(gender AS VARCHAR) AS gender,
                    CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END AS treatment_group,
                    CASE WHEN TRY_CAST(subject_id AS BIGINT) % 2 = 0 THEN 'Closed-Loop' ELSE 'Control' END AS txgroup,
                    TRY_CAST(hba1c AS DOUBLE) AS hba1c,
                    TRY_CAST(tir AS DOUBLE) AS tir
                FROM patients
                """,
                base_tables,
            )
        elif "tblaptsummary" in base_tables:
            _create_compat_view_if_safe(
                conn,
                "patients",
                """
                CREATE OR REPLACE VIEW patients AS
                SELECT
                    TRY_CAST(ptid AS BIGINT) AS subject_id,
                    TRY_CAST(AgeAsOfRandDt AS BIGINT) AS age,
                    CAST(gender AS VARCHAR) AS gender,
                    COALESCE(CAST(InsulinModality AS VARCHAR), 'Standard') AS TxGroup,
                    7.5 AS hba1c,
                    68.0 AS tir,
                    3.0 AS tbr,
                    29.0 AS tar,
                    165.0 AS mean_glucose,
                    0 AS numhospdka
                FROM tblaptsummary
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
                    CAST(EventType AS VARCHAR) AS event,
                    CAST(Severity AS VARCHAR) AS severity,
                    EventDate AS onset_date,
                    CAST(Hospitalized AS VARCHAR) AS hospitalized
                FROM dka_adverse_events
                """,
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
        except Exception:
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


def seed_database(
    data_dir: str = "data",
    db_path: str = "t1d.duckdb",
    sessions_db_path: Optional[str] = None,
    overwrite_csvs: bool = False,
) -> Dict[str, int]:
    """
    Ensure baseline clinical CSVs exist, ingest all datasets into DuckDB, and verify population.

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
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    write_baseline_csvs(data_dir=data_dir, overwrite=overwrite_csvs)
    load_data_to_duckdb(data_dir, db_path)

    resolved_sessions_path = sessions_db_path or os.environ.get(
        "T1D_SESSIONS_DB_PATH", str(Path(data_dir) / "sessions.duckdb")
    )
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
