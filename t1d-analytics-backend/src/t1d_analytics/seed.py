"""Deterministic synthetic T1D clinical dataset generator, session transcript sanitizer, and database seeder."""

import csv
import json
import shutil
from pathlib import Path

import duckdb
from pydantic import ValidationError

from t1d_analytics.api_schemas import ChatMessageModel
from t1d_analytics.ingestion import load_data_to_duckdb
from t1d_analytics.jaeb_cohort_sql import build_jaeb_beck2023_cohort_table
from t1d_analytics.seed_verifier import (
    BASE_TABLE_NUMERIC_COLUMNS,
    BASELINE_CLINICAL_TABLES,
    COMPATIBILITY_CLINICAL_VIEWS,
    collect_jaeb_table_counts,
    ensure_canonical_clinical_views,
    find_authentic_jaeb_source_db,
    is_jaeb_authentic_duckdb_file,
    resolve_non_empty_db_path,
    verify_seeded_clinical_tables,
)
from t1d_analytics.session_store import open_sessions_db_conn
from t1d_analytics.sql_utils import global_read_only_duckdb_pool

SeedValue = str | int | float


def generate_baseline_patients() -> list[dict[str, SeedValue]]:
    """
    Generate a deterministic cohort of 100 representative T1D clinical trial patients.

    Returns:
        list[dict[str, SeedValue]]: List of patient row dictionaries covering canonical T1D columns.

    """
    rows: list[dict[str, SeedValue]] = []
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


def generate_baseline_cgm_readings() -> list[dict[str, SeedValue]]:
    """
    Generate deterministic continuous glucose monitoring (CGM) traces across patients.

    Returns:
        list[dict[str, SeedValue]]: List of CGM reading dictionaries.

    """
    rows: list[dict[str, SeedValue]] = []
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


def generate_baseline_hba1c_labs() -> list[dict[str, SeedValue]]:
    """
    Generate deterministic longitudinal HbA1c and C-Peptide laboratory measurements.

    Returns:
        list[dict[str, SeedValue]]: List of lab measurement dictionaries.

    """
    rows: list[dict[str, SeedValue]] = []
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


def generate_baseline_dka_events() -> list[dict[str, SeedValue]]:
    """
    Generate deterministic Diabetic Ketoacidosis (DKA) and severe hypoglycemia events.

    Returns:
        list[dict[str, SeedValue]]: List of adverse event dictionaries.

    """
    rows: list[dict[str, SeedValue]] = []
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


def write_baseline_csvs(data_dir: str = "data", overwrite: bool = False) -> list[Path]:
    """
    Write baseline clinical CSV files into the target data directory if missing or overwrite is True.

    Args:
        data_dir: Directory where CSV files should be written.
        overwrite: Whether to overwrite existing CSV files of the same name.

    Returns:
        list[Path]: Paths to the baseline CSV files ensured on disk.

    """
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[dict[str, SeedValue]]] = {
        "patients.csv": generate_baseline_patients(),
        "cgm_readings.csv": generate_baseline_cgm_readings(),
        "hba1c_labs.csv": generate_baseline_hba1c_labs(),
        "dka_adverse_events.csv": generate_baseline_dka_events(),
    }

    written_paths: list[Path] = []
    for filename, rows in datasets.items():
        file_path = out_dir / filename
        if rows and (overwrite or not file_path.exists()):
            fieldnames = list(rows[0].keys())
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        written_paths.append(file_path)

    return written_paths


def _is_legacy_sum_varchar_error_message(msg: ChatMessageModel) -> bool:
    """
    Determine whether a `ChatMessageModel` is a legacy `sum(VARCHAR)` or `avg(VARCHAR)` Binder Error.

    Args:
        msg: Validated `ChatMessageModel` instance.

    Returns:
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
    messages: list[ChatMessageModel],
    session_id: str = "",
) -> tuple[list[ChatMessageModel], bool]:
    """
    Filter out legacy `sum(VARCHAR)` / `NumSevHypo` binder error messages from a session transcript.

    Args:
        messages: Ordered list of `ChatMessageModel` items in the session.
        session_id: Optional session identifier for `session-default-welcome` checks.

    Returns:
        tuple[list[ChatMessageModel], bool]: `(cleaned_messages, was_modified)`.

    """
    cleaned: list[ChatMessageModel] = []
    changed = False
    total = len(messages)
    for idx, item in enumerate(messages):
        is_welcome_err = session_id == "session-default-welcome" and (
            bool(item.is_error) or "binder error" in item.content.lower()
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

    Args:
        conn: Open writable DuckDB connection to the sessions database.

    Returns:
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
        sessions_db_path: Path to the persistent `sessions.duckdb` file.

    Returns:
        int: Number of legacy error sessions sanitized.

    """
    from t1d_analytics.db_security import validate_sessions_db_path

    validated_path = validate_sessions_db_path(sessions_db_path)
    Path(validated_path).parent.mkdir(parents=True, exist_ok=True)
    sess_conn = open_sessions_db_conn(db_path=validated_path)
    try:
        return sanitize_legacy_error_sessions(sess_conn)
    finally:
        sess_conn.close()


def seed_database(
    data_dir: str = "data",
    db_path: str = "t1d.duckdb",
    sessions_db_path: str | None = None,
    overwrite_csvs: bool = False,
) -> dict[str, int]:
    """
    Ensure baseline clinical CSVs exist, ingest all datasets into DuckDB, and verify population.

    Args:
        data_dir: Directory containing clinical datasets.
        db_path: Path to the target DuckDB analytics database file.
        sessions_db_path: Optional path to the persistent chat sessions DuckDB file.
        overwrite_csvs: Whether to overwrite existing baseline CSV files.

    Returns:
        dict[str, int]: Mapping of clinical table names to row counts in the populated DuckDB file.

    Raises:
        SeedVerificationError: If fewer than 4 tables exist or any required clinical table has 0 rows.

    """
    db_path = resolve_non_empty_db_path(db_path)
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_sessions_path = resolve_non_empty_db_path(
        sessions_db_path,
        env_var="T1D_SESSIONS_DB_PATH",
        default_path=str(Path(data_dir) / "sessions.duckdb"),
    )

    is_jaeb = is_jaeb_authentic_duckdb_file(db_file)
    if not is_jaeb and not overwrite_csvs:
        jaeb_source = find_authentic_jaeb_source_db(db_file)
        if jaeb_source is not None:
            global_read_only_duckdb_pool.close_path(str(db_path))
            shutil.copyfile(jaeb_source, db_file)
            is_jaeb = True

    if is_jaeb:
        seed_sessions_database(resolved_sessions_path)
        ensure_canonical_clinical_views(str(db_path))
        return collect_jaeb_table_counts(str(db_path))

    write_baseline_csvs(data_dir=data_dir, overwrite=overwrite_csvs)
    load_data_to_duckdb(data_dir, db_path)

    seed_sessions_database(resolved_sessions_path)
    table_counts = verify_seeded_clinical_tables(db_path)
    ensure_canonical_clinical_views(db_path)

    return table_counts


__all__ = [
    "BASELINE_CLINICAL_TABLES",
    "BASE_TABLE_NUMERIC_COLUMNS",
    "COMPATIBILITY_CLINICAL_VIEWS",
    "SeedValue",
    "build_jaeb_beck2023_cohort_table",
    "ensure_canonical_clinical_views",
    "filter_legacy_error_messages",
    "generate_baseline_cgm_readings",
    "generate_baseline_dka_events",
    "generate_baseline_hba1c_labs",
    "generate_baseline_patients",
    "sanitize_legacy_error_sessions",
    "seed_database",
    "seed_sessions_database",
    "write_baseline_csvs",
]
