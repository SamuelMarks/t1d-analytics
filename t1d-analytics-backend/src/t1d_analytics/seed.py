"""Deterministic T1D clinical dataset generator and DuckDB database seeder."""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import duckdb

from t1d_analytics.analytics import load_data_to_duckdb
from t1d_analytics.session_store import open_sessions_db_conn

BASELINE_CLINICAL_TABLES = (
    "patients",
    "cgm_readings",
    "hba1c_labs",
    "dka_adverse_events",
)


def generate_baseline_patients() -> List[Dict[str, object]]:
    """
    Generate a deterministic cohort of 100 representative T1D clinical trial patients.

    Returns
    -------
        List[Dict[str, object]]: List of patient row dictionaries covering canonical T1D columns.

    """
    rows: List[Dict[str, object]] = []
    for i in range(1, 101):
        age = 8 + ((i * 7) % 62)
        gender = "F" if i % 2 == 0 else "M"
        hba1c = round(5.8 + ((i * 13) % 45) * 0.1, 1)
        tir = round(max(30.0, min(92.0, 95.0 - (hba1c - 5.5) * 10.5)), 1)
        tbr = round(1.0 + ((i * 3) % 60) * 0.1, 1)
        tar = round(max(0.0, 100.0 - tir - tbr), 1)
        mean_cgm = round(95.0 + (hba1c - 5.0) * 28.5, 1)
        num_dka = 1 if (hba1c >= 9.0 and i % 3 == 0) else (2 if i == 99 else 0)
        rows.append(
            {
                "PtID": i,
                "Age": age,
                "Gender": gender,
                "HbA1c": hba1c,
                "TIR": tir,
                "TBR": tbr,
                "TAR": tar,
                "MeanCGM": mean_cgm,
                "Pt_NumHospDKA": num_dka,
            }
        )
    return rows


def generate_baseline_cgm_readings() -> List[Dict[str, object]]:
    """
    Generate deterministic continuous glucose monitoring (CGM) traces across patients.

    Returns
    -------
        List[Dict[str, object]]: List of CGM reading dictionaries.

    """
    rows: List[Dict[str, object]] = []
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


def generate_baseline_hba1c_labs() -> List[Dict[str, object]]:
    """
    Generate deterministic longitudinal HbA1c and C-Peptide laboratory measurements.

    Returns
    -------
        List[Dict[str, object]]: List of lab measurement dictionaries.

    """
    rows: List[Dict[str, object]] = []
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


def generate_baseline_dka_events() -> List[Dict[str, object]]:
    """
    Generate deterministic Diabetic Ketoacidosis (DKA) and severe hypoglycemia events.

    Returns
    -------
        List[Dict[str, object]]: List of adverse event dictionaries.

    """
    rows: List[Dict[str, object]] = []
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

    datasets: Dict[str, List[Dict[str, object]]] = {
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
    sess_conn = open_sessions_db_conn(db_path=resolved_sessions_path)
    sess_conn.close()

    conn = duckdb.connect(db_path, read_only=True)
    try:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        user_tables = [t for t in tables if not t.startswith("_t1d_")]
        table_counts: Dict[str, int] = {}
        for t_name in user_tables:
            safe_name = t_name.replace('"', '""')
            cnt_row = conn.execute(f'SELECT COUNT(*) FROM "{safe_name}"').fetchone()
            row_count = int(cnt_row[0]) if cnt_row else 0
            table_counts[t_name] = row_count
    finally:
        conn.close()

    if len(table_counts) < 4:
        raise RuntimeError(
            f"Seed verification failed for {db_path}: expected >= 4 clinical tables, found {len(table_counts)}."
        )

    for req_table in BASELINE_CLINICAL_TABLES:
        if table_counts.get(req_table, 0) <= 0:
            raise RuntimeError(
                f"Seed verification failed for {db_path}: required table '{req_table}' is missing or empty."
            )

    return table_counts
