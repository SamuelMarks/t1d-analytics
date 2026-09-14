"""Analytics module for querying downloaded T1D datasets."""

import csv
import json
import os
import re
import zipfile
from pathlib import Path

import duckdb

from t1d_analytics.diagnostics import DatabaseStatusCode, check_database_health
from t1d_analytics.i18n import get_translator


def extract_zips(data_dir_str: str) -> None:
    """
    Extract any zip files found in the data directory safely without path traversal.

    Args:
    ----
        data_dir_str: Directory path containing zip files to extract.

    """
    _ = get_translator()
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print(_("Data directory {} does not exist.", data_dir))
        return

    print(_("Checking for zip files to extract..."))
    found_zips = False
    for path in data_dir.rglob("*.zip"):
        found_zips = True
        extract_dir = path.with_suffix("")
        if not extract_dir.exists():
            print(_("Extracting {}...", path.name))
            extract_dir.mkdir(parents=True, exist_ok=True)
            resolved_extract_dir = extract_dir.resolve()
            try:
                with zipfile.ZipFile(path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        target_path = (resolved_extract_dir / member).resolve()
                        if not str(target_path).startswith(str(resolved_extract_dir)):
                            print(
                                _(
                                    "Skipping unsafe entry {} in {} (path traversal detected).",
                                    member,
                                    path.name,
                                )
                            )
                            continue
                        zip_ref.extract(member, extract_dir)
            except zipfile.BadZipFile:
                print(_("Failed to extract {} (Bad Zip File).", path.name))

    if not found_zips:
        print(_("No zip files found to extract."))
    else:
        print(_("Extraction complete."))


def load_data_to_duckdb(data_dir_str: str, db_path: str) -> None:
    """
    Load all CSV and TXT files from the data directory into DuckDB tables.

    Args:
    ----
        data_dir_str: Directory containing extracted CSV/TXT datasets.
        db_path: Path to the target DuckDB database file.

    """
    _ = get_translator()
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print(_("Data directory {} does not exist.", data_dir))
        return

    print(_("Connecting to DuckDB at {}...", db_path))
    conn = duckdb.connect(db_path, read_only=False)

    print(_("Scanning for CSV and TXT files..."))
    data_files = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.txt"))
    # Filter out MacOS hidden files or unhelpful docs
    data_files = [
        f
        for f in data_files
        if not f.name.startswith("._")
        and "Glossary" not in f.name
        and "Protocol" not in f.name
    ]

    if not data_files:
        print(_("No tabular files found."))
        conn.close()
        return

    # Load reverse map for column renaming
    reverse_map = {}
    matches_file = Path(__file__).parent / "likely_matches.json"
    if matches_file.exists():
        with open(matches_file, "r") as f:
            matches = json.load(f)
            for std_name, orig_names in matches.items():
                for orig in orig_names:
                    reverse_map[orig.lower()] = std_name

    loaded_tables = []
    for data_file in data_files:
        # Create a safe table name from the file name
        safe_stem = re.sub(r"[^a-zA-Z0-9_]", "_", data_file.stem).lower()
        table_name = safe_stem.strip("_") or "dataset"
        # Ensure it starts with a letter
        if not table_name[0].isalpha():
            table_name = "t_" + table_name

        try:
            # 1. Detect Encoding (utf-8-sig vs utf-16 vs utf-8)
            with open(data_file, "rb") as f_rb:
                raw_bytes = f_rb.read(4)
            if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
                py_encoding = "utf-16"
                duckdb_encoding = "utf-16"
            elif raw_bytes[:3] == b"\xef\xbb\xbf":
                py_encoding = "utf-8-sig"
                duckdb_encoding = "utf-8"
            else:
                py_encoding = "utf-8"
                duckdb_encoding = "utf-8"

            # 2. Detect Separator using csv.Sniffer with fallback
            sep = ","
            try:
                with open(data_file, "r", encoding=py_encoding, errors="ignore") as f:
                    sample = "".join([f.readline() for _ in range(5)])
                if sample.strip():
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
                    sep = dialect.delimiter
            except Exception:
                with open(data_file, "r", encoding=py_encoding, errors="ignore") as f:
                    first_line = f.readline()
                if "|" in first_line:
                    sep = "|"
                elif "\t" in first_line:
                    sep = "\t"

            duckdb_sep = "\\t" if sep == "\t" else sep

            print(
                _(
                    "Loading {} (encoding={}, sep='{}') into table {}...",
                    data_file.name,
                    duckdb_encoding,
                    duckdb_sep,
                    table_name,
                )
            )

            # Create temp view to inspect columns
            escaped_path = str(data_file).replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM read_csv('{escaped_path}', auto_detect=true, sep='{duckdb_sep}', encoding='{duckdb_encoding}')"
            )
            cols = conn.execute("DESCRIBE temp_view").fetchall()

            select_exprs = []
            seen_std_names = set()
            for c in cols:
                orig_col = c[0]
                std_name = reverse_map.get(orig_col.lower(), orig_col)

                # Failsafe: avoid duplicate column names in the same table
                if std_name.lower() in seen_std_names:
                    std_name = orig_col  # fallback to original if collision still happens somehow

                # If still duplicate (e.g. original name conflicts with a mapped name)
                counter = 1
                base_std_name = std_name
                while std_name.lower() in seen_std_names:
                    counter += 1
                    std_name = f"{base_std_name}_{counter}"

                seen_std_names.add(std_name.lower())

                escaped_orig = orig_col.replace('"', '""')
                escaped_std = std_name.replace('"', '""')
                if std_name != orig_col:
                    select_exprs.append(f'"{escaped_orig}" AS "{escaped_std}"')
                else:
                    select_exprs.append(f'"{escaped_orig}"')

            select_sql = ",\n                ".join(select_exprs)

            # Check if table already exists to make script idempotent
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            if table_name in tables:
                print(_("Table {} already exists, skipping...", table_name))
                conn.execute("DROP VIEW temp_view")
                continue

            query = (
                f'CREATE TABLE "{table_name}" AS SELECT \n'
                f"                {select_sql} \n"
                f"            FROM temp_view"
            )
            conn.execute(query)
            conn.execute("DROP VIEW temp_view")

            loaded_tables.append(table_name)
        except Exception as e:
            print(_("Failed to load {}: {}", data_file.name, e))

    if loaded_tables:
        print(_("Successfully populated {} tables in {}.", len(loaded_tables), db_path))
    conn.close()


def get_database_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """
    Extract the database schema as a text description, including sample data.

    Args:
    ----
        conn: The DuckDB connection.

    Returns:
    -------
        The database schema string.

    """
    tables = conn.execute("SHOW TABLES").fetchall()
    schema_parts = []
    for (table_name,) in tables:
        columns = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        col_desc = ", ".join([f"{c[0]} ({c[1]})" for c in columns])

        try:
            sample = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 1').fetchone()
            sample_desc = (
                f"Sample row: {sample}" if sample else "Sample row: (empty table)"
            )
        except Exception:
            sample_desc = "Sample row: (could not fetch)"

        schema_parts.append(f"Table: {table_name}\nColumns: {col_desc}\n{sample_desc}")
    return "\n\n".join(schema_parts)


def is_sql_query(query: str) -> bool:
    """
    Determine whether a user input string is a SQL query rather than natural language.

    Args:
    ----
        query: The raw query string entered by the user.

    Returns:
    -------
        True if the query appears to be a SQL statement, False if natural language.

    """
    # Strip SQL comments (-- ... and /* ... */) and whitespace
    cleaned = re.sub(r"--[^\n]*\n?", " ", query)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        return False

    first_word = cleaned.split()[0].lower()
    sql_keywords = ("select", "with", "describe", "pragma", "explain")
    if first_word in sql_keywords:
        return True

    if first_word == "show":
        words = cleaned.lower().split()
        nl_indicators = {
            "me",
            "the",
            "a",
            "an",
            "us",
            "our",
            "all",
            "how",
            "what",
            "where",
            "which",
            "any",
            "patients",
            "trials",
            "datasets",
        }
        if len(words) > 1 and words[1] in nl_indicators:
            return False
        return True

    return False


def handle_natural_language(
    conn: duckdb.DuckDBPyConnection, query: str, model: str = "gemma4"
) -> None:
    """
    Translate natural language to SQL using a local LLM via any-llm and execute it.

    Args:
    ----
        conn: DB connection.
        query: The natural language query.
        model: Model name for translation.

    """
    _ = get_translator()
    try:
        from any_llm import AnyLLM  # type: ignore
    except ImportError:
        print(_("Error: any-llm-sdk[ollama] is not installed. Please install it."))
        return

    print(_("Thinking..."))
    schema = get_database_schema(conn)

    prompt = f"""You are a DuckDB SQL expert. Given the following database schema for Type 1 Diabetes (T1D) clinical trial datasets:

{schema}

Context & Rules for T1D Analytics:
1. Terminology & Aliases:
   - "Lows" / "Hypoglycemia": Look for Time Below Range (TBR, <70 or <54 mg/dL), `pct_time_under_70`, `hypo`.
   - "Highs" / "Hyperglycemia": Look for Time Above Range (TAR, >180 or >250 mg/dL), `pct_time_over_180`, `hyper`.
   - "In-range" / "TIR": Look for Time in Range (70-180 mg/dL), `pct_time_in_range_70_180`, `tir`, `cgm_in_range`.
   - "A1C" / "HbA1c": Look for glycated hemoglobin, `a1c`, `hba1c`.
   - "Variability": Look for Coefficient of Variation (`cv`), Standard Deviation (`sd`).
   - "Average sugar": Look for Mean Glucose, `mean_cgm`, Glucose Management Indicator (`gmi`).
   - "Adverse Events": "DKA" (Diabetic Ketoacidosis), "SH" (Severe Hypoglycemia).
   - Demographics: "Duration" (years with diabetes), "Age", "BMI", "Gender/Sex", "Weight".
   - Treatment: "Pump" (CSII), "Injections/Shots" (MDI), "Loop" (AID/Closed-Loop), "Insulin" (TDD, Basal, Bolus).
   - Study Timepoints: "Baseline" (visit 0 or baseline flag), "End of trial/Follow-up" (max visit month).
2. SQL Generation Rules for Multi-Trial Data:
   - Trial Names: Acronyms (DCLP3, DCLP5, Pedap, etc.) usually correspond directly to table names.
   - Aggregating Combined Trials: To calculate metrics for multiple trials "together", "combined", or "overall", you MUST use a CTE with `UNION ALL` to combine raw rows from all requested tables *before* applying aggregate functions like `AVG()`.
   - Comparing Individual Trials: To calculate metrics for multiple trials "individually" or "each", use `UNION ALL` with an identifying literal string column (e.g., `SELECT 'DCLP3' as trial, AVG(...) ... UNION ALL SELECT 'DCLP5'...`).
   - Aligning UNION Columns: Clinical trial tables often have different schemas. When using `UNION ALL`, you MUST explicitly `SELECT` and alias only the specific columns needed (e.g. `age`, `tir`) to ensure they align perfectly in number, order, and type across all blocks. NEVER use `SELECT *` with `UNION ALL`.
   - Linking Data: If questions span patient demographics and CGM outcomes, use `JOIN` on the patient/subject ID columns (e.g., `pt_id`, `subject_id`).
3. Formatting & Logic:
   - Handle colloquial/informal language intuitively (e.g., "betwixt" -> `BETWEEN`, "kids/pediatric" -> `age < 18`, "adults" -> `age >= 18`).
   - Account for string vs. numeric data types; use `CAST(col AS FLOAT)` if computing averages on textual columns that contain numbers.
   - Ignore NULL values appropriately when calculating averages or sums.
4. Strict Schema Enforcement:
   - CRITICAL: You must ONLY use the exact table names and column names provided in the schema above.
   - Do NOT invent, guess, or hallucinate table names (e.g., do not use "Visit1" or "demographics" if they are not in the schema).
   - If a requested concept (like "demographics") is not a table, find the relevant columns within the existing tables (e.g., `visits` or `patients`).

Write a SQL query that answers the user's request. 
You may provide an explanation of your thought process as SQL comments (`--`) before the query.
Return ONLY the raw SQL query, with no markdown formatting and no code blocks.
The query should be a valid DuckDB SQL SELECT statement.

User request: {query}
"""
    try:
        api_base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not api_base.startswith("http://") and not api_base.startswith("https://"):
            api_base = f"http://{api_base}"

        llm = AnyLLM.create("ollama")
        response = llm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        sql_query = response.choices[0].message.content.strip()

        # Remove potential markdown block if the model ignores instructions
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            if len(lines) > 2:
                sql_query = "\n".join(lines[1:-1])
            else:
                sql_query = sql_query.replace("```", "")

        # Clean up if model still outputs 'sql'
        if sql_query.lower().startswith("sql\n"):
            sql_query = sql_query[4:]
        elif sql_query.lower().startswith("duckdb\n"):
            sql_query = sql_query[7:]

        print(_("Generated SQL: \n{}\n", sql_query))
        print(_("Executing...\n"))

        result = conn.sql(sql_query)
        if result:
            result.show()

    except Exception as e:
        print(_("Failed to generate or execute query: {}", e))


def run_query_repl(db_path: str, model: str = "gemma4") -> None:
    """
    Run the interactive query interface.

    Args:
    ----
        db_path: Path to DB.
        model: Local LLM model name for query translation.

    """
    _ = get_translator()
    if not Path(db_path).exists():
        print(_("Database {} does not exist.", db_path))
        print(_("Please run the 'load' command first to populate the database."))
        return

    db_health = check_database_health(db_path)
    if db_health.status_code == DatabaseStatusCode.EMPTY_DB:
        print(_("Warning: Database {} contains 0 tables.", db_path))
        print(_("Please run the 'load' command first to populate the database."))
    elif db_health.status_code == DatabaseStatusCode.MISSING_INITIAL_DATA:
        print(
            _(
                "Warning: Database {} lacks standard initial clinical trial datasets.",
                db_path,
            )
        )

    print(_("Connecting to DuckDB at {}...", db_path))
    conn = duckdb.connect(db_path, read_only=True)

    print("\n" + "=" * 50)
    print(_("Welcome to T1D Analytics Interface!"))
    print(_("You can enter:"))
    print(
        _("  - Standard SQL queries (starting with SELECT, WITH, SHOW, DESCRIBE, etc.)")
    )
    print(_("  - Natural language queries (will be translated to SQL via LLM)"))
    print(_("  - 'exit' or 'quit' to close."))
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + _("Exiting."))
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in ("exit", "quit"):
            print(_("Exiting."))
            break

        if is_sql_query(user_input):
            try:
                result = conn.sql(user_input)
                # Show results nicely
                if result:
                    result.show()
            except Exception as e:
                print(_("SQL Error: {}", e))
        else:
            handle_natural_language(conn, user_input, model=model)

    conn.close()
