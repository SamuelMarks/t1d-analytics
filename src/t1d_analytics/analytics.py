"""Analytics module for querying downloaded T1D datasets."""

import zipfile
from pathlib import Path

import duckdb


def extract_zips(data_dir_str: str) -> None:
    """Extract any zip files found in the data directory."""
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return

    print("Checking for zip files to extract...")
    found_zips = False
    for path in data_dir.rglob("*.zip"):
        found_zips = True
        extract_dir = path.with_suffix("")
        if not extract_dir.exists():
            print(f"Extracting {path.name}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                print(f"Failed to extract {path.name} (Bad Zip File).")

    if not found_zips:
        print("No zip files found to extract.")
    else:
        print("Extraction complete.")


def load_data_to_duckdb(data_dir_str: str, db_path: str) -> None:
    """Load all CSV and TXT files from the data directory into DuckDB tables."""
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return

    print(f"Connecting to DuckDB at {db_path}...")
    conn = duckdb.connect(db_path)

    print("Scanning for CSV and TXT files...")
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
        print("No tabular files found.")
        conn.close()
        return

    loaded_tables = []
    for data_file in data_files:
        # Create a safe table name from the file name
        table_name = data_file.stem.replace(" ", "_").replace("-", "_").lower()
        # Ensure it starts with a letter
        if not table_name[0].isalpha():
            table_name = "t_" + table_name

        try:
            # 1. Detect Encoding (utf-16 vs utf-8)
            with open(data_file, "rb") as f:
                raw_bytes = f.read(2)
            encoding = "utf-16" if raw_bytes == b"\xff\xfe" else "utf-8"

            # 2. Detect Separator
            with open(data_file, "r", encoding=encoding, errors="ignore") as f:
                first_line = f.readline()

            sep = ","
            if "|" in first_line:
                sep = "|"
            elif "\t" in first_line:
                sep = "\\t"

            print(
                f"Loading {data_file.name} (encoding={encoding}, sep='{sep}') into table {table_name}..."
            )

            query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv('{str(data_file)}', auto_detect=true, sep='{sep}', encoding='{encoding}')"
            conn.execute(query)
            loaded_tables.append(table_name)
        except Exception as e:
            print(f"Failed to load {data_file.name}: {e}")

    if loaded_tables:
        print(f"Successfully populated {len(loaded_tables)} tables in {db_path}.")
    conn.close()


def get_database_schema(conn: duckdb.DuckDBPyConnection) -> str:
    """Extract the database schema as a text description, including sample data."""
    tables = conn.execute("SHOW TABLES").fetchall()
    schema_parts = []
    for (table_name,) in tables:
        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
        col_desc = ", ".join([f"{c[0]} ({c[1]})" for c in columns])

        try:
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchone()
            sample_desc = (
                f"Sample row: {sample}" if sample else "Sample row: (empty table)"
            )
        except Exception:
            sample_desc = "Sample row: (could not fetch)"

        schema_parts.append(f"Table: {table_name}\nColumns: {col_desc}\n{sample_desc}")
    return "\n\n".join(schema_parts)


def handle_natural_language(conn: duckdb.DuckDBPyConnection, query: str) -> None:
    """Translate natural language to SQL using a local LLM via any-llm and execute it."""
    try:
        from any_llm import AnyLLM
    except ImportError:
        print("Error: any-llm-sdk[ollama] is not installed. Please install it.")
        return

    print("Thinking...")
    schema = get_database_schema(conn)

    prompt = f"""You are a DuckDB SQL expert. Given the following database schema for Type 1 Diabetes (T1D) clinical trial datasets:

{schema}

Write a SQL query that answers the user's request. 
Return ONLY the raw SQL query, with no markdown formatting, no code blocks, and no explanations.
The query should be a valid DuckDB SQL SELECT statement.

User request: {query}
"""
    try:
        llm = AnyLLM.create("ollama")
        response = llm.completion(
            model="gemma4",
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

        print(f"Generated SQL: \n{sql_query}\n")
        print("Executing...\n")

        result = conn.sql(sql_query)
        if result:
            result.show()

    except Exception as e:
        print(f"Failed to generate or execute query: {e}")


def run_query_repl(db_path: str) -> None:
    """Run the interactive query interface."""
    if not Path(db_path).exists():
        print(f"Database {db_path} does not exist.")
        print("Please run the 'load' command first to populate the database.")
        return

    print(f"Connecting to DuckDB at {db_path}...")
    conn = duckdb.connect(db_path)

    print("\n" + "=" * 50)
    print("Welcome to T1D Analytics Interface!")
    print("You can enter:")
    print("  - Standard SQL queries (starting with SELECT, WITH, SHOW, DESCRIBE, etc.)")
    print("  - Natural language queries (will be translated to SQL via LLM)")
    print("  - 'exit' or 'quit' to close.")
    print("=" * 50 + "\n")

    sql_keywords = ("select", "with", "show", "describe", "pragma")

    while True:
        try:
            user_input = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in ("exit", "quit"):
            print("Exiting.")
            break

        if lower_input.startswith(sql_keywords):
            try:
                result = conn.sql(user_input)
                # Show results nicely
                if result:
                    result.show()
            except Exception as e:
                print(f"SQL Error: {e}")
        else:
            handle_natural_language(conn, user_input)

    conn.close()
