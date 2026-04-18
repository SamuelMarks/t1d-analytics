"""FastAPI application for the T1D Analytics web interface."""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Union

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from t1d_analytics.analytics import get_database_schema

SqlValue = Union[str, int, float, bool, None]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="T1D Analytics API", description="API for querying T1D datasets.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    message: str
    model: str = "gemma4"
    db_path: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    content: str
    sqlResult: Optional[List[Dict[str, SqlValue]]] = None
    sqlQuery: Optional[str] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    size: Optional[int] = None


class ModelsResponse(BaseModel):
    """Response model for the models endpoint."""

    models: List[ModelInfo]


def execute_sql(db_path: str, query: str) -> List[Dict[str, SqlValue]]:
    """
    Execute a SQL query and return results as a list of dictionaries.

    Args:
        db_path: Path to the DuckDB database file.
        query: SQL query to execute.

    Returns:
        List of dictionaries representing the rows returned.

    Raises:
        ValueError: If there is an error executing the query.

    """
    logger.info(f"Executing SQL query against database at {db_path}:\n{query}")
    try:
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(query)

        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
        output = [dict(zip(columns, row)) for row in rows]
        conn.close()
        logger.info(f"SQL query executed successfully. Returned {len(output)} rows.")
        return output
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise ValueError(f"backend.sqlExecution|{e}")


def generate_sql_from_nl(
    db_path: str, nl_query: str, model_name: str = "gemma4"
) -> str:
    """
    Translate natural language to SQL using any-llm.

    Args:
        db_path: Path to the DuckDB database file.
        nl_query: Natural language question.
        model_name: The Ollama model to use.

    Returns:
        The generated SQL query string.

    Raises:
        RuntimeError: If any-llm-sdk is missing or LLM API fails.
        ValueError: If reading schema fails.

    """
    logger.info(
        f"Translating natural language to SQL: '{nl_query}' using model '{model_name}'"
    )
    try:
        from any_llm import AnyLLM
    except ImportError:
        logger.error("any-llm-sdk[ollama] is not installed.")
        raise RuntimeError("backend.missingSdk|any-llm-sdk[ollama]")

    try:
        conn = duckdb.connect(db_path, read_only=True)
        schema = get_database_schema(conn)
        conn.close()
        logger.debug(f"Retrieved schema: {schema[:200]}... (truncated)")
    except Exception as e:
        logger.error(f"Failed to read schema: {e}")
        raise ValueError(f"backend.readSchemaFailed|{e}")

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
You may provide an explanation of your thought process before or after the query.
Ensure the SQL query is enclosed in a markdown code block (e.g. ```sql ... ```).
The query should be a valid DuckDB SQL SELECT statement.

User request: {nl_query}"""

    try:
        logger.info(f"Sending prompt to LLM ({model_name})...")
        llm = AnyLLM.create("ollama")
        response = llm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        full_response = response.choices[0].message.content.strip()
        logger.info(f"LLM response:\n{full_response}")

        import re
        match = re.search(r"```(?:sql|duckdb)?\n?(.*?)\n?```", full_response, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(1).strip()
        else:
            # Fallback if no code block
            sql_query = full_response
            if sql_query.lower().startswith("sql\n"):
                sql_query = sql_query[4:]
            elif sql_query.lower().startswith("duckdb\n"):
                sql_query = sql_query[7:]
            sql_query = sql_query.replace("```", "").strip()

        return full_response, sql_query
    except Exception as e:
        logger.error(f"LLM translation error: {e}")
        raise RuntimeError(f"backend.llmTranslationError|{e}")


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Handle incoming chat messages and execute SQL or NLP queries.

    Args:
        request: The incoming ChatRequest.

    Returns:
        A ChatResponse indicating success or error.

    Raises:
        HTTPException: If request parameters are invalid.

    """
    if not request.message.strip():
        logger.warning("Received empty message.")
        raise HTTPException(status_code=400, detail="backend.emptyMessage")

    try:
        logger.info(
            f"Received chat request: model='{request.model}', message='{request.message}'"
        )
        db_path = request.db_path or os.environ.get("T1D_DB_PATH", "t1d.duckdb")
        if request.model == "sql":
            sql_query = request.message
            content_prefix = "backend.literalSql"
            results = execute_sql(db_path, sql_query)
            return ChatResponse(
                content=content_prefix, sqlResult=results, sqlQuery=sql_query
            )
        else:
            full_response, sql_query = generate_sql_from_nl(
                db_path, request.message, model_name=request.model
            )
            return ChatResponse(content=full_response, sqlQuery=sql_query)
    except ValueError as ve:
        logger.error(f"Database error during chat request: {ve}")
        return ChatResponse(
            content="backend.errorDbExecution", error=str(ve)
        )
    except RuntimeError as re:
        logger.error(f"NLP error during chat request: {re}")
        return ChatResponse(content="backend.errorNlpTranslation", error=str(re))
    except Exception as e:
        logger.exception("Unexpected error during chat request.")
        return ChatResponse(content="backend.errorUnexpected", error=str(e))


class TableDataResponse(BaseModel):
    """Response model for table data."""

    rows: List[Dict[str, SqlValue]]


@app.get("/api/table/{table_name}", response_model=TableDataResponse)
def get_table_data(
    table_name: str, limit: int = 25, offset: int = 0
) -> TableDataResponse:
    """Return paginated rows from a specific table."""
    db_path = os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    # Validate table name to prevent SQL injection
    if not table_name.isidentifier():
        raise HTTPException(status_code=400, detail="backend.invalidTable")

    try:
        conn = duckdb.connect(db_path, read_only=True)
        # Check if table exists
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if table_name not in tables:
            conn.close()
            raise HTTPException(status_code=404, detail="backend.tableNotFound")

        # Limit to reasonable maximum
        limit = min(limit, 1000)

        query = f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}"
        result = conn.execute(query)
        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
        output = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return TableDataResponse(rows=output)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching table data: {e}")
        raise HTTPException(status_code=500, detail=f"backend.serverError|{e}")


class ColumnInfo(BaseModel):
    """Information about a column in a table."""

    name: str
    type: str


class TableInfo(BaseModel):
    """Information about a table and its columns."""

    name: str
    columns: List[ColumnInfo]


class SchemaResponse(BaseModel):
    """Response model for database schema."""

    tables: List[TableInfo]


@app.get("/api/schema", response_model=SchemaResponse)
def get_schema() -> SchemaResponse:
    """Return structured schema for the frontend Schema Explorer."""
    db_path = os.environ.get("T1D_DB_PATH", "t1d.duckdb")
    try:
        conn = duckdb.connect(db_path, read_only=True)
        # Fetch tables
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        schema_data = []
        for table in tables:
            # Fetch columns for each table
            columns = conn.execute(f"DESCRIBE {table}").fetchall()
            col_info = [ColumnInfo(name=c[0], type=c[1]) for c in columns]
            schema_data.append(TableInfo(name=table, columns=col_info))

        conn.close()
        return SchemaResponse(tables=schema_data)
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
        return SchemaResponse(tables=[])


@app.get("/api/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """
    List available local Ollama models.

    Returns:
        A list of available models.

    """
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        models = []
        for model in data.get("models", []):
            models.append(ModelInfo(name=model.get("name"), size=model.get("size")))

        return ModelsResponse(models=models)
    except urllib.error.URLError as e:
        logger.error(f"Failed to connect to local Ollama instance: {e}")
        return ModelsResponse(models=[ModelInfo(name="gemma4")])  # Fallback to default
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return ModelsResponse(models=[ModelInfo(name="gemma4")])  # Fallback
