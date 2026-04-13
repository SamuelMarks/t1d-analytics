"""FastAPI application for the T1D Analytics web interface."""

from typing import Any, Dict, List, Optional

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from t1d_analytics.analytics import get_database_schema

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
    db_path: str = "t1d.duckdb"


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    content: str
    sqlResult: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


def execute_sql(db_path: str, query: str) -> List[Dict[str, Any]]:
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
    try:
        conn = duckdb.connect(db_path)
        result = conn.execute(query)

        columns = [desc[0] for desc in result.description] if result.description else []
        rows = result.fetchall()
        output = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return output
    except Exception as e:
        raise ValueError(f"SQL execution error: {e}")


def generate_sql_from_nl(db_path: str, nl_query: str) -> str:
    """
    Translate natural language to SQL using any-llm.

    Args:
        db_path: Path to the DuckDB database file.
        nl_query: Natural language question.

    Returns:
        The generated SQL query string.

    Raises:
        RuntimeError: If any-llm-sdk is missing or LLM API fails.
        ValueError: If reading schema fails.

    """
    try:
        from any_llm import AnyLLM
    except ImportError:
        raise RuntimeError("any-llm-sdk[ollama] is not installed.")

    try:
        conn = duckdb.connect(db_path)
        schema = get_database_schema(conn)
        conn.close()
    except Exception as e:
        raise ValueError(f"Failed to read schema: {e}")

    prompt = f"You are a DuckDB SQL expert. Given the following database schema for Type 1 Diabetes (T1D) clinical trial datasets:\n\n{schema}\n\nWrite a SQL query that answers the user's request.\nReturn ONLY the raw SQL query, with no markdown formatting, no code blocks, and no explanations.\nThe query should be a valid DuckDB SQL SELECT statement.\n\nUser request: {nl_query}"

    try:
        llm = AnyLLM.create("ollama")
        response = llm.completion(
            model="gemma4",
            messages=[{"role": "user", "content": prompt}],
        )
        sql_query = response.choices[0].message.content.strip()

        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            if len(lines) > 2:
                sql_query = "\n".join(lines[1:-1])
            else:
                sql_query = sql_query.replace("```", "")

        if sql_query.lower().startswith("sql\n"):
            sql_query = sql_query[4:]
        elif sql_query.lower().startswith("duckdb\n"):
            sql_query = sql_query[7:]

        return sql_query.strip()
    except Exception as e:
        raise RuntimeError(f"LLM translation error: {e}")


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
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        if request.model == "sql":
            sql_query = request.message
            content_prefix = f"Executed literal SQL:\n```sql\n{sql_query}\n```"
        else:
            sql_query = generate_sql_from_nl(request.db_path, request.message)
            content_prefix = f"Generated SQL:\n```sql\n{sql_query}\n```"

        results = execute_sql(request.db_path, sql_query)

        return ChatResponse(content=content_prefix, sqlResult=results)
    except ValueError as ve:
        return ChatResponse(
            content="Error executing database operation.", error=str(ve)
        )
    except RuntimeError as re:
        return ChatResponse(content="Error during NLP translation.", error=str(re))
    except Exception as e:
        return ChatResponse(content="An unexpected error occurred.", error=str(e))
