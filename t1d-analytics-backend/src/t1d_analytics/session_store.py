"""DuckDB-backed persistent storage for chat conversation sessions."""

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import duckdb

from t1d_analytics.api_schemas import (
    ChatMessageModel,
    CreateSessionRequest,
    SessionListResponse,
    SessionModel,
)
from t1d_analytics.errors import SessionNotFoundError
from t1d_analytics.sql_utils import global_read_only_duckdb_pool


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


def open_sessions_db_conn(
    db_path: Optional[str] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> duckdb.DuckDBPyConnection:
    """
    Open connection to persistent chat sessions database and ensure schema exists.

    Args:
    ----
        db_path: Optional path to the database file.
        sleep_fn: Injectable sleep function for lock retry backoff.

    Returns:
    -------
        duckdb.DuckDBPyConnection: Database connection.

    """
    if db_path is None:
        env_db = os.environ.get("T1D_SESSIONS_DB") or os.environ.get(
            "T1D_SESSIONS_DB_PATH"
        )
        if env_db:
            db_path = env_db
        else:
            sessions_dir = Path("./data")
            sessions_dir.mkdir(parents=True, exist_ok=True)
            db_path = str((sessions_dir / "sessions.duckdb").resolve())
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    global_read_only_duckdb_pool.close_path(db_path)
    max_retries = 5
    conn: Optional[duckdb.DuckDBPyConnection] = None
    for attempt in range(max_retries - 1):
        try:
            conn = duckdb.connect(db_path, read_only=False)
            break
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                sleep_fn(0.05 * (attempt + 1))
            else:
                raise
    else:
        conn = duckdb.connect(db_path, read_only=False)

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR PRIMARY KEY,
                title VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                messages_json VARCHAR
            )
            """
        )
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def managed_sessions_db_conn(
    db_path: Optional[str] = None,
    open_fn: Callable[
        [Optional[str]], duckdb.DuckDBPyConnection
    ] = open_sessions_db_conn,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager wrapping `open_sessions_db_conn` that guarantees `conn.close()` on exit.

    Args:
    ----
        db_path: Optional path to the sessions database file.
        open_fn: Injectable callable opening the sessions DuckDB connection.

    Yields:
    ------
        duckdb.DuckDBPyConnection: Open DuckDB connection to the sessions database.

    """
    conn = open_fn(db_path)
    try:
        yield conn
    finally:
        conn.close()


class SessionStore:
    """Thread-safe repository managing persistent chat sessions in DuckDB."""

    def __init__(
        self,
        conn_factory: Callable[
            [Optional[str]], ContextManager[duckdb.DuckDBPyConnection]
        ] = managed_sessions_db_conn,
        lock: Optional[threading.Lock] = None,
    ) -> None:
        """
        Initialize the SessionStore with a connection factory and synchronization lock.

        Args:
        ----
            conn_factory: Context-manager factory yielding an active DuckDB connection.
            lock: Optional threading.Lock to synchronize concurrent writes.

        """
        self._lock = lock or threading.Lock()
        self._conn_factory = conn_factory

    @contextmanager
    def connect(
        self, db_path: Optional[str] = None
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        """
        Yield an active DuckDB connection with the chat_sessions table initialized.

        Args:
        ----
            db_path: Optional path to the DuckDB database file.

        Yields:
        ------
            duckdb.DuckDBPyConnection: Active DuckDB connection.

        """
        with self._conn_factory(db_path) as conn:
            yield conn

    @staticmethod
    def _deserialize_messages(
        messages_json: str,
        session_id: str = "",
    ) -> List[ChatMessageModel]:
        """
        Parse a JSON string of chat messages into a list of ChatMessageModel instances.

        Args:
        ----
            messages_json: Raw JSON string from DuckDB.
            session_id: Optional session ID used to sanitize legacy error messages.

        Returns:
        -------
            List[ChatMessageModel]: Deserialized messages, or empty list on parse failure.

        """
        try:
            msgs_raw = json.loads(messages_json)
            models = [ChatMessageModel(**m) for m in msgs_raw]
            cleaned, _ = filter_legacy_error_messages(models, session_id=session_id)
            return cleaned
        except Exception:
            return []

    def list_sessions(self, db_path: Optional[str] = None) -> SessionListResponse:
        """
        List all persistent conversation sessions sorted by updated_at descending.

        Args:
        ----
            db_path: Optional path to the sessions database.

        Returns:
        -------
            SessionListResponse: List of stored conversation sessions.

        """
        with self._lock:
            with self.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT session_id, title, strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S'), messages_json FROM chat_sessions ORDER BY updated_at DESC"
                ).fetchall()

        sessions: List[SessionModel] = [
            SessionModel(
                session_id=r[0],
                title=r[1],
                created_at=str(r[2]),
                updated_at=str(r[3]),
                messages=self._deserialize_messages(r[4], session_id=str(r[0])),
            )
            for r in rows
        ]
        return SessionListResponse(sessions=sessions)

    def upsert_session(
        self, request: CreateSessionRequest, db_path: Optional[str] = None
    ) -> SessionModel:
        """
        Create or update a persistent conversation session.

        Args:
        ----
            request: Session creation or update payload.
            db_path: Optional path to the sessions database.

        Returns:
        -------
            SessionModel: The saved conversation session.

        """
        session_id = request.session_id or f"session-{uuid.uuid4()}"
        title = request.title
        if not title:
            title = request.messages[0].content[:40] if request.messages else "New Chat"

        messages_json = json.dumps([m.model_dump() for m in request.messages])
        with self._lock:
            with self.connect(db_path) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM chat_sessions WHERE session_id = ?", [session_id]
                ).fetchone()
                if exists:
                    conn.execute(
                        "UPDATE chat_sessions SET title = ?, updated_at = now(), messages_json = ? WHERE session_id = ?",
                        [title, messages_json, session_id],
                    )
                else:
                    conn.execute(
                        "INSERT INTO chat_sessions VALUES (?, ?, now(), now(), ?)",
                        [session_id, title, messages_json],
                    )
                row = conn.execute(
                    "SELECT strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S') FROM chat_sessions WHERE session_id = ?",
                    [session_id],
                ).fetchone()

        created_at = str(row[0]) if row else ""
        updated_at = str(row[1]) if row else ""

        return SessionModel(
            session_id=session_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            messages=request.messages,
        )

    def get_session(
        self, session_id: str, db_path: Optional[str] = None
    ) -> SessionModel:
        """
        Retrieve a specific conversation session by ID.

        Args:
        ----
            session_id: Unique session identifier.
            db_path: Optional path to the sessions database.

        Returns:
        -------
            SessionModel: The found conversation session.

        Raises:
        ------
            SessionNotFoundError: If session is not found.

        """
        with self._lock:
            with self.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT session_id, title, strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S'), messages_json FROM chat_sessions WHERE session_id = ?",
                    [session_id],
                ).fetchone()
        if not row:
            raise SessionNotFoundError(
                "Session not found", params={"session_id": session_id}
            )

        return SessionModel(
            session_id=row[0],
            title=row[1],
            created_at=str(row[2]),
            updated_at=str(row[3]),
            messages=self._deserialize_messages(row[4]),
        )

    def delete_session(
        self, session_id: str, db_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Delete a conversation session by ID.

        Args:
        ----
            session_id: Unique session identifier.
            db_path: Optional path to the sessions database.

        Returns:
        -------
            Dict[str, str]: Deletion confirmation payload.

        Raises:
        ------
            SessionNotFoundError: If session is not found.

        """
        with self._lock:
            with self.connect(db_path) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM chat_sessions WHERE session_id = ?", [session_id]
                ).fetchone()
                if not exists:
                    raise SessionNotFoundError(
                        "Session not found", params={"session_id": session_id}
                    )
                conn.execute(
                    "DELETE FROM chat_sessions WHERE session_id = ?", [session_id]
                )
        return {"status": "deleted", "session_id": session_id}
