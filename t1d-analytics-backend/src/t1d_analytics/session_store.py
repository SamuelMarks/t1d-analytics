"""DuckDB-backed persistent storage for chat conversation sessions and normalized messages."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol

import duckdb
from pydantic import ValidationError

from t1d_analytics.api_schemas import (
    ChatMessageModel,
    CreateSessionRequest,
    SessionListResponse,
    SessionModel,
)
from t1d_analytics.db_security import validate_sessions_db_path
from t1d_analytics.duckdb_pool import acquire_duckdb_connection_with_retry
from t1d_analytics.errors import InvalidDatabasePathError, SessionNotFoundError

logger = logging.getLogger(__name__)

_INITIALIZED_SESSION_DBS: set[str] = set()
_INIT_LOCK = threading.Lock()


def clear_initialized_session_dbs() -> None:
    """
    Clear the cache of initialized session database paths.

    Returns:
        None

    """
    with _INIT_LOCK:
        _INITIALIZED_SESSION_DBS.clear()


def _ensure_sessions_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create the `chat_sessions` and normalized `chat_messages` tables if absent.

    Args:
        conn: Open writable DuckDB connection to the sessions database.

    """
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            session_id VARCHAR,
            seq INTEGER,
            role VARCHAR,
            content VARCHAR,
            message_json VARCHAR,
            PRIMARY KEY (session_id, seq)
        )
        """
    )


def open_sessions_db_conn(
    db_path: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> duckdb.DuckDBPyConnection:
    """
    Open connection to persistent chat sessions database and ensure normalized schema exists.

    Args:
        db_path: Optional path to the database file.
        sleep_fn: Injectable sleep function for lock retry backoff.

    Returns:
        duckdb.DuckDBPyConnection: Database connection.

    Raises:
        InvalidDatabasePathError: If `db_path` fails sandboxing or its parent directory is absent.
        DatabaseLockedError: If lock retries are exhausted.

    """
    is_default_target = db_path is None
    validated_db_path = validate_sessions_db_path(db_path)
    resolved_path = Path(validated_db_path)
    resolved_key = str(resolved_path)
    existed_before = resolved_path.exists()
    if is_default_target:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    elif not resolved_path.parent.exists():
        raise InvalidDatabasePathError(params={"path": str(db_path)})

    conn = acquire_duckdb_connection_with_retry(
        db_path=validated_db_path,
        read_only=False,
        max_attempts=5,
        base_delay_seconds=0.05,
        sleep_fn=sleep_fn,
        connect_fn=duckdb.connect,
    )

    with _INIT_LOCK:
        if not existed_before:
            _INITIALIZED_SESSION_DBS.discard(resolved_key)
        needs_ddl = resolved_key not in _INITIALIZED_SESSION_DBS

    if needs_ddl:
        try:
            _ensure_sessions_schema(conn)
        except (duckdb.Error, OSError, RuntimeError):
            conn.close()
            raise
        with _INIT_LOCK:
            _INITIALIZED_SESSION_DBS.add(resolved_key)
    return conn


@contextmanager
def managed_sessions_db_conn(
    db_path: str | None = None,
    open_fn: Callable[[str | None], duckdb.DuckDBPyConnection] = open_sessions_db_conn,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager wrapping `open_sessions_db_conn` that guarantees `conn.close()` on exit.

    Args:
        db_path: Optional path to the sessions database file.
        open_fn: Injectable callable opening the sessions DuckDB connection.

    Yields:
        duckdb.DuckDBPyConnection: Open DuckDB connection to the sessions database.

    """
    conn = open_fn(db_path)
    try:
        yield conn
    finally:
        conn.close()


class SessionDbConnectionProtocol(Protocol):
    """Structural protocol for DuckDB session database connections."""

    def execute(
        self,
        query: str,
        parameters: Sequence[str | int] | None = None,
    ) -> SessionDbConnectionProtocol:
        """Execute a parameterized SQL query."""
        ...

    def fetchone(self) -> tuple[str | int | None, ...] | None:
        """Fetch one result row."""
        ...

    def fetchall(self) -> list[tuple[str | int | None, ...]]:
        """Fetch all result rows."""
        ...


class SessionStore:
    """Thread-safe repository managing persistent chat sessions and normalized messages in DuckDB."""

    def __init__(
        self,
        conn_factory: Callable[
            [str | None], AbstractContextManager[SessionDbConnectionProtocol]
        ] = managed_sessions_db_conn,
        lock: threading.Lock | None = None,
    ) -> None:
        """
        Initialize the SessionStore with a connection factory and synchronization lock.

        Args:
            conn_factory: Context-manager factory yielding an active DuckDB connection.
            lock: Optional threading.Lock to synchronize concurrent writes.

        """
        self._lock = lock or threading.Lock()
        self._conn_factory = conn_factory

    @contextmanager
    def connect(
        self, db_path: str | None = None
    ) -> Iterator[SessionDbConnectionProtocol]:
        """
        Yield an active DuckDB connection with `chat_sessions` and `chat_messages` initialized.

        Args:
            db_path: Optional path to the DuckDB database file.

        Yields:
            duckdb.DuckDBPyConnection: Active DuckDB connection.

        """
        with self._conn_factory(db_path) as conn:
            yield conn

    @staticmethod
    def _deserialize_messages(
        messages_json: str,
    ) -> list[ChatMessageModel]:
        """
        Parse a JSON array string of chat messages into `ChatMessageModel` instances.

        Args:
            messages_json: Raw JSON string from DuckDB.

        Returns:
            List[ChatMessageModel]: Deserialized messages, or empty list on parse failure.

        """
        try:
            msgs_raw = json.loads(messages_json)
            return [ChatMessageModel(**m) for m in msgs_raw]
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
            ValidationError,
        ) as exc:
            logger.warning(f"Failed to deserialize session messages JSON: {exc}")
            return []

    @staticmethod
    def _deserialize_single_message(message_json: str) -> ChatMessageModel | None:
        """
        Parse a single normalized `chat_messages.message_json` row into a `ChatMessageModel`.

        Args:
            message_json: JSON object string for a single message.

        Returns:
            Optional[ChatMessageModel]: Parsed `ChatMessageModel`, or None if malformed.

        """
        try:
            raw_obj = json.loads(message_json)
            return ChatMessageModel(**raw_obj)
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
            ValidationError,
        ) as exc:
            logger.warning(f"Failed to deserialize normalized chat message: {exc}")
            return None

    def list_sessions(
        self,
        db_path: str | None = None,
        include_messages: bool = True,
    ) -> SessionListResponse:
        """
        List persistent conversation sessions sorted by `updated_at` descending.

        When `include_messages=False`, projects session metadata columns only without
        reading `messages_json` or `chat_messages`. When `include_messages=True`, loads
        normalized rows from `chat_messages` and falls back to `messages_json` only for
        legacy unmigrated rows.

        Args:
            db_path: Optional path to the sessions database.
            include_messages: Whether to hydrate conversation messages (default: True).

        Returns:
            SessionListResponse: List of stored conversation sessions.

        """
        with self._lock:
            with self.connect(db_path) as conn:
                if not include_messages:
                    meta_rows = conn.execute(
                        "SELECT session_id, title, strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S') FROM chat_sessions ORDER BY updated_at DESC"
                    ).fetchall()
                    return SessionListResponse(
                        sessions=[
                            SessionModel(
                                session_id=str(r[0]),
                                title=str(r[1]),
                                created_at=str(r[2]),
                                updated_at=str(r[3]),
                                messages=[],
                            )
                            for r in meta_rows
                        ]
                    )

                meta_rows = conn.execute(
                    "SELECT session_id, title, strftime(created_at, '%Y-%m-%d %H:%M:%S'), strftime(updated_at, '%Y-%m-%d %H:%M:%S') FROM chat_sessions ORDER BY updated_at DESC"
                ).fetchall()
                msg_rows = conn.execute(
                    "SELECT session_id, message_json FROM chat_messages ORDER BY session_id, seq ASC"
                ).fetchall()
                by_session: dict[str, list[ChatMessageModel]] = {}
                for s_id, m_json in msg_rows:
                    parsed_msg = self._deserialize_single_message(str(m_json))
                    if parsed_msg is not None:
                        by_session.setdefault(str(s_id), []).append(parsed_msg)

                missing_ids = [
                    str(r[0]) for r in meta_rows if str(r[0]) not in by_session
                ]
                legacy_by_session: dict[str, list[ChatMessageModel]] = {}
                if missing_ids:
                    placeholders = ", ".join("?" for _ in missing_ids)
                    legacy_rows = conn.execute(
                        f"SELECT session_id, messages_json FROM chat_sessions WHERE session_id IN ({placeholders}) AND messages_json IS NOT NULL",
                        missing_ids,
                    ).fetchall()
                    for l_id, l_json in legacy_rows:
                        legacy_by_session[str(l_id)] = self._deserialize_messages(
                            str(l_json)
                        )

        sessions: list[SessionModel] = [
            SessionModel(
                session_id=str(r[0]),
                title=str(r[1]),
                created_at=str(r[2]),
                updated_at=str(r[3]),
                messages=by_session.get(
                    str(r[0]), legacy_by_session.get(str(r[0]), [])
                ),
            )
            for r in meta_rows
        ]
        return SessionListResponse(sessions=sessions)

    def upsert_session(
        self, request: CreateSessionRequest, db_path: str | None = None
    ) -> SessionModel:
        """
        Create or update a persistent conversation session and its normalized `chat_messages` rows.

        Args:
            request: Session creation or update payload.
            db_path: Optional path to the sessions database.

        Returns:
            SessionModel: The saved conversation session.

        """
        session_id = request.session_id or f"session-{uuid.uuid4()}"
        title = request.title
        if not title:
            title = request.messages[0].content[:40] if request.messages else "New Chat"

        messages_json = json.dumps(
            [m.model_dump(by_alias=True) for m in request.messages]
        )
        with self._lock:
            with self.connect(db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                try:
                    exists = conn.execute(
                        "SELECT 1 FROM chat_sessions WHERE session_id = ?",
                        [session_id],
                    ).fetchone()
                    if exists:
                        conn.execute(
                            "UPDATE chat_sessions SET title = ?, updated_at = now(), messages_json = ? WHERE session_id = ?",
                            [title, messages_json, session_id],
                        )
                        conn.execute(
                            "DELETE FROM chat_messages WHERE session_id = ?",
                            [session_id],
                        )
                    else:
                        conn.execute(
                            "INSERT INTO chat_sessions VALUES (?, ?, now(), now(), ?)",
                            [session_id, title, messages_json],
                        )
                    for seq, msg in enumerate(request.messages):
                        conn.execute(
                            "INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?)",
                            [
                                session_id,
                                seq,
                                msg.role,
                                msg.content,
                                json.dumps(msg.model_dump(by_alias=True)),
                            ],
                        )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
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

    def get_session(self, session_id: str, db_path: str | None = None) -> SessionModel:
        """
        Retrieve a specific conversation session by ID.

        Args:
            session_id: Unique session identifier.
            db_path: Optional path to the sessions database.

        Returns:
            SessionModel: The found conversation session.

        Raises:
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
                norm_rows = conn.execute(
                    "SELECT message_json FROM chat_messages WHERE session_id = ? ORDER BY seq ASC",
                    [session_id],
                ).fetchall()

        if norm_rows:
            messages = [
                parsed
                for (m_json,) in norm_rows
                if (parsed := self._deserialize_single_message(str(m_json))) is not None
            ]
        else:
            messages = self._deserialize_messages(str(row[4])) if row[4] else []

        return SessionModel(
            session_id=str(row[0]),
            title=str(row[1]),
            created_at=str(row[2]),
            updated_at=str(row[3]),
            messages=messages,
        )

    def delete_session(
        self, session_id: str, db_path: str | None = None
    ) -> dict[str, str]:
        """
        Delete a conversation session and its normalized messages by ID.

        Args:
            session_id: Unique session identifier.
            db_path: Optional path to the sessions database.

        Returns:
            Dict[str, str]: Deletion confirmation payload.

        Raises:
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
                conn.execute("BEGIN TRANSACTION")
                try:
                    conn.execute(
                        "DELETE FROM chat_messages WHERE session_id = ?",
                        [session_id],
                    )
                    conn.execute(
                        "DELETE FROM chat_sessions WHERE session_id = ?",
                        [session_id],
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
        return {"status": "deleted", "session_id": session_id}
