"""Persistent chat session CRUD endpoints (`/api/sessions`)."""

from typing import Dict, Optional

from fastapi import APIRouter, Depends

from t1d_analytics.api_schemas import (
    CreateSessionRequest,
    SessionListResponse,
    SessionModel,
)
from t1d_analytics.session_store import SessionStore

sessions_router = APIRouter()


def _resolve_store_dependency() -> SessionStore:
    """
    Delegate session store dependency resolution to `t1d_analytics.api.get_session_store`.

    Returns
    -------
        SessionStore: Active conversation session store.

    """
    import t1d_analytics.api as api_mod

    return api_mod.get_session_store()


@sessions_router.get("/api/sessions", response_model=SessionListResponse)
def list_sessions(
    db_path: Optional[str] = None,
    store: SessionStore = Depends(_resolve_store_dependency),
) -> SessionListResponse:
    """
    List all persistent conversation sessions.

    Args:
    ----
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        SessionListResponse: List of stored conversation sessions sorted by updated_at.

    """
    return store.list_sessions(db_path=db_path)


@sessions_router.post("/api/sessions", response_model=SessionModel)
def save_session(
    request: CreateSessionRequest,
    db_path: Optional[str] = None,
    store: SessionStore = Depends(_resolve_store_dependency),
) -> SessionModel:
    """
    Create or update a persistent conversation session.

    Args:
    ----
        request: The session creation or update payload.
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        SessionModel: The saved conversation session.

    """
    return store.upsert_session(request=request, db_path=db_path)


@sessions_router.get("/api/sessions/{session_id}", response_model=SessionModel)
def get_session(
    session_id: str,
    db_path: Optional[str] = None,
    store: SessionStore = Depends(_resolve_store_dependency),
) -> SessionModel:
    """
    Retrieve a specific conversation session by ID.

    Args:
    ----
        session_id: The ID of the session to retrieve.
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        SessionModel: The found conversation session.

    """
    return store.get_session(session_id=session_id, db_path=db_path)


@sessions_router.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: str,
    db_path: Optional[str] = None,
    store: SessionStore = Depends(_resolve_store_dependency),
) -> Dict[str, str]:
    """
    Delete a conversation session.

    Args:
    ----
        session_id: The ID of the session to delete.
        db_path: Optional path to the sessions database.
        store: Injected SessionStore dependency.

    Returns:
    -------
        Dict[str, str]: Deletion confirmation message.

    """
    return store.delete_session(session_id=session_id, db_path=db_path)
