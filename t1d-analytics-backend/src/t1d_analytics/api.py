"""FastAPI application factory and HTTP middleware for the T1D Analytics web interface."""

import logging
import os
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from t1d_analytics.api_dependencies import (
    AppServices,
    configure_api_logging,
    create_default_app_services,
    get_app_services,
    get_cors_origins,
    get_request_locale,
    get_sessions_conn,
    parse_error,
    register_default_app_services,
    resolve_chat_job_manager,
    resolve_session_store,
    resolve_training_job_manager,
    verify_api_auth,
)
from t1d_analytics.diagnostics import (
    check_database_health,
    log_startup_diagnostics,
    run_system_diagnostics,
)
from t1d_analytics.errors import FrontendBuildNotFoundError, T1DError
from t1d_analytics.i18n import get_translator
from t1d_analytics.ingestion import load_data_to_duckdb
from t1d_analytics.job_manager import ChatTurnJobManager
from t1d_analytics.rate_limiter import (
    BaseRateLimiter,
    DuckDbRateLimiter,
    MemoryRateLimiter,
    close_all_rate_limiters,
)
from t1d_analytics.routers.chat_routes import chat_router
from t1d_analytics.routers.session_routes import sessions_router
from t1d_analytics.routers.status_routes import status_router
from t1d_analytics.routers.table_routes import tables_router
from t1d_analytics.routers.training_routes import training_router
from t1d_analytics.seed import ensure_canonical_clinical_views, seed_database
from t1d_analytics.session_store import SessionStore
from t1d_analytics.sql_utils import close_read_only_duckdb_pool
from t1d_analytics.training_runner import TrainingJobManager

logger = logging.getLogger(__name__)

_FRONTEND_DIST: Path = (
    Path(__file__).resolve().parents[3] / "t1d-analytics-web" / "dist"
)


def resolve_api_route_rate_limit(path: str) -> tuple[int, float] | None:
    """
    Resolve the `(max_requests, window_seconds)` rate-limit tier for an HTTP request path.

    Liveness and readiness probes (`/api/health`, `/api/ready`) and non-API static paths
    are exempt from rate limiting. Expensive export and training routes receive tighter
    budgets than interactive chat/SQL and lightweight metadata/session routes.

    Args:
        path: URL path of the incoming HTTP request.

    Returns:
        Optional[Tuple[int, float]]: `(limit, window_seconds)` tuple if rate-limited, else None.

    """
    if not path.startswith("/api/") or path in {"/api/health", "/api/ready"}:
        return None
    if path.startswith(("/api/export/", "/api/training/jobs")):
        return (60, 60.0)
    if (
        path.startswith(("/api/chat", "/api/eval/"))
        or path == "/api/execute-sql"
        or (path.startswith("/api/table/") and path.endswith("/profile"))
    ):
        return (120, 60.0)
    return (240, 60.0)


@asynccontextmanager
async def lifespan(target_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager configuring logging, auto-populating empty databases, and cleaning up pools.

    Args:
        target_app: The FastAPI application instance entering lifespan.

    Yields:
        None upon server startup.

    """
    configure_api_logging()
    db_path = os.environ.get("T1D_DB_PATH") or "data/t1d.duckdb"
    data_dir = os.environ.get("T1D_DATA_DIR") or "data"
    sessions_db_path = os.environ.get("T1D_SESSIONS_DB_PATH") or "data/sessions.duckdb"
    db_health = check_database_health(db_path=db_path)
    if (
        not db_health.exists
        or db_health.table_count == 0
        or db_health.total_rows == 0
        or not db_health.has_initial_data
    ):
        try:
            seed_database(
                data_dir=data_dir,
                db_path=db_path,
                sessions_db_path=sessions_db_path,
            )
            load_data_to_duckdb(data_dir, db_path)
            ensure_canonical_clinical_views(db_path)
        except (T1DError, duckdb.Error, OSError, RuntimeError, ValueError) as exc:
            logger.warning(f"Startup auto-ingestion failed for {db_path}: {exc}")
    run_system_diagnostics(db_path=db_path)
    log_startup_diagnostics(db_path=db_path)
    mount_frontend_static_assets(target_app)
    try:
        yield
    finally:
        app_limiter = getattr(target_app.state, "rate_limiter", None)
        if isinstance(app_limiter, BaseRateLimiter):
            app_limiter.close()
        training_mgr = getattr(target_app.state, "training_job_manager", None)
        if isinstance(training_mgr, TrainingJobManager):
            training_mgr.shutdown(wait=False)
        chat_mgr = getattr(target_app.state, "chat_job_manager", None)
        if isinstance(chat_mgr, ChatTurnJobManager):
            chat_mgr.shutdown(wait=False)
        close_read_only_duckdb_pool()
        close_all_rate_limiters()


async def t1d_error_exception_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    """
    Translate typed domain `T1DError` exceptions into structured JSON responses.

    Args:
        _request: Incoming FastAPI request.
        exc: Raised exception (expected to be a `T1DError` subclass).

    Returns:
        JSONResponse: HTTP response with `exc.status_code` and structured `detail`.

    """
    if isinstance(exc, T1DError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.to_dict()},
            headers=exc.headers,
        )
    api_err = parse_error(exc)
    return JSONResponse(status_code=500, content={"detail": api_err.model_dump()})


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unhandled exceptions globally and return a localized structured JSON error payload.

    Args:
        request: The incoming FastAPI HTTP request.
        exc: The unhandled exception.

    Returns:
        JSONResponse: Status 500 and localized error payload.

    """
    lang = get_request_locale(request)
    _ = get_translator(lang)

    logger.exception(f"Unhandled exception: {exc}")
    raw_error = str(exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error_code": "backend.serverError",
                "message": _("Internal server error: {}", raw_error),
                "params": {"error": raw_error},
            }
        },
    )


async def serve_chat_spa_entrypoint(chat_id: str, request: Request) -> Response:
    """
    Serve the compiled SPA `index.html` entrypoint for shareable `/chats/{chat_id}` deep-links.

    Args:
        chat_id: Target chat session identifier from the URL path.
        request: Active FastAPI HTTP request carrying optional `app.state.frontend_dist`.

    Returns:
        Response: `FileResponse` containing the frontend `index.html`.

    Raises:
        FrontendBuildNotFoundError: Status 404 when the compiled frontend `index.html` is absent.

    """
    _ = chat_id
    env_dist = os.environ.get("T1D_WEB_DIST_DIR")
    configured_dist = getattr(request.app.state, "frontend_dist", None)
    dist_dir = (
        Path(env_dist)
        if env_dist
        else (configured_dist if isinstance(configured_dist, Path) else _FRONTEND_DIST)
    )
    index_file = dist_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file), media_type="text/html")
    raise FrontendBuildNotFoundError(
        "Frontend index.html not found",
        params={"path": str(index_file)},
    )


def _create_app_rate_limiter() -> BaseRateLimiter:
    """
    Instantiate a fresh `BaseRateLimiter` according to `T1D_RATE_LIMITER_BACKEND`.

    Returns:
        BaseRateLimiter: Isolated `DuckDbRateLimiter` or `MemoryRateLimiter` instance.

    """
    backend = os.environ.get("T1D_RATE_LIMITER_BACKEND", "memory").lower().strip()
    if backend == "duckdb":
        return DuckDbRateLimiter()
    return MemoryRateLimiter()


def _resolve_middleware_services(request: Request) -> AppServices:
    """
    Resolve active `AppServices` inside HTTP middleware, respecting FastAPI `dependency_overrides`.

    Args:
        request: Incoming FastAPI HTTP request.

    Returns:
        AppServices: Overridden or per-application `AppServices` bundle.

    """
    override_fn = request.app.dependency_overrides.get(get_app_services)
    if callable(override_fn):
        overridden: AppServices = override_fn()
        return overridden
    return get_app_services(request)


def mount_frontend_static_assets(
    target_app: FastAPI | None = None,
    frontend_dist: Path | None = None,
) -> None:
    """
    Configure debug static file serving if the `DEBUG` environment variable is enabled.

    Args:
        target_app: Optional FastAPI application instance to mount static files onto (defaults to `app`).
        frontend_dist: Optional explicit frontend build directory override.

    """
    resolved_app = target_app if target_app is not None else app
    if os.environ.get("DEBUG"):
        env_dist = os.environ.get("T1D_WEB_DIST_DIR")
        state_dist = getattr(resolved_app.state, "frontend_dist", None)
        dist_path = (
            frontend_dist
            if frontend_dist is not None
            else (
                Path(env_dist)
                if env_dist
                else (state_dist if isinstance(state_dist, Path) else _FRONTEND_DIST)
            )
        )
        if dist_path.exists():
            logger.info(f"DEBUG mode enabled. Serving static frontend from {dist_path}")
            resolved_app.mount(
                "/", StaticFiles(directory=str(dist_path), html=True), name="static"
            )
        else:
            logger.warning(
                f"DEBUG mode enabled but static dist folder not found at {dist_path}"
            )


def create_app(frontend_dist: Path | None = None) -> FastAPI:
    """
    Construct a fully isolated `FastAPI` application instance with fresh stateful services.

    Every invocation creates independent `FastAPI`, `AppServices`, `SessionStore`,
    `TrainingJobManager`, `ChatTurnJobManager`, and `BaseRateLimiter` instances.

    Args:
        frontend_dist: Optional path to compiled frontend `dist/` directory.

    Returns:
        FastAPI: Configured FastAPI application instance.

    """
    new_app = FastAPI(
        title="T1D Analytics API",
        description="API for querying T1D datasets.",
        lifespan=lifespan,
    )

    allowed_origins = get_cors_origins()
    allow_all_origins = allowed_origins == ["*"]

    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @new_app.middleware("http")
    async def auth_and_rate_limit_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Enforce optional API key authentication and uniform route-tier rate limiting across `/api/*`."""
        try:
            verify_api_auth(request)
            tier = resolve_api_route_rate_limit(request.url.path)
            if tier is not None:
                svc = _resolve_middleware_services(request)
                svc.enforce_rate_limit(request, limit=tier[0], window_seconds=tier[1])
                request.state.rate_limit_enforced = True
        except T1DError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.to_dict()},
                headers=exc.headers,
            )
        return await call_next(request)

    new_app.add_exception_handler(T1DError, t1d_error_exception_handler)
    new_app.add_exception_handler(Exception, global_exception_handler)

    session_store = SessionStore(
        conn_factory=get_sessions_conn,
        lock=threading.Lock(),
    )
    training_job_manager = TrainingJobManager()
    chat_job_manager = ChatTurnJobManager()
    rate_limiter = _create_app_rate_limiter()

    new_app.state.session_store = session_store
    new_app.state.training_job_manager = training_job_manager
    new_app.state.chat_job_manager = chat_job_manager
    new_app.state.rate_limiter = rate_limiter
    if frontend_dist is not None:
        new_app.state.frontend_dist = frontend_dist

    services = create_default_app_services(
        target_app=new_app,
        session_store=session_store,
        training_job_manager=training_job_manager,
        chat_job_manager=chat_job_manager,
        rate_limiter=rate_limiter,
        session_store_getter=lambda: resolve_session_store(new_app, session_store),
        training_job_manager_getter=lambda: resolve_training_job_manager(new_app),
        chat_job_manager_getter=lambda: resolve_chat_job_manager(new_app),
    )
    new_app.state.services = services

    new_app.include_router(sessions_router)
    new_app.include_router(chat_router)
    new_app.include_router(tables_router)
    new_app.include_router(status_router)
    new_app.include_router(training_router)
    new_app.add_api_route(
        "/chats/{chat_id}",
        serve_chat_spa_entrypoint,
        methods=["GET"],
        include_in_schema=False,
        response_model=None,
    )

    return new_app


app = create_app()
register_default_app_services(app.state.services)

__all__ = [
    "app",
    "create_app",
    "global_exception_handler",
    "lifespan",
    "mount_frontend_static_assets",
    "resolve_api_route_rate_limit",
    "serve_chat_spa_entrypoint",
    "t1d_error_exception_handler",
]
