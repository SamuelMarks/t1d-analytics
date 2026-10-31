"""Domain-focused FastAPI routers for the T1D Analytics API."""

from t1d_analytics.routers.chat_routes import chat_router as chat_router
from t1d_analytics.routers.session_routes import sessions_router as sessions_router
from t1d_analytics.routers.status_routes import status_router as status_router
from t1d_analytics.routers.table_routes import tables_router as tables_router
from t1d_analytics.routers.training_routes import training_router as training_router

__all__ = [
    "chat_router",
    "sessions_router",
    "status_router",
    "tables_router",
    "training_router",
]
