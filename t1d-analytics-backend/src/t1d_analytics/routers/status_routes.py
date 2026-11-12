"""System status, readiness, deep health, LLM provider status, and model catalog endpoints."""

import urllib.request

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from t1d_analytics.api_dependencies import AppServices, get_app_services
from t1d_analytics.api_schemas import (
    ClinicalTemplateItemSchema,
    DatabaseHealthSchema,
    ModelsResponse,
    OllamaStatusSchema,
    ProvidersStatusResponse,
    ProviderStatusEntrySchema,
    StatusResponse,
    TemplatesCatalogResponse,
)
from t1d_analytics.clinical_query_catalog import get_clinical_query_catalog
from t1d_analytics.diagnostics import (
    HealthStatus,
    SystemHealthDict,
    get_system_health,
    run_system_diagnostics,
)
from t1d_analytics.llm_client import (
    fetch_available_models,
    probe_providers_status,
)

status_router = APIRouter()


@status_router.get("/api/status", response_model=StatusResponse)
def get_status(
    request: Request,
    db_path: str | None = None,
    require_ready: bool = False,
    svc: AppServices = Depends(get_app_services),
) -> StatusResponse | JSONResponse:
    """
    Return comprehensive system status, table counts, row counts, and LLM availability.

    Args:
        request: FastAPI request instance.
        db_path: Optional custom database path.
        require_ready: If True, return HTTP 503 when database is unpopulated or not ready.
        svc: Injected AppServices dependency bundle.

    Returns:
        Union[StatusResponse, JSONResponse]: StatusResponse or JSONResponse with status details.

    """
    target_db = svc.validate_db_path(db_path) if db_path else None
    lang = svc.get_request_locale(request)
    health = get_system_health(db_path=target_db, lang=lang)
    db = health.database

    is_ready = (
        db.connected
        and db.table_count > 0
        and db.total_rows > 0
        and health.sessions_database_healthy
    )
    status_str = "ready" if is_ready else health.status
    resp = StatusResponse(
        status=status_str,
        tables_count=db.table_count,
        total_rows=db.total_rows,
        required_tables_present=db.has_initial_data,
        backend_accessible=health.backend_accessible,
        version=health.version,
        database=DatabaseHealthSchema.model_validate(db.to_dict()),
        ollama=OllamaStatusSchema.model_validate(health.ollama.to_dict()),
        providers={
            k: ProviderStatusEntrySchema.model_validate(v.to_dict())
            for k, v in health.providers.items()
        },
    )
    if require_ready and (
        status_str != "ready" or db.table_count == 0 or db.total_rows == 0
    ):
        return JSONResponse(status_code=503, content=resp.model_dump())
    return resp


@status_router.get("/api/ready", response_model=None)
def get_ready(
    request: Request,
    db_path: str | None = None,
    svc: AppServices = Depends(get_app_services),
) -> StatusResponse | JSONResponse:
    """
    Readiness probe endpoint returning HTTP 200 when database is populated or HTTP 503 otherwise.

    Args:
        request: FastAPI request instance.
        db_path: Optional custom database path.
        svc: Injected AppServices dependency bundle.

    Returns:
        Union[StatusResponse, JSONResponse]: StatusResponse or 503 JSONResponse.

    """
    return get_status(request=request, db_path=db_path, require_ready=True, svc=svc)


@status_router.get("/api/health", response_model=None)
def get_health(
    request: Request,
    db_path: str | None = None,
    svc: AppServices = Depends(get_app_services),
) -> SystemHealthDict | JSONResponse:
    """
    Deep health check endpoint returning HTTP 503 if the DuckDB database is missing or empty.

    Args:
        request: FastAPI request instance.
        db_path: Optional custom database path.
        svc: Injected AppServices dependency bundle.

    Returns:
        Union[SystemHealthDict, JSONResponse]: Dictionary or HTTP 503 JSONResponse containing system health details.

    """
    target_db = svc.validate_db_path(db_path) if db_path else None
    lang = svc.get_request_locale(request)
    health = run_system_diagnostics(db_path=target_db, lang=lang)
    db = health.database
    payload = health.to_dict()
    if (
        health.status in ("error", HealthStatus.UNHEALTHY.value)
        or db.status == HealthStatus.UNHEALTHY
        or db.table_count == 0
        or db.total_rows == 0
    ):
        return JSONResponse(status_code=503, content=payload)
    return payload


@status_router.get("/api/providers/status", response_model=ProvidersStatusResponse)
def get_providers_status() -> ProvidersStatusResponse:
    """
    Report connection health and credential status for all supported LLM providers.

    Returns:
        ProvidersStatusResponse: Status for ollama, openai, anthropic, google, and configured providers.

    """
    return probe_providers_status(urlopen_fn=urllib.request.urlopen)


@status_router.get("/api/models", response_model=ModelsResponse)
def list_models(
    include_cloud: bool = True,
    configured_models: str | None = None,
    svc: AppServices = Depends(get_app_services),
) -> ModelsResponse:
    """
    List available local Ollama models and configured any-llm cloud providers.

    Args:
        include_cloud: Whether to include cloud and configured any-llm provider models.
        configured_models: Optional comma-separated list of custom configured provider/model IDs.
        svc: Injected AppServices dependency bundle.

    Returns:
        ModelsResponse: List of available local and cloud models.

    """
    return fetch_available_models(
        include_cloud=include_cloud,
        configured_models=configured_models,
        cloud_catalog=svc.cloud_provider_models,
        urlopen_fn=urllib.request.urlopen,
    )


@status_router.get("/api/templates", response_model=TemplatesCatalogResponse)
def list_clinical_templates() -> TemplatesCatalogResponse:
    """
    Return the canonical starter and Academic Table 1 clinical SQL query catalog.

    Returns:
        TemplatesCatalogResponse: Canonical list of starter and Table 1 templates.

    """
    items = [
        ClinicalTemplateItemSchema.model_validate(entry)
        for entry in get_clinical_query_catalog()
    ]
    return TemplatesCatalogResponse(templates=items)
