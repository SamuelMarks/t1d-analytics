"""Background model training job CRUD endpoints (`/api/training/jobs`)."""

from fastapi import APIRouter, Depends

from t1d_analytics.api_dependencies import AppServices, get_app_services
from t1d_analytics.api_schemas import TrainingJobRequest
from t1d_analytics.errors import (
    InvalidTrainingBackendError,
    TrainingJobNotFoundError,
)
from t1d_analytics.job_manager import TrainingJobDict, TrainingJobManager
from t1d_analytics.models import TrainingBackend, TrainingJobConfig

training_router = APIRouter()


def _resolve_job_manager_dependency(
    svc: AppServices = Depends(get_app_services),
) -> TrainingJobManager:
    """
    Resolve the active `TrainingJobManager` via injected `AppServices`.

    Args:
        svc: Injected AppServices dependency bundle.

    Returns:
        TrainingJobManager: Active background training job manager.

    """
    return svc.get_training_job_manager()


@training_router.post("/api/training/jobs")
def submit_training_job(
    job_req: TrainingJobRequest,
    job_manager: TrainingJobManager = Depends(_resolve_job_manager_dependency),
    svc: AppServices = Depends(get_app_services),
) -> TrainingJobDict:
    """
    Submit a background training job using the configured accelerator backend.

    Args:
        job_req: Training job specification.
        job_manager: Injected TrainingJobManager dependency.
        svc: Injected AppServices dependency bundle.

    Returns:
        TrainingJobDict: Job ID and initial status.

    Raises:
        InvalidTrainingBackendError: If requested training backend is unsupported.

    """
    validated_dataset_path = svc.validate_workspace_artifact_path(job_req.dataset_path)
    validated_output_dir = svc.validate_workspace_artifact_path(job_req.output_dir)

    backend_val = job_req.backend.lower()
    if backend_val == "cpu":
        backend_val = "local-cpu"
    elif backend_val == "gpu":
        backend_val = "local-gpu"
    elif backend_val == "tpu":
        backend_val = "remote-tpu-maxtext"

    try:
        backend_enum = TrainingBackend(backend_val)
    except ValueError as exc:
        raise InvalidTrainingBackendError(
            params={"backend": job_req.backend},
        ) from exc

    config = TrainingJobConfig(
        dataset_path=validated_dataset_path,
        output_dir=validated_output_dir,
        backend=backend_enum,
        model_name=job_req.model_name,
        dry_run=job_req.dry_run,
    )
    job_id = job_manager.submit_job(config)
    job = job_manager.get_job(job_id)
    return job if job else {"job_id": job_id, "status": "pending"}


@training_router.get("/api/training/jobs")
def list_training_jobs(
    job_manager: TrainingJobManager = Depends(_resolve_job_manager_dependency),
) -> dict[str, list[TrainingJobDict]]:
    """
    List all tracked background training jobs.

    Args:
        job_manager: Injected TrainingJobManager dependency.

    Returns:
        Dict[str, List[TrainingJobDict]]: Dictionary containing list of training jobs.

    """
    return {"jobs": job_manager.list_jobs()}


@training_router.get("/api/training/jobs/{job_id}")
def get_training_job(
    job_id: str,
    job_manager: TrainingJobManager = Depends(_resolve_job_manager_dependency),
) -> TrainingJobDict:
    """
    Retrieve details and execution metrics for a specific background training job.

    Args:
        job_id: Unique job identifier string.
        job_manager: Injected TrainingJobManager dependency.

    Returns:
        TrainingJobDict: Job status and execution results.

    Raises:
        TrainingJobNotFoundError: 404 if the job ID is not found.

    """
    job = job_manager.get_job(job_id)
    if not job:
        raise TrainingJobNotFoundError(params={"job_id": job_id})
    return job


@training_router.delete("/api/training/jobs/{job_id}")
def cancel_training_job(
    job_id: str,
    job_manager: TrainingJobManager = Depends(_resolve_job_manager_dependency),
) -> TrainingJobDict:
    """
    Request cancellation of a running background training job.

    Args:
        job_id: Unique job identifier string.
        job_manager: Injected TrainingJobManager dependency.

    Returns:
        TrainingJobDict: Cancellation outcome.

    Raises:
        TrainingJobNotFoundError: 404 if the job ID is not found.

    """
    cancelled = job_manager.cancel_job(job_id)
    if not cancelled:
        raise TrainingJobNotFoundError(params={"job_id": job_id})
    job = job_manager.get_job(job_id)
    return job if job else {"job_id": job_id, "status": "cancelled"}
