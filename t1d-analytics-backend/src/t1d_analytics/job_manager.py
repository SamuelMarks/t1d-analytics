"""Thread-safe asynchronous background manager for training jobs."""

import datetime
import threading
import uuid
from dataclasses import field
from typing import Callable, Dict, List, Optional, Protocol, Union

from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from t1d_analytics.models import TrainingBackend, TrainingJobConfig
from t1d_analytics.sql_utils import SqlValue

JobMetricValue = Union[SqlValue, List[SqlValue], Dict[str, SqlValue]]


class TrainingJobDict(TypedDict, total=False):
    """Serialized dictionary representation of a background training job."""

    job_id: str
    status: str
    backend: str
    model_name: str
    dataset_path: str
    output_dir: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    metrics: Dict[str, JobMetricValue]
    error: Optional[str]


class TrainingRunnerProtocol(Protocol):
    """Protocol for training runner instances resolved by TrainingJobManager."""

    def run_training(
        self,
        config: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """Execute training job for the given configuration."""
        ...


@dataclass
class TrainingJobState:
    """
    Strongly-typed state record for a background training job.

    Attributes
    ----------
        job_id: Unique job identifier string.
        status: Current lifecycle state ('pending', 'running', 'completed', 'failed', 'cancelled').
        backend: Execution backend identifier string.
        model_name: Name of the model being trained.
        dataset_path: Path to the input training dataset.
        output_dir: Destination directory for checkpoints and metrics.
        created_at: ISO-8601 UTC timestamp when the job was submitted.
        started_at: Optional ISO-8601 UTC timestamp when execution began.
        completed_at: Optional ISO-8601 UTC timestamp when execution finished.
        metrics: Dictionary of training loss and performance metrics.
        error: Optional error message if the job failed.

    """

    job_id: str
    status: str
    backend: str
    model_name: str
    dataset_path: str
    output_dir: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: Dict[str, JobMetricValue] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> TrainingJobDict:
        """
        Convert the job state into a plain dictionary for JSON serialization.

        Returns
        -------
            TrainingJobDict: Dictionary representation of the job state.

        """
        return {
            "job_id": self.job_id,
            "status": self.status,
            "backend": self.backend,
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metrics": dict(self.metrics),
            "error": self.error,
        }


class TrainingJobManager:
    """Thread-safe asynchronous background manager for training jobs."""

    def __init__(
        self,
        runner_resolver: Callable[[TrainingBackend], TrainingRunnerProtocol],
    ) -> None:
        """
        Initialize the job manager with thread locks and registry collections.

        Args:
        ----
            runner_resolver: Factory callable resolving a TrainingBackend to a runner instance.

        """
        self._lock = threading.Lock()
        self._jobs: Dict[str, TrainingJobState] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._runner_resolver = runner_resolver

    def submit_job(self, config: TrainingJobConfig) -> str:
        """
        Submit a new training job to run in a background daemon thread.

        Args:
        ----
            config: TrainingJobConfig defining dataset, model, hyperparameters, and backend.

        Returns:
        -------
            str: Unique job ID string.

        """
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = TrainingJobState(
                job_id=job_id,
                status="pending",
                backend=config.backend.value,
                model_name=config.model_name,
                dataset_path=config.dataset_path,
                output_dir=config.output_dir,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            self._cancel_flags[job_id] = cancel_event

        def _worker() -> None:
            """Execute training runner asynchronously within background thread."""
            with self._lock:
                self._jobs[job_id].status = "running"
                self._jobs[job_id].started_at = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()

            runner = self._runner_resolver(config.backend)

            try:
                if cancel_event.is_set():
                    with self._lock:
                        self._jobs[job_id].status = "cancelled"
                    return

                res = runner.run_training(config, cancel_event=cancel_event)
                with self._lock:
                    if cancel_event.is_set():
                        self._jobs[job_id].status = "cancelled"
                    else:
                        self._jobs[job_id].status = "completed"
                        self._jobs[job_id].metrics = res
            except Exception as e:
                with self._lock:
                    self._jobs[job_id].status = "failed"
                    self._jobs[job_id].error = str(e)
            finally:
                with self._lock:
                    self._jobs[job_id].completed_at = datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return job_id

    def get_job(self, job_id: str) -> Optional[TrainingJobDict]:
        """
        Retrieve current status and metrics of a training job.

        Args:
        ----
            job_id: The job ID string.

        Returns:
        -------
            Optional[TrainingJobDict]: Job details dictionary or None if not found.

        """
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list_jobs(self) -> List[TrainingJobDict]:
        """
        List all tracked training jobs.

        Returns
        -------
            List[TrainingJobDict]: List of all job information dictionaries.

        """
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of a running or pending training job.

        Args:
        ----
            job_id: The job ID to cancel.

        Returns:
        -------
            bool: True if job was found and cancellation was signaled, False otherwise.

        """
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._cancel_flags[job_id].set()
            if self._jobs[job_id].status in ("pending", "running"):
                self._jobs[job_id].status = "cancelled"
            return True
