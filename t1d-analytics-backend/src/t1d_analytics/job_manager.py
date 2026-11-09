"""Thread-safe asynchronous background manager for training and chat jobs."""

import datetime
import threading
import uuid
from dataclasses import field
from typing import Callable, Dict, List, Literal, Optional, Protocol, Union

from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from t1d_analytics.api_schemas import (
    ApiError,
    ChatJobStatus,
    ChatRequest,
    ChatResponse,
    NlToSqlEvalRequest,
    NlToSqlEvalResponse,
)
from t1d_analytics.errors import T1DError, exception_to_error_detail
from t1d_analytics.models import TrainingBackend, TrainingJobConfig
from t1d_analytics.sql_utils import SqlValue

JobMetricValue = Union[SqlValue, List[SqlValue], Dict[str, SqlValue]]
ChatJobPayloadValue = Union[
    SqlValue,
    List[Dict[str, SqlValue]],
    Dict[
        str,
        Union[SqlValue, Dict[str, SqlValue], List[str], List[Dict[str, SqlValue]]],
    ],
]


class ChatJobDict(TypedDict, total=False):
    """Serialized dictionary representation of an asynchronous chat or evaluation job."""

    job_id: str
    status: ChatJobStatus
    created_at: str
    completed_at: Optional[str]
    partial_content: str
    content: str
    sqlQuery: Optional[str]
    sqlResult: Optional[List[Dict[str, SqlValue]]]
    is_synthetic_data: bool
    is_rule_based_sql: bool
    generation_mode: str
    llm_error: Optional[str]
    llm_error_code: Optional[str]
    data_provenance: str
    gold_comparison: Optional[
        Dict[
            str,
            Union[SqlValue, Dict[str, SqlValue], List[str], List[Dict[str, SqlValue]]],
        ]
    ]
    error: Optional[Dict[str, Union[SqlValue, Dict[str, SqlValue]]]]
    question: Optional[str]
    model: Optional[str]


@dataclass
class ChatJobState:
    """
    Strongly-typed state record for an asynchronous chat or NL-to-SQL evaluation job.

    Attributes
    ----------
        job_id: Unique job identifier string ('chatjob-<uuid>').
        status: Current lifecycle state ('pending', 'complete', 'failed').
        created_at: ISO-8601 UTC timestamp when the job was submitted.
        completed_at: Optional ISO-8601 UTC timestamp when execution finished.
        partial_content: Live accumulated token buffer while 'pending'.
        result: Completed ChatResponse record when status is 'complete' or 'failed'.
        eval_result: Completed NlToSqlEvalResponse record for evaluation jobs.
        error: Optional structured ApiError if the job failed.

    """

    job_id: str
    status: Literal["pending", "complete", "failed"]
    created_at: str
    completed_at: Optional[str] = None
    partial_content: str = ""
    result: Optional[ChatResponse] = None
    eval_result: Optional[NlToSqlEvalResponse] = None
    error: Optional[ApiError] = None

    def to_chat_response(self) -> ChatResponse:
        """
        Convert the current job state into a `ChatResponse` model for polling endpoints.

        Returns
        -------
            ChatResponse: Pending ticket or completed `ChatResponse` record.

        """
        if self.status == "pending":
            return ChatResponse(
                job_id=self.job_id,
                status="pending",
                partial_content=self.partial_content,
                content=self.partial_content,
            )
        if self.result is not None:
            resolved_status: ChatJobStatus = (
                "failed" if self.status == "failed" else "complete"
            )
            return self.result.model_copy(
                update={
                    "job_id": self.job_id,
                    "status": resolved_status,
                    "partial_content": self.partial_content or self.result.content,
                }
            )
        return ChatResponse(
            job_id=self.job_id,
            status="failed",
            partial_content=self.partial_content,
            content=self.partial_content,
            error=self.error
            or ApiError(error_code="backend.errorUnexpected", params={}),
        )

    def to_eval_response(
        self, fallback_question: str = "", fallback_model: str = "gemma:2b"
    ) -> NlToSqlEvalResponse:
        """
        Convert the current job state into an `NlToSqlEvalResponse` model for evaluation polling endpoints.

        Args:
        ----
            fallback_question: Fallback question string if `eval_result` is not yet populated.
            fallback_model: Fallback model identifier if `eval_result` is not yet populated.

        Returns:
        -------
            NlToSqlEvalResponse: Pending ticket or completed `NlToSqlEvalResponse` record.

        """
        if self.status == "pending":
            return NlToSqlEvalResponse(
                job_id=self.job_id,
                status="pending",
                question=fallback_question,
                model=fallback_model,
                partial_content=self.partial_content,
                content=self.partial_content,
            )
        if self.eval_result is not None:
            resolved_status: ChatJobStatus = (
                "failed" if self.status == "failed" else "complete"
            )
            return self.eval_result.model_copy(
                update={
                    "job_id": self.job_id,
                    "status": resolved_status,
                    "partial_content": self.partial_content or self.eval_result.content,
                }
            )
        return NlToSqlEvalResponse(
            job_id=self.job_id,
            status="failed",
            question=fallback_question,
            model=fallback_model,
            partial_content=self.partial_content,
            content=self.partial_content,
            error=self.error
            or ApiError(error_code="backend.errorUnexpected", params={}),
        )

    def to_dict(self) -> Dict[str, ChatJobPayloadValue]:
        """
        Serialize the job state to a dictionary suitable for JSON polling responses.

        When `status == "pending"`, returns the lightweight pending envelope with
        `partial_content` and `content`. When `status` is `"complete"` or `"failed"`,
        merges the full `ChatResponse` (or `NlToSqlEvalResponse`) dictionary with
        `job_id`, `status`, `created_at`, `completed_at`, and `partial_content`.

        Returns
        -------
            Dict[str, ChatJobPayloadValue]: Serialized job dictionary.

        """
        if self.status == "pending":
            return {
                "job_id": self.job_id,
                "status": "pending",
                "created_at": self.created_at,
                "completed_at": self.completed_at,
                "partial_content": self.partial_content,
                "content": self.partial_content,
            }
        chat_resp = self.to_chat_response()
        merged = chat_resp.to_dict()
        merged["job_id"] = self.job_id
        merged["status"] = self.status
        merged["created_at"] = self.created_at
        merged["completed_at"] = self.completed_at
        merged["partial_content"] = self.partial_content or chat_resp.content
        return merged


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
        self._done_flags: Dict[str, threading.Event] = {}
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
        done_event = threading.Event()
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
            self._done_flags[job_id] = done_event

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
            except (
                T1DError,
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
            ) as e:
                with self._lock:
                    self._jobs[job_id].status = "failed"
                    self._jobs[job_id].error = str(e)
            finally:
                with self._lock:
                    self._jobs[job_id].completed_at = datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return job_id

    def wait_for_job(
        self, job_id: str, timeout: float = 5.0
    ) -> Optional[TrainingJobDict]:
        """
        Wait deterministically for a background training job to reach a terminal state.

        Args:
        ----
            job_id: The job ID string to wait on.
            timeout: Maximum seconds to wait before returning the current state.

        Returns:
        -------
            Optional[TrainingJobDict]: Job details dictionary or None if not found.

        """
        with self._lock:
            done_event = self._done_flags.get(job_id)
        if done_event is not None:
            done_event.wait(timeout=timeout)
        return self.get_job(job_id)

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

    def prune_completed_jobs(self, max_retained: int = 100) -> int:
        """
        Prune terminal jobs ('completed', 'failed', 'cancelled') when job history exceeds `max_retained`.

        Args:
        ----
            max_retained: Maximum number of terminal jobs to retain in memory.

        Returns:
        -------
            int: Number of pruned terminal job records.

        """
        with self._lock:
            terminal_ids = [
                jid
                for jid, state in self._jobs.items()
                if state.status in ("completed", "failed", "cancelled")
            ]
            excess = max(0, len(terminal_ids) - max(0, max_retained))
            for jid in terminal_ids[:excess]:
                self._jobs.pop(jid, None)
                self._cancel_flags.pop(jid, None)
                self._done_flags.pop(jid, None)
            return excess


class ChatJobManager:
    """Thread-safe asynchronous background manager for chat and NL-to-SQL evaluation jobs."""

    def __init__(self) -> None:
        """Initialize the chat job manager with thread locks and registry collections."""
        self._lock = threading.Lock()
        self._jobs: Dict[str, ChatJobState] = {}
        self._done_flags: Dict[str, threading.Event] = {}
        self._eval_meta: Dict[str, tuple[str, str]] = {}

    def submit_chat_job(
        self,
        request: ChatRequest,
        worker_fn: Callable[[ChatRequest, Callable[[str], None]], ChatResponse],
    ) -> ChatJobDict:
        """
        Submit a new chat turn to execute asynchronously in a background daemon thread.

        Args:
        ----
            request: Validated `ChatRequest` payload.
            worker_fn: Callable executing the chat turn and accepting `(request, on_progress)`.

        Returns:
        -------
            ChatJobDict: Immediate pending job ticket (`{"job_id": ..., "status": "pending", ...}`).

        """
        self.prune_completed_chat_jobs(max_retained=200)
        job_id = f"chatjob-{uuid.uuid4().hex[:12]}"
        done_event = threading.Event()
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        job_state = ChatJobState(
            job_id=job_id,
            status="pending",
            created_at=created_at,
            partial_content="",
        )
        with self._lock:
            self._jobs[job_id] = job_state
            self._done_flags[job_id] = done_event

        def _on_progress(chunk: str) -> None:
            """Append a live partial token or progress text to the active job state."""
            if not chunk:
                return
            with self._lock:
                job_state.partial_content += chunk

        def _worker() -> None:
            """Execute the chat turn in the background and transition job state to complete or failed."""
            try:
                resp = worker_fn(request, _on_progress)
                with self._lock:
                    job_state.status = "complete"
                    job_state.result = resp.model_copy(
                        update={
                            "job_id": job_id,
                            "status": "complete",
                            "partial_content": job_state.partial_content
                            or resp.content,
                        }
                    )
                    job_state.error = resp.error
                    if not job_state.partial_content:
                        job_state.partial_content = resp.content
            except (
                T1DError,
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                err_detail = exception_to_error_detail(exc)
                api_err = ApiError(
                    error_code=err_detail["error_code"],
                    params=err_detail["params"],
                )
                with self._lock:
                    job_state.status = "failed"
                    job_state.error = api_err
                    job_state.result = ChatResponse(
                        job_id=job_id,
                        status="failed",
                        content=job_state.partial_content,
                        partial_content=job_state.partial_content,
                        error=api_err,
                    )
            finally:
                with self._lock:
                    job_state.completed_at = datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return {
            "job_id": job_id,
            "status": "pending",
            "created_at": created_at,
            "completed_at": None,
            "partial_content": "",
            "content": "",
        }

    def submit_eval_job(
        self,
        request: NlToSqlEvalRequest,
        worker_fn: Callable[
            [NlToSqlEvalRequest, Callable[[str], None]], NlToSqlEvalResponse
        ],
    ) -> ChatJobDict:
        """
        Submit a new NL-to-SQL evaluation turn to execute asynchronously in a background daemon thread.

        Args:
        ----
            request: Validated `NlToSqlEvalRequest` payload.
            worker_fn: Callable executing the evaluation turn and accepting `(request, on_progress)`.

        Returns:
        -------
            ChatJobDict: Immediate pending job ticket (`{"job_id": ..., "status": "pending", ...}`).

        """
        self.prune_completed_chat_jobs(max_retained=200)
        job_id = f"chatjob-{uuid.uuid4().hex[:12]}"
        done_event = threading.Event()
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        eval_state = ChatJobState(
            job_id=job_id,
            status="pending",
            created_at=created_at,
            partial_content="",
        )
        with self._lock:
            self._jobs[job_id] = eval_state
            self._done_flags[job_id] = done_event
            self._eval_meta[job_id] = (request.question, request.model)

        def _on_progress(chunk: str) -> None:
            """Append a live partial token or progress text to the active evaluation job state."""
            if not chunk:
                return
            with self._lock:
                eval_state.partial_content += chunk

        def _worker() -> None:
            """Execute the evaluation turn in the background and transition job state to complete or failed."""
            try:
                eval_resp = worker_fn(request, _on_progress)
                chat_equiv = ChatResponse(
                    job_id=job_id,
                    status="complete",
                    content=eval_resp.content,
                    partial_content=eval_resp.content,
                    sqlQuery=eval_resp.sqlQuery,
                    sqlResult=eval_resp.sqlResult,
                    gold_comparison=eval_resp.gold_comparison,
                    is_rule_based_sql=eval_resp.is_rule_based_sql,
                    generation_mode=eval_resp.generation_mode,
                    is_synthetic_data=eval_resp.is_synthetic_data,
                    data_provenance=eval_resp.data_provenance,
                    error=eval_resp.error,
                )
                with self._lock:
                    eval_state.status = "complete"
                    eval_state.eval_result = eval_resp.model_copy(
                        update={
                            "job_id": job_id,
                            "status": "complete",
                            "partial_content": eval_state.partial_content
                            or eval_resp.content,
                        }
                    )
                    eval_state.result = chat_equiv
                    eval_state.error = eval_resp.error
                    if not eval_state.partial_content:
                        eval_state.partial_content = eval_resp.content
            except (
                T1DError,
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                err_detail = exception_to_error_detail(exc)
                api_err = ApiError(
                    error_code=err_detail["error_code"],
                    params=err_detail["params"],
                )
                with self._lock:
                    eval_state.status = "failed"
                    eval_state.error = api_err
                    eval_state.eval_result = NlToSqlEvalResponse(
                        job_id=job_id,
                        status="failed",
                        question=request.question,
                        model=request.model,
                        content=eval_state.partial_content,
                        partial_content=eval_state.partial_content,
                        error=api_err,
                    )
                    eval_state.result = ChatResponse(
                        job_id=job_id,
                        status="failed",
                        content=eval_state.partial_content,
                        partial_content=eval_state.partial_content,
                        error=api_err,
                    )
            finally:
                with self._lock:
                    eval_state.completed_at = datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return {
            "job_id": job_id,
            "status": "pending",
            "created_at": created_at,
            "completed_at": None,
            "partial_content": "",
            "content": "",
            "question": request.question,
            "model": request.model,
        }

    def get_chat_job(self, job_id: str) -> Optional[Dict[str, ChatJobPayloadValue]]:
        """
        Retrieve the current status or completed `ChatResponse` dictionary for `job_id`.

        Args:
        ----
            job_id: Target chat job identifier.

        Returns:
        -------
            Optional[Dict[str, ChatJobPayloadValue]]: Serialized job dictionary or None if not found.

        """
        with self._lock:
            state = self._jobs.get(job_id)
            return state.to_dict() if state is not None else None

    def get_chat_job_response(self, job_id: str) -> Optional[ChatResponse]:
        """
        Retrieve the current status or completed `ChatResponse` model for `job_id`.

        Args:
        ----
            job_id: Target chat job identifier.

        Returns:
        -------
            Optional[ChatResponse]: `ChatResponse` model (with `status="pending"` or `"complete"`) or None.

        """
        with self._lock:
            state = self._jobs.get(job_id)
            return state.to_chat_response() if state is not None else None

    def get_eval_job_response(self, job_id: str) -> Optional[NlToSqlEvalResponse]:
        """
        Retrieve the current status or completed `NlToSqlEvalResponse` model for `job_id`.

        Args:
        ----
            job_id: Target evaluation job identifier.

        Returns:
        -------
            Optional[NlToSqlEvalResponse]: `NlToSqlEvalResponse` model or None if not found.

        """
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            q, m = self._eval_meta.get(job_id, ("", "gemma:2b"))
            return state.to_eval_response(fallback_question=q, fallback_model=m)

    def wait_for_chat_job(
        self, job_id: str, timeout: float = 5.0
    ) -> Optional[Dict[str, ChatJobPayloadValue]]:
        """
        Wait deterministically for a background chat job to reach a terminal state.

        Args:
        ----
            job_id: Target chat job identifier.
            timeout: Maximum seconds to wait before returning the current state.

        Returns:
        -------
            Optional[Dict[str, ChatJobPayloadValue]]: Job dictionary or None if not found.

        """
        with self._lock:
            done_event = self._done_flags.get(job_id)
        if done_event is not None:
            done_event.wait(timeout=timeout)
        return self.get_chat_job(job_id)

    def prune_completed_chat_jobs(self, max_retained: int = 200) -> int:
        """
        Prune terminal chat jobs ('complete', 'failed') when retained history exceeds `max_retained`.

        Args:
        ----
            max_retained: Maximum number of terminal chat jobs to retain in memory.

        Returns:
        -------
            int: Number of pruned terminal chat job records.

        """
        with self._lock:
            terminal_ids = [
                jid
                for jid, state in self._jobs.items()
                if state.status in ("complete", "failed")
            ]
            excess = max(0, len(terminal_ids) - max(0, max_retained))
            for jid in terminal_ids[:excess]:
                self._jobs.pop(jid, None)
                self._done_flags.pop(jid, None)
                self._eval_meta.pop(jid, None)
            return excess


global_chat_job_manager = ChatJobManager()
