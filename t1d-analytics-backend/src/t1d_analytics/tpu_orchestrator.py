"""Google Cloud TPU MaxText orchestration, XPK/gcloud command builders, and GCS synchronization."""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from t1d_analytics.errors import (
    HardwareUnavailableError,
    TpuOrchestrationError,
    TpuPollingTimeoutError,
    TpuPreemptionError,
)
from t1d_analytics.gcs_storage import (
    GcsBlobProtocol,
    GcsBucketProtocol,
    GcsClientProtocol,
    Md5BlobProtocol,
    download_checkpoints_from_gcs,
    parse_gcs_uri,
    sync_dataset_to_gcs,
    verify_gcs_blob_md5,
)
from t1d_analytics.hf_models import BaseTrainingRunner
from t1d_analytics.job_manager import JobMetricValue
from t1d_analytics.models import MaxTextConfig, TrainingJobConfig


@runtime_checkable
class SubprocessExecutorProtocol(Protocol):
    """Injectable protocol for executing CLI subprocess commands."""

    def __call__(
        self,
        cmd: Sequence[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a command and return a CompletedProcess record."""
        ...


def _build_xpk_create_command(
    xpk_bin: str,
    workload_name: str,
    cluster_name: str,
    max_cfg: MaxTextConfig,
) -> List[str]:
    """
    Build the `xpk workload create` CLI command vector for TPU workload dispatch.

    Args:
    ----
        xpk_bin: Path to the `xpk` executable.
        workload_name: Dispatched workload identifier.
        cluster_name: Target GKE TPU cluster name.
        max_cfg: MaxText hardware and topology configuration.

    Returns:
    -------
        List[str]: Argument list for `subprocess.run`.

    """
    return [
        xpk_bin,
        "workload",
        "create",
        f"--workload={workload_name}",
        f"--cluster={cluster_name}",
        f"--tpu-type={max_cfg.accelerator_type.value}",
        f"--num-slices={max_cfg.num_slices}",
        f"--zone={max_cfg.zone}",
    ]


def _build_gcloud_ssh_command(
    gcloud_bin: str,
    workload_name: str,
    max_cfg: MaxTextConfig,
    config_filename: str,
) -> List[str]:
    """
    Build the `gcloud compute tpus tpu-vm ssh` CLI command vector for direct TPU VM execution.

    Args:
    ----
        gcloud_bin: Path to the `gcloud` executable.
        workload_name: Target TPU VM instance name.
        max_cfg: MaxText hardware and zone configuration.
        config_filename: Name of the generated MaxText YAML file.

    Returns:
    -------
        List[str]: Argument list for `subprocess.run`.

    """
    return [
        gcloud_bin,
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        workload_name,
        f"--zone={max_cfg.zone}",
        f"--command=maxtext {config_filename}",
    ]


class RemoteTpuMaxTextRunner(BaseTrainingRunner):
    """Training runner orchestrating MaxText jobs on Google Cloud TPU clusters."""

    def __init__(
        self,
        subprocess_executor: Optional[SubprocessExecutorProtocol] = None,
    ) -> None:
        """
        Initialize the Cloud TPU MaxText runner with an optional subprocess executor.

        Args:
        ----
            subprocess_executor: Optional callable matching `SubprocessExecutorProtocol`
                (defaults to `subprocess.run`).

        """
        self._subprocess_executor = subprocess_executor

    def _run_subprocess(
        self,
        cmd: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute a CLI command via the configured `SubprocessExecutorProtocol` or `subprocess.run`.

        Args:
        ----
            cmd: Command argument sequence.

        Returns:
        -------
            subprocess.CompletedProcess[str]: Completed process record.

        """
        if self._subprocess_executor is not None:
            return self._subprocess_executor(
                cmd, capture_output=True, text=True, check=False
            )
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def validate_environment(self) -> bool:
        """
        Validate that Google Cloud CLI or Libscript toolchain is installed.

        Returns
        -------
            bool: True if remote TPU environment tools are present.

        Raises
        ------
            HardwareUnavailableError: If gcloud or xpk CLI is missing.

        """
        has_gcloud = shutil.which("gcloud") is not None
        has_libscript = (
            Path(os.environ.get("LIBSCRIPT_ROOT_DIR", "~/repos/libscript"))
            .expanduser()
            .exists()
        )
        if not (has_gcloud or has_libscript):
            raise HardwareUnavailableError(
                "Google Cloud CLI (gcloud) or Libscript orchestrator not found. "
                "Install Google Cloud CLI or set LIBSCRIPT_ROOT_DIR."
            )
        return True

    def generate_maxtext_yaml(self, config: TrainingJobConfig) -> str:
        """
        Generate MaxText YAML configuration file content for Cloud TPU execution.

        Args:
        ----
            config: Structured training configuration.

        Returns:
        -------
            str: Generated YAML configuration content.

        """
        max_cfg: MaxTextConfig = config.maxtext_config or MaxTextConfig(
            project_id=os.environ.get("GCP_PROJECT_ID", "t1d-analytics-gcp")
        )
        max_cfg.validate_tfrc_alignment()

        hparams = config.hyperparameters
        batch_size = hparams.batch_size if hparams else 4
        lr = hparams.learning_rate if hparams else 2e-5
        steps = (hparams.num_epochs if hparams else 3) * 50

        full_params = max_cfg.build_full_maxtext_parameters()
        lines = [
            f"# Generated MaxText Configuration for {config.model_name}",
            f"model_name: {config.model_name}",
            f"base_output_directory: '{max_cfg.bucket_url}/models/{config.model_name}'",
            f"dataset_path: '{config.dataset_path}'",
            f"learning_rate: {lr}",
            f"per_device_batch_size: {batch_size}",
            f"steps: {steps}",
            f"hardware: '{max_cfg.accelerator_type.value}'",
            f"zone: '{max_cfg.zone}'",
            f"num_slices: {max_cfg.num_slices}",
        ]
        for k, v in full_params.items():
            if k not in (
                "model_name",
                "base_output_directory",
                "dataset_path",
                "learning_rate",
                "per_device_batch_size",
                "steps",
                "hardware",
                "zone",
                "num_slices",
            ):
                lines.append(f"{k}: {v}")
        nl = chr(10)
        return nl.join(lines) + nl

    def _dispatch_xpk_workload(
        self,
        config: TrainingJobConfig,
        max_cfg: MaxTextConfig,
        config_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        """
        Dispatch MaxText workload to GKE TPU cluster using XPK toolchain.

        Args:
        ----
            config: Training job configuration.
            max_cfg: MaxText infrastructure and hardware parameters.
            config_path: Path to the generated MaxText YAML configuration file.

        Returns:
        -------
            subprocess.CompletedProcess[str]: Process execution record.

        Raises:
        ------
            TpuOrchestrationError: If dispatch fails or returns non-zero code.

        """
        workload_name = f"t1d-{config.model_name.replace('/', '-').replace(':', '-')}"
        cluster_name = f"tpu-cluster-{max_cfg.accelerator_type.value}"

        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            gcloud_bin = shutil.which("gcloud")
            if not gcloud_bin:
                raise TpuOrchestrationError(
                    "Neither 'xpk' nor 'gcloud' CLI is installed."
                )
            cmd = _build_gcloud_ssh_command(
                gcloud_bin, workload_name, max_cfg, config_path.name
            )
        else:
            cmd = _build_xpk_create_command(
                xpk_bin, workload_name, cluster_name, max_cfg
            )

        proc = self._run_subprocess(cmd)
        if proc.returncode != 0:
            raise TpuOrchestrationError(f"TPU dispatch failed: {proc.stderr.strip()}")
        return proc

    @staticmethod
    def _parse_gcloud_describe_response(
        workload_name: str,
        zone: str,
        res: subprocess.CompletedProcess[str],
    ) -> Optional[str]:
        """
        Parse a single `gcloud compute tpus tpu-vm describe` subprocess result.

        Args:
        ----
            workload_name: TPU VM instance identifier.
            zone: GCP availability zone.
            res: Completed subprocess result.

        Returns:
        -------
            Optional[str]: `'SUCCESS'` if ready, or `None` if still provisioning/pending.

        Raises:
        ------
            TpuPreemptionError: If the instance state or healthDescription indicates preemption.
            TpuOrchestrationError: If the instance encountered a fatal error or is not found.

        """
        if res.returncode == 0:
            try:
                data = json.loads(res.stdout)
                state = str(data.get("state", "")).upper()
                health_desc = str(data.get("healthDescription", "")).upper()
                if state == "PREEMPTED" or "PREEMPTED" in health_desc:
                    raise TpuPreemptionError(
                        f"Workload '{workload_name}' was PREEMPTED on spot TPU capacity."
                    )
                if "ERROR" in health_desc or state == "FAILED":
                    raise TpuOrchestrationError(
                        f"Workload '{workload_name}' encountered fatal error: {data.get('healthDescription')}"
                    )
                if state in ("READY", "ACTIVE", "RUNNING", "STOPPED", "TERMINATED"):
                    return "SUCCESS"
            except json.JSONDecodeError:
                if "READY" in res.stdout.upper():
                    return "SUCCESS"
        elif "NOT_FOUND" in res.stderr.upper():
            raise TpuOrchestrationError(
                f"Workload '{workload_name}' not found in zone '{zone}'."
            )
        return None

    @staticmethod
    def _parse_xpk_describe_response(
        workload_name: str,
        res: subprocess.CompletedProcess[str],
    ) -> Optional[str]:
        """
        Parse a single `xpk workload describe` subprocess result.

        Args:
        ----
            workload_name: Dispatched XPK workload identifier.
            res: Completed subprocess result.

        Returns:
        -------
            Optional[str]: `'SUCCESS'` or `'FAILED'` if terminal, or `None` if still running/pending.

        Raises:
        ------
            TpuPreemptionError: If output indicates spot TPU preemption.
            TpuOrchestrationError: If output indicates fatal XLA compilation error.

        """
        if res.returncode == 0:
            stdout_upper = res.stdout.upper()
            if "PREEMPTED" in stdout_upper:
                raise TpuPreemptionError(
                    f"Workload '{workload_name}' was PREEMPTED on spot TPU capacity."
                )
            if "XLA" in stdout_upper and "ERROR" in stdout_upper:
                raise TpuOrchestrationError(
                    f"Workload '{workload_name}' encountered fatal XLA compilation error."
                )
            if (
                "SUCCESS" in stdout_upper
                or "FINISHED" in stdout_upper
                or "COMPLETED" in stdout_upper
            ):
                return "SUCCESS"
            if "FAILED" in stdout_upper or "ERROR" in stdout_upper:
                return "FAILED"
        return None

    def _build_gcloud_describe_cmd(self, workload_name: str, zone: str) -> List[str]:
        """
        Build the `gcloud compute tpus tpu-vm describe` command and validate `gcloud` availability.

        Args:
        ----
            workload_name: Identifier of the TPU VM instance.
            zone: GCP availability zone of the TPU VM instance.

        Returns:
        -------
            List[str]: Subprocess argument list for describing the TPU VM.

        Raises:
        ------
            TpuOrchestrationError: If Google Cloud CLI (`gcloud`) is not installed.

        """
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            raise TpuOrchestrationError("Google Cloud CLI (gcloud) is not installed.")
        return [
            gcloud_bin,
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            workload_name,
            f"--zone={zone}",
            "--format=json",
        ]

    def _build_gcloud_delete_cmd(self, workload_name: str, zone: str) -> List[str]:
        """
        Build the `gcloud compute tpus tpu-vm delete` command and validate `gcloud` availability.

        Args:
        ----
            workload_name: Identifier of the TPU VM instance.
            zone: GCP availability zone of the TPU VM instance.

        Returns:
        -------
            List[str]: Subprocess argument list for deleting the TPU VM.

        Raises:
        ------
            TpuOrchestrationError: If Google Cloud CLI (`gcloud`) is not installed.

        """
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            raise TpuOrchestrationError("Google Cloud CLI (gcloud) is not installed.")
        return [
            gcloud_bin,
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            workload_name,
            f"--zone={zone}",
            "--quiet",
        ]

    def _step_gcloud_tpu_poll(
        self,
        cmd: Sequence[str],
        workload_name: str,
        zone: str,
    ) -> Optional[str]:
        """
        Execute a single `gcloud compute tpus tpu-vm describe` command and parse its output.

        Args:
        ----
            cmd: Prepared gcloud describe command sequence.
            workload_name: TPU VM instance identifier.
            zone: GCP availability zone.

        Returns:
        -------
            Optional[str]: `'SUCCESS'` if ready, or `None` if still pending.

        """
        res = self._run_subprocess(cmd)
        return self._parse_gcloud_describe_response(workload_name, zone, res)

    def _poll_gcloud_tpu_status_sync(
        self,
        workload_name: str,
        zone: str,
        timeout_seconds: float = 30.0,
        cancel_event: Optional[threading.Event] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> str:
        """
        Poll TPU VM workload status synchronously using Google Cloud CLI (gcloud).

        Args:
        ----
            workload_name: Identifier of the TPU VM instance.
            zone: GCP availability zone of the TPU VM instance.
            timeout_seconds: Timeout threshold in seconds.
            cancel_event: Optional cooperative cancellation event.
            sleep_fn: Injectable sleep function for backoff intervals.

        Returns:
        -------
            str: Normalized lifecycle status ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuOrchestrationError: If gcloud CLI is missing, instance is not found, cancelled, or preemption occurs.
            TpuPollingTimeoutError: If polling exceeds timeout threshold.

        """
        cmd = self._build_gcloud_describe_cmd(workload_name, zone)
        start = time.time()
        while time.time() - start < timeout_seconds:
            if cancel_event is not None and cancel_event.is_set():
                self._reclaim_gcloud_tpu_workload_sync(workload_name, zone)
                raise TpuOrchestrationError(
                    f"Workload '{workload_name}' cancelled and reclaimed."
                )
            parsed = self._step_gcloud_tpu_poll(cmd, workload_name, zone)
            if parsed is not None:
                return parsed
            sleep_fn(min(0.02, timeout_seconds / 2.0))

        raise TpuPollingTimeoutError(
            f"Workload '{workload_name}' polling timed out after {timeout_seconds} seconds."
        )

    async def _poll_gcloud_tpu_status(
        self,
        workload_name: str,
        zone: str,
        timeout_seconds: float = 30.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Poll TPU VM workload status using Google Cloud CLI (gcloud).

        Args:
        ----
            workload_name: Identifier of the TPU VM instance.
            zone: GCP availability zone of the TPU VM instance.
            timeout_seconds: Timeout threshold in seconds.
            cancel_event: Optional cooperative cancellation event.

        Returns:
        -------
            str: Normalized lifecycle status ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuOrchestrationError: If gcloud CLI is missing, instance is not found, cancelled, or preemption occurs.
            TpuPollingTimeoutError: If polling exceeds timeout threshold.

        """
        return self._poll_gcloud_tpu_status_sync(
            workload_name=workload_name,
            zone=zone,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )

    def _reclaim_gcloud_tpu_workload_sync(
        self,
        workload_name: str,
        zone: str,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> bool:
        """
        Reclaim and delete TPU VM instance synchronously using Google Cloud CLI (gcloud).

        Args:
        ----
            workload_name: Identifier of the TPU VM instance to delete.
            zone: GCP availability zone of the TPU VM.
            max_retries: Maximum deletion retry attempts upon failure.
            sleep_fn: Injectable sleep function for retry backoff.

        Returns:
        -------
            bool: True if workload resource was successfully deleted.

        Raises:
        ------
            TpuOrchestrationError: If deletion fails after maximum retries or gcloud CLI is missing.

        """
        cmd = self._build_gcloud_delete_cmd(workload_name, zone)
        last_error = ""
        for attempt in range(1, max_retries + 1):
            res = self._run_subprocess(cmd)
            if res.returncode == 0:
                return True
            last_error = res.stderr.strip()
            if attempt < max_retries:
                sleep_fn(0.01 * (2 ** (attempt - 1)))

        raise TpuOrchestrationError(
            f"Failed to reclaim TPU VM workload '{workload_name}' via gcloud: {last_error}"
        )

    async def _reclaim_gcloud_tpu_workload(
        self,
        workload_name: str,
        zone: str,
        max_retries: int = 3,
    ) -> bool:
        """
        Reclaim and delete TPU VM instance using Google Cloud CLI (gcloud).

        Args:
        ----
            workload_name: Identifier of the TPU VM instance to delete.
            zone: GCP availability zone of the TPU VM.
            max_retries: Maximum deletion retry attempts upon failure.

        Returns:
        -------
            bool: True if workload resource was successfully deleted.

        Raises:
        ------
            TpuOrchestrationError: If deletion fails after maximum retries or gcloud CLI is missing.

        """
        return self._reclaim_gcloud_tpu_workload_sync(
            workload_name=workload_name,
            zone=zone,
            max_retries=max_retries,
        )

    @staticmethod
    def _build_xpk_describe_cmd(
        xpk_bin: str, workload_name: str, max_cfg: MaxTextConfig
    ) -> List[str]:
        """
        Build the `xpk workload describe` command list.

        Args:
        ----
            xpk_bin: Path to the `xpk` executable.
            workload_name: Identifier of the dispatched training workload.
            max_cfg: MaxText infrastructure configuration.

        Returns:
        -------
            List[str]: Subprocess argument list for describing the XPK workload.

        """
        return [
            xpk_bin,
            "workload",
            "describe",
            f"--workload={workload_name}",
            f"--cluster=tpu-cluster-{max_cfg.accelerator_type.value}",
        ]

    @staticmethod
    def _build_xpk_delete_cmd(
        xpk_bin: str, workload_name: str, max_cfg: MaxTextConfig
    ) -> List[str]:
        """
        Build the `xpk workload delete` command list.

        Args:
        ----
            xpk_bin: Path to the `xpk` executable.
            workload_name: Identifier of the workload to delete.
            max_cfg: MaxText infrastructure configuration.

        Returns:
        -------
            List[str]: Subprocess argument list for deleting the XPK workload.

        """
        return [
            xpk_bin,
            "workload",
            "delete",
            f"--workload={workload_name}",
            f"--cluster=tpu-cluster-{max_cfg.accelerator_type.value}",
        ]

    def _step_xpk_poll_or_cancel(
        self,
        xpk_bin: str,
        cmd: Sequence[str],
        workload_name: str,
        max_cfg: MaxTextConfig,
        cancel_event: Optional[threading.Event],
    ) -> Optional[str]:
        """
        Check cancellation or execute a single `xpk workload describe` step.

        Args:
        ----
            xpk_bin: Path to `xpk` executable.
            cmd: Prepared `xpk workload describe` command sequence.
            workload_name: Dispatched workload identifier.
            max_cfg: MaxText infrastructure configuration.
            cancel_event: Optional cooperative cancellation event.

        Returns:
        -------
            Optional[str]: Terminal status string (`'SUCCESS'` or `'FAILED'`) or `None` if pending.

        Raises:
        ------
            TpuOrchestrationError: If cancelled or fatal XLA error occurs.
            TpuPreemptionError: If workload is preempted.

        """
        if cancel_event is not None and cancel_event.is_set():
            self._execute_xpk_reclaim_command(xpk_bin, workload_name, max_cfg)
            raise TpuOrchestrationError(
                f"Workload '{workload_name}' cancelled and reclaimed."
            )
        res = self._run_subprocess(cmd)
        return self._parse_xpk_describe_response(workload_name, res)

    def _poll_workload_status_sync(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
        timeout_seconds: float = 30.0,
        cancel_event: Optional[threading.Event] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> str:
        """
        Poll workload execution status synchronously from XPK or Google Cloud TPU API.

        Args:
        ----
            workload_name: Identifier of the dispatched training workload.
            max_cfg: MaxText infrastructure configuration.
            timeout_seconds: Timeout threshold in seconds.
            cancel_event: Optional cooperative cancellation event.
            sleep_fn: Injectable sleep function for polling intervals.

        Returns:
        -------
            str: Lifecycle status string ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuPollingTimeoutError: If polling exceeds specified timeout duration.
            TpuOrchestrationError: If workload is preempted, cancelled, or encounters an execution failure.

        """
        if cancel_event is not None and cancel_event.is_set():
            self._reclaim_tpu_workload_sync(workload_name, max_cfg)
            raise TpuOrchestrationError(
                f"Workload '{workload_name}' cancelled and reclaimed."
            )

        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            return self._poll_gcloud_tpu_status_sync(
                workload_name=workload_name,
                zone=max_cfg.zone,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
                sleep_fn=sleep_fn,
            )

        cmd = self._build_xpk_describe_cmd(xpk_bin, workload_name, max_cfg)
        start = time.time()
        while time.time() - start < timeout_seconds:
            parsed = self._step_xpk_poll_or_cancel(
                xpk_bin, cmd, workload_name, max_cfg, cancel_event
            )
            if parsed is not None:
                return parsed
            sleep_fn(min(0.02, timeout_seconds / 2.0))

        raise TpuPollingTimeoutError(
            f"Workload '{workload_name}' polling timed out after {timeout_seconds} seconds."
        )

    async def poll_workload_status(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
        timeout_seconds: float = 30.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Poll workload execution status from XPK or Google Cloud TPU API.

        Args:
        ----
            workload_name: Identifier of the dispatched training workload.
            max_cfg: MaxText infrastructure configuration.
            timeout_seconds: Timeout threshold in seconds.
            cancel_event: Optional cooperative cancellation event.

        Returns:
        -------
            str: Lifecycle status string ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuPollingTimeoutError: If polling exceeds specified timeout duration.
            TpuOrchestrationError: If workload is preempted, cancelled, or encounters an execution failure.

        """
        if cancel_event is not None and cancel_event.is_set():
            await self.reclaim_tpu_workload(workload_name, max_cfg)
            raise TpuOrchestrationError(
                f"Workload '{workload_name}' cancelled and reclaimed."
            )
        if not shutil.which("xpk"):
            return await self._poll_gcloud_tpu_status(
                workload_name=workload_name,
                zone=max_cfg.zone,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        return self._poll_workload_status_sync(
            workload_name=workload_name,
            max_cfg=max_cfg,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )

    def _execute_xpk_reclaim_command(
        self,
        xpk_bin: str,
        workload_name: str,
        max_cfg: MaxTextConfig,
    ) -> bool:
        """
        Execute `xpk workload delete` for a TPU workload and validate return code.

        Args:
        ----
            xpk_bin: Path to the `xpk` executable.
            workload_name: Identifier of the workload to delete.
            max_cfg: MaxText infrastructure parameters.

        Returns:
        -------
            bool: True if workload deletion succeeded.

        Raises:
        ------
            TpuOrchestrationError: If `xpk workload delete` exits with a non-zero code.

        """
        cmd = self._build_xpk_delete_cmd(xpk_bin, workload_name, max_cfg)
        res = self._run_subprocess(cmd)
        if res.returncode != 0:
            raise TpuOrchestrationError(
                f"Failed to reclaim TPU workload: {res.stderr.strip()}"
            )
        return True

    def _reclaim_tpu_workload_sync(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
    ) -> bool:
        """
        Reclaim and delete queued or terminated TPU workload resources synchronously.

        Args:
        ----
            workload_name: Identifier of the workload to reclaim.
            max_cfg: MaxText infrastructure parameters.

        Returns:
        -------
            bool: True if resources were reclaimed.

        Raises:
        ------
            TpuOrchestrationError: If reclamation command returns non-zero error.

        """
        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            return self._reclaim_gcloud_tpu_workload_sync(
                workload_name=workload_name,
                zone=max_cfg.zone,
            )
        return self._execute_xpk_reclaim_command(xpk_bin, workload_name, max_cfg)

    async def reclaim_tpu_workload(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
    ) -> bool:
        """
        Reclaim and delete queued or terminated TPU workload resources.

        Args:
        ----
            workload_name: Identifier of the workload to reclaim.
            max_cfg: MaxText infrastructure parameters.

        Returns:
        -------
            bool: True if resources were reclaimed.

        Raises:
        ------
            TpuOrchestrationError: If reclamation command returns non-zero error.

        """
        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            return await self._reclaim_gcloud_tpu_workload(
                workload_name=workload_name,
                zone=max_cfg.zone,
            )
        return self._execute_xpk_reclaim_command(xpk_bin, workload_name, max_cfg)

    async def download_checkpoints_from_gcs(
        self,
        bucket_url: str,
        output_dir: Path,
        max_retries: int = 3,
        validate_checksum: bool = True,
        gcs_client: Optional[GcsClientProtocol] = None,
    ) -> Path:
        """
        Download trained model checkpoints from Google Cloud Storage to local directory.

        Args:
        ----
            bucket_url: GCS bucket URI containing model checkpoints.
            output_dir: Destination local directory path.
            max_retries: Maximum download retry attempts.
            validate_checksum: Whether to verify download checksum against manifest.
            gcs_client: Optional injectable GCS client instance.

        Returns:
        -------
            Path: Path to the local directory containing downloaded checkpoints.

        """
        return await download_checkpoints_from_gcs(
            bucket_url=bucket_url,
            output_dir=output_dir,
            max_retries=max_retries,
            validate_checksum=validate_checksum,
            gcs_client=gcs_client,
        )

    def _dispatch_with_preemption_retries(
        self,
        config: TrainingJobConfig,
        max_cfg: MaxTextConfig,
        config_path: Path,
        workload_name: str,
        cancel_event: Optional[threading.Event],
    ) -> str:
        """
        Dispatch an XPK/gcloud TPU workload and retry up to `max_preemption_retries` times on preemption.

        Args:
        ----
            config: Training job configuration.
            max_cfg: MaxText infrastructure configuration.
            config_path: Path to the generated MaxText YAML file.
            workload_name: Dispatched workload identifier.
            cancel_event: Optional cooperative cancellation event.

        Returns:
        -------
            str: Terminal workload status string.

        Raises:
        ------
            RuntimeError: If the workload fails or exhausts preemption retries (after reclaiming resources).

        """
        self._dispatch_xpk_workload(config, max_cfg, config_path)
        try:
            return self._poll_workload_status_sync(
                workload_name,
                max_cfg,
                timeout_seconds=5,
                cancel_event=cancel_event,
            )
        except RuntimeError as e:
            if "PREEMPTED" not in str(e):
                self._reclaim_tpu_workload_sync(workload_name, max_cfg)
                raise
            for _ in range(max_cfg.max_preemption_retries):
                self._dispatch_xpk_workload(config, max_cfg, config_path)
                try:
                    return self._poll_workload_status_sync(
                        workload_name,
                        max_cfg,
                        timeout_seconds=2,
                        cancel_event=cancel_event,
                    )
                except RuntimeError:
                    continue
            self._reclaim_tpu_workload_sync(workload_name, max_cfg)
            raise

    @staticmethod
    def _write_job_manifest(
        out_path: Path,
        workload_name: str,
        tpu_cluster: str,
        config_path: Path,
        model_name: str,
        status_str: str,
    ) -> None:
        """
        Write `job_manifest.json` summarizing the dispatched TPU training job.

        Args:
        ----
            out_path: Output directory path.
            workload_name: Dispatched workload identifier.
            tpu_cluster: Target TPU cluster accelerator string.
            config_path: Path to `maxtext_config.yml`.
            model_name: Trained model name.
            status_str: Resolved dispatch/polling status string.

        """
        manifest_path = out_path / "job_manifest.json"
        with open(manifest_path, "w") as mf:
            json.dump(
                {
                    "workload_name": workload_name,
                    "tpu_cluster": tpu_cluster,
                    "config_file": str(config_path),
                    "output_dir": str(out_path),
                    "model_name": model_name,
                    "dispatched_at": time.time(),
                    "status": status_str,
                },
                mf,
                indent=2,
            )

    def run_training(
        self,
        config: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Dispatch MaxText workload to Cloud TPU cluster.

        Args:
        ----
            config: Training job configuration.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Dict[str, JobMetricValue]: Remote workload launch details and generated configuration.

        """
        self.prepare_dataset(config)
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if (
            config.maxtext_config
            and config.maxtext_config.bucket_url
            and (
                config.maxtext_config.auto_sync_gcs
                or os.environ.get("T1D_AUTO_SYNC_GCS") == "1"
            )
            and not config.dry_run
        ):
            sync_dataset_to_gcs(
                Path(config.dataset_path), config.maxtext_config.bucket_url
            )

        yaml_content = self.generate_maxtext_yaml(config)
        config_path = out_path / "maxtext_config.yml"
        with open(config_path, "w") as f:
            f.write(yaml_content)

        workload_name = f"t1d-{config.model_name.replace('/', '-').replace(':', '-')}"
        tpu_cluster = (
            config.maxtext_config.accelerator_type.value
            if config.maxtext_config
            else "v4-8"
        )

        status_str = "dispatched"
        if not config.dry_run:
            should_dispatch = (
                config.maxtext_config is not None
                and config.maxtext_config.execute_tpu_dispatch
            ) or os.environ.get("T1D_EXECUTE_TPU_DISPATCH") == "1"
            if should_dispatch:
                max_cfg_dispatch = config.maxtext_config or MaxTextConfig(
                    project_id=os.environ.get("GCP_PROJECT_ID", "t1d-analytics-gcp")
                )
                status_str = self._dispatch_with_preemption_retries(
                    config=config,
                    max_cfg=max_cfg_dispatch,
                    config_path=config_path,
                    workload_name=workload_name,
                    cancel_event=cancel_event,
                )

            self._write_job_manifest(
                out_path=out_path,
                workload_name=workload_name,
                tpu_cluster=tpu_cluster,
                config_path=config_path,
                model_name=config.model_name,
                status_str=status_str,
            )

        return {
            "status": "dry_run_completed" if config.dry_run else status_str,
            "backend": config.backend.value,
            "model_name": config.model_name,
            "config_file": str(config_path),
            "output_dir": str(out_path),
            "tpu_cluster": tpu_cluster,
            "workload_name": workload_name,
        }


__all__ = [
    "GcsBlobProtocol",
    "GcsBucketProtocol",
    "GcsClientProtocol",
    "Md5BlobProtocol",
    "RemoteTpuMaxTextRunner",
    "SubprocessExecutorProtocol",
    "_build_gcloud_ssh_command",
    "_build_xpk_create_command",
    "download_checkpoints_from_gcs",
    "parse_gcs_uri",
    "sync_dataset_to_gcs",
    "verify_gcs_blob_md5",
]
