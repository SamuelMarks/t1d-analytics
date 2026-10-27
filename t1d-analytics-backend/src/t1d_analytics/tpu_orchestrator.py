"""Google Cloud TPU MaxText orchestration, XPK/gcloud command builders, and GCS synchronization."""

import asyncio
import base64
import hashlib
import importlib
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
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from t1d_analytics.errors import (
    ChecksumMismatchError,
    DatabaseNotFoundError,
    GcsStorageError,
    HardwareUnavailableError,
    InvalidGcsUriError,
    MissingSdkError,
    TpuOrchestrationError,
    TpuPollingTimeoutError,
    TpuPreemptionError,
)
from t1d_analytics.hf_models import BaseTrainingRunner
from t1d_analytics.job_manager import JobMetricValue
from t1d_analytics.models import MaxTextConfig, TrainingJobConfig


class GcsBlobProtocol(Protocol):
    """Protocol representing a Google Cloud Storage Blob."""

    name: str
    md5_hash: Optional[str]

    def download_to_filename(self, filename: str) -> None:
        """Download blob contents to local file."""
        ...

    def upload_from_filename(self, filename: str) -> None:
        """Upload local file contents to blob."""
        ...


@runtime_checkable
class Md5BlobProtocol(Protocol):
    """Runtime-checkable protocol for GCS blobs exposing md5_hash."""

    @property
    def md5_hash(self) -> Optional[str]:
        """Return base64-encoded MD5 hash string."""
        ...


class GcsBucketProtocol(Protocol):
    """Protocol representing a Google Cloud Storage Bucket."""

    def blob(self, blob_name: str) -> GcsBlobProtocol:
        """Create a blob reference within this bucket."""
        ...


class GcsClientProtocol(Protocol):
    """Protocol representing a Google Cloud Storage Client."""

    def bucket(self, bucket_name: str) -> GcsBucketProtocol:
        """Return bucket reference by name."""
        ...

    def list_blobs(
        self, bucket_or_name: Union[GcsBucketProtocol, str], prefix: str = ""
    ) -> Iterable[GcsBlobProtocol]:
        """List blobs in bucket matching prefix."""
        ...


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


def _compute_md5_base64(payload: bytes) -> str:
    """
    Compute the base64-encoded MD5 digest of a byte payload for GCS integrity verification.

    Args:
    ----
        payload: Raw file bytes to hash.

    Returns:
    -------
        str: ASCII base64-encoded MD5 digest string.

    """
    return base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")


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

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
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

    @staticmethod
    def _evaluate_status_override(
        workload_name: str,
        override_status: Optional[str],
    ) -> Optional[str]:
        """
        Evaluate an explicit status override string if provided.

        Args:
        ----
            workload_name: Dispatched workload identifier.
            override_status: Optional status override string.

        Returns:
        -------
            Optional[str]: Resolved status override or `None` if no override is active.

        Raises:
        ------
            TpuPreemptionError: If override is `'PREEMPTED'`.
            TpuOrchestrationError: If override is `'FAILED'` or `'XLA_ERROR'`.

        """
        if not override_status:
            return None
        if override_status == "PREEMPTED":
            raise TpuPreemptionError(
                f"Workload '{workload_name}' was PREEMPTED on spot TPU capacity."
            )
        if override_status in ("FAILED", "XLA_ERROR"):
            raise TpuOrchestrationError(
                f"Workload '{workload_name}' encountered fatal XLA compilation error."
            )
        return override_status

    def _poll_gcloud_tpu_status_sync(
        self,
        workload_name: str,
        zone: str,
        timeout_seconds: float = 30.0,
    ) -> str:
        """
        Poll TPU VM workload status synchronously using Google Cloud CLI (gcloud).

        Args:
        ----
            workload_name: Identifier of the TPU VM instance.
            zone: GCP availability zone of the TPU VM instance.
            timeout_seconds: Timeout threshold in seconds.

        Returns:
        -------
            str: Normalized lifecycle status ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuOrchestrationError: If gcloud CLI is missing, instance is not found, or preemption occurs.
            TpuPollingTimeoutError: If polling exceeds timeout threshold.

        """
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            raise TpuOrchestrationError("Google Cloud CLI (gcloud) is not installed.")

        cmd = [
            gcloud_bin,
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            workload_name,
            f"--zone={zone}",
            "--format=json",
        ]
        start = time.time()
        while time.time() - start < timeout_seconds:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            parsed = self._parse_gcloud_describe_response(workload_name, zone, res)
            if parsed is not None:
                return parsed
            time.sleep(min(0.02, timeout_seconds / 2.0))

        raise TpuPollingTimeoutError(
            f"Workload '{workload_name}' polling timed out after {timeout_seconds} seconds."
        )

    async def _poll_gcloud_tpu_status(
        self,
        workload_name: str,
        zone: str,
        timeout_seconds: float = 30.0,
    ) -> str:
        """
        Poll TPU VM workload status using Google Cloud CLI (gcloud).

        Args:
        ----
            workload_name: Identifier of the TPU VM instance.
            zone: GCP availability zone of the TPU VM instance.
            timeout_seconds: Timeout threshold in seconds.

        Returns:
        -------
            str: Normalized lifecycle status ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuOrchestrationError: If gcloud CLI is missing, instance is not found, or preemption occurs.
            TpuPollingTimeoutError: If polling exceeds timeout threshold.

        """
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            raise TpuOrchestrationError("Google Cloud CLI (gcloud) is not installed.")

        cmd = [
            gcloud_bin,
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            workload_name,
            f"--zone={zone}",
            "--format=json",
        ]
        start = time.time()
        while time.time() - start < timeout_seconds:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            parsed = self._parse_gcloud_describe_response(workload_name, zone, res)
            if parsed is not None:
                return parsed
            await asyncio.sleep(2.0)

        raise TpuPollingTimeoutError(
            f"Workload '{workload_name}' polling timed out after {timeout_seconds} seconds."
        )

    def _reclaim_gcloud_tpu_workload_sync(
        self,
        workload_name: str,
        zone: str,
        max_retries: int = 3,
    ) -> bool:
        """
        Reclaim and delete TPU VM instance synchronously using Google Cloud CLI (gcloud).

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
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            raise TpuOrchestrationError("Google Cloud CLI (gcloud) is not installed.")

        cmd = [
            gcloud_bin,
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            workload_name,
            f"--zone={zone}",
            "--quiet",
        ]
        last_error = ""
        for attempt in range(1, max_retries + 1):
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return True
            last_error = res.stderr.strip()
            if attempt < max_retries:
                time.sleep(0.01 * (2 ** (attempt - 1)))

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
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            raise TpuOrchestrationError("Google Cloud CLI (gcloud) is not installed.")

        cmd = [
            gcloud_bin,
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            workload_name,
            f"--zone={zone}",
            "--quiet",
        ]
        last_error = ""
        for attempt in range(1, max_retries + 1):
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return True
            last_error = res.stderr.strip()
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        raise TpuOrchestrationError(
            f"Failed to reclaim TPU VM workload '{workload_name}' via gcloud: {last_error}"
        )

    def _poll_workload_status_sync(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
        timeout_seconds: float = 30.0,
        status_override: Optional[str] = None,
    ) -> str:
        """
        Poll workload execution status synchronously from XPK or Google Cloud TPU API.

        Args:
        ----
            workload_name: Identifier of the dispatched training workload.
            max_cfg: MaxText infrastructure configuration.
            timeout_seconds: Timeout threshold in seconds.
            status_override: Optional explicit status override string.

        Returns:
        -------
            str: Lifecycle status string ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuPollingTimeoutError: If polling exceeds specified timeout duration.
            TpuOrchestrationError: If workload is preempted or encounters an execution failure.

        """
        resolved_override = self._evaluate_status_override(
            workload_name,
            status_override or os.environ.get("T1D_TPU_STATUS_OVERRIDE"),
        )
        if resolved_override is not None:
            return resolved_override

        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            return self._poll_gcloud_tpu_status_sync(
                workload_name=workload_name,
                zone=max_cfg.zone,
                timeout_seconds=timeout_seconds,
            )

        cmd = [
            xpk_bin,
            "workload",
            "describe",
            f"--workload={workload_name}",
            f"--cluster=tpu-cluster-{max_cfg.accelerator_type.value}",
        ]
        start = time.time()
        while time.time() - start < timeout_seconds:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            parsed = self._parse_xpk_describe_response(workload_name, res)
            if parsed is not None:
                return parsed
            time.sleep(min(0.02, timeout_seconds / 2.0))

        raise TpuPollingTimeoutError(
            f"Workload '{workload_name}' polling timed out after {timeout_seconds} seconds."
        )

    async def poll_workload_status(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
        timeout_seconds: float = 30.0,
        status_override: Optional[str] = None,
    ) -> str:
        """
        Poll workload execution status from XPK or Google Cloud TPU API.

        Args:
        ----
            workload_name: Identifier of the dispatched training workload.
            max_cfg: MaxText infrastructure configuration.
            timeout_seconds: Timeout threshold in seconds.
            status_override: Optional explicit status override string.

        Returns:
        -------
            str: Lifecycle status string ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TpuPollingTimeoutError: If polling exceeds specified timeout duration.
            TpuOrchestrationError: If workload is preempted or encounters an execution failure.

        """
        resolved_override = self._evaluate_status_override(
            workload_name,
            status_override or os.environ.get("T1D_TPU_STATUS_OVERRIDE"),
        )
        if resolved_override is not None:
            return resolved_override

        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            return await self._poll_gcloud_tpu_status(
                workload_name=workload_name,
                zone=max_cfg.zone,
                timeout_seconds=timeout_seconds,
            )

        cmd = [
            xpk_bin,
            "workload",
            "describe",
            f"--workload={workload_name}",
            f"--cluster=tpu-cluster-{max_cfg.accelerator_type.value}",
        ]
        start = time.time()
        while time.time() - start < timeout_seconds:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            parsed = self._parse_xpk_describe_response(workload_name, res)
            if parsed is not None:
                return parsed
            await asyncio.sleep(2.0)

        raise TpuPollingTimeoutError(
            f"Workload '{workload_name}' polling timed out after {timeout_seconds} seconds."
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
        cluster_name = f"tpu-cluster-{max_cfg.accelerator_type.value}"
        cmd = [
            xpk_bin,
            "workload",
            "delete",
            f"--workload={workload_name}",
            f"--cluster={cluster_name}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
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

        Raises:
        ------
            InvalidGcsUriError: If bucket URI does not begin with 'gs://'.
            ChecksumMismatchError: If checksum mismatches.
            GcsStorageError: If gcloud storage copy command fails after retries.

        """
        if not bucket_url.startswith("gs://"):
            raise InvalidGcsUriError(
                f"Invalid GCS bucket URI '{bucket_url}'. Must start with 'gs://'."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        if gcs_client is not None:
            client = gcs_client
        else:
            try:
                gcs_pkg = importlib.import_module("google.cloud.storage")
            except ImportError as e:
                raise MissingSdkError(
                    "The 'google-cloud-storage' Python package is required. "
                    "Install via 'pip install google-cloud-storage'."
                ) from e
            client = gcs_pkg.Client()

        bucket_name = bucket_url[5:].rstrip("/").split("/")[0]
        prefix = "/".join(bucket_url[5:].rstrip("/").split("/")[1:])

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                bucket = client.bucket(bucket_name)
                blobs = list(client.list_blobs(bucket, prefix=prefix))
                if not blobs:
                    raise GcsStorageError(
                        f"No checkpoint files found at '{bucket_url}'."
                    )
                for blob in blobs:
                    rel_name = blob.name[len(prefix) :].lstrip("/")
                    dest_file = output_dir / rel_name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(dest_file))
                    remote_md5 = (
                        blob.md5_hash
                        if validate_checksum and isinstance(blob, Md5BlobProtocol)
                        else None
                    )
                    if (
                        isinstance(remote_md5, str)
                        and remote_md5
                        and dest_file.exists()
                    ):
                        local_md5 = _compute_md5_base64(dest_file.read_bytes())
                        if local_md5 != remote_md5:
                            raise ChecksumMismatchError(
                                f"Checksum mismatch for '{dest_file.name}': expected {remote_md5}, got {local_md5}"
                            )
                return output_dir
            except ValueError:
                raise
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        raise GcsStorageError(
            f"Checkpoint download failed after {max_retries} attempts: {last_err}"
        ) from last_err

    def _invoke_status_poll_sync(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
        timeout_seconds: float,
    ) -> str:
        """
        Invoke workload status polling synchronously, honoring any patched `poll_workload_status`.

        Args:
        ----
            workload_name: Dispatched workload identifier.
            max_cfg: MaxText infrastructure configuration.
            timeout_seconds: Polling timeout in seconds.

        Returns:
        -------
            str: Resolved workload status string.

        """
        res = self.poll_workload_status(
            workload_name, max_cfg, timeout_seconds=timeout_seconds
        )
        if asyncio.iscoroutine(res):
            return str(asyncio.run(res))
        return str(res)

    def _invoke_reclaim_sync(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
    ) -> bool:
        """
        Invoke TPU workload reclamation synchronously, honoring any patched `reclaim_tpu_workload`.

        Args:
        ----
            workload_name: Dispatched workload identifier.
            max_cfg: MaxText infrastructure configuration.

        Returns:
        -------
            bool: True if resources were reclaimed.

        """
        res = self.reclaim_tpu_workload(workload_name, max_cfg)
        if asyncio.iscoroutine(res):
            return bool(asyncio.run(res))
        return bool(res)

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
        _ = cancel_event
        self.prepare_dataset(config)
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if (
            config.maxtext_config
            and config.maxtext_config.bucket_url
            and os.environ.get("T1D_AUTO_SYNC_GCS") == "1"
            and not config.dry_run
        ):
            from t1d_analytics import training_runner as _tr_mod

            _tr_mod.sync_dataset_to_gcs(
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
            if os.environ.get("T1D_EXECUTE_TPU_DISPATCH") == "1":
                max_cfg_dispatch = config.maxtext_config or MaxTextConfig(
                    project_id=os.environ.get("GCP_PROJECT_ID", "t1d-analytics-gcp")
                )
                self._dispatch_xpk_workload(config, max_cfg_dispatch, config_path)
                try:
                    status_str = self._invoke_status_poll_sync(
                        workload_name, max_cfg_dispatch, timeout_seconds=5
                    )
                except RuntimeError as e:
                    if "PREEMPTED" in str(e):
                        status_str = "PREEMPTED_RETRY_EXHAUSTED"
                        for _ in range(max_cfg_dispatch.max_preemption_retries):
                            self._dispatch_xpk_workload(
                                config, max_cfg_dispatch, config_path
                            )
                            try:
                                status_str = self._invoke_status_poll_sync(
                                    workload_name,
                                    max_cfg_dispatch,
                                    timeout_seconds=2,
                                )
                                break
                            except RuntimeError:
                                continue
                        if status_str == "PREEMPTED_RETRY_EXHAUSTED":
                            self._invoke_reclaim_sync(workload_name, max_cfg_dispatch)
                            raise
                    else:
                        self._invoke_reclaim_sync(workload_name, max_cfg_dispatch)
                        raise

            manifest_path = out_path / "job_manifest.json"
            with open(manifest_path, "w") as mf:
                json.dump(
                    {
                        "workload_name": workload_name,
                        "tpu_cluster": tpu_cluster,
                        "config_file": str(config_path),
                        "output_dir": str(out_path),
                        "model_name": config.model_name,
                        "dispatched_at": time.time(),
                        "status": status_str,
                    },
                    mf,
                    indent=2,
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


def sync_dataset_to_gcs(
    local_path: Path,
    gcs_bucket: str,
    max_retries: int = 3,
    validate_checksum: bool = True,
    sleep_fn: Optional[Callable[[float], None]] = None,
    gcs_client: Optional[GcsClientProtocol] = None,
) -> str:
    """
    Synchronize exported dataset file to Google Cloud Storage bucket with retries and checksums.

    Args:
    ----
        local_path: Local dataset filepath.
        gcs_bucket: Destination GCS bucket URL (e.g., 'gs://bucket-name').
        max_retries: Number of upload retry attempts.
        validate_checksum: Whether to compute and verify MD5 checksum.
        sleep_fn: Optional injectable sleep function for retry backoff.
        gcs_client: Optional injectable GCS client instance.

    Returns:
    -------
        str: Remote GCS target URI.

    Raises:
    ------
        InvalidGcsUriError: If bucket URL is invalid.
        DatabaseNotFoundError: If local dataset file does not exist.
        GcsStorageError: If gcloud storage upload fails after retries.

    """
    if not gcs_bucket.startswith("gs://"):
        raise InvalidGcsUriError(
            f"Invalid GCS bucket URI '{gcs_bucket}'. Must start with 'gs://'."
        )

    if not local_path.exists():
        raise DatabaseNotFoundError(f"Local file not found at '{local_path}'.")

    bucket_clean = gcs_bucket.rstrip("/")
    dest_uri = f"{bucket_clean}/{local_path.name}"

    if gcs_client is not None:
        client = gcs_client
    else:
        try:
            gcs_pkg = importlib.import_module("google.cloud.storage")
        except ImportError as e:
            raise MissingSdkError(
                "The 'google-cloud-storage' Python package is required. "
                "Install via 'pip install google-cloud-storage'."
            ) from e
        client = gcs_pkg.Client()

    bucket_name = gcs_bucket[5:].rstrip("/").split("/")[0]
    blob_prefix = "/".join(gcs_bucket[5:].rstrip("/").split("/")[1:])
    blob_name = f"{blob_prefix}/{local_path.name}".lstrip("/")

    local_md5 = (
        _compute_md5_base64(local_path.read_bytes()) if validate_checksum else ""
    )

    backoff_sleep = sleep_fn or time.sleep
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.md5_hash = local_md5
            blob.upload_from_filename(str(local_path))
            return dest_uri
        except Exception as e:
            last_err = e
            backoff_sleep(0.05 * (2 ** (attempt - 1)))

    raise GcsStorageError(
        f"GCS upload failed after {max_retries} attempts: {last_err}"
    ) from last_err
