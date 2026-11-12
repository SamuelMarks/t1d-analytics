"""Unit tests for Cloud TPU MaxText orchestration, XPK/gcloud command builders, and GCS synchronization."""

import asyncio
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from t1d_analytics.errors import TpuOrchestrationError, TpuPreemptionError
from t1d_analytics.gcs_storage import (
    GcsBlobProtocol,
    GcsBucketProtocol,
    GcsClientProtocol,
    Md5BlobProtocol,
    compute_md5_base64,
    sync_dataset_to_gcs,
)
from t1d_analytics.models import (
    MaxTextConfig,
    TpuAcceleratorType,
    TpuSchedulingType,
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.tpu_orchestrator import (
    RemoteTpuMaxTextRunner,
    SubprocessRunnerProtocol,
)


class FakeSubprocessRunner:
    """Explicit contract-verified fake implementing `SubprocessRunnerProtocol`."""

    def __init__(
        self,
        responses: Sequence[subprocess.CompletedProcess[str]] | None = None,
        *,
        default_stdout: str = '{"state": "READY"}',
        default_stderr: str = "",
        default_returncode: int = 0,
        on_call: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
        | None = None,
    ) -> None:
        """
        Initialize fake subprocess runner with queued responses or dynamic handler.

        Args:
            responses: Optional FIFO queue of `CompletedProcess` records.
            default_stdout: Default stdout when queue is empty or absent.
            default_stderr: Default stderr when queue is empty or absent.
            default_returncode: Default exit code when queue is empty or absent.
            on_call: Optional dynamic handler invoked per command vector.

        """
        self.commands: list[list[str]] = []
        self._queue: list[subprocess.CompletedProcess[str]] = (
            list(responses) if responses is not None else []
        )
        self._default_stdout = default_stdout
        self._default_stderr = default_stderr
        self._default_returncode = default_returncode
        self._on_call = on_call

    def __call__(
        self,
        cmd: Sequence[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """
        Record executed command vector and return configured `CompletedProcess`.

        Args:
            cmd: Command argument sequence.
            capture_output: Whether stdout/stderr capture was requested.
            text: Whether text mode was requested.
            check: Whether non-zero exit checking was requested.

        Returns:
            subprocess.CompletedProcess[str]: Configured fake completion record.

        """
        _ = (capture_output, text, check)
        self.commands.append(list(cmd))
        if self._on_call is not None:
            return self._on_call(cmd)
        if self._queue:
            queued = self._queue.pop(0)
            return subprocess.CompletedProcess(
                args=list(cmd),
                returncode=queued.returncode,
                stdout=queued.stdout,
                stderr=queued.stderr,
            )
        return subprocess.CompletedProcess(
            args=list(cmd),
            returncode=self._default_returncode,
            stdout=self._default_stdout,
            stderr=self._default_stderr,
        )


class FakeGcsBlob:
    """Contract-verified fake implementing GcsBlobProtocol and Md5BlobProtocol."""

    def __init__(self, name: str, payload: bytes = b"checkpoint_bytes") -> None:
        """
        Initialize fake GCS blob.

        Args:
            name: Object key within the bucket.
            payload: Raw byte payload for MD5 calculation and download simulation.

        """
        self.name = name
        self.payload = payload
        self.md5_hash: str | None = compute_md5_base64(payload)
        self.uploaded_from: str | None = None
        self.downloaded_to: str | None = None

    def download_to_filename(self, filename: str) -> None:
        """
        Write blob payload to destination filename.

        Args:
            filename: Local destination file path string.

        """
        self.downloaded_to = filename
        Path(filename).write_bytes(self.payload)

    def upload_from_filename(self, filename: str) -> None:
        """
        Record uploaded filename.

        Args:
            filename: Local source file path string.

        """
        self.uploaded_from = filename


class FakeGcsBucket:
    """Contract-verified fake implementing GcsBucketProtocol."""

    def __init__(self, blobs: list[FakeGcsBlob]) -> None:
        """
        Initialize fake bucket with blob list.

        Args:
            blobs: Backing blob list mutated on new uploads.

        """
        self._blobs = blobs

    def blob(self, blob_name: str) -> GcsBlobProtocol:
        """
        Create and register a new FakeGcsBlob.

        Args:
            blob_name: Target object key name.

        Returns:
            GcsBlobProtocol: Registered fake blob instance.

        """
        new_blob = FakeGcsBlob(blob_name)
        self._blobs.append(new_blob)
        return new_blob


class FakeGcsClient:
    """Contract-verified fake implementing GcsClientProtocol."""

    def __init__(self, blobs: list[FakeGcsBlob] | None = None) -> None:
        """
        Initialize fake GCS client.

        Args:
            blobs: Optional initial blob list.

        """
        self.blobs: list[FakeGcsBlob] = list(blobs) if blobs is not None else []

    def bucket(self, bucket_name: str) -> GcsBucketProtocol:
        """
        Return a FakeGcsBucket.

        Args:
            bucket_name: Bucket name string.

        Returns:
            GcsBucketProtocol: Fake bucket bound to `self.blobs`.

        """
        _ = bucket_name
        return FakeGcsBucket(self.blobs)

    def list_blobs(
        self, bucket_or_name: GcsBucketProtocol | str, prefix: str = ""
    ) -> Iterable[GcsBlobProtocol]:
        """
        Return blobs matching prefix.

        Args:
            bucket_or_name: Bucket reference or name.
            prefix: Object prefix filter.

        Returns:
            Iterable[GcsBlobProtocol]: Snapshot of fake blobs.

        """
        _ = (bucket_or_name, prefix)
        return list(self.blobs)


def _proc(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    """
    Build a typed `CompletedProcess[str]` response fixture for `FakeSubprocessRunner`.

    Args:
        stdout: Standard output payload.
        stderr: Standard error payload.
        returncode: Process exit code.

    Returns:
        subprocess.CompletedProcess[str]: Typed process completion fixture.

    """
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_remote_tpu_maxtext_runner(tmp_path: Path) -> None:
    """Test RemoteTpuMaxTextRunner configuration generation, default subprocess runner, and job dispatch."""
    fake_sub = FakeSubprocessRunner()
    assert isinstance(fake_sub, SubprocessRunnerProtocol)
    runner = RemoteTpuMaxTextRunner(subprocess_runner=fake_sub)

    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        assert runner.validate_environment() is True

    with patch("shutil.which", return_value=None):
        with patch("pathlib.Path.exists", return_value=True):
            assert runner.validate_environment() is True

    with patch("shutil.which", return_value=None):
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="Google Cloud CLI"):
                runner.validate_environment()

    ds = tmp_path / "data.parquet"
    ds.write_text("data", encoding="utf-8")
    out = tmp_path / "tpu_out"

    maxtext_cfg = MaxTextConfig(
        project_id="test-proj",
        zone="us-central2-b",
        accelerator_type=TpuAcceleratorType.V4_8,
        scheduling_type=TpuSchedulingType.SPOT,
        bucket_url="gs://my-bucket",
    )
    config = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(out),
        hyperparameters=TrainingHyperparameters(
            batch_size=8, num_epochs=2, learning_rate=3e-5
        ),
        maxtext_config=maxtext_cfg,
        dry_run=False,
    )
    res = runner.run_training(config)
    assert res["status"] == "dispatched"
    config_file = Path(str(res["config_file"]))
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "model_name: gemma-7b" in content
    assert "hardware: v4-8" in content
    assert "zone: us-central2-b" in content

    config_dry = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "tpu_dry"),
        dry_run=True,
    )
    res_dry = runner.run_training(config_dry)
    assert res_dry["status"] == "dry_run_completed"

    # Exercise default subprocess.run delegation when subprocess_runner is None using real 'echo' binary
    echo_bin = shutil.which("echo")
    assert echo_bin is not None
    default_runner = RemoteTpuMaxTextRunner()
    with patch("shutil.which", side_effect=lambda b: echo_bin if b == "xpk" else None):
        real_proc = default_runner.dispatch_xpk_workload(
            config, maxtext_cfg, config_file
        )
        assert real_proc.returncode == 0
        assert "workload create" in real_proc.stdout


def test_sync_dataset_to_gcs(tmp_path: Path) -> None:
    """Test sync_dataset_to_gcs upload, injectable GcsClientProtocol, validation, and error branches."""
    local_file = tmp_path / "data.parquet"
    local_file.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid GCS bucket URI"):
        sync_dataset_to_gcs(local_file, "s3://invalid-bucket")

    with pytest.raises(FileNotFoundError, match="Local file not found"):
        sync_dataset_to_gcs(tmp_path / "missing.parquet", "gs://my-bucket")

    fake_client: GcsClientProtocol = FakeGcsClient()
    uri_injected = sync_dataset_to_gcs(
        local_file, "gs://my-bucket/datasets", gcs_client=fake_client
    )
    assert uri_injected == "gs://my-bucket/datasets/data.parquet"

    with patch.dict(
        "sys.modules",
        {"google": None, "google.cloud": None, "google.cloud.storage": None},
    ):
        with pytest.raises(RuntimeError, match="google-cloud-storage"):
            sync_dataset_to_gcs(local_file, "gs://my-bucket")

    mock_gcs = MagicMock()
    mock_client = MagicMock()
    mock_gcs.Client.return_value = mock_client
    mock_google = MagicMock()
    mock_cloud = MagicMock()
    mock_google.cloud = mock_cloud
    mock_cloud.storage = mock_gcs
    with patch.dict(
        "sys.modules",
        {
            "google": mock_google,
            "google.cloud": mock_cloud,
            "google.cloud.storage": mock_gcs,
        },
    ):
        uri = sync_dataset_to_gcs(local_file, "gs://my-bucket/")
        assert uri == "gs://my-bucket/data.parquet"
        uri2 = sync_dataset_to_gcs(local_file, "gs://my-bucket")
        assert uri2 == "gs://my-bucket/data.parquet"

    mock_err_gcs = MagicMock()
    mock_err_client = MagicMock()
    mock_err_blob = MagicMock()
    mock_err_blob.upload_from_filename.side_effect = RuntimeError("Network timeout")
    mock_err_client.bucket.return_value.blob.return_value = mock_err_blob
    mock_err_gcs.Client.return_value = mock_err_client
    with patch.dict(
        "sys.modules",
        {
            "google": mock_google,
            "google.cloud": mock_cloud,
            "google.cloud.storage": mock_err_gcs,
        },
    ):
        with pytest.raises(RuntimeError, match="GCS upload failed after 2 attempts"):
            sync_dataset_to_gcs(
                local_file,
                "gs://my-bucket",
                max_retries=2,
                sleep_fn=lambda _: None,
            )


def test_remote_tpu_maxtext_runner_auto_sync_gcs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test RemoteTpuMaxTextRunner auto-sync to GCS and job manifest serialization."""
    runner = RemoteTpuMaxTextRunner(subprocess_runner=FakeSubprocessRunner())
    monkeypatch.setenv("T1D_AUTO_SYNC_GCS", "1")

    ds = tmp_path / "dataset.parquet"
    ds.write_text("dataset", encoding="utf-8")
    out = tmp_path / "tpu_out_sync"

    maxtext_cfg = MaxTextConfig(
        project_id="test-proj",
        zone="us-central2-b",
        accelerator_type=TpuAcceleratorType.V4_8,
        scheduling_type=TpuSchedulingType.ON_DEMAND,
        bucket_url="gs://test-bucket",
    )
    config = TrainingJobConfig(
        model_name="gemma-4-sql",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(out),
        maxtext_config=maxtext_cfg,
        dry_run=False,
    )

    with patch("t1d_analytics.tpu_orchestrator.sync_dataset_to_gcs") as mock_sync:
        mock_sync.return_value = "gs://test-bucket/dataset.parquet"
        res = runner.run_training(config)
        mock_sync.assert_called_once_with(ds, "gs://test-bucket")

    assert res["status"] == "dispatched"
    assert res["tpu_cluster"] == "v4-8"
    assert "gemma-4-sql" in str(res["workload_name"])
    manifest = out / "job_manifest.json"
    assert manifest.exists()


def test_remote_tpu_dispatch_xpk_and_gcloud_branches(tmp_path: Path) -> None:
    """Test dispatch_xpk_workload under xpk binary, gcloud fallback, and non-zero exit via SubprocessRunnerProtocol."""
    ds = tmp_path / "ds.parquet"
    ds.write_text("data", encoding="utf-8")
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("config", encoding="utf-8")
    config = TrainingJobConfig(
        model_name="gemma-tpu",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out"),
    )
    max_cfg = MaxTextConfig(project_id="test-proj")

    ok_fake = FakeSubprocessRunner(default_stdout="OK")
    runner_ok = RemoteTpuMaxTextRunner(subprocess_runner=ok_fake)

    with patch(
        "shutil.which",
        side_effect=lambda bin_name: "/usr/bin/xpk" if bin_name == "xpk" else None,
    ):
        res_xpk = runner_ok.dispatch_xpk_workload(config, max_cfg, cfg_path)
        assert res_xpk.returncode == 0
        assert ok_fake.commands[-1][:3] == ["/usr/bin/xpk", "workload", "create"]

    with patch(
        "shutil.which",
        side_effect=lambda bin_name: (
            "/usr/bin/gcloud" if bin_name == "gcloud" else None
        ),
    ):
        res_gcloud = runner_ok.dispatch_xpk_workload(config, max_cfg, cfg_path)
        assert res_gcloud.returncode == 0
        assert ok_fake.commands[-1][:4] == [
            "/usr/bin/gcloud",
            "compute",
            "tpus",
            "tpu-vm",
        ]

    with patch("shutil.which", return_value=None):
        with pytest.raises(
            RuntimeError, match="Neither 'xpk' nor 'gcloud' CLI is installed"
        ):
            runner_ok.dispatch_xpk_workload(config, max_cfg, cfg_path)

    err_fake = FakeSubprocessRunner(default_returncode=1, default_stderr="Cluster down")
    runner_err = RemoteTpuMaxTextRunner(subprocess_runner=err_fake)
    with patch(
        "shutil.which",
        side_effect=lambda bin_name: "/usr/bin/xpk" if bin_name == "xpk" else None,
    ):
        with pytest.raises(RuntimeError, match="TPU dispatch failed: Cluster down"):
            runner_err.dispatch_xpk_workload(config, max_cfg, cfg_path)


def test_remote_tpu_poll_workload_status_branches() -> None:
    """Test poll_workload_status under xpk and gcloud polling, success, failure, and timeout."""
    max_cfg = MaxTextConfig(project_id="test-proj")

    runner_missing = RemoteTpuMaxTextRunner(subprocess_runner=FakeSubprocessRunner())
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner_missing.poll_workload_status("t1d-test", max_cfg))

    gcloud_ready_fake = FakeSubprocessRunner(
        default_stdout=json.dumps({"state": "READY"})
    )
    runner_gcloud = RemoteTpuMaxTextRunner(subprocess_runner=gcloud_ready_fake)
    with patch(
        "shutil.which",
        side_effect=lambda cmd: "/usr/bin/gcloud" if cmd == "gcloud" else None,
    ):
        assert (
            asyncio.run(runner_gcloud.poll_workload_status("t1d-test", max_cfg))
            == "SUCCESS"
        )
        assert gcloud_ready_fake.commands[-1][0] == "/usr/bin/gcloud"

    with patch("shutil.which", return_value="/usr/bin/xpk"):
        runner_succ = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(default_stdout="Status: SUCCESS")
        )
        assert (
            asyncio.run(runner_succ.poll_workload_status("t1d-test", max_cfg))
            == "SUCCESS"
        )

        runner_fail = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(default_stdout="Status: FAILED")
        )
        assert (
            asyncio.run(runner_fail.poll_workload_status("t1d-test", max_cfg))
            == "FAILED"
        )

        runner_pend = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(default_stdout="Status: PENDING")
        )
        with pytest.raises(TimeoutError, match="polling timed out"):
            asyncio.run(
                runner_pend.poll_workload_status(
                    "t1d-test", max_cfg, timeout_seconds=0.01
                )
            )

        runner_rc1 = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_returncode=1, default_stdout=""
            )
        )
        with pytest.raises(TimeoutError, match="polling timed out"):
            asyncio.run(
                runner_rc1.poll_workload_status(
                    "t1d-test", max_cfg, timeout_seconds=0.01
                )
            )


def test_remote_tpu_download_checkpoints_from_gcs(tmp_path: Path) -> None:
    """Test download_checkpoints_from_gcs with injectable FakeGcsClient, checksum verification, and errors."""
    runner = RemoteTpuMaxTextRunner(subprocess_runner=FakeSubprocessRunner())
    out_dir = tmp_path / "ckpts"

    with pytest.raises(ValueError, match="Invalid GCS bucket URI"):
        asyncio.run(runner.download_checkpoints_from_gcs("s3://bucket/test", out_dir))

    fake_blob = FakeGcsBlob("test/model.ckpt", b"checkpoint_payload")
    fake_client = FakeGcsClient([fake_blob])
    res_dir = asyncio.run(
        runner.download_checkpoints_from_gcs(
            "gs://bucket/test", out_dir, gcs_client=fake_client
        )
    )
    assert res_dir == out_dir
    assert (out_dir / "model.ckpt").read_bytes() == b"checkpoint_payload"

    fake_blob.md5_hash = "invalid_md5=="
    with pytest.raises(ValueError, match="Checksum mismatch"):
        asyncio.run(
            runner.download_checkpoints_from_gcs(
                "gs://bucket/test",
                out_dir,
                validate_checksum=True,
                gcs_client=fake_client,
            )
        )

    mock_blob = MagicMock()
    mock_blob.name = "test/model2.ckpt"
    mock_blob.md5_hash = None
    mock_gcs = MagicMock()
    mock_gcs.Client.return_value.list_blobs.return_value = [mock_blob]

    with patch.dict("sys.modules", {"google.cloud.storage": mock_gcs}):
        asyncio.run(runner.download_checkpoints_from_gcs("gs://bucket/test", out_dir))
        mock_blob.download_to_filename.assert_called()

    with patch.dict("sys.modules", {"google.cloud.storage": None}):
        with pytest.raises(RuntimeError, match="google-cloud-storage"):
            asyncio.run(
                runner.download_checkpoints_from_gcs("gs://bucket/test", out_dir)
            )

    empty_client = FakeGcsClient([])
    with pytest.raises(
        RuntimeError, match="No checkpoint files found at 'gs://bucket/test'"
    ):
        asyncio.run(
            runner.download_checkpoints_from_gcs(
                "gs://bucket/test", out_dir, max_retries=1, gcs_client=empty_client
            )
        )


def test_remote_tpu_run_training_with_dispatch_flag(tmp_path: Path) -> None:
    """Test RemoteTpuMaxTextRunner.run_training when T1D_EXECUTE_TPU_DISPATCH=1."""
    fake_sub = FakeSubprocessRunner(default_stdout="Status: SUCCESS")
    runner = RemoteTpuMaxTextRunner(subprocess_runner=fake_sub)
    ds = tmp_path / "ds.parquet"
    ds.write_text("dataset", encoding="utf-8")
    out = tmp_path / "tpu_dispatched_out"

    config = TrainingJobConfig(
        model_name="gemma-tpu-test",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=False,
    )
    with (
        patch.dict("os.environ", {"T1D_EXECUTE_TPU_DISPATCH": "1"}),
        patch("shutil.which", return_value="/usr/bin/xpk"),
    ):
        res = runner.run_training(config)
        assert res["status"] == "SUCCESS"
        assert (out / "job_manifest.json").exists()


def test_tpu_workload_status_override_and_xla_preempted() -> None:
    """Test poll_workload_status with PREEMPTED status and XLA errors via SubprocessRunnerProtocol."""
    max_cfg = MaxTextConfig(project_id="p1")

    with patch("shutil.which", return_value="/usr/bin/xpk"):
        runner_succ = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout="Workload workload-1 | Status: SUCCESS | Slice: v4-8"
            )
        )
        assert (
            asyncio.run(
                runner_succ.poll_workload_status(
                    "workload-1", max_cfg, timeout_seconds=1
                )
            )
            == "SUCCESS"
        )

        runner_preempt = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout="STATUS: PREEMPTED by node eviction"
            )
        )
        with pytest.raises(TpuPreemptionError, match="was PREEMPTED"):
            asyncio.run(
                runner_preempt.poll_workload_status(
                    "workload-1", max_cfg, timeout_seconds=1
                )
            )

        runner_xla = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout="FATAL: XLA COMPILATION ERROR"
            )
        )
        with pytest.raises(TpuOrchestrationError, match="fatal XLA compilation error"):
            asyncio.run(
                runner_xla.poll_workload_status(
                    "workload-1", max_cfg, timeout_seconds=1
                )
            )


def test_tpu_workload_reclamation() -> None:
    """Test reclaim_tpu_workload across xpk and gcloud modes via SubprocessRunnerProtocol."""
    max_cfg = MaxTextConfig(project_id="p1")
    runner_ok = RemoteTpuMaxTextRunner(subprocess_runner=FakeSubprocessRunner())

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner_ok.reclaim_tpu_workload("workload-1", max_cfg))

    gcloud_fake = FakeSubprocessRunner()
    runner_gcloud = RemoteTpuMaxTextRunner(subprocess_runner=gcloud_fake)
    with patch(
        "shutil.which",
        side_effect=lambda cmd: "/usr/bin/gcloud" if cmd == "gcloud" else None,
    ):
        assert (
            asyncio.run(runner_gcloud.reclaim_tpu_workload("workload-1", max_cfg))
            is True
        )
        assert gcloud_fake.commands[-1][:4] == [
            "/usr/bin/gcloud",
            "compute",
            "tpus",
            "tpu-vm",
        ]

    with patch("shutil.which", return_value="/usr/bin/xpk"):
        assert (
            asyncio.run(runner_ok.reclaim_tpu_workload("workload-1", max_cfg)) is True
        )

        runner_fail = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_returncode=1, default_stderr="Cluster not reachable"
            )
        )
        with pytest.raises(RuntimeError, match="Failed to reclaim TPU workload"):
            asyncio.run(runner_fail.reclaim_tpu_workload("workload-1", max_cfg))


def test_remote_tpu_run_training_preemption_and_retries(tmp_path: Path) -> None:
    """Test RemoteTpuMaxTextRunner preemption retry and fatal failure handling via SubprocessRunnerProtocol."""
    ds = tmp_path / "tpu_ds.parquet"
    ds.write_text("ds", encoding="utf-8")
    out = tmp_path / "tpu_out_preempt"

    cfg = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=False,
        maxtext_config=MaxTextConfig(project_id="p1", max_preemption_retries=2),
    )

    with (
        patch.dict(os.environ, {"T1D_EXECUTE_TPU_DISPATCH": "1"}),
        patch("shutil.which", return_value="/usr/bin/xpk"),
    ):
        preempt_fake = FakeSubprocessRunner(
            on_call=lambda cmd: (
                _proc("Status: PREEMPTED") if "describe" in cmd else _proc("ok")
            )
        )
        runner_preempt = RemoteTpuMaxTextRunner(subprocess_runner=preempt_fake)
        with pytest.raises(RuntimeError, match="PREEMPTED"):
            runner_preempt.run_training(cfg)
        assert any("delete" in c for c in preempt_fake.commands)

        fail_fake = FakeSubprocessRunner(
            on_call=lambda cmd: (
                _proc("Fatal XLA compilation ERROR", stderr="OOM")
                if "describe" in cmd
                else _proc("ok")
            )
        )
        runner_fail = RemoteTpuMaxTextRunner(subprocess_runner=fail_fake)
        with pytest.raises(RuntimeError, match="XLA compilation error"):
            runner_fail.run_training(cfg)
        assert any("delete" in c for c in fail_fake.commands)

        describe_attempt = [0]

        def _recover_handler(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
            """Simulate first XPK describe preempting, second returning non-zero, and third succeeding."""
            if "describe" in cmd:
                describe_attempt[0] += 1
                if describe_attempt[0] == 1:
                    return _proc("Status: PREEMPTED")
                if describe_attempt[0] == 2:
                    return _proc("", stderr="transient", returncode=1)
                return _proc("Status: SUCCESS")
            return _proc("ok")

        runner_recover = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(on_call=_recover_handler)
        )
        res_retry_succ = runner_recover.run_training(cfg)
        assert res_retry_succ["status"] == "SUCCESS"
        assert describe_attempt[0] == 3


def test_poll_gcloud_tpu_status_realistic_lifecycle_states() -> None:
    """Test gcloud lifecycle states (CREATING, READY, RUNNING, PREEMPTED, STOPPED, TERMINATED, non-zero exit)."""
    ready_fake = FakeSubprocessRunner(default_stdout=json.dumps({"state": "READY"}))
    runner_ready = RemoteTpuMaxTextRunner(subprocess_runner=ready_fake)

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner_ready.poll_gcloud_tpu_status("node-1", "us-central2-b"))
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(
                runner_ready.reclaim_gcloud_tpu_workload("node-1", "us-central2-b")
            )

    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        assert (
            asyncio.run(runner_ready.poll_gcloud_tpu_status("node-1", "us-central2-b"))
            == "SUCCESS"
        )
        assert ready_fake.commands[-1] == [
            "/usr/bin/gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            "node-1",
            "--zone=us-central2-b",
            "--format=json",
        ]

        for halted_state in ("STOPPED", "TERMINATED"):
            halted_runner = RemoteTpuMaxTextRunner(
                subprocess_runner=FakeSubprocessRunner(
                    default_stdout=json.dumps({"state": halted_state})
                )
            )
            with pytest.raises(RuntimeError, match="terminal halted state"):
                asyncio.run(
                    halted_runner.poll_gcloud_tpu_status("node-1", "us-central2-b")
                )

        # CREATING -> RUNNING transition
        creating_to_running = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                responses=[
                    _proc(json.dumps({"state": "CREATING"})),
                    _proc(json.dumps({"state": "RUNNING"})),
                ]
            )
        )
        assert (
            asyncio.run(
                creating_to_running.poll_gcloud_tpu_status("node-1", "us-central2-b")
            )
            == "RUNNING"
        )

        # Async reclaim_gcloud_tpu_workload retry + eventual success and exhaustion
        reclaim_retry = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                responses=[
                    _proc("", stderr="transient err", returncode=1),
                    _proc("deleted", returncode=0),
                ]
            )
        )
        assert (
            asyncio.run(
                reclaim_retry.reclaim_gcloud_tpu_workload(
                    "node-1", "us-central2-b", max_retries=2
                )
            )
            is True
        )

        reclaim_fail = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_returncode=1, default_stderr="perm err"
            )
        )
        with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM"):
            asyncio.run(
                reclaim_fail.reclaim_gcloud_tpu_workload(
                    "node-1", "us-central2-b", max_retries=2
                )
            )

        # YAML escaping and format_yaml_scalar checks
        inj_model = """gemma'inject
evil: true"""
        inj_ds = """/tmp/ds'"
injected: 1.jsonl"""
        inj_cfg = TrainingJobConfig(
            model_name=inj_model,
            backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
            dataset_path=inj_ds,
            output_dir="/tmp/out",
            maxtext_config=MaxTextConfig(project_id="proj-1"),
        )
        yaml_str = runner_ready.generate_maxtext_yaml(inj_cfg)
        assert "injected: 1" not in yaml_str.splitlines()
        assert "evil: true" not in yaml_str.splitlines()
        assert runner_ready.format_yaml_scalar(True) == "true"
        assert runner_ready.format_yaml_scalar(False) == "false"

        text_ready_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout="State: READY (non-json output)"
            )
        )
        assert (
            asyncio.run(
                text_ready_runner.poll_gcloud_tpu_status("node-1", "us-central2-b")
            )
            == "SUCCESS"
        )

        # Preempted, healthDescription preempted, fatal hardware error, and non-zero NOT_FOUND exit
        preempt_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout=json.dumps({"state": "PREEMPTED"})
            )
        )
        with pytest.raises(RuntimeError, match="was PREEMPTED"):
            asyncio.run(
                preempt_runner.poll_gcloud_tpu_status("node-1", "us-central2-b")
            )

        health_preempt_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout=json.dumps(
                    {"state": "STOPPING", "healthDescription": "Node was preempted"}
                )
            )
        )
        with pytest.raises(RuntimeError, match="was PREEMPTED"):
            asyncio.run(
                health_preempt_runner.poll_gcloud_tpu_status("node-1", "us-central2-b")
            )

        hw_err_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout=json.dumps(
                    {"state": "RUNNING", "healthDescription": "Fatal hardware ERROR"}
                )
            )
        )
        with pytest.raises(RuntimeError, match="encountered fatal error"):
            asyncio.run(hw_err_runner.poll_gcloud_tpu_status("node-1", "us-central2-b"))

        nf_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_returncode=1,
                default_stderr="ERROR: (gcloud) NOT_FOUND: Resource not found",
            )
        )
        with pytest.raises(RuntimeError, match="not found in zone"):
            asyncio.run(nf_runner.poll_gcloud_tpu_status("node-1", "us-central2-b"))


def test_poll_gcloud_tpu_status_timeout_and_cancellation() -> None:
    """Test poll_gcloud_tpu_status and poll_workload_status_sync timeout and cancellation branches."""
    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        for pending_fake in (
            FakeSubprocessRunner(default_stdout=json.dumps({"state": "CREATING"})),
            FakeSubprocessRunner(default_stdout="Some garbled text without status"),
            FakeSubprocessRunner(
                default_returncode=1, default_stderr="Connection reset by peer"
            ),
        ):
            pending_runner = RemoteTpuMaxTextRunner(subprocess_runner=pending_fake)
            with pytest.raises(TimeoutError, match="polling timed out"):
                asyncio.run(
                    pending_runner.poll_gcloud_tpu_status(
                        "node-1", "us-central2-b", timeout_seconds=0.05
                    )
                )

    max_cfg_sync = MaxTextConfig(project_id="p-sync", zone="us-central2-b")
    with patch("shutil.which", return_value="/usr/bin/xpk"):
        succ_sync_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(default_stdout="Status: SUCCESS")
        )
        assert (
            succ_sync_runner.poll_workload_status_sync("wl-sync", max_cfg_sync)
            == "SUCCESS"
        )

    cancel_ev = threading.Event()
    cancel_ev.set()
    with patch("shutil.which", return_value="/usr/bin/xpk"):
        reclaim_ok_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner()
        )
        with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
            reclaim_ok_runner.poll_workload_status_sync(
                "wl-sync", max_cfg_sync, cancel_event=cancel_ev
            )
        with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
            asyncio.run(
                reclaim_ok_runner.poll_workload_status(
                    "wl-sync", max_cfg_sync, cancel_event=cancel_ev
                )
            )

        mid_ev_sync = threading.Event()

        def _mid_sync_handler(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
            """Set cancel event on first describe call and succeed on subsequent delete."""
            if "describe" in cmd:
                mid_ev_sync.set()
                return _proc("Status: PENDING")
            return _proc("deleted")

        mid_sync_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(on_call=_mid_sync_handler)
        )
        with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
            mid_sync_runner.poll_workload_status_sync(
                "wl-sync",
                max_cfg_sync,
                timeout_seconds=1.0,
                cancel_event=mid_ev_sync,
            )

        mid_ev_async = threading.Event()

        def _mid_async_handler(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
            """Set cancel event on first async describe call and succeed on delete."""
            if "describe" in cmd:
                mid_ev_async.set()
                return _proc("Status: PENDING")
            return _proc("deleted")

        mid_async_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(on_call=_mid_async_handler)
        )
        with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
            asyncio.run(
                mid_async_runner.poll_workload_status(
                    "wl-sync",
                    max_cfg_sync,
                    timeout_seconds=1.0,
                    cancel_event=mid_ev_async,
                )
            )

    with patch(
        "shutil.which",
        side_effect=lambda b: "/usr/bin/gcloud" if b == "gcloud" else None,
    ):
        gcloud_cancel_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner()
        )
        with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
            gcloud_cancel_runner.poll_gcloud_tpu_status_sync(
                "wl-sync",
                "us-central2-b",
                timeout_seconds=1.0,
                cancel_event=cancel_ev,
            )
        with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
            asyncio.run(
                gcloud_cancel_runner.poll_gcloud_tpu_status(
                    "wl-sync",
                    "us-central2-b",
                    timeout_seconds=1.0,
                    cancel_event=cancel_ev,
                )
            )


def test_reclaim_gcloud_tpu_workload_retries() -> None:
    """Test reclaim_gcloud_tpu_workload and sync/async reclamation command flags and retry behavior."""
    reclaim_fake = FakeSubprocessRunner(default_stdout=json.dumps({"state": "READY"}))
    runner = RemoteTpuMaxTextRunner(subprocess_runner=reclaim_fake)
    max_cfg_sync = MaxTextConfig(project_id="p-sync", zone="us-central2-b")

    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        assert (
            asyncio.run(runner.reclaim_gcloud_tpu_workload("node-1", "us-central2-b"))
            is True
        )
        assert reclaim_fake.commands[-1] == [
            "/usr/bin/gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            "node-1",
            "--zone=us-central2-b",
            "--quiet",
        ]

        fail_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_returncode=1, default_stderr="PermissionDenied"
            )
        )
        with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM workload"):
            asyncio.run(
                fail_runner.reclaim_gcloud_tpu_workload(
                    "node-1", "us-central2-b", max_retries=2
                )
            )

    with patch("shutil.which", return_value="/usr/bin/xpk"):
        assert runner.reclaim_tpu_workload_sync("wl-sync", max_cfg_sync) is True
        assert reclaim_fake.commands[-1] == [
            "/usr/bin/xpk",
            "workload",
            "delete",
            "--workload=wl-sync",
            "--cluster=tpu-cluster-v4-8",
        ]
        succ_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(default_stdout="Status: SUCCESS")
        )
        assert (
            succ_runner.poll_workload_status_sync(
                "wl-sync", max_cfg_sync, timeout_seconds=1.0
            )
            == "SUCCESS"
        )
        pend_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(default_stdout="Status: PENDING")
        )
        with pytest.raises(TimeoutError):
            pend_runner.poll_workload_status_sync(
                "wl-sync", max_cfg_sync, timeout_seconds=0.01
            )
        del_err_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_returncode=1, default_stderr="delete err"
            )
        )
        with pytest.raises(RuntimeError, match="Failed to reclaim TPU workload"):
            del_err_runner.reclaim_tpu_workload_sync("wl-sync", max_cfg_sync)

    with patch(
        "shutil.which",
        side_effect=lambda b: "/usr/bin/gcloud" if b == "gcloud" else None,
    ):
        assert runner.poll_workload_status_sync("wl-sync", max_cfg_sync) == "SUCCESS"
        assert runner.reclaim_tpu_workload_sync("wl-sync", max_cfg_sync) is True
        creating_runner = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_stdout=json.dumps({"state": "CREATING"})
            )
        )
        with pytest.raises(TimeoutError):
            creating_runner.poll_gcloud_tpu_status_sync(
                "wl-sync", "us-central2-b", timeout_seconds=0.01
            )
        gcloud_del_err = RemoteTpuMaxTextRunner(
            subprocess_runner=FakeSubprocessRunner(
                default_returncode=1, default_stderr="delete err"
            )
        )
        with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM workload"):
            gcloud_del_err.reclaim_gcloud_tpu_workload_sync(
                "wl-sync", "us-central2-b", max_retries=2
            )

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError):
            runner.poll_gcloud_tpu_status_sync("wl-sync", "us-central2-b")
        with pytest.raises(RuntimeError):
            runner.reclaim_gcloud_tpu_workload_sync("wl-sync", "us-central2-b")
        with pytest.raises(RuntimeError):
            runner.reclaim_tpu_workload_sync("wl-sync", max_cfg_sync)


def test_tpu_orchestrator_protocol_stubs(tmp_path: Path) -> None:
    """Verify GcsClientProtocol and SubprocessRunnerProtocol contracts end-to-end."""
    uploaded_files: dict[str, bytes] = {}

    class _InMemoryBlob:
        """In-memory GCS blob satisfying GcsBlobProtocol and Md5BlobProtocol."""

        def __init__(self, name: str) -> None:
            """
            Initialize blob with name.

            Args:
                name: Object key string.

            """
            self.name = name
            self.md5_hash: str | None = None

        def upload_from_filename(self, filename: str) -> None:
            """
            Store uploaded file bytes and compute base64 MD5 digest.

            Args:
                filename: Source file path string.

            """
            raw = Path(filename).read_bytes()
            uploaded_files[self.name] = raw
            self.md5_hash = compute_md5_base64(raw)

        def download_to_filename(self, filename: str) -> None:
            """
            Write stored bytes to destination file.

            Args:
                filename: Destination file path string.

            """
            Path(filename).write_bytes(uploaded_files.get(self.name, b"ckpt-weights"))

    blobs_by_name: dict[str, _InMemoryBlob] = {}

    class _InMemoryBucket:
        """In-memory GCS bucket satisfying GcsBucketProtocol."""

        def blob(self, blob_name: str) -> GcsBlobProtocol:
            """
            Return or create an in-memory blob.

            Args:
                blob_name: Object key string.

            Returns:
                GcsBlobProtocol: In-memory blob instance.

            """
            if blob_name not in blobs_by_name:
                blobs_by_name[blob_name] = _InMemoryBlob(blob_name)
            return blobs_by_name[blob_name]

    class _InMemoryClient:
        """In-memory GCS client satisfying GcsClientProtocol."""

        def bucket(self, bucket_name: str) -> GcsBucketProtocol:
            """
            Return in-memory bucket.

            Args:
                bucket_name: Bucket name string.

            Returns:
                GcsBucketProtocol: In-memory bucket instance.

            """
            _ = bucket_name
            return _InMemoryBucket()

        def list_blobs(
            self, bucket_or_name: GcsBucketProtocol | str, prefix: str = ""
        ) -> Iterable[GcsBlobProtocol]:
            """
            List in-memory blobs matching prefix.

            Args:
                bucket_or_name: Bucket reference or name.
                prefix: Object prefix filter.

            Returns:
                Iterable[GcsBlobProtocol]: Matching in-memory blobs.

            """
            _ = bucket_or_name
            return [b for name, b in blobs_by_name.items() if name.startswith(prefix)]

    client = _InMemoryClient()
    src_dataset = tmp_path / "train.jsonl"
    src_dataset.write_text(
        json.dumps({"prompt": "hi", "sql": "SELECT 1"}), encoding="utf-8"
    )
    gcs_uri = sync_dataset_to_gcs(
        src_dataset, "gs://t1d-bucket/datasets", gcs_client=client
    )
    assert gcs_uri == "gs://t1d-bucket/datasets/train.jsonl"
    assert isinstance(blobs_by_name["datasets/train.jsonl"], Md5BlobProtocol)

    echo_fake = FakeSubprocessRunner(
        on_call=lambda cmd: _proc("Status: SUCCESS " + " ".join(cmd))
    )
    assert isinstance(echo_fake, SubprocessRunnerProtocol)
    runner_with_exec = RemoteTpuMaxTextRunner(subprocess_runner=echo_fake)

    dl_dir = tmp_path / "downloaded"
    resolved_dir = asyncio.run(
        runner_with_exec.download_checkpoints_from_gcs(
            "gs://t1d-bucket/datasets", dl_dir, gcs_client=client
        )
    )
    assert resolved_dir == dl_dir
    assert (dl_dir / "train.jsonl").read_text(
        encoding="utf-8"
    ) == src_dataset.read_text(encoding="utf-8")

    max_cfg_exec = MaxTextConfig(
        project_id="stanford-t1d",
        zone="us-central2-b",
        accelerator_type=TpuAcceleratorType.V4_8,
        num_slices=1,
        execute_tpu_dispatch=True,
    )
    job_cfg = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(src_dataset),
        output_dir=str(tmp_path / "exec_out"),
        dry_run=False,
        maxtext_config=max_cfg_exec,
    )
    with patch("shutil.which", return_value="/usr/bin/xpk"):
        train_res = runner_with_exec.run_training(job_cfg)
    assert train_res["status"] == "SUCCESS"
    assert len(echo_fake.commands) >= 2
    create_cmd = " ".join(echo_fake.commands[0])
    assert "/usr/bin/xpk workload create" in create_cmd
    assert "--workload=t1d-gemma-2b" in create_cmd
    assert "--cluster=tpu-cluster-v4-8" in create_cmd
    assert "--zone=us-central2-b" in create_cmd
    assert "--tpu-type=v4-8" in create_cmd
    assert "--num-slices=1" in create_cmd
    describe_cmd = " ".join(echo_fake.commands[1])
    assert "/usr/bin/xpk workload describe" in describe_cmd
