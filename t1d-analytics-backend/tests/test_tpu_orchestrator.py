"""Unit tests for Cloud TPU MaxText orchestration, XPK/gcloud command builders, and GCS synchronization."""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Union
from unittest.mock import MagicMock, patch

import pytest

from t1d_analytics.gcs_storage import (
    GcsBlobProtocol,
    GcsBucketProtocol,
    GcsClientProtocol,
    _compute_md5_base64,
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
from t1d_analytics.tpu_orchestrator import RemoteTpuMaxTextRunner


@pytest.fixture(autouse=True)
def safe_subprocess() -> Generator[None, None, None]:
    """Intercept gcloud, xpk, and gsutil commands with mock success unless overridden."""
    original_run = subprocess.run

    def safe_run(
        cmd_arg: Union[str, Sequence[str]],
        capture_output: bool = False,
        text: bool = False,
        timeout: Optional[float] = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Intercept gcloud, xpk, and gsutil commands with mock success."""
        cmd_str = " ".join(cmd_arg) if isinstance(cmd_arg, list) else str(cmd_arg)

        if "gcloud" in cmd_str or "xpk" in cmd_str or "gsutil" in cmd_str:
            return subprocess.CompletedProcess(
                args=cmd_str, returncode=0, stdout='{"state": "READY"}', stderr=""
            )

        return original_run(
            cmd_arg,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=check,
        )

    with patch("subprocess.run", safe_run):
        yield


class FakeGcsBlob:
    """Contract-verified fake implementing GcsBlobProtocol and Md5BlobProtocol."""

    def __init__(self, name: str, payload: bytes = b"checkpoint_bytes") -> None:
        """Initialize fake GCS blob."""
        self.name = name
        self.payload = payload
        self.md5_hash: Optional[str] = _compute_md5_base64(payload)
        self.uploaded_from: Optional[str] = None
        self.downloaded_to: Optional[str] = None

    def download_to_filename(self, filename: str) -> None:
        """Write blob payload to destination filename."""
        self.downloaded_to = filename
        Path(filename).write_bytes(self.payload)

    def upload_from_filename(self, filename: str) -> None:
        """Record uploaded filename."""
        self.uploaded_from = filename


class FakeGcsBucket:
    """Contract-verified fake implementing GcsBucketProtocol."""

    def __init__(self, blobs: List[FakeGcsBlob]) -> None:
        """Initialize fake bucket with blob list."""
        self._blobs = blobs

    def blob(self, blob_name: str) -> GcsBlobProtocol:
        """Create and register a new FakeGcsBlob."""
        new_blob = FakeGcsBlob(blob_name)
        self._blobs.append(new_blob)
        return new_blob


class FakeGcsClient:
    """Contract-verified fake implementing GcsClientProtocol."""

    def __init__(self, blobs: Optional[List[FakeGcsBlob]] = None) -> None:
        """Initialize fake GCS client."""
        self.blobs: List[FakeGcsBlob] = list(blobs) if blobs is not None else []

    def bucket(self, bucket_name: str) -> GcsBucketProtocol:
        """Return a FakeGcsBucket."""
        _ = bucket_name
        return FakeGcsBucket(self.blobs)

    def list_blobs(
        self, bucket_or_name: Union[GcsBucketProtocol, str], prefix: str = ""
    ) -> Iterable[GcsBlobProtocol]:
        """Return blobs matching prefix."""
        _ = (bucket_or_name, prefix)
        return list(self.blobs)


def test_remote_tpu_maxtext_runner(tmp_path: Path) -> None:
    """Test RemoteTpuMaxTextRunner configuration generation and job dispatch."""
    runner = RemoteTpuMaxTextRunner()

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
    ds.write_text("data")
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
    content = config_file.read_text()
    assert "model_name: gemma-7b" in content
    assert "hardware: 'v4-8'" in content
    assert "zone: 'us-central2-b'" in content

    config_dry = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "tpu_dry"),
        dry_run=True,
    )
    res_dry = runner.run_training(config_dry)
    assert res_dry["status"] == "dry_run_completed"


def test_sync_dataset_to_gcs(tmp_path: Path) -> None:
    """Test sync_dataset_to_gcs upload, injectable GcsClientProtocol, validation, and error branches."""
    local_file = tmp_path / "data.parquet"
    local_file.write_text("test")

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
        with pytest.raises(
            RuntimeError,
            match="google-cloud-storage",
        ):
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
    runner = RemoteTpuMaxTextRunner()
    monkeypatch.setenv("T1D_AUTO_SYNC_GCS", "1")

    ds = tmp_path / "dataset.parquet"
    ds.write_text("dataset")
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
    """Test _dispatch_xpk_workload under xpk binary, gcloud fallback, and errors."""
    runner = RemoteTpuMaxTextRunner()
    ds = tmp_path / "ds.parquet"
    ds.write_text("data")
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("config")
    config = TrainingJobConfig(
        model_name="gemma-tpu",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out"),
    )
    max_cfg = MaxTextConfig(project_id="test-proj")

    with patch(
        "shutil.which",
        side_effect=lambda bin_name: "/usr/bin/xpk" if bin_name == "xpk" else None,
    ):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="OK"
            ),
        ):
            res_xpk = runner._dispatch_xpk_workload(config, max_cfg, cfg_path)
            assert res_xpk.returncode == 0

    with patch(
        "shutil.which",
        side_effect=lambda bin_name: (
            "/usr/bin/gcloud" if bin_name == "gcloud" else None
        ),
    ):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="OK"
            ),
        ):
            res_gcloud = runner._dispatch_xpk_workload(config, max_cfg, cfg_path)
            assert res_gcloud.returncode == 0

    with patch("shutil.which", return_value=None):
        with pytest.raises(
            RuntimeError,
            match="Neither 'xpk' nor 'gcloud' CLI is installed",
        ):
            runner._dispatch_xpk_workload(config, max_cfg, cfg_path)

    with patch(
        "shutil.which",
        side_effect=lambda bin_name: "/usr/bin/xpk" if bin_name == "xpk" else None,
    ):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stderr="Cluster down"
            ),
        ):
            with pytest.raises(RuntimeError, match="TPU dispatch failed: Cluster down"):
                runner._dispatch_xpk_workload(config, max_cfg, cfg_path)


def test_remote_tpu_poll_workload_status_branches() -> None:
    """Test poll_workload_status under xpk/gcloud polling, success, failure, and timeout."""
    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="test-proj")

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner.poll_workload_status("t1d-test", max_cfg))

    def mock_gcloud_only(cmd: str) -> Optional[str]:
        """Return gcloud path only when cmd is gcloud."""
        return "/usr/bin/gcloud" if cmd == "gcloud" else None

    with patch("shutil.which", side_effect=mock_gcloud_only):
        with patch(
            "t1d_analytics.tpu_orchestrator.RemoteTpuMaxTextRunner._poll_gcloud_tpu_status",
            return_value="SUCCESS",
        ) as mock_g:
            assert (
                asyncio.run(runner.poll_workload_status("t1d-test", max_cfg))
                == "SUCCESS"
            )
            mock_g.assert_called_once()

    with patch("shutil.which", return_value="/usr/bin/xpk"):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: SUCCESS"
            ),
        ):
            assert (
                asyncio.run(runner.poll_workload_status("t1d-test", max_cfg))
                == "SUCCESS"
            )

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: FAILED"
            ),
        ):
            assert (
                asyncio.run(runner.poll_workload_status("t1d-test", max_cfg))
                == "FAILED"
            )

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: PENDING"
            ),
        ):
            with pytest.raises(TimeoutError, match="polling timed out"):
                asyncio.run(
                    runner.poll_workload_status(
                        "t1d-test", max_cfg, timeout_seconds=0.01
                    )
                )

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        ):
            with pytest.raises(TimeoutError, match="polling timed out"):
                asyncio.run(
                    runner.poll_workload_status(
                        "t1d-test", max_cfg, timeout_seconds=0.01
                    )
                )


def test_remote_tpu_download_checkpoints_from_gcs(tmp_path: Path) -> None:
    """Test download_checkpoints_from_gcs with injectable FakeGcsClient, checksum verification, and errors."""
    runner = RemoteTpuMaxTextRunner()
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
    runner = RemoteTpuMaxTextRunner()
    ds = tmp_path / "ds.parquet"
    ds.write_text("dataset")
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
        patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: SUCCESS", stderr=""
            ),
        ),
    ):
        res = runner.run_training(config)
        assert res["status"] == "SUCCESS"
        manifest_path = out / "job_manifest.json"
        assert manifest_path.exists()


def test_tpu_workload_status_override_and_xla_preempted() -> None:
    """Test poll_workload_status with PREEMPTED status and XLA errors via SubprocessExecutorProtocol."""
    from t1d_analytics.errors import TpuOrchestrationError, TpuPreemptionError

    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="p1")

    with patch("shutil.which", return_value="/usr/bin/xpk"):
        mock_proc_success = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Workload workload-1 | Status: SUCCESS | Slice: v4-8",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_proc_success):
            assert (
                asyncio.run(
                    runner.poll_workload_status(
                        "workload-1", max_cfg, timeout_seconds=1
                    )
                )
                == "SUCCESS"
            )

        mock_proc_preempt = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="STATUS: PREEMPTED by node eviction",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_proc_preempt):
            with pytest.raises(TpuPreemptionError, match="was PREEMPTED"):
                asyncio.run(
                    runner.poll_workload_status(
                        "workload-1", max_cfg, timeout_seconds=1
                    )
                )

        mock_proc_xla = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="FATAL: XLA COMPILATION ERROR",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_proc_xla):
            with pytest.raises(
                TpuOrchestrationError, match="fatal XLA compilation error"
            ):
                asyncio.run(
                    runner.poll_workload_status(
                        "workload-1", max_cfg, timeout_seconds=1
                    )
                )


def test_tpu_workload_reclamation() -> None:
    """Test reclaim_tpu_workload across xpk and gcloud modes."""
    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="p1")

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner.reclaim_tpu_workload("workload-1", max_cfg))

    def mock_gcloud_reclaim(cmd: str) -> Optional[str]:
        """Return gcloud path only when cmd is gcloud."""
        return "/usr/bin/gcloud" if cmd == "gcloud" else None

    with patch("shutil.which", side_effect=mock_gcloud_reclaim):
        with patch(
            "t1d_analytics.tpu_orchestrator.RemoteTpuMaxTextRunner._reclaim_gcloud_tpu_workload",
            return_value=True,
        ) as mock_rg:
            assert (
                asyncio.run(runner.reclaim_tpu_workload("workload-1", max_cfg)) is True
            )
            mock_rg.assert_called_once()

    mock_succ = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("shutil.which", return_value="/usr/bin/xpk"):
        with patch("subprocess.run", return_value=mock_succ):
            assert (
                asyncio.run(runner.reclaim_tpu_workload("workload-1", max_cfg)) is True
            )

    mock_fail = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Cluster not reachable"
    )
    with patch("shutil.which", return_value="/usr/bin/xpk"):
        with patch("subprocess.run", return_value=mock_fail):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU workload"):
                asyncio.run(runner.reclaim_tpu_workload("workload-1", max_cfg))


def test_remote_tpu_run_training_preemption_and_retries(tmp_path: Path) -> None:
    """Test RemoteTpuMaxTextRunner preemption retry and general failure handling via SubprocessExecutorProtocol."""
    ds = tmp_path / "tpu_ds.parquet"
    ds.write_text("ds")
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
        preempt_cmds: list[list[str]] = []

        def _always_preempt_exec(
            cmd: Sequence[str],
            capture_output: bool = True,
            text: bool = True,
            check: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            """Simulate XPK describe returning PREEMPTED and delete succeeding."""
            _ = (capture_output, text, check)
            preempt_cmds.append(list(cmd))
            if "describe" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="Status: PREEMPTED", stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ok", stderr=""
            )

        runner_preempt = RemoteTpuMaxTextRunner(
            subprocess_executor=_always_preempt_exec
        )
        with pytest.raises(RuntimeError, match="PREEMPTED"):
            runner_preempt.run_training(cfg)
        assert any("delete" in c for c in preempt_cmds)

        fail_cmds: list[list[str]] = []

        def _fatal_fail_exec(
            cmd: Sequence[str],
            capture_output: bool = True,
            text: bool = True,
            check: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            """Simulate XPK describe returning fatal XLA compilation error and delete succeeding."""
            _ = (capture_output, text, check)
            fail_cmds.append(list(cmd))
            if "describe" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="Fatal XLA compilation ERROR",
                    stderr="OOM",
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ok", stderr=""
            )

        runner_fail = RemoteTpuMaxTextRunner(subprocess_executor=_fatal_fail_exec)
        with pytest.raises(RuntimeError, match="XLA compilation error"):
            runner_fail.run_training(cfg)
        assert any("delete" in c for c in fail_cmds)

        describe_attempt = 0

        def _recover_on_retry_exec(
            cmd: Sequence[str],
            capture_output: bool = True,
            text: bool = True,
            check: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            """Simulate first XPK describe preempting, second returning non-zero, and third succeeding."""
            nonlocal describe_attempt
            _ = (capture_output, text, check)
            if "describe" in cmd:
                describe_attempt += 1
                if describe_attempt == 1:
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout="Status: PREEMPTED", stderr=""
                    )
                if describe_attempt == 2:
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=1, stdout="", stderr="transient"
                    )
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="Status: SUCCESS", stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ok", stderr=""
            )

        runner_recover = RemoteTpuMaxTextRunner(
            subprocess_executor=_recover_on_retry_exec
        )
        res_retry_succ = runner_recover.run_training(cfg)
        assert res_retry_succ["status"] == "SUCCESS"
        assert describe_attempt == 3


def test_poll_gcloud_tpu_status_terminal_states() -> None:
    """Test _poll_gcloud_tpu_status terminal READY, STOPPED, and non-JSON READY responses and CLI flags."""
    recorded_cmds: list[list[str]] = []

    def _recording_executor(
        cmd: Sequence[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Record gcloud describe CLI command and return READY JSON."""
        _ = (capture_output, text, check)
        recorded_cmds.append(list(cmd))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"state": "READY"}', stderr=""
        )

    runner = RemoteTpuMaxTextRunner(subprocess_executor=_recording_executor)
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner._reclaim_gcloud_tpu_workload("node-1", "us-central2-b"))

    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        assert (
            asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))
            == "SUCCESS"
        )
        assert recorded_cmds[-1] == [
            "/usr/bin/gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            "node-1",
            "--zone=us-central2-b",
            "--format=json",
        ]

        runner_default = RemoteTpuMaxTextRunner()
        mock_proc_stopped = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"state": "STOPPED"}', stderr=""
        )
        with patch("subprocess.run", return_value=mock_proc_stopped):
            assert (
                asyncio.run(
                    runner_default._poll_gcloud_tpu_status("node-1", "us-central2-b")
                )
                == "SUCCESS"
            )

        mock_proc_text_ready = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="State: READY (non-json output)", stderr=""
        )
        with patch("subprocess.run", return_value=mock_proc_text_ready):
            assert (
                asyncio.run(
                    runner_default._poll_gcloud_tpu_status("node-1", "us-central2-b")
                )
                == "SUCCESS"
            )


def test_poll_gcloud_tpu_status_preemption_and_errors() -> None:
    """Test _poll_gcloud_tpu_status preemption, fatal hardware error, and NOT_FOUND states."""
    runner = RemoteTpuMaxTextRunner()
    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        mock_proc_preempt = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"state": "PREEMPTED"}', stderr=""
        )
        with patch("subprocess.run", return_value=mock_proc_preempt):
            with pytest.raises(RuntimeError, match="was PREEMPTED"):
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))

        mock_proc_health_preempt = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"state": "STOPPING", "healthDescription": "Node was preempted"}',
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_proc_health_preempt):
            with pytest.raises(RuntimeError, match="was PREEMPTED"):
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))

        mock_proc_err = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"state": "RUNNING", "healthDescription": "Fatal hardware ERROR"}',
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_proc_err):
            with pytest.raises(RuntimeError, match="encountered fatal error"):
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))

        mock_proc_nf = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="ERROR: (gcloud) NOT_FOUND: Resource not found",
        )
        with patch("subprocess.run", return_value=mock_proc_nf):
            with pytest.raises(RuntimeError, match="not found in zone"):
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))


def test_poll_gcloud_tpu_status_timeout_and_cancellation() -> None:
    """Test _poll_gcloud_tpu_status and _poll_workload_status_sync timeout and cancellation branches."""
    import threading

    runner = RemoteTpuMaxTextRunner()
    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        for pending_proc in (
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"state": "CREATING"}', stderr=""
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="Some garbled text without status",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="Connection reset by peer"
            ),
        ):
            with patch("subprocess.run", return_value=pending_proc):
                with pytest.raises(TimeoutError, match="polling timed out"):
                    asyncio.run(
                        runner._poll_gcloud_tpu_status(
                            "node-1", "us-central2-b", timeout_seconds=0.05
                        )
                    )

    max_cfg_sync = MaxTextConfig(project_id="p-sync", zone="us-central2-b")
    with (
        patch("shutil.which", return_value="/usr/bin/xpk"),
        patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: SUCCESS", stderr=""
            ),
        ),
    ):
        assert runner._poll_workload_status_sync("wl-sync", max_cfg_sync) == "SUCCESS"

    cancel_ev = threading.Event()
    cancel_ev.set()
    with patch("shutil.which", return_value="/usr/bin/xpk"):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        ):
            with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
                runner._poll_workload_status_sync(
                    "wl-sync", max_cfg_sync, cancel_event=cancel_ev
                )
            with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
                asyncio.run(
                    runner.poll_workload_status(
                        "wl-sync", max_cfg_sync, cancel_event=cancel_ev
                    )
                )

        mid_ev_sync = threading.Event()

        def _set_mid_sync(
            cmd: Sequence[str],
            capture_output: bool = True,
            text: bool = True,
            check: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            """Set cancel event on first describe call."""
            _ = (cmd, capture_output, text, check)
            mid_ev_sync.set()
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="Status: PENDING", stderr=""
            )

        with patch("subprocess.run", side_effect=_set_mid_sync):
            with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
                runner._poll_workload_status_sync(
                    "wl-sync",
                    max_cfg_sync,
                    timeout_seconds=1.0,
                    cancel_event=mid_ev_sync,
                )

        mid_ev_async = threading.Event()

        def _set_mid_async(
            cmd: Sequence[str],
            capture_output: bool = True,
            text: bool = True,
            check: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            """Set cancel event on first async describe call."""
            _ = (cmd, capture_output, text, check)
            mid_ev_async.set()
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="Status: PENDING", stderr=""
            )

        with (
            patch("subprocess.run", side_effect=_set_mid_async),
            patch("asyncio.sleep", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
                asyncio.run(
                    runner.poll_workload_status(
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
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        ):
            with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
                runner._poll_gcloud_tpu_status_sync(
                    "wl-sync",
                    "us-central2-b",
                    timeout_seconds=1.0,
                    cancel_event=cancel_ev,
                )
            with pytest.raises(RuntimeError, match="cancelled and reclaimed"):
                asyncio.run(
                    runner._poll_gcloud_tpu_status(
                        "wl-sync",
                        "us-central2-b",
                        timeout_seconds=1.0,
                        cancel_event=cancel_ev,
                    )
                )


def test_reclaim_gcloud_tpu_workload_retries() -> None:
    """Test _reclaim_gcloud_tpu_workload and sync/async reclamation command flags and retry behavior."""
    recorded_cmds: list[list[str]] = []

    def _reclaim_executor(
        cmd: Sequence[str],
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Record gcloud/xpk delete commands and return success."""
        _ = (capture_output, text, check)
        recorded_cmds.append(list(cmd))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"state": "READY"}', stderr=""
        )

    runner = RemoteTpuMaxTextRunner(subprocess_executor=_reclaim_executor)
    max_cfg_sync = MaxTextConfig(project_id="p-sync", zone="us-central2-b")

    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        assert (
            asyncio.run(runner._reclaim_gcloud_tpu_workload("node-1", "us-central2-b"))
            is True
        )
        assert recorded_cmds[-1] == [
            "/usr/bin/gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            "node-1",
            "--zone=us-central2-b",
            "--quiet",
        ]

    runner_default = RemoteTpuMaxTextRunner()
    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        mock_fail = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="PermissionDenied"
        )
        with patch("subprocess.run", return_value=mock_fail):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM workload"):
                asyncio.run(
                    runner_default._reclaim_gcloud_tpu_workload(
                        "node-1", "us-central2-b", max_retries=2
                    )
                )

    with patch("shutil.which", return_value="/usr/bin/xpk"):
        assert runner._reclaim_tpu_workload_sync("wl-sync", max_cfg_sync) is True
        assert recorded_cmds[-1] == [
            "/usr/bin/xpk",
            "workload",
            "delete",
            "--workload=wl-sync",
            "--cluster=tpu-cluster-v4-8",
        ]
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: SUCCESS"
            ),
        ):
            assert (
                runner_default._poll_workload_status_sync(
                    "wl-sync", max_cfg_sync, timeout_seconds=1.0
                )
                == "SUCCESS"
            )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: PENDING"
            ),
        ):
            with pytest.raises(TimeoutError):
                runner_default._poll_workload_status_sync(
                    "wl-sync", max_cfg_sync, timeout_seconds=0.01
                )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stderr="delete err"
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU workload"):
                runner_default._reclaim_tpu_workload_sync("wl-sync", max_cfg_sync)

    with patch(
        "shutil.which",
        side_effect=lambda b: "/usr/bin/gcloud" if b == "gcloud" else None,
    ):
        assert runner._poll_workload_status_sync("wl-sync", max_cfg_sync) == "SUCCESS"
        assert runner._reclaim_tpu_workload_sync("wl-sync", max_cfg_sync) is True
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"state": "CREATING"}'
            ),
        ):
            with pytest.raises(TimeoutError):
                runner_default._poll_gcloud_tpu_status_sync(
                    "wl-sync", "us-central2-b", timeout_seconds=0.01
                )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stderr="delete err"
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM workload"):
                runner_default._reclaim_gcloud_tpu_workload_sync(
                    "wl-sync", "us-central2-b", max_retries=2
                )
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError):
            runner_default._poll_gcloud_tpu_status_sync("wl-sync", "us-central2-b")
        with pytest.raises(RuntimeError):
            runner_default._reclaim_gcloud_tpu_workload_sync("wl-sync", "us-central2-b")
        with pytest.raises(RuntimeError):
            runner_default._reclaim_tpu_workload_sync("wl-sync", max_cfg_sync)


def test_tpu_orchestrator_protocol_stubs(tmp_path: Path) -> None:
    """Verify GcsClientProtocol and SubprocessExecutorProtocol contracts through upload_dataset_to_gcs and download_artifacts_from_gcs."""
    import base64
    import hashlib
    from typing import Iterable, Sequence, Union

    from t1d_analytics.gcs_storage import (
        GcsBlobProtocol,
        GcsBucketProtocol,
        Md5BlobProtocol,
    )
    from t1d_analytics.tpu_orchestrator import SubprocessExecutorProtocol

    uploaded_files: dict[str, bytes] = {}

    class _InMemoryBlob:
        """In-memory GCS blob satisfying GcsBlobProtocol and Md5BlobProtocol."""

        def __init__(self, name: str) -> None:
            """Initialize blob with name."""
            self.name = name
            self.md5_hash: Optional[str] = None

        def upload_from_filename(self, filename: str) -> None:
            """Store uploaded file bytes and compute base64 MD5 digest."""
            raw = Path(filename).read_bytes()
            uploaded_files[self.name] = raw
            self.md5_hash = base64.b64encode(hashlib.md5(raw).digest()).decode("ascii")

        def download_to_filename(self, filename: str) -> None:
            """Write stored bytes to destination file."""
            Path(filename).write_bytes(uploaded_files.get(self.name, b"ckpt-weights"))

    blobs_by_name: dict[str, _InMemoryBlob] = {}

    class _InMemoryBucket:
        """In-memory GCS bucket satisfying GcsBucketProtocol."""

        def blob(self, blob_name: str) -> GcsBlobProtocol:
            """Return or create an in-memory blob."""
            if blob_name not in blobs_by_name:
                blobs_by_name[blob_name] = _InMemoryBlob(blob_name)
            return blobs_by_name[blob_name]

    class _InMemoryClient:
        """In-memory GCS client satisfying GcsClientProtocol."""

        def bucket(self, bucket_name: str) -> GcsBucketProtocol:
            """Return in-memory bucket."""
            _ = bucket_name
            return _InMemoryBucket()

        def list_blobs(
            self, bucket_or_name: Union[GcsBucketProtocol, str], prefix: str = ""
        ) -> Iterable[GcsBlobProtocol]:
            """List in-memory blobs matching prefix."""
            _ = bucket_or_name
            return [b for name, b in blobs_by_name.items() if name.startswith(prefix)]

    client = _InMemoryClient()
    runner = RemoteTpuMaxTextRunner()

    src_dataset = tmp_path / "train.jsonl"
    src_dataset.write_text('{"prompt": "hi", "sql": "SELECT 1"}\n', encoding="utf-8")
    gcs_uri = sync_dataset_to_gcs(
        src_dataset, "gs://t1d-bucket/datasets", gcs_client=client
    )
    assert gcs_uri == "gs://t1d-bucket/datasets/train.jsonl"
    assert isinstance(blobs_by_name["datasets/train.jsonl"], Md5BlobProtocol)

    dl_dir = tmp_path / "downloaded"
    resolved_dir = asyncio.run(
        runner.download_checkpoints_from_gcs(
            "gs://t1d-bucket/datasets", dl_dir, gcs_client=client
        )
    )
    assert resolved_dir == dl_dir
    assert (dl_dir / "train.jsonl").read_text(
        encoding="utf-8"
    ) == src_dataset.read_text(encoding="utf-8")

    recorded_commands: list[list[str]] = []

    class _EchoExecutor:
        """Subprocess executor recording commands and returning realistic XPK status output."""

        def __call__(
            self,
            cmd: Sequence[str],
            capture_output: bool = True,
            text: bool = True,
            check: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            """Record command arguments and return CompletedProcess."""
            _ = (capture_output, text, check)
            recorded_commands.append(list(cmd))
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="Status: SUCCESS\n" + " ".join(cmd),
                stderr="",
            )

    executor = _EchoExecutor()
    assert isinstance(executor, SubprocessExecutorProtocol)
    runner_with_exec = RemoteTpuMaxTextRunner(subprocess_executor=executor)
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
    assert len(recorded_commands) >= 2
    create_cmd = " ".join(recorded_commands[0])
    assert "/usr/bin/xpk workload create" in create_cmd
    assert "--workload=t1d-gemma-2b" in create_cmd
    assert "--cluster=tpu-cluster-v4-8" in create_cmd
    assert "--zone=us-central2-b" in create_cmd
    assert "--tpu-type=v4-8" in create_cmd
    assert "--num-slices=1" in create_cmd
    describe_cmd = " ".join(recorded_commands[1])
    assert "/usr/bin/xpk workload describe" in describe_cmd
