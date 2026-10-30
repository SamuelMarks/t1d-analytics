"""Unit tests for Cloud TPU MaxText orchestration, XPK/gcloud command builders, and GCS synchronization."""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Union
from unittest.mock import MagicMock, patch

import pytest

from t1d_analytics.models import (
    MaxTextConfig,
    TpuAcceleratorType,
    TpuSchedulingType,
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.tpu_orchestrator import (
    GcsBlobProtocol,
    GcsBucketProtocol,
    GcsClientProtocol,
    RemoteTpuMaxTextRunner,
    _compute_md5_base64,
    sync_dataset_to_gcs,
)


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
    """Test poll_workload_status under status override, success, failure, and timeout."""
    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="test-proj")

    assert (
        asyncio.run(
            runner.poll_workload_status("t1d-test", max_cfg, status_override="SUCCESS")
        )
        == "SUCCESS"
    )

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
    """Test poll_workload_status overrides, PREEMPTED status, and XLA errors."""
    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="p1")

    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": "PREEMPTED"}):
        with pytest.raises(RuntimeError, match="was PREEMPTED on spot TPU capacity"):
            asyncio.run(runner.poll_workload_status("workload-1", max_cfg))

    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": "XLA_ERROR"}):
        with pytest.raises(RuntimeError, match="fatal XLA compilation error"):
            asyncio.run(runner.poll_workload_status("workload-1", max_cfg))

    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": "PROVISIONING"}):
        assert (
            asyncio.run(runner.poll_workload_status("workload-1", max_cfg))
            == "PROVISIONING"
        )

    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": ""}):
        with patch("shutil.which", return_value="/usr/bin/xpk"):
            mock_proc_preempt = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="STATUS: PREEMPTED by node eviction",
                stderr="",
            )
            with patch("subprocess.run", return_value=mock_proc_preempt):
                with pytest.raises(RuntimeError, match="was PREEMPTED"):
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
                with pytest.raises(RuntimeError, match="fatal XLA compilation error"):
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
    """Test RemoteTpuMaxTextRunner preemption retry and general failure handling."""
    runner = RemoteTpuMaxTextRunner()
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
        with patch(
            "t1d_analytics.tpu_orchestrator.RemoteTpuMaxTextRunner.poll_workload_status",
            side_effect=RuntimeError("was PREEMPTED"),
        ):
            with patch(
                "t1d_analytics.tpu_orchestrator.RemoteTpuMaxTextRunner.reclaim_tpu_workload"
            ) as mock_reclaim:
                with pytest.raises(RuntimeError, match="was PREEMPTED"):
                    runner.run_training(cfg)
                mock_reclaim.assert_called_once()

        with patch(
            "t1d_analytics.tpu_orchestrator.RemoteTpuMaxTextRunner.poll_workload_status",
            side_effect=RuntimeError("Generic cluster error"),
        ):
            with patch(
                "t1d_analytics.tpu_orchestrator.RemoteTpuMaxTextRunner.reclaim_tpu_workload"
            ) as mock_reclaim2:
                with pytest.raises(RuntimeError, match="Generic cluster error"):
                    runner.run_training(cfg)
                mock_reclaim2.assert_called_once()

        with patch.object(
            runner,
            "poll_workload_status",
            new=MagicMock(side_effect=[RuntimeError("was PREEMPTED"), "SUCCESS"]),
        ):
            res_retry_succ = runner.run_training(cfg)
            assert res_retry_succ["status"] == "SUCCESS"
        with patch.object(
            runner, "reclaim_tpu_workload", new=MagicMock(return_value=True)
        ):
            assert (
                runner._invoke_reclaim_sync("wl-1", MaxTextConfig(project_id="p1"))
                is True
            )

        async def _fake_async_poll(
            w_name: str, m_cfg: MaxTextConfig, timeout_seconds: float = 5.0
        ) -> str:
            """Return SUCCESS from async coroutine."""
            _ = (w_name, m_cfg, timeout_seconds)
            return "SUCCESS"

        async def _fake_async_reclaim(w_name: str, m_cfg: MaxTextConfig) -> bool:
            """Return True from async coroutine."""
            _ = (w_name, m_cfg)
            return True

        with patch.object(runner, "poll_workload_status", new=_fake_async_poll):
            assert (
                runner._invoke_status_poll_sync(
                    "wl-1", MaxTextConfig(project_id="p1"), timeout_seconds=1.0
                )
                == "SUCCESS"
            )
        with patch.object(runner, "reclaim_tpu_workload", new=_fake_async_reclaim):
            assert (
                runner._invoke_reclaim_sync("wl-1", MaxTextConfig(project_id="p1"))
                is True
            )


def test_poll_gcloud_tpu_status_and_reclaim_states() -> None:
    """Test _poll_gcloud_tpu_status, _reclaim_gcloud_tpu_workload, and synchronous helpers."""
    runner = RemoteTpuMaxTextRunner()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))
        with pytest.raises(RuntimeError, match=r"Google Cloud CLI .* is not installed"):
            asyncio.run(runner._reclaim_gcloud_tpu_workload("node-1", "us-central2-b"))

    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        mock_proc_ready = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"state": "READY"}', stderr=""
        )
        with patch("subprocess.run", return_value=mock_proc_ready):
            assert (
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))
                == "SUCCESS"
            )

        mock_proc_stopped = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"state": "STOPPED"}', stderr=""
        )
        with patch("subprocess.run", return_value=mock_proc_stopped):
            assert (
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))
                == "SUCCESS"
            )

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

        mock_proc_text_ready = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="State: READY (non-json output)", stderr=""
        )
        with patch("subprocess.run", return_value=mock_proc_text_ready):
            assert (
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))
                == "SUCCESS"
            )

        mock_proc_nf = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="ERROR: (gcloud) NOT_FOUND: Resource not found",
        )
        with patch("subprocess.run", return_value=mock_proc_nf):
            with pytest.raises(RuntimeError, match="not found in zone"):
                asyncio.run(runner._poll_gcloud_tpu_status("node-1", "us-central2-b"))

        mock_proc_pending = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"state": "CREATING"}', stderr=""
        )
        with patch("subprocess.run", return_value=mock_proc_pending):
            with pytest.raises(TimeoutError, match="polling timed out"):
                asyncio.run(
                    runner._poll_gcloud_tpu_status(
                        "node-1", "us-central2-b", timeout_seconds=0.08
                    )
                )

        mock_proc_bad_json = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Some garbled text without status",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_proc_bad_json):
            with pytest.raises(TimeoutError, match="polling timed out"):
                asyncio.run(
                    runner._poll_gcloud_tpu_status(
                        "node-1", "us-central2-b", timeout_seconds=0.08
                    )
                )

        mock_proc_transient = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Connection reset by peer"
        )
        with patch("subprocess.run", return_value=mock_proc_transient):
            with pytest.raises(TimeoutError, match="polling timed out"):
                asyncio.run(
                    runner._poll_gcloud_tpu_status(
                        "node-1", "us-central2-b", timeout_seconds=0.08
                    )
                )

        mock_ok = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with patch("subprocess.run", return_value=mock_ok):
            assert (
                asyncio.run(
                    runner._reclaim_gcloud_tpu_workload("node-1", "us-central2-b")
                )
                is True
            )

        mock_fail = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="PermissionDenied"
        )
        with patch("subprocess.run", return_value=mock_fail):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM workload"):
                asyncio.run(
                    runner._reclaim_gcloud_tpu_workload(
                        "node-1", "us-central2-b", max_retries=2
                    )
                )

    max_cfg_sync = MaxTextConfig(project_id="p-sync", zone="us-central2-b")
    assert (
        runner._poll_workload_status_sync(
            "wl-sync", max_cfg_sync, status_override="SUCCESS"
        )
        == "SUCCESS"
    )
    with patch("shutil.which", return_value="/usr/bin/xpk"):
        assert runner._reclaim_tpu_workload_sync("wl-sync", max_cfg_sync) is True
        assert runner._invoke_reclaim_sync("wl-sync", max_cfg_sync) is True
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Status: SUCCESS"
            ),
        ):
            assert (
                runner._poll_workload_status_sync(
                    "wl-sync", max_cfg_sync, timeout_seconds=1.0
                )
                == "SUCCESS"
            )
            assert (
                runner._invoke_status_poll_sync(
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
                runner._poll_workload_status_sync(
                    "wl-sync", max_cfg_sync, timeout_seconds=0.01
                )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stderr="delete err"
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU workload"):
                runner._reclaim_tpu_workload_sync("wl-sync", max_cfg_sync)

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
                runner._poll_gcloud_tpu_status_sync(
                    "wl-sync", "us-central2-b", timeout_seconds=0.01
                )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stderr="delete err"
            ),
        ):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM workload"):
                runner._reclaim_gcloud_tpu_workload_sync(
                    "wl-sync", "us-central2-b", max_retries=2
                )
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError):
            runner._poll_gcloud_tpu_status_sync("wl-sync", "us-central2-b")
        with pytest.raises(RuntimeError):
            runner._reclaim_gcloud_tpu_workload_sync("wl-sync", "us-central2-b")
        with pytest.raises(RuntimeError):
            runner._reclaim_tpu_workload_sync("wl-sync", max_cfg_sync)
        with pytest.raises(RuntimeError):
            runner._invoke_reclaim_sync("wl-sync", max_cfg_sync)


def test_tpu_orchestrator_protocol_stubs() -> None:
    """Test protocol stub methods and properties in tpu_orchestrator."""
    from typing import Iterable, Sequence, Union

    from t1d_analytics.tpu_orchestrator import (
        GcsBlobProtocol,
        GcsBucketProtocol,
        Md5BlobProtocol,
        SubprocessExecutorProtocol,
    )

    class _ConcreteBlob:
        """Concrete blob satisfying GcsBlobProtocol and Md5BlobProtocol."""

        name = "blob"
        md5_hash: Optional[str] = None

        def download_to_filename(self, filename: str) -> None:
            """No-op download."""

        def upload_from_filename(self, filename: str) -> None:
            """No-op upload."""

    class _ConcreteBucket:
        """Concrete bucket satisfying GcsBucketProtocol."""

        def blob(self, blob_name: str) -> GcsBlobProtocol:
            """Return concrete blob."""
            return _ConcreteBlob()

    class _ConcreteClient:
        """Concrete client satisfying GcsClientProtocol."""

        def bucket(self, bucket_name: str) -> GcsBucketProtocol:
            """Return concrete bucket."""
            return _ConcreteBucket()

        def list_blobs(
            self, bucket_or_name: Union[GcsBucketProtocol, str], prefix: str = ""
        ) -> Iterable[GcsBlobProtocol]:
            """Return empty blob list."""
            return []

    class _ConcreteExecutor:
        """Concrete executor satisfying SubprocessExecutorProtocol."""

        def __call__(
            self,
            cmd: Sequence[str],
            capture_output: bool = True,
            text: bool = True,
            check: bool = False,
        ) -> subprocess.CompletedProcess[str]:
            """Return completed process."""
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

    blob = _ConcreteBlob()
    bucket = _ConcreteBucket()
    client = _ConcreteClient()
    executor = _ConcreteExecutor()
    blob.download_to_filename("f.txt")
    blob.upload_from_filename("f.txt")
    assert isinstance(blob, Md5BlobProtocol)
    assert isinstance(bucket.blob("b"), _ConcreteBlob)
    assert isinstance(client.bucket("bkt"), _ConcreteBucket)
    assert list(client.list_blobs("bkt")) == []
    assert isinstance(executor, SubprocessExecutorProtocol)
    assert executor(["echo"]).returncode == 0
    passthrough_proc = subprocess.run(
        ["echo", "passthrough"], capture_output=True, text=True
    )
    assert passthrough_proc.returncode == 0
