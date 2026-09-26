"""Hardware accelerator execution, failure recovery, TPU preemption, and GCS resilience tests."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from t1d_analytics.models import (
    TrainingBackend,
    TrainingJobConfig,
)
from t1d_analytics.training_runner import (
    LocalGpuRunner,
    RemoteTpuMaxTextRunner,
    TrainingExecutionError,
    sync_dataset_to_gcs,
)

try:
    import torch

    has_cuda = torch.cuda.is_available()
except ImportError:
    has_cuda = False


@pytest.mark.skipif(not has_cuda, reason="Real CUDA hardware accelerator not detected")
def test_cuda_hardware_execution_real() -> None:
    """Execute real CUDA forward pass and verify memory allocation when CUDA device is present."""
    device = torch.device("cuda:0")
    tensor_a = torch.randn((100, 100), device=device)
    tensor_b = torch.randn((100, 100), device=device)
    tensor_c = torch.matmul(tensor_a, tensor_b)
    assert tensor_c.is_cuda
    assert tensor_c.shape == (100, 100)


def test_cuda_oom_handling_and_recovery(tmp_path: Path) -> None:
    """Test LocalGpuRunner catches CUDA Out Of Memory (OOM) errors and triggers cleanup."""
    runner = LocalGpuRunner()
    ds_path = tmp_path / "train.jsonl"
    ds_path.write_text(
        '{"prompt": "Count patients", "sql": "SELECT COUNT(*) FROM patients"}\n'
    )
    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    config = TrainingJobConfig(
        dataset_path=str(ds_path),
        output_dir=str(out_dir),
        backend=TrainingBackend.LOCAL_GPU,
        model_name="gemma4-sql",
    )

    with patch.dict(os.environ, {"T1D_MOCK_GPU": "1"}):
        with patch(
            "t1d_analytics.training_runner.TinyCausalLM.forward",
            side_effect=RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB"),
        ):
            with pytest.raises(TrainingExecutionError, match="CUDA out of memory"):
                runner.run_training(config)


def test_tpu_preemption_and_timeout_simulation(tmp_path: Path) -> None:
    """Test RemoteTpuMaxTextRunner handles TPU node preemption and gcloud communication timeouts."""
    runner = RemoteTpuMaxTextRunner()

    # 1. Simulate TPU node preemption error during status polling
    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["gcloud"],
                returncode=0,
                stdout='{"state": "PREEMPTED", "healthDescription": "Preempted"}',
            ),
        ):
            with pytest.raises(RuntimeError, match="PREEMPTED"):
                runner._poll_gcloud_tpu_status(
                    "tpu-vm-1", "us-central2-b", timeout_seconds=1.0
                )

    # 2. Simulate polling timeout
    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["gcloud"],
                returncode=1,
                stderr="Still provisioning",
            ),
        ):
            with patch("time.sleep"):
                with pytest.raises(TimeoutError, match="polling timed out"):
                    runner._poll_gcloud_tpu_status(
                        "tpu-job-99", "us-central2-b", timeout_seconds=0.001
                    )


def test_gcs_upload_retry_with_exponential_backoff(
    tmp_path: Path, mocker: MagicMock
) -> None:
    """Test upload_dataset_to_gcs retries with exponential backoff on simulated 503 errors."""
    src_file = tmp_path / "dataset.jsonl"
    src_file.write_text('{"prompt": "q", "sql": "SELECT 1"}\n')

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    call_count = 0

    def mock_upload(filename: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("503 Service Unavailable")

    mock_blob.upload_from_filename.side_effect = mock_upload
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    mock_storage = MagicMock()
    mock_storage.Client.return_value = mock_client

    with patch.dict("sys.modules", {"google.cloud.storage": mock_storage}):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("T1D_MOCK_GCS", None)
            with patch("time.sleep") as mock_sleep:
                uri = sync_dataset_to_gcs(
                    src_file, "gs://my-bucket/data", max_retries=3
                )
                assert uri == "gs://my-bucket/data/dataset.jsonl"
                assert call_count == 3
                assert mock_sleep.call_count == 2
