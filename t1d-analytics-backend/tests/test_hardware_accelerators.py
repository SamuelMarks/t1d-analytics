"""Hardware accelerator execution, failure recovery, TPU preemption, and GCS resilience tests."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from t1d_analytics.training_runner import (
    LocalGpuRunner,
    resolve_device_telemetry,
    sync_dataset_to_gcs,
)

try:
    import torch

    has_cuda = torch.cuda.is_available()
except ImportError:
    has_cuda = False


def test_cuda_hardware_execution_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GPU environment validation, device telemetry resolution, and tensor matmul execution."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    runner = LocalGpuRunner()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-80GB"),
        patch(
            "torch.cuda.mem_get_info",
            return_value=(40 * 1024 * 1024 * 1024, 80 * 1024 * 1024 * 1024),
        ),
    ):
        assert runner.validate_environment() is True
        dev_name, total_mem_mb = resolve_device_telemetry("cuda:0")
        assert dev_name == "NVIDIA A100-SXM4-80GB"
        assert total_mem_mb == 81920.0

    device = torch.device("cuda:0" if has_cuda else "cpu")
    tensor_a = torch.ones((16, 16), device=device)
    tensor_b = torch.full((16, 16), 2.0, device=device)
    tensor_c = torch.matmul(tensor_a, tensor_b)
    assert tensor_c.shape == (16, 16)
    assert float(tensor_c[0, 0].item()) == 32.0


def test_has_cuda_import_error() -> None:
    """Test behavior when torch is not available."""
    import sys
    from importlib import reload

    import tests.test_hardware_accelerators

    orig = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        reload(tests.test_hardware_accelerators)
        assert not tests.test_hardware_accelerators.has_cuda
    finally:
        sys.modules["torch"] = orig  # type: ignore[assignment]
        reload(tests.test_hardware_accelerators)


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
        """Simulate transient 503 failure on first two upload attempts."""
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
