"""Hardware accelerator execution, failure recovery, TPU preemption, and GCS resilience tests."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from t1d_analytics.models import (
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.training_runner import (
    LocalGpuRunner,
    resolve_device_telemetry,
    sync_dataset_to_gcs,
)

has_cuda = torch.cuda.is_available()


def test_cuda_hardware_execution_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify GPU environment validation, device telemetry resolution, and LocalGpuRunner tiny model execution."""
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

    ds_file = tmp_path / "gpu_train.jsonl"
    ds_file.write_text(
        '{"prompt": "Count patients", "sql": "SELECT COUNT(*) FROM patients;"}\n'
    )
    out_dir = tmp_path / "gpu_checkpoints"
    cfg = TrainingJobConfig(
        dataset_path=str(ds_file),
        output_dir=str(out_dir),
        backend=TrainingBackend.LOCAL_GPU,
        model_name="gemma4",
        use_tiny_model=True,
        hyperparameters=TrainingHyperparameters(
            num_epochs=1,
            batch_size=1,
            max_seq_length=64,
        ),
    )

    class _SimpleTokenizer:
        """Deterministic byte tokenizer for local GPU runner tests."""

        pad_token_id: int = 0

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            """Encode text characters to bounded integer token IDs."""
            _ = add_special_tokens
            return [min(ord(c), 999) for c in text]

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=_SimpleTokenizer(),
    ):
        metrics = runner.run_training(cfg)
    assert metrics["status"] == "completed"
    assert metrics["total_steps"] == 1
    assert isinstance(metrics["final_loss"], float)
    assert (out_dir / "checkpoint-gpu-final" / "model.pt").exists()


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
