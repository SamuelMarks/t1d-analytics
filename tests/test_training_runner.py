"""Unit tests for training runners and hardware orchestration."""

import os
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from t1d_analytics.models import (
    MaxTextConfig,
    TpuAcceleratorType,
    TpuSchedulingType,
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.training_runner import (
    DeterministicSubwordTokenizer,
    Gemma4SqlRunner,
    HuggingFaceCausalLMFactory,
    HuggingFaceCausalLMRunner,
    InsufficientDataError,
    LocalCpuRunner,
    LocalGpuRunner,
    RemoteTpuMaxTextRunner,
    TinyCausalLM,
    TinyCausalLMFactory,
    TrainingExecutionError,
    get_training_runner,
    prepare_torch_dataset,
    resolve_device_telemetry,
    sync_dataset_to_gcs,
)

_real_torch_device = torch.device


class _MockTorchDeviceType(type):
    """Metaclass for torch.device mock to preserve isinstance checks in PyTorch accelerator utils."""

    def __instancecheck__(cls, instance: object) -> bool:
        """Verify instance is a real torch.device instance."""
        return isinstance(instance, _real_torch_device)


class MockTorchDevice(metaclass=_MockTorchDeviceType):
    """Mock class redirecting torch.device instantiation to CPU while retaining type identity."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Return a real CPU torch device."""
        return _real_torch_device("cpu")


def test_prepare_dataset_raises_when_missing(tmp_path: Path) -> None:
    """Test that prepare_dataset raises FileNotFoundError for nonexistent paths."""
    runner = LocalCpuRunner()
    config = TrainingJobConfig(
        model_name="test",
        dataset_path=str(tmp_path / "missing.parquet"),
    )
    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        runner.prepare_dataset(config)


def test_prepare_dataset_success(tmp_path: Path) -> None:
    """Test that prepare_dataset returns existing Path."""
    ds = tmp_path / "data.parquet"
    ds.write_text("content")
    runner = LocalCpuRunner()
    config = TrainingJobConfig(model_name="test", dataset_path=str(ds))
    assert runner.prepare_dataset(config) == ds


def test_local_cpu_runner_execution_and_dry_run(tmp_path: Path) -> None:
    """Test LocalCpuRunner execution paths for normal, dry-run, and empty dataset modes."""
    ds = tmp_path / "data.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
    )
    out = tmp_path / "cpu_out"

    runner = LocalCpuRunner()
    assert runner.validate_environment() is True

    # Normal execution
    config = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(out),
        hyperparameters=TrainingHyperparameters(
            batch_size=2, num_epochs=1, learning_rate=1e-4
        ),
        dry_run=False,
    )
    res = runner.run_training(config)
    assert res["status"] == "completed"
    assert res["total_steps"] == 1
    assert res["final_loss"] > 0.0
    assert (out / "checkpoint-final" / "training_args.json").exists()

    # Dry-run execution with default hyperparameters (None)
    out_dry = tmp_path / "cpu_out_dry"
    config_dry = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(out_dry),
        dry_run=True,
    )
    res_dry = runner.run_training(config_dry)
    assert res_dry["status"] == "dry_run_completed"
    assert res_dry["final_loss"] == 0.0
    assert not (out_dry / "checkpoint-final").exists()

    # Empty dataset raises InsufficientDataError
    empty_ds = tmp_path / "empty.jsonl"
    empty_ds.write_text("")
    config_empty = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(empty_ds),
        output_dir=str(out),
    )
    with pytest.raises(InsufficientDataError):
        runner.run_training(config_empty)


def test_local_gpu_runner_validation_and_execution(tmp_path: Path) -> None:
    """Test LocalGpuRunner environment validation and training runs."""
    runner = LocalGpuRunner()

    # Test with CUDA_VISIBLE_DEVICES
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
        assert runner.validate_environment() is True

    # Test with T1D_MOCK_GPU
    mock_no_accel = MagicMock()
    mock_no_accel.cuda.is_available.return_value = False
    mock_no_accel.backends.mps.is_available.return_value = False
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "T1D_MOCK_GPU": "1"}):
        with patch.dict("sys.modules", {"torch": mock_no_accel}):
            assert runner.validate_environment() is True

    # Test failure when no GPU detected
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "T1D_MOCK_GPU": ""}):
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(RuntimeError, match="No compatible GPU device"):
                runner.validate_environment()

    # Test torch CUDA branch
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "T1D_MOCK_GPU": ""}):
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert runner.validate_environment() is True

    # Test torch Apple MPS branch
    mock_torch_mps = MagicMock()
    mock_torch_mps.cuda.is_available.return_value = False
    mock_torch_mps.backends.mps.is_available.return_value = True
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "T1D_MOCK_GPU": ""}):
        with patch.dict("sys.modules", {"torch": mock_torch_mps}):
            assert runner.validate_environment() is True

    # Test torch with neither CUDA nor MPS available
    mock_torch_none = MagicMock()
    mock_torch_none.cuda.is_available.return_value = False
    mock_torch_none.backends.mps.is_available.return_value = False
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "T1D_MOCK_GPU": ""}):
        with patch.dict("sys.modules", {"torch": mock_torch_none}):
            with pytest.raises(RuntimeError, match="No compatible GPU device"):
                runner.validate_environment()

    # Test execution
    ds = tmp_path / "data_gpu.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
    )
    out = tmp_path / "gpu_out"
    config = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=False,
    )
    res = runner.run_training(config)
    assert res["status"] == "completed"
    assert (out / "checkpoint-gpu-final" / "gpu_metadata.json").exists()

    # Test dry run
    out_dry = tmp_path / "gpu_dry"
    config_dry = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out_dry),
        dry_run=True,
    )
    res_dry = runner.run_training(config_dry)
    assert res_dry["status"] == "dry_run_completed"

    # Test empty dataset raises InsufficientDataError
    empty_ds = tmp_path / "empty_gpu.jsonl"
    empty_ds.write_text("")
    config_empty = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(empty_ds),
        output_dir=str(out),
    )
    with pytest.raises(InsufficientDataError):
        runner.run_training(config_empty)


def test_remote_tpu_maxtext_runner(tmp_path: Path) -> None:
    """Test RemoteTpuMaxTextRunner configuration generation and job dispatch."""
    runner = RemoteTpuMaxTextRunner()

    # Environment validation
    with patch.dict(os.environ, {"T1D_MOCK_TPU": "1"}):
        assert runner.validate_environment() is True

    with patch.dict(os.environ, {"T1D_MOCK_TPU": ""}):
        with patch("shutil.which", return_value="/usr/bin/gcloud"):
            assert runner.validate_environment() is True

        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.exists", return_value=True):
                assert runner.validate_environment() is True

        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(RuntimeError, match="Google Cloud CLI"):
                    runner.validate_environment()

    # YAML generation & training execution
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
    config_file = Path(res["config_file"])
    assert config_file.exists()
    content = config_file.read_text()
    assert "model_name: gemma-7b" in content
    assert "hardware: 'v4-8'" in content
    assert "zone: 'us-central2-b'" in content

    # Dry-run dispatch
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
    """Test sync_dataset_to_gcs upload, validation, and error branches."""
    local_file = tmp_path / "data.parquet"
    local_file.write_text("test")

    # Invalid bucket URL
    with pytest.raises(ValueError, match="Invalid GCS bucket URI"):
        sync_dataset_to_gcs(local_file, "s3://invalid-bucket")

    # Nonexistent local file
    with pytest.raises(FileNotFoundError, match="Local file not found"):
        sync_dataset_to_gcs(tmp_path / "missing.parquet", "gs://my-bucket")

    # Missing google-cloud-storage library
    with patch.dict(os.environ, {"T1D_MOCK_GCS": ""}):
        with patch.dict("sys.modules", {"google.cloud.storage": None}):
            with pytest.raises(
                RuntimeError,
                match="The 'google-cloud-storage' Python package is required",
            ):
                sync_dataset_to_gcs(local_file, "gs://my-bucket")

    # Mock GCS upload
    with patch.dict(os.environ, {"T1D_MOCK_GCS": "1"}):
        uri = sync_dataset_to_gcs(local_file, "gs://my-bucket/")
        assert uri == "gs://my-bucket/data.parquet"

    # Native library upload success
    mock_gcs = MagicMock()
    mock_client = MagicMock()
    mock_gcs.Client.return_value = mock_client
    with patch.dict(os.environ, {"T1D_MOCK_GCS": ""}):
        with patch.dict("sys.modules", {"google.cloud.storage": mock_gcs}):
            uri = sync_dataset_to_gcs(local_file, "gs://my-bucket")
            assert uri == "gs://my-bucket/data.parquet"

    # Native library upload failure with retry exhaustion
    mock_err_gcs = MagicMock()
    mock_err_client = MagicMock()
    mock_err_blob = MagicMock()
    mock_err_blob.upload_from_filename.side_effect = RuntimeError("Network timeout")
    mock_err_client.bucket.return_value.blob.return_value = mock_err_blob
    mock_err_gcs.Client.return_value = mock_err_client
    with patch.dict(os.environ, {"T1D_MOCK_GCS": ""}):
        with patch.dict("sys.modules", {"google.cloud.storage": mock_err_gcs}):
            with pytest.raises(
                RuntimeError,
                match="GCS upload failed after 3 attempts.*Network timeout",
            ):
                sync_dataset_to_gcs(local_file, "gs://my-bucket", max_retries=3)


def test_get_training_runner() -> None:
    """Test get_training_runner factory returns correct instances."""
    assert isinstance(get_training_runner(TrainingBackend.LOCAL_CPU), LocalCpuRunner)
    assert isinstance(get_training_runner(TrainingBackend.GEMMA_4_SQL), Gemma4SqlRunner)
    assert isinstance(get_training_runner(TrainingBackend.LOCAL_GPU), LocalGpuRunner)
    assert isinstance(
        get_training_runner(TrainingBackend.REMOTE_TPU_MAXTEXT), RemoteTpuMaxTextRunner
    )
    assert isinstance(get_training_runner(TrainingBackend("local-cpu")), LocalCpuRunner)


def test_local_cpu_runner_with_real_datasets(tmp_path: Path) -> None:
    """Test LocalCpuRunner dataset inspection, step progression, and metrics saving."""
    import duckdb

    runner = LocalCpuRunner()

    # 1. Real JSONL dataset
    jsonl_ds = tmp_path / "train.jsonl"
    jsonl_ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n{"prompt": "q2", "completion": "SELECT 2"}\n{"prompt": "q3", "completion": "SELECT 3"}\n{"prompt": "q4", "completion": "SELECT 4"}\n'
    )
    out_jsonl = tmp_path / "cpu_jsonl_out"

    config_jsonl = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(jsonl_ds),
        output_dir=str(out_jsonl),
        hyperparameters=TrainingHyperparameters(
            batch_size=2, num_epochs=2, learning_rate=1e-4
        ),
        dry_run=False,
    )
    res_jsonl = runner.run_training(config_jsonl)
    assert res_jsonl["status"] == "completed"
    assert res_jsonl["total_steps"] == 4  # 4 samples // 2 batch_size * 2 epochs
    assert len(res_jsonl["loss_history"]) == 4
    metrics_file = out_jsonl / "checkpoint-final" / "training_metrics.json"
    assert metrics_file.exists()

    # 2. Real Parquet dataset via DuckDB
    pq_ds = tmp_path / "train.parquet"
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE sft AS SELECT range AS id, 'SELECT 1' AS sql FROM range(6)"
    )
    conn.execute(f"COPY sft TO '{pq_ds}' (FORMAT PARQUET)")
    conn.close()

    out_pq = tmp_path / "cpu_pq_out"
    config_pq = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(pq_ds),
        output_dir=str(out_pq),
        hyperparameters=TrainingHyperparameters(
            batch_size=3, num_epochs=1, learning_rate=2e-5
        ),
        dry_run=False,
    )
    res_pq = runner.run_training(config_pq)
    assert res_pq["status"] == "completed"
    assert res_pq["total_steps"] == 2  # 6 samples // 3 batch_size * 1 epoch

    # 3. Unreadable dataset file triggers InsufficientDataError
    unreadable_ds = tmp_path / "corrupt.jsonl"
    unreadable_ds.write_text("corrupt")
    out_corrupt = tmp_path / "cpu_corrupt_out"
    config_corrupt = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(unreadable_ds),
        output_dir=str(out_corrupt),
        hyperparameters=TrainingHyperparameters(batch_size=2, num_epochs=2),
        dry_run=False,
    )
    with patch(
        "builtins.open", side_effect=[IOError("Read error"), MagicMock(), MagicMock()]
    ):
        with pytest.raises(InsufficientDataError):
            runner.run_training(config_corrupt)


def test_local_gpu_runner_real_datasets_and_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test LocalGpuRunner with real datasets, CUDA telemetry, and MPS device detection."""
    import duckdb

    runner = LocalGpuRunner()
    monkeypatch.setenv("T1D_MOCK_GPU", "1")

    # 1. Real JSONL dataset with CUDA mock
    jsonl_ds = tmp_path / "gpu_train.jsonl"
    jsonl_ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
        '{"prompt": "q3", "completion": "SELECT 3"}\n'
        '{"prompt": "q4", "completion": "SELECT 4"}\n'
    )
    out_cuda = tmp_path / "gpu_cuda_out"

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA A100"):
            with patch("torch.cuda.max_memory_allocated", return_value=52428800):
                with patch(
                    "torch.cuda.mem_get_info",
                    return_value=(500 * 1024 * 1024, 1000 * 1024 * 1024),
                ):
                    with (
                        patch("torch.device", MockTorchDevice),
                        patch.object(
                            torch.optim.Optimizer,
                            "_accelerator_graph_capture_health_check",
                            lambda self: None,
                            create=True,
                        ),
                        patch.object(
                            torch.optim.Optimizer,
                            "_cuda_graph_capture_health_check",
                            lambda self: None,
                            create=True,
                        ),
                    ):
                        config_cuda = TrainingJobConfig(
                            model_name="gemma-7b",
                            backend=TrainingBackend.LOCAL_GPU,
                            dataset_path=str(jsonl_ds),
                            output_dir=str(out_cuda),
                            hyperparameters=TrainingHyperparameters(
                                batch_size=2, num_epochs=1
                            ),
                            dry_run=False,
                        )
                        res_cuda = runner.run_training(config_cuda)
                        assert res_cuda["status"] == "completed"
                        assert res_cuda["device"] == "cuda:0"
                        assert res_cuda["mixed_precision"] == "fp16"
                        assert res_cuda["total_steps"] == 2
                        metrics_file = (
                            out_cuda / "checkpoint-gpu-final" / "training_metrics.json"
                        )
                        assert metrics_file.exists()

    # 2. Real Parquet dataset with MPS mock
    pq_ds = tmp_path / "gpu_train.parquet"
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE sft AS SELECT range AS id, 'SELECT ' || range AS sql FROM range(4)"
    )
    conn.execute(f"COPY sft TO '{pq_ds}' (FORMAT PARQUET)")
    conn.close()

    out_mps = tmp_path / "gpu_mps_out"
    monkeypatch.delenv("T1D_MOCK_GPU", raising=False)
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.device", MockTorchDevice):
                config_mps = TrainingJobConfig(
                    model_name="gemma-7b",
                    backend=TrainingBackend.LOCAL_GPU,
                    dataset_path=str(pq_ds),
                    output_dir=str(out_mps),
                    hyperparameters=TrainingHyperparameters(batch_size=2, num_epochs=1),
                    dry_run=False,
                )
                res_mps = runner.run_training(config_mps)
                assert res_mps["device"] == "mps"
                assert res_mps["mixed_precision"] == "fp32"


def test_remote_tpu_maxtext_runner_auto_sync_gcs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test RemoteTpuMaxTextRunner auto-sync to GCS and job manifest serialization."""
    runner = RemoteTpuMaxTextRunner()
    monkeypatch.setenv("T1D_MOCK_TPU", "1")
    monkeypatch.setenv("T1D_MOCK_GCS", "1")
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

    with patch("t1d_analytics.training_runner.sync_dataset_to_gcs") as mock_sync:
        mock_sync.return_value = "gs://test-bucket/dataset.parquet"
        res = runner.run_training(config)
        mock_sync.assert_called_once_with(ds, "gs://test-bucket")

    assert res["status"] == "dispatched"
    assert res["tpu_cluster"] == "v4-8"
    assert "gemma-4-sql" in res["workload_name"]
    manifest = out / "job_manifest.json"
    assert manifest.exists()


def test_count_dataset_samples_unsupported_suffix_and_duckdb_error(
    tmp_path: Path,
) -> None:
    """Test _count_dataset_samples returns 0 for unsupported suffixes and unreadable parquet."""
    runner = LocalCpuRunner()
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("hello world")
    assert runner._count_dataset_samples(txt_file) == 0

    bad_pq = tmp_path / "corrupt.parquet"
    bad_pq.write_text("not a parquet")
    assert runner._count_dataset_samples(bad_pq) == 0


def test_local_gpu_runner_torch_import_error(tmp_path: Path) -> None:
    """Test LocalGpuRunner raises TrainingExecutionError when torch is not importable."""
    runner = LocalGpuRunner()
    ds = tmp_path / "data.jsonl"
    ds.write_text('{"prompt": "q1", "completion": "SELECT 1"}\n')
    out = tmp_path / "gpu_no_torch"

    config = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=True,
    )
    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(
            TrainingExecutionError, match="GPU training execution failed"
        ):
            runner.run_training(config)


def test_local_gpu_runner_torch_cpu_fallback(tmp_path: Path) -> None:
    """Test LocalGpuRunner fallback to CPU when torch is available but has neither CUDA nor MPS."""
    runner = LocalGpuRunner()
    ds = tmp_path / "data.jsonl"
    ds.write_text('{"prompt": "q1", "completion": "SELECT 1"}\n')
    out = tmp_path / "gpu_cpu_fallback"

    config = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=True,
    )
    with patch.dict(os.environ, {"T1D_MOCK_GPU": ""}):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                res = runner.run_training(config)
                assert res["device"] == "cpu"


def test_get_training_runner_gemma_4_sql() -> None:
    """Test get_training_runner returns Gemma4SqlRunner for GEMMA_4_SQL backend."""
    runner = get_training_runner(TrainingBackend.GEMMA_4_SQL)
    assert isinstance(runner, Gemma4SqlRunner)


def test_gemma_4_sql_runner_validate_environment_success() -> None:
    """Test Gemma4SqlRunner environment validation success."""
    runner = Gemma4SqlRunner()
    with patch(
        "t1d_analytics.gemma_bridge.check_gemma_sql_installed", return_value=True
    ):
        assert runner.validate_environment() is True


def test_gemma_4_sql_runner_validate_environment_failure() -> None:
    """Test Gemma4SqlRunner environment validation raises RuntimeError when missing."""
    runner = Gemma4SqlRunner()
    with patch(
        "t1d_analytics.gemma_bridge.check_gemma_sql_installed", return_value=False
    ):
        with pytest.raises(
            RuntimeError, match="gemma-4-sql toolchain is not installed"
        ):
            runner.validate_environment()


def test_gemma_4_sql_runner_run_training_dry_run(tmp_path: Path) -> None:
    """Test Gemma4SqlRunner dry-run mode for pretrain, posttrain, and sft stages."""
    runner = Gemma4SqlRunner()
    ds = tmp_path / "train.parquet"
    ds.write_text("data")
    out = tmp_path / "gemma_out"

    # Stage: pretrain
    config_pretrain = TrainingJobConfig(
        model_name="gemma-4-pretrain",
        backend=TrainingBackend.GEMMA_4_SQL,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=True,
    )
    res_pre = runner.run_training(config_pretrain)
    assert res_pre["status"] == "dry_run_completed"
    assert res_pre["stage"] == "pretrain"
    assert (out / "gemma_sql_pretrain_config.yml").exists()

    # Stage: posttrain
    config_post = TrainingJobConfig(
        model_name="gemma-4-dpo",
        backend=TrainingBackend.GEMMA_4_SQL,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=True,
    )
    res_post = runner.run_training(config_post)
    assert res_post["status"] == "dry_run_completed"
    assert res_post["stage"] == "posttrain"

    # Stage: sft (default)
    config_sft = TrainingJobConfig(
        model_name="gemma-4-sft",
        backend=TrainingBackend.GEMMA_4_SQL,
        dataset_path=str(ds),
        output_dir=str(out),
        hyperparameters=TrainingHyperparameters(
            batch_size=8, num_epochs=2, learning_rate=3e-5
        ),
        dry_run=True,
    )
    res_sft = runner.run_training(config_sft)
    assert res_sft["status"] == "dry_run_completed"
    assert res_sft["stage"] == "sft"


def test_gemma_4_sql_runner_run_training_execution(tmp_path: Path) -> None:
    """Test Gemma4SqlRunner execution delegating to gemma_bridge.run_gemma_sql_train."""
    import subprocess

    runner = Gemma4SqlRunner()
    ds = tmp_path / "train.parquet"
    ds.write_text("data")
    out = tmp_path / "gemma_exec_out"

    config = TrainingJobConfig(
        model_name="gemma-4-sql",
        backend=TrainingBackend.GEMMA_4_SQL,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=False,
    )
    mock_completed = subprocess.CompletedProcess(
        args=["gemma-4-sql"],
        returncode=0,
        stdout="[MOCK] Success\n",
        stderr="",
    )
    with patch(
        "t1d_analytics.gemma_bridge.run_gemma_sql_train", return_value=mock_completed
    ) as mock_train:
        res = runner.run_training(config)
        assert res["status"] == "completed"
        assert res["returncode"] == 0
        assert "[MOCK] Success" in res["stdout"]
        mock_train.assert_called_once()


def test_prepare_torch_dataset_missing_and_unsupported(tmp_path: Path) -> None:
    """Test prepare_torch_dataset error conditions for missing files and unsupported formats."""
    missing = tmp_path / "nonexistent.jsonl"
    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        prepare_torch_dataset(missing)

    unsupported = tmp_path / "data.csv"
    unsupported.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        prepare_torch_dataset(unsupported)


def test_prepare_torch_dataset_jsonl_parsing_and_error(tmp_path: Path) -> None:
    """Test prepare_torch_dataset JSONL format reading, malformed row skipping, and empty error."""
    # Valid and malformed mixed
    jsonl_file = tmp_path / "valid.jsonl"
    jsonl_file.write_text(
        '{"prompt": "Show glucose", "sql": "SELECT glucose FROM cgm"}\n'
        "\n"
        "not a json\n"
        '{"question": "Average a1c", "completion": "SELECT AVG(a1c) FROM patient"}\n'
    )
    dataset = prepare_torch_dataset(jsonl_file, max_seq_length=32)
    assert len(dataset) == 2
    assert dataset[0][0].shape[0] == 31
    assert dataset[0][1].shape[0] == 31

    # Empty JSONL raises ValueError
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("\n\n")
    with pytest.raises(ValueError, match="contains no valid training pairs"):
        prepare_torch_dataset(empty_file)

    # Read error
    with patch("builtins.open", side_effect=IOError("Disk error")):
        with pytest.raises(ValueError, match="Failed to read JSONL dataset"):
            prepare_torch_dataset(jsonl_file)


def test_prepare_torch_dataset_parquet_parsing_and_error(tmp_path: Path) -> None:
    """Test prepare_torch_dataset Parquet format reading and error handling."""
    import duckdb

    pq_file = tmp_path / "clinical.parquet"
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE pairs AS SELECT 'What is TBR?' AS prompt, 'SELECT tbr FROM cgm' AS sql"
    )
    conn.execute(f"COPY pairs TO '{pq_file}' (FORMAT PARQUET)")
    conn.close()

    dataset = prepare_torch_dataset(pq_file, max_seq_length=64)
    assert len(dataset) == 1

    # Parquet read error
    bad_pq = tmp_path / "corrupted.parquet"
    bad_pq.write_text("invalid parquet binary")
    with pytest.raises(ValueError, match="Failed to read Parquet dataset"):
        prepare_torch_dataset(bad_pq)


def test_tiny_causal_lm_forward() -> None:
    """Test TinyCausalLM initialization and forward pass tensor shape."""
    import torch

    model = TinyCausalLM(vocab_size=100, hidden_dim=32)
    input_ids = torch.tensor([[1, 5, 9, 2], [3, 7, 8, 4]], dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 4, 100)


def test_local_cpu_runner_real_torch_training_and_model_save(tmp_path: Path) -> None:
    """Test LocalCpuRunner real training loop with PyTorch generating model.pt and dry-run."""
    runner = LocalCpuRunner()
    ds = tmp_path / "train.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
    )
    out = tmp_path / "cpu_real_out"

    # Real execution
    config = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(out),
        hyperparameters=TrainingHyperparameters(
            batch_size=1, num_epochs=1, learning_rate=1e-3
        ),
        dry_run=False,
    )
    res = runner.run_training(config)
    assert res["status"] == "completed"
    assert (out / "checkpoint-final" / "model.pt").exists()
    assert len(res["loss_history"]) == 2

    # Dry run execution with prepare_torch_dataset
    out_dry = tmp_path / "cpu_real_dry"
    config_dry = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(out_dry),
        dry_run=True,
    )
    res_dry = runner.run_training(config_dry)
    assert res_dry["status"] == "dry_run_completed"
    assert res_dry["final_loss"] == 0.0


def test_prepare_torch_dataset_edge_cases(tmp_path: Path) -> None:
    """Test prepare_torch_dataset with non-dict rows and empty text strings."""
    jsonl_file = tmp_path / "edge_cases.jsonl"
    jsonl_file.write_text(
        "123\n"  # non-dict line
        '"plain string"\n'  # non-dict line
        '{"prompt": "", "completion": ""}\n'  # empty text -> formatted with prompt tokens
    )
    dataset = prepare_torch_dataset(jsonl_file, max_seq_length=16)
    assert len(dataset) == 1
    assert dataset[0][0].tolist()[0] == 4  # <start_of_turn> token ID


def test_prepare_torch_dataset_parquet_empty_and_no_cols(tmp_path: Path) -> None:
    """Test prepare_torch_dataset when Parquet has no rows or columns cannot be resolved."""
    import duckdb

    pq_empty = tmp_path / "empty_rows.parquet"
    conn = duckdb.connect()
    conn.execute("CREATE TABLE empty_table (prompt VARCHAR, completion VARCHAR)")
    conn.execute(f"COPY empty_table TO '{pq_empty}' (FORMAT PARQUET)")
    conn.close()

    with pytest.raises(ValueError, match="contains no valid training pairs"):
        prepare_torch_dataset(pq_empty)

    mock_rel = MagicMock()
    mock_rel.description = None
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_rel
    with patch("duckdb.connect", return_value=mock_conn):
        with pytest.raises(ValueError, match="Parquet file contains no schema columns"):
            prepare_torch_dataset(pq_empty)


def test_local_cpu_runner_training_batch_break(tmp_path: Path) -> None:
    """Test LocalCpuRunner training loop break when step_idx reaches total_steps across uneven batches."""
    runner = LocalCpuRunner()
    ds = tmp_path / "three_samples.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
        '{"prompt": "q3", "completion": "SELECT 3"}\n'
    )
    out = tmp_path / "cpu_uneven_out"
    config = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(out),
        hyperparameters=TrainingHyperparameters(batch_size=2, num_epochs=1),
        dry_run=False,
    )
    res = runner.run_training(config)
    assert res["status"] == "completed"
    assert res["total_steps"] == 1


def test_local_cpu_runner_fallback_exception_paths(tmp_path: Path) -> None:
    """Test LocalCpuRunner exception fallback paths for dry-run and normal execution modes."""
    import duckdb

    runner = LocalCpuRunner()
    pq_file = tmp_path / "valid.parquet"
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS id, 'SELECT 1' AS sql FROM range(4)")
    conn.execute(f"COPY t TO '{pq_file}' (FORMAT PARQUET)")
    conn.close()

    out_normal = tmp_path / "cpu_fallback_normal"
    config_normal = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(pq_file),
        output_dir=str(out_normal),
        hyperparameters=TrainingHyperparameters(batch_size=2, num_epochs=1),
        dry_run=False,
    )
    with patch(
        "t1d_analytics.training_runner.prepare_torch_dataset",
        side_effect=RuntimeError("Mock torch failure"),
    ):
        with pytest.raises(
            TrainingExecutionError, match="CPU training execution failed"
        ):
            runner.run_training(config_normal)

    out_dry = tmp_path / "cpu_fallback_dry"
    config_dry = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(pq_file),
        output_dir=str(out_dry),
        hyperparameters=TrainingHyperparameters(batch_size=2, num_epochs=1),
        dry_run=True,
    )
    with patch(
        "t1d_analytics.training_runner.prepare_torch_dataset",
        side_effect=RuntimeError("Mock torch failure"),
    ):
        with pytest.raises(
            TrainingExecutionError, match="CPU training execution failed"
        ):
            runner.run_training(config_dry)


def test_local_gpu_runner_real_training_cuda_pipeline(tmp_path: Path) -> None:
    """Test LocalGpuRunner authentic training execution with CUDA telemetry and checkpoints."""
    runner = LocalGpuRunner()
    ds = tmp_path / "train_gpu.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
        '{"prompt": "q3", "completion": "SELECT 3"}\n'
    )
    out = tmp_path / "gpu_real_cuda_out"

    config = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out),
        hyperparameters=TrainingHyperparameters(
            batch_size=2, num_epochs=1, learning_rate=1e-3
        ),
        dry_run=False,
    )

    with patch.dict("os.environ", {"T1D_MOCK_GPU": "0"}):
        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-40GB"
            ):
                with patch(
                    "torch.cuda.mem_get_info",
                    return_value=(100 * 1024 * 1024, 40 * 1024 * 1024 * 1024),
                ):
                    with patch(
                        "torch.cuda.max_memory_allocated",
                        return_value=1024 * 1024 * 64,
                    ):
                        with (
                            patch("torch.device", MockTorchDevice),
                            patch.object(
                                torch.optim.Optimizer,
                                "_accelerator_graph_capture_health_check",
                                lambda self: None,
                                create=True,
                            ),
                            patch.object(
                                torch.optim.Optimizer,
                                "_cuda_graph_capture_health_check",
                                lambda self: None,
                                create=True,
                            ),
                        ):
                            res = runner.run_training(config)
                            assert res["status"] == "completed"
                            assert res["device"] == "cuda:0"
                            assert res["mixed_precision"] == "fp16"
                            assert (out / "checkpoint-gpu-final" / "model.pt").exists()
                            assert (
                                out / "checkpoint-gpu-final" / "gpu_metadata.json"
                            ).exists()


def test_local_gpu_runner_real_training_mps_and_dry_run(tmp_path: Path) -> None:
    """Test LocalGpuRunner real training execution with MPS device and dry-run mode."""
    runner = LocalGpuRunner()
    ds = tmp_path / "train_gpu.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
    )
    out_mps = tmp_path / "gpu_real_mps_out"

    config_mps = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out_mps),
        hyperparameters=TrainingHyperparameters(batch_size=1, num_epochs=1),
        dry_run=False,
    )
    with patch.dict("os.environ", {"T1D_MOCK_GPU": "0"}):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                with patch("torch.device", MockTorchDevice):
                    res_mps = runner.run_training(config_mps)
                    assert res_mps["device"] == "mps"
                    assert res_mps["mixed_precision"] == "fp32"
                    assert (out_mps / "checkpoint-gpu-final" / "model.pt").exists()

    # Dry run
    out_dry = tmp_path / "gpu_real_dry"
    config_dry = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out_dry),
        dry_run=True,
    )
    with patch.dict("os.environ", {"T1D_MOCK_GPU": "1"}):
        with patch("torch.device", MockTorchDevice):
            res_dry = runner.run_training(config_dry)
            assert res_dry["status"] == "dry_run_completed"
            assert res_dry["final_loss"] == 0.0


def test_local_gpu_runner_fallback_on_exception(tmp_path: Path) -> None:
    """Test LocalGpuRunner exception fallback for dry-run and normal runs."""
    runner = LocalGpuRunner()
    ds = tmp_path / "train_gpu.jsonl"
    ds.write_text('{"prompt": "q1", "completion": "SELECT 1"}\n')
    out = tmp_path / "gpu_fallback_out"

    config = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=False,
    )
    with patch.dict("os.environ", {"T1D_MOCK_GPU": "1"}):
        with patch(
            "t1d_analytics.training_runner.prepare_torch_dataset",
            side_effect=RuntimeError("GPU tensor error"),
        ):
            with pytest.raises(
                TrainingExecutionError, match="GPU training execution failed"
            ):
                runner.run_training(config)

    out_dry = tmp_path / "gpu_fallback_dry"
    config_dry = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out_dry),
        dry_run=True,
    )
    with patch.dict("os.environ", {"T1D_MOCK_GPU": "1"}):
        with patch(
            "t1d_analytics.training_runner.prepare_torch_dataset",
            side_effect=RuntimeError("GPU tensor error"),
        ):
            with pytest.raises(
                TrainingExecutionError, match="GPU training execution failed"
            ):
                runner.run_training(config_dry)


def test_local_gpu_runner_device_info_exceptions(tmp_path: Path) -> None:
    """Test LocalGpuRunner handling exceptions when probing GPU device name and VRAM."""
    runner = LocalGpuRunner()
    ds = tmp_path / "train_gpu_exc.jsonl"
    ds.write_text('{"prompt": "q1", "completion": "SELECT 1"}\n')
    out = tmp_path / "gpu_exc_out"

    config = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(out),
        dry_run=True,
    )

    with patch.dict("os.environ", {"T1D_MOCK_GPU": "0"}):
        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "torch.cuda.get_device_name",
                side_effect=RuntimeError("Device name error"),
            ):
                with patch(
                    "torch.cuda.mem_get_info",
                    side_effect=RuntimeError("Mem info error"),
                ):
                    with patch("torch.cuda.max_memory_allocated", return_value=0):
                        with patch("torch.device", MockTorchDevice):
                            res = runner.run_training(config)
                            assert res["status"] == "dry_run_completed"
                            assert res["device"] == "cuda:0"


def test_remote_tpu_dispatch_xpk_and_gcloud_branches(tmp_path: Path) -> None:
    """Test _dispatch_xpk_workload under mock, xpk binary, gcloud fallback, and errors."""
    import subprocess

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

    # 1. Mock TPU mode
    with patch.dict("os.environ", {"T1D_MOCK_TPU": "1"}):
        res = runner._dispatch_xpk_workload(config, max_cfg, cfg_path)
        assert res.returncode == 0
        assert "[MOCK]" in res.stdout

    # 2. Real XPK binary present
    with patch.dict("os.environ", {"T1D_MOCK_TPU": "0"}):
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

        # 3. XPK missing, gcloud fallback
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

        # 4. Neither XPK nor gcloud
        with patch("shutil.which", return_value=None):
            with pytest.raises(
                RuntimeError,
                match="Neither 'xpk' nor 'gcloud' CLI is installed",
            ):
                runner._dispatch_xpk_workload(config, max_cfg, cfg_path)

        # 5. Failure return code
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
                with pytest.raises(
                    RuntimeError, match="TPU dispatch failed: Cluster down"
                ):
                    runner._dispatch_xpk_workload(config, max_cfg, cfg_path)


def test_remote_tpu_poll_workload_status_branches() -> None:
    """Test poll_workload_status under mock, success, failure, and timeout."""
    import subprocess

    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="test-proj")

    # Mock TPU mode
    with patch.dict("os.environ", {"T1D_MOCK_TPU": "1"}):
        assert runner.poll_workload_status("t1d-test", max_cfg) == "SUCCESS"

    with patch.dict("os.environ", {"T1D_MOCK_TPU": "0"}):
        # Neither XPK nor gcloud binary -> raises
        with patch("shutil.which", return_value=None):
            with pytest.raises(
                RuntimeError, match="Google Cloud CLI .* is not installed"
            ):
                runner.poll_workload_status("t1d-test", max_cfg)

        # No XPK binary, gcloud binary available -> delegates to _poll_gcloud_tpu_status
        def mock_gcloud_only(cmd: str) -> Optional[str]:
            return "/usr/bin/gcloud" if cmd == "gcloud" else None

        with patch("shutil.which", side_effect=mock_gcloud_only):
            with patch.object(
                runner, "_poll_gcloud_tpu_status", return_value="SUCCESS"
            ) as mock_g:
                assert runner.poll_workload_status("t1d-test", max_cfg) == "SUCCESS"
                mock_g.assert_called_once()

        # XPK success
        with patch("shutil.which", return_value="/usr/bin/xpk"):
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="Status: SUCCESS"
                ),
            ):
                assert runner.poll_workload_status("t1d-test", max_cfg) == "SUCCESS"

            # XPK failure
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="Status: FAILED"
                ),
            ):
                assert runner.poll_workload_status("t1d-test", max_cfg) == "FAILED"

            # Polling timeout
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="Status: PENDING"
                ),
            ):
                with pytest.raises(TimeoutError, match="polling timed out"):
                    runner.poll_workload_status(
                        "t1d-test", max_cfg, timeout_seconds=0.01
                    )

            # Polling with non-zero returncode retrying until timeout
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=1, stdout=""
                ),
            ):
                with pytest.raises(TimeoutError, match="polling timed out"):
                    runner.poll_workload_status(
                        "t1d-test", max_cfg, timeout_seconds=0.01
                    )


def test_remote_tpu_download_checkpoints_from_gcs(tmp_path: Path) -> None:
    """Test download_checkpoints_from_gcs uri validation, mock mode, and real gcloud execution."""
    runner = RemoteTpuMaxTextRunner()
    out_dir = tmp_path / "ckpts"

    # Invalid URI
    with pytest.raises(ValueError, match="Invalid GCS bucket URI"):
        runner.download_checkpoints_from_gcs("s3://bucket/test", out_dir)

    # Mock mode
    with patch.dict("os.environ", {"T1D_MOCK_GCS": "1"}):
        res_dir = runner.download_checkpoints_from_gcs("gs://bucket/test", out_dir)
        assert (res_dir / "checkpoint_tpu" / "model_manifest.json").exists()

    with patch.dict("os.environ", {"T1D_MOCK_GCS": "0"}):
        # Missing google-cloud-storage library
        with patch.dict("sys.modules", {"google.cloud.storage": None}):
            with pytest.raises(
                RuntimeError,
                match="The 'google-cloud-storage' Python package is required",
            ):
                runner.download_checkpoints_from_gcs("gs://bucket/test", out_dir)

        # Native download success
        mock_gcs = MagicMock()
        mock_client = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "test/model.bin"
        mock_client.list_blobs.return_value = [mock_blob]
        mock_gcs.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google.cloud.storage": mock_gcs}):
            res = runner.download_checkpoints_from_gcs("gs://bucket/test", out_dir)
            assert res == out_dir
            mock_blob.download_to_filename.assert_called_once()

        # No checkpoint files found
        mock_empty_client = MagicMock()
        mock_empty_client.list_blobs.return_value = []
        mock_empty_gcs = MagicMock()
        mock_empty_gcs.Client.return_value = mock_empty_client
        with patch.dict("sys.modules", {"google.cloud.storage": mock_empty_gcs}):
            with pytest.raises(RuntimeError, match="No checkpoint files found at"):
                runner.download_checkpoints_from_gcs("gs://bucket/test", out_dir)

        # Download failure with retry exhaustion
        mock_err_gcs = MagicMock()
        mock_err_client = MagicMock()
        mock_err_client.list_blobs.side_effect = RuntimeError("GCS connection reset")
        mock_err_gcs.Client.return_value = mock_err_client
        with patch.dict("sys.modules", {"google.cloud.storage": mock_err_gcs}):
            with pytest.raises(
                RuntimeError,
                match="Checkpoint download failed after 3 attempts.*GCS connection reset",
            ):
                runner.download_checkpoints_from_gcs(
                    "gs://bucket/test", out_dir, max_retries=3
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
    with patch.dict(
        "os.environ",
        {"T1D_MOCK_TPU": "1", "T1D_EXECUTE_TPU_DISPATCH": "1"},
    ):
        res = runner.run_training(config)
        assert res["status"] == "SUCCESS"
        manifest_path = out / "job_manifest.json"
        assert manifest_path.exists()


def test_prepare_torch_dataset_with_tokenizer(tmp_path: Path) -> None:
    """Test prepare_torch_dataset with custom tokenizer, prompt masking, padding, and truncation."""
    ds = tmp_path / "train_tok.jsonl"
    ds.write_text('{"prompt": "question", "completion": "answer"}\n')

    class MockTokenizer:
        """Mock tokenizer for testing encode and pad_token_id."""

        pad_token_id: int = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            """Encode text into integer token ids."""
            return [ord(c) % 50 for c in text]

    mock_tok = MockTokenizer()
    # Test padding
    res_pad = prepare_torch_dataset(ds, max_seq_length=128, tokenizer=mock_tok)
    assert len(res_pad) == 1
    inps, labels = res_pad[0]
    assert inps.shape[0] == 127
    assert labels.shape[0] == 127
    # Verify prompt tokens are masked with -100
    assert (labels == -100).any()

    # Test truncation
    res_trunc = prepare_torch_dataset(ds, max_seq_length=10, tokenizer=mock_tok)
    assert len(res_trunc) == 1
    assert res_trunc[0][0].shape[0] == 9


def test_prepare_torch_dataset_tokenizer_name_and_failure(tmp_path: Path) -> None:
    """Test prepare_torch_dataset loading tokenizer_name and falling back when loading fails."""
    ds = tmp_path / "train_tok_name.jsonl"
    ds.write_text('{"prompt": "q", "completion": "a"}\n')

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.pad_token_id = 0

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        res = prepare_torch_dataset(ds, tokenizer_name="google/gemma-4-2b")
        assert len(res) == 1

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=RuntimeError("Tokenizer load error"),
    ):
        res_fallback = prepare_torch_dataset(ds, tokenizer_name="google/gemma-4-2b")
        assert len(res_fallback) == 1


def test_tiny_causal_lm_factory() -> None:
    """Test TinyCausalLMFactory model creation and device assignment."""
    factory = TinyCausalLMFactory()
    model = factory.create_model("test-model", device="cpu")
    assert isinstance(model, TinyCausalLM)

    with patch("torch.device", side_effect=RuntimeError("Invalid device")):
        model_bad_dev = factory.create_model("test-model", device="bad-device")
        assert isinstance(model_bad_dev, TinyCausalLM)


def test_huggingface_causal_lm_factory_all_branches() -> None:
    """Test HuggingFaceCausalLMFactory quantization, LoRA, error branches, and device mapping."""
    factory = HuggingFaceCausalLMFactory()

    # Missing transformers
    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(RuntimeError, match="transformers.*is required"):
            factory.create_model("gemma-2b")

    # Missing peft with use_lora
    mock_model = MagicMock()
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ):
        with patch.dict("sys.modules", {"peft": None}):
            with pytest.raises(RuntimeError, match="peft.*is required"):
                factory.create_model("gemma-2b", use_lora=True)

    # Peft error
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ):
        mock_peft = MagicMock()
        mock_peft.get_peft_model.side_effect = RuntimeError("Peft failed")
        with patch.dict("sys.modules", {"peft": mock_peft}):
            with pytest.raises(RuntimeError, match="Failed to apply LoRA"):
                factory.create_model("gemma-2b", use_lora=True)

    # Pretrained load error
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=RuntimeError("Model download failed"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load Hugging Face model"):
            factory.create_model("gemma-2b")

    # Quantization 4bit and 8bit
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ):
        with patch("transformers.BitsAndBytesConfig") as mock_bnb:
            factory.create_model("gemma-2b", quantization="4bit")
            mock_bnb.assert_called()
            factory.create_model("gemma-2b", quantization="8bit")
            mock_bnb.assert_called()

    # BitsAndBytesConfig exception fallback
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ):
        with patch(
            "transformers.BitsAndBytesConfig", side_effect=RuntimeError("BNB fail")
        ):
            res_bnb_fail = factory.create_model("gemma-2b", quantization="4bit")
            assert res_bnb_fail == mock_model

    # Successful LoRA application
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ):
        mock_peft = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft.get_peft_model.return_value = mock_peft_model
        with patch.dict("sys.modules", {"peft": mock_peft}):
            res_model = factory.create_model("gemma-2b", use_lora=True)
            assert res_model == mock_peft_model


def test_gradient_accumulation_in_cpu_and_gpu(tmp_path: Path) -> None:
    """Test gradient accumulation stepping in LocalCpuRunner and LocalGpuRunner."""
    ds = tmp_path / "data_accum.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
        '{"prompt": "q3", "completion": "SELECT 3"}\n'
        '{"prompt": "q4", "completion": "SELECT 4"}\n'
    )
    # CPU with gradient accumulation steps = 2
    cpu_runner = LocalCpuRunner()
    cfg_cpu = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "cpu_accum_out"),
        hyperparameters=TrainingHyperparameters(
            batch_size=1, num_epochs=1, gradient_accumulation_steps=2
        ),
        dry_run=False,
    )
    res_cpu = cpu_runner.run_training(cfg_cpu)
    assert res_cpu["status"] == "completed"

    # GPU with gradient accumulation steps = 2
    gpu_runner = LocalGpuRunner()
    cfg_gpu = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "gpu_accum_out"),
        hyperparameters=TrainingHyperparameters(
            batch_size=1, num_epochs=1, gradient_accumulation_steps=2
        ),
        dry_run=False,
    )
    with patch.dict(os.environ, {"T1D_MOCK_GPU": "1"}):
        with patch("torch.device", MockTorchDevice):
            res_gpu = gpu_runner.run_training(cfg_gpu)
            assert res_gpu["status"] == "completed"


def test_huggingface_causal_lm_runner_lifecycle(tmp_path: Path) -> None:
    """Test HuggingFaceCausalLMRunner validation, dry-run, evaluation, and training run."""
    import torch.nn as nn

    runner = HuggingFaceCausalLMRunner()
    assert runner.validate_environment() is True

    # Missing transformers in environment validation
    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(RuntimeError, match="HuggingFace execution requires"):
            runner.validate_environment()

    ds = tmp_path / "hf_train.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
    )
    val_ds = tmp_path / "hf_val.jsonl"
    val_ds.write_text('{"prompt": "qv", "completion": "SELECT v"}\n')
    out_dir = tmp_path / "hf_out"

    # Test dry run
    cfg_dry = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "hf_dry"),
        dry_run=True,
    )
    with patch.object(HuggingFaceCausalLMFactory, "create_model") as mock_create:
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = MagicMock()
        mock_model.return_value = mock_outputs
        mock_create.return_value = mock_model
        res_dry = runner.run_training(cfg_dry)
        assert res_dry["status"] == "dry_run_completed"
        assert res_dry["final_loss"] == 0.0

    # Test full training run with validation split
    cfg_full = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        validation_dataset_path=str(val_ds),
        output_dir=str(out_dir),
        hyperparameters=TrainingHyperparameters(
            batch_size=1, num_epochs=1, gradient_accumulation_steps=2
        ),
        dry_run=False,
    )

    class MockHFModel(nn.Module):
        """Mock HuggingFace model module."""

        def __init__(self) -> None:
            """Initialize embedding and projection."""
            super().__init__()
            self.embed = nn.Embedding(1000, 32)
            self.head = nn.Linear(32, 1000)

        def forward(self, input_ids: Any) -> Any:
            """Forward pass returning mock logits."""
            x = self.embed(input_ids)
            logits = self.head(x)
            return logits

        def save_pretrained(self, save_dir: str) -> None:
            """Mock save_pretrained."""
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / "config.json").write_text("{}")

    real_mock_model = MockHFModel()
    with patch.object(
        HuggingFaceCausalLMFactory, "create_model", return_value=real_mock_model
    ):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                res_full = runner.run_training(cfg_full)
                assert res_full["status"] == "completed"
                assert "eval_loss" in res_full["eval_metrics"]
                assert "perplexity" in res_full["eval_metrics"]
                assert (out_dir / "checkpoint-hf-final" / "config.json").exists()

    # Empty dataset error
    empty_ds = tmp_path / "hf_empty.jsonl"
    empty_ds.write_text("")
    cfg_empty = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(empty_ds),
        output_dir=str(out_dir),
    )
    with pytest.raises(InsufficientDataError):
        runner.run_training(cfg_empty)

    # Training exception error
    with patch.object(
        HuggingFaceCausalLMFactory,
        "create_model",
        side_effect=RuntimeError("Instantiation error"),
    ):
        with pytest.raises(
            TrainingExecutionError, match="Hugging Face training execution failed"
        ):
            runner.run_training(cfg_full)


def test_huggingface_causal_lm_runner_evaluate_overflow() -> None:
    """Test HuggingFaceCausalLMRunner._evaluate overflow branch when loss is high."""
    import torch

    runner = HuggingFaceCausalLMRunner()
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.train.return_value = None
    mock_logits = torch.randn(1, 10, 100)

    # Force large loss
    with patch("torch.nn.CrossEntropyLoss") as mock_loss_cls:
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value.item.return_value = 100.0
        mock_loss_cls.return_value = mock_loss_fn
        mock_model.return_value = mock_logits

        fake_eval_data = [
            (torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.long))
        ]
        metrics = runner._evaluate(
            mock_model, fake_eval_data, device=torch.device("cpu")
        )
        assert metrics["eval_loss"] == 100.0
        assert metrics["perplexity"] > 0


def test_get_training_runner_huggingface() -> None:
    """Test get_training_runner returns HuggingFaceCausalLMRunner for HUGGINGFACE backend."""
    runner = get_training_runner(TrainingBackend.HUGGINGFACE)
    assert isinstance(runner, HuggingFaceCausalLMRunner)


def test_hf_runner_and_factory_remaining_branches(tmp_path: Path) -> None:
    """Test edge cases in HuggingFace runner and factory (cuda dtype, scheduler None, overflow, non-existent val path, torch.save)."""
    import torch
    import torch.nn as nn

    factory = HuggingFaceCausalLMFactory()

    # 1. Cuda device branch for torch_dtype
    mock_model = MagicMock()
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ):
        res_cuda = factory.create_model("gemma-2b", device="cuda:0")
        assert res_cuda == mock_model

    # 2. to_fn raises exception in factory
    mock_model_bad_to = MagicMock()
    mock_model_bad_to.to.side_effect = RuntimeError("to error")
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model_bad_to,
    ):
        res_bad_to = factory.create_model("gemma-2b", device="cpu")
        assert res_bad_to == mock_model_bad_to

    # 3. TinyCausalLMFactory when torch is None
    tiny_factory = TinyCausalLMFactory()
    with patch("t1d_analytics.training_runner.torch", None):
        res_tiny_no_torch = tiny_factory.create_model("test", device="cpu")
        assert isinstance(res_tiny_no_torch, TinyCausalLM)

    # 4. OverflowError in _evaluate
    runner = HuggingFaceCausalLMRunner()
    eval_model = MagicMock()
    eval_model.eval.return_value = None
    eval_model.train.return_value = None
    eval_model.return_value = torch.randn(1, 4, 1000)
    fake_eval = [(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))]
    with patch("math.exp", side_effect=OverflowError):
        metrics = runner._evaluate(eval_model, fake_eval, device=torch.device("cpu"))
        assert metrics["perplexity"] == 999999.0

    # 5. CUDA device resolution in HuggingFaceCausalLMRunner
    ds = tmp_path / "hf_edge.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n{"prompt": "q2", "completion": "SELECT 2"}\n'
    )
    cfg_cuda = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        validation_dataset_path=str(tmp_path / "nonexistent_val.jsonl"),
        output_dir=str(tmp_path / "hf_cuda_out"),
        dry_run=True,
    )
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.device", MockTorchDevice):
            with patch.object(
                HuggingFaceCausalLMFactory, "create_model"
            ) as mock_create:
                mock_hf_m = MagicMock()
                mock_hf_m.to.side_effect = RuntimeError("to fail")
                mock_create.return_value = mock_hf_m
                res_c = runner.run_training(cfg_cuda)
                assert res_c["status"] == "dry_run_completed"

    # 5b. MPS device resolution in HuggingFaceCausalLMRunner
    cfg_mps = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        validation_dataset_path=str(tmp_path / "nonexistent_val.jsonl"),
        output_dir=str(tmp_path / "hf_mps_out"),
        dry_run=True,
    )
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.device", MockTorchDevice),
        patch.object(HuggingFaceCausalLMFactory, "create_model") as mock_create_mps,
    ):
        mock_hf_m_mps = MagicMock()
        mock_hf_m_mps.to.side_effect = RuntimeError("to fail")
        mock_create_mps.return_value = mock_hf_m_mps
        res_m = runner.run_training(cfg_mps)
        assert res_m["status"] == "dry_run_completed"

    # 6. Model without save_pretrained (falls back to torch.save) and scheduler None
    class SimpleModel(nn.Module):
        """Simple model without save_pretrained."""

        def __init__(self) -> None:
            """Initialize parameters."""
            super().__init__()
            self.emb = nn.Embedding(1000, 16)
            self.fc = nn.Linear(16, 1000)

        def forward(self, x: Any) -> Any:
            """Forward."""
            return self.fc(self.emb(x))

    simple_m = SimpleModel()
    cfg_train_simple = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "hf_simple_out"),
        hyperparameters=TrainingHyperparameters(batch_size=1, num_epochs=1),
        dry_run=False,
    )
    with patch.object(
        HuggingFaceCausalLMFactory, "create_model", return_value=simple_m
    ):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            with patch.object(torch.optim.lr_scheduler, "CosineAnnealingLR", None):
                res_train_simple = runner.run_training(cfg_train_simple)
                assert res_train_simple["status"] == "completed"
                assert (
                    tmp_path / "hf_simple_out" / "checkpoint-hf-final" / "model.pt"
                ).exists()

    # 7. LocalCpuRunner and LocalGpuRunner with scheduler None
    cpu_r = LocalCpuRunner()
    cfg_cpu = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "cpu_no_sched"),
        dry_run=False,
    )
    with patch.object(torch.optim.lr_scheduler, "CosineAnnealingLR", None):
        res_cpu_no_sched = cpu_r.run_training(cfg_cpu)
        assert res_cpu_no_sched["status"] == "completed"

    gpu_r = LocalGpuRunner()
    cfg_gpu = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "gpu_no_sched"),
        dry_run=False,
    )
    with patch.dict(os.environ, {"T1D_MOCK_GPU": "1"}):
        with patch("torch.device", MockTorchDevice):
            with patch.object(torch.optim.lr_scheduler, "CosineAnnealingLR", None):
                res_gpu_no_sched = gpu_r.run_training(cfg_gpu)
                assert res_gpu_no_sched["status"] == "completed"


def test_hf_exact_branches_coverage(tmp_path: Path) -> None:
    """Test the last specific edge branches in HuggingFace runner and factory."""
    import torch.nn as nn

    factory = HuggingFaceCausalLMFactory()
    runner = HuggingFaceCausalLMRunner()

    # 1. HuggingFaceCausalLMFactory with torch is None
    with patch("t1d_analytics.training_runner.torch", None):
        mock_m = MagicMock()
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_m
        ):
            res = factory.create_model("gemma-2b")
            assert res == mock_m

    # 2. bnb_config is not None (skips model.to)
    mock_m2 = MagicMock()
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_m2
    ):
        with patch("transformers.BitsAndBytesConfig", return_value=MagicMock()):
            res2 = factory.create_model("gemma-2b", quantization="4bit")
            assert res2 == mock_m2

    # 3. Model without callable to_fn in runner, and inner batch loop break
    ds = tmp_path / "hf_break.jsonl"
    ds.write_text(
        '{"prompt": "q1", "completion": "SELECT 1"}\n'
        '{"prompt": "q2", "completion": "SELECT 2"}\n'
        '{"prompt": "q3", "completion": "SELECT 3"}\n'
    )

    class ModelNoTo(nn.Module):
        """Model with to attribute as non-callable."""

        to = "not_callable"  # type: ignore[assignment]

        def __init__(self) -> None:
            """Init."""
            super().__init__()
            self.emb = nn.Embedding(1000, 16)
            self.fc = nn.Linear(16, 1000)

        def forward(self, x: Any) -> Any:
            """Forward."""
            return self.fc(self.emb(x))

    m_no_to = ModelNoTo()
    cfg_break = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "hf_break_out"),
        hyperparameters=TrainingHyperparameters(batch_size=2, num_epochs=1),
        dry_run=False,
    )
    with patch.object(HuggingFaceCausalLMFactory, "create_model", return_value=m_no_to):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            res_break = runner.run_training(cfg_break)
            assert res_break["status"] == "completed"
            assert res_break["total_steps"] == 1


def test_tpu_workload_status_override_and_xla_preempted() -> None:
    """Test poll_workload_status overrides, PREEMPTED status, and XLA errors."""
    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="p1")

    # PREEMPTED override
    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": "PREEMPTED"}):
        with pytest.raises(RuntimeError, match="was PREEMPTED on spot TPU capacity"):
            runner.poll_workload_status("workload-1", max_cfg)

    # XLA_ERROR override
    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": "XLA_ERROR"}):
        with pytest.raises(RuntimeError, match="fatal XLA compilation error"):
            runner.poll_workload_status("workload-1", max_cfg)

    # Custom override
    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": "PROVISIONING"}):
        assert runner.poll_workload_status("workload-1", max_cfg) == "PROVISIONING"

    # Subprocess output parsing for PREEMPTED and XLA ERROR
    with patch.dict(os.environ, {"T1D_TPU_STATUS_OVERRIDE": "", "T1D_MOCK_TPU": ""}):
        with patch("shutil.which", return_value="/usr/bin/xpk"):
            mock_proc_preempt = MagicMock()
            mock_proc_preempt.returncode = 0
            mock_proc_preempt.stdout = "STATUS: PREEMPTED by node eviction"
            with patch("subprocess.run", return_value=mock_proc_preempt):
                with pytest.raises(RuntimeError, match="was PREEMPTED"):
                    runner.poll_workload_status(
                        "workload-1", max_cfg, timeout_seconds=1
                    )

            mock_proc_xla = MagicMock()
            mock_proc_xla.returncode = 0
            mock_proc_xla.stdout = "FATAL: XLA COMPILATION ERROR"
            with patch("subprocess.run", return_value=mock_proc_xla):
                with pytest.raises(RuntimeError, match="fatal XLA compilation error"):
                    runner.poll_workload_status(
                        "workload-1", max_cfg, timeout_seconds=1
                    )


def test_tpu_workload_reclamation() -> None:
    """Test reclaim_tpu_workload in mock and subprocess modes."""
    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="p1")

    # Mock mode
    with patch.dict(os.environ, {"T1D_MOCK_TPU": "1"}):
        assert runner.reclaim_tpu_workload("workload-1", max_cfg) is True

    # Subprocess without xpk or gcloud -> raises
    with patch.dict(os.environ, {"T1D_MOCK_TPU": ""}):
        with patch("shutil.which", return_value=None):
            with pytest.raises(
                RuntimeError, match="Google Cloud CLI .* is not installed"
            ):
                runner.reclaim_tpu_workload("workload-1", max_cfg)

    # Subprocess without xpk, with gcloud -> delegates to _reclaim_gcloud_tpu_workload
    def mock_gcloud_reclaim(cmd: str) -> Optional[str]:
        return "/usr/bin/gcloud" if cmd == "gcloud" else None

    with patch.dict(os.environ, {"T1D_MOCK_TPU": ""}):
        with patch("shutil.which", side_effect=mock_gcloud_reclaim):
            with patch.object(
                runner, "_reclaim_gcloud_tpu_workload", return_value=True
            ) as mock_rg:
                assert runner.reclaim_tpu_workload("workload-1", max_cfg) is True
                mock_rg.assert_called_once()

    # Subprocess with xpk success
    mock_succ = MagicMock()
    mock_succ.returncode = 0
    with patch.dict(os.environ, {"T1D_MOCK_TPU": ""}):
        with patch("shutil.which", return_value="/usr/bin/xpk"):
            with patch("subprocess.run", return_value=mock_succ):
                assert runner.reclaim_tpu_workload("workload-1", max_cfg) is True

    # Subprocess with xpk failure
    mock_fail = MagicMock()
    mock_fail.returncode = 1
    mock_fail.stderr = "Cluster not reachable"
    with patch.dict(os.environ, {"T1D_MOCK_TPU": ""}):
        with patch("shutil.which", return_value="/usr/bin/xpk"):
            with patch("subprocess.run", return_value=mock_fail):
                with pytest.raises(
                    RuntimeError, match="Failed to reclaim TPU workload"
                ):
                    runner.reclaim_tpu_workload("workload-1", max_cfg)


def test_gcs_mock_failure_and_corruption(tmp_path: Path) -> None:
    """Test GCS mock upload and download failure and checksum corruption branches."""
    local_file = tmp_path / "data.parquet"
    local_file.write_text("test")
    out_dir = tmp_path / "gcs_dl_out"

    runner = RemoteTpuMaxTextRunner()

    # Upload mock failure
    with patch.dict(os.environ, {"T1D_MOCK_GCS": "1", "T1D_MOCK_GCS_FAIL": "1"}):
        with pytest.raises(RuntimeError, match="Mock upload failure"):
            sync_dataset_to_gcs(local_file, "gs://my-bucket")

    # Upload mock corruption
    with patch.dict(os.environ, {"T1D_MOCK_GCS": "1", "T1D_MOCK_GCS_CORRUPT": "1"}):
        with pytest.raises(
            ValueError, match="Checksum verification failed for uploaded dataset"
        ):
            sync_dataset_to_gcs(local_file, "gs://my-bucket")

    # Download mock failure
    with patch.dict(os.environ, {"T1D_MOCK_GCS": "1", "T1D_MOCK_GCS_FAIL": "1"}):
        with pytest.raises(RuntimeError, match="Mock error"):
            runner.download_checkpoints_from_gcs("gs://my-bucket/ckpt", out_dir)

    # Download mock corruption
    with patch.dict(os.environ, {"T1D_MOCK_GCS": "1", "T1D_MOCK_GCS_CORRUPT": "1"}):
        with pytest.raises(
            ValueError, match="Checksum verification failed for downloaded checkpoint"
        ):
            runner.download_checkpoints_from_gcs("gs://my-bucket/ckpt", out_dir)


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

    # Preemption retry exhaustion
    with patch.dict(
        os.environ,
        {
            "T1D_EXECUTE_TPU_DISPATCH": "1",
            "T1D_MOCK_TPU": "1",
        },
    ):
        with patch.object(
            runner,
            "poll_workload_status",
            side_effect=RuntimeError("was PREEMPTED"),
        ):
            with patch.object(runner, "reclaim_tpu_workload") as mock_reclaim:
                with pytest.raises(RuntimeError, match="was PREEMPTED"):
                    runner.run_training(cfg)
                mock_reclaim.assert_called_once()

        # Non-preemption failure
        with patch.object(
            runner,
            "poll_workload_status",
            side_effect=RuntimeError("Generic cluster error"),
        ):
            with patch.object(runner, "reclaim_tpu_workload") as mock_reclaim2:
                with pytest.raises(RuntimeError, match="Generic cluster error"):
                    runner.run_training(cfg)
                mock_reclaim2.assert_called_once()

    # Preemption retry success on second attempt
    with patch.dict(
        os.environ,
        {
            "T1D_EXECUTE_TPU_DISPATCH": "1",
            "T1D_MOCK_TPU": "1",
        },
    ):
        with patch.object(
            runner,
            "poll_workload_status",
            side_effect=[RuntimeError("was PREEMPTED"), "SUCCESS"],
        ):
            res_retry_succ = runner.run_training(cfg)
            assert res_retry_succ["status"] == "SUCCESS"


def test_poll_gcloud_tpu_status_missing_binary() -> None:
    """Test _poll_gcloud_tpu_status raises when gcloud is missing."""
    runner = RemoteTpuMaxTextRunner()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Google Cloud CLI .* is not installed"):
            runner._poll_gcloud_tpu_status("node-1", "us-central2-b")


def test_poll_gcloud_tpu_status_states_and_errors() -> None:
    """Test _poll_gcloud_tpu_status for normal states, preemption, failures, non-json, not found, and timeout."""
    runner = RemoteTpuMaxTextRunner()
    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        # 1. State READY
        mock_proc_ready = MagicMock(returncode=0, stdout='{"state": "READY"}')
        with patch("subprocess.run", return_value=mock_proc_ready):
            assert (
                runner._poll_gcloud_tpu_status("node-1", "us-central2-b") == "SUCCESS"
            )

        # 2. State STOPPED
        mock_proc_stopped = MagicMock(returncode=0, stdout='{"state": "STOPPED"}')
        with patch("subprocess.run", return_value=mock_proc_stopped):
            assert (
                runner._poll_gcloud_tpu_status("node-1", "us-central2-b") == "SUCCESS"
            )

        # 3. State PREEMPTED
        mock_proc_preempt = MagicMock(returncode=0, stdout='{"state": "PREEMPTED"}')
        with patch("subprocess.run", return_value=mock_proc_preempt):
            with pytest.raises(RuntimeError, match="was PREEMPTED"):
                runner._poll_gcloud_tpu_status("node-1", "us-central2-b")

        # 4. healthDescription PREEMPTED
        mock_proc_health_preempt = MagicMock(
            returncode=0,
            stdout='{"state": "STOPPING", "healthDescription": "Node was preempted"}',
        )
        with patch("subprocess.run", return_value=mock_proc_health_preempt):
            with pytest.raises(RuntimeError, match="was PREEMPTED"):
                runner._poll_gcloud_tpu_status("node-1", "us-central2-b")

        # 5. healthDescription ERROR
        mock_proc_err = MagicMock(
            returncode=0,
            stdout='{"state": "RUNNING", "healthDescription": "Fatal hardware ERROR"}',
        )
        with patch("subprocess.run", return_value=mock_proc_err):
            with pytest.raises(RuntimeError, match="encountered fatal error"):
                runner._poll_gcloud_tpu_status("node-1", "us-central2-b")

        # 6. Non-JSON but output contains READY
        mock_proc_text_ready = MagicMock(
            returncode=0, stdout="State: READY (non-json output)"
        )
        with patch("subprocess.run", return_value=mock_proc_text_ready):
            assert (
                runner._poll_gcloud_tpu_status("node-1", "us-central2-b") == "SUCCESS"
            )

        # 7. Non-zero returncode with NOT_FOUND in stderr
        mock_proc_nf = MagicMock(
            returncode=1, stderr="ERROR: (gcloud) NOT_FOUND: Resource not found"
        )
        with patch("subprocess.run", return_value=mock_proc_nf):
            with pytest.raises(RuntimeError, match="not found in zone"):
                runner._poll_gcloud_tpu_status("node-1", "us-central2-b")

        # 8. Timeout
        mock_proc_pending = MagicMock(returncode=0, stdout='{"state": "CREATING"}')
        with patch("subprocess.run", return_value=mock_proc_pending):
            with pytest.raises(TimeoutError, match="polling timed out"):
                runner._poll_gcloud_tpu_status(
                    "node-1", "us-central2-b", timeout_seconds=0.08
                )

        # 9. JSONDecodeError without READY (times out)
        mock_proc_bad_json = MagicMock(
            returncode=0, stdout="Some garbled text without status"
        )
        with patch("subprocess.run", return_value=mock_proc_bad_json):
            with pytest.raises(TimeoutError, match="polling timed out"):
                runner._poll_gcloud_tpu_status(
                    "node-1", "us-central2-b", timeout_seconds=0.08
                )

        # 10. Non-zero returncode without NOT_FOUND (transient error, times out)
        mock_proc_transient = MagicMock(returncode=1, stderr="Connection reset by peer")
        with patch("subprocess.run", return_value=mock_proc_transient):
            with pytest.raises(TimeoutError, match="polling timed out"):
                runner._poll_gcloud_tpu_status(
                    "node-1", "us-central2-b", timeout_seconds=0.08
                )


def test_reclaim_gcloud_tpu_workload_lifecycle() -> None:
    """Test _reclaim_gcloud_tpu_workload missing binary, success, and retry failure."""
    runner = RemoteTpuMaxTextRunner()
    # Missing binary
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Google Cloud CLI .* is not installed"):
            runner._reclaim_gcloud_tpu_workload("node-1", "us-central2-b")

    with patch("shutil.which", return_value="/usr/bin/gcloud"):
        # Success
        mock_ok = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", return_value=mock_ok):
            assert (
                runner._reclaim_gcloud_tpu_workload("node-1", "us-central2-b") is True
            )

        # Retry failure
        mock_fail = MagicMock(returncode=1, stderr="PermissionDenied")
        with patch("subprocess.run", return_value=mock_fail):
            with pytest.raises(RuntimeError, match="Failed to reclaim TPU VM workload"):
                runner._reclaim_gcloud_tpu_workload(
                    "node-1", "us-central2-b", max_retries=2
                )


def test_poll_and_reclaim_workload_delegation_without_xpk() -> None:
    """Test poll_workload_status and reclaim_tpu_workload delegate to gcloud methods when xpk is absent."""
    runner = RemoteTpuMaxTextRunner()
    max_cfg = MaxTextConfig(project_id="p1", zone="us-central2-b")

    def mock_which(cmd: str) -> Optional[str]:
        if cmd == "xpk":
            return None
        return "/usr/bin/gcloud"

    with patch("shutil.which", side_effect=mock_which):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("T1D_MOCK_TPU", None)
            os.environ.pop("T1D_TPU_STATUS_OVERRIDE", None)
            with patch.object(
                runner, "_poll_gcloud_tpu_status", return_value="SUCCESS"
            ) as mock_p:
                res = runner.poll_workload_status("wl-1", max_cfg)
                assert res == "SUCCESS"
                mock_p.assert_called_once_with(
                    workload_name="wl-1", zone="us-central2-b", timeout_seconds=30.0
                )

            with patch.object(
                runner, "_reclaim_gcloud_tpu_workload", return_value=True
            ) as mock_r:
                res2 = runner.reclaim_tpu_workload("wl-1", max_cfg)
                assert res2 is True
                mock_r.assert_called_once_with(
                    workload_name="wl-1", zone="us-central2-b"
                )


def test_local_cpu_runner_architecture_selection(tmp_path: Path) -> None:
    """Test LocalCpuRunner model factory selection for tiny, transformer, auto, and HF execution."""
    import torch

    runner = LocalCpuRunner()
    ds = tmp_path / "dataset.jsonl"
    ds.write_text('{"prompt": "question", "completion": "answer"}\n')

    # 1. Explicit tiny model
    cfg_tiny = TrainingJobConfig(
        model_name="test-tiny",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_tiny"),
        use_tiny_model=True,
        dry_run=True,
    )
    with patch.object(
        TinyCausalLMFactory, "create_model", wraps=TinyCausalLMFactory().create_model
    ) as mock_tiny:
        res = runner.run_training(cfg_tiny)
        assert res["status"] == "dry_run_completed"
        mock_tiny.assert_called_once()

    # 2. Explicit transformer model architecture (HuggingFace)
    cfg_hf = TrainingJobConfig(
        model_name="google/gemma-2b",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_hf"),
        model_architecture="transformer",
        dry_run=True,
    )
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(logits=torch.zeros((1, 5, 100)))
    with patch.object(
        HuggingFaceCausalLMFactory, "create_model", return_value=mock_model
    ) as mock_hf:
        res_hf = runner.run_training(cfg_hf)
        assert res_hf["status"] == "dry_run_completed"
        mock_hf.assert_called_once_with(
            model_name="google/gemma-2b",
            device="cpu",
            quantization=None,
            use_lora=False,
        )

    # 3. Full training run with HuggingFace model and save_pretrained
    cfg_hf_train = TrainingJobConfig(
        model_name="google/gemma-2b",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_hf_train"),
        model_architecture="transformer",
        dry_run=False,
        hyperparameters=TrainingHyperparameters(num_epochs=1, batch_size=1),
    )
    mock_model_train = MagicMock()
    mock_model_train.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
    grad_logits = torch.randn(1, 127, 1000, requires_grad=True)
    mock_model_train.return_value = MagicMock(logits=grad_logits)
    mock_model_train.save_pretrained = MagicMock()

    with patch.object(
        HuggingFaceCausalLMFactory, "create_model", return_value=mock_model_train
    ):
        res_train = runner.run_training(cfg_hf_train)
        assert res_train["status"] == "completed"
        mock_model_train.save_pretrained.assert_called_once()


def test_local_gpu_runner_architecture_selection(tmp_path: Path) -> None:
    """Test LocalGpuRunner model factory selection with quantization, LoRA, and HF model execution."""
    import torch

    runner = LocalGpuRunner()
    ds = tmp_path / "dataset_gpu.jsonl"
    ds.write_text('{"prompt": "question", "completion": "answer"}\n')

    # 1. Explicit tiny model on GPU
    cfg_tiny_gpu = TrainingJobConfig(
        model_name="custom-model",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_gpu_tiny"),
        use_tiny_model=True,
        dry_run=True,
    )
    with patch.dict(os.environ, {"T1D_MOCK_GPU": "1"}):
        with patch.object(
            TinyCausalLMFactory,
            "create_model",
            wraps=TinyCausalLMFactory().create_model,
        ) as mock_tiny:
            res = runner.run_training(cfg_tiny_gpu)
            assert res["status"] == "dry_run_completed"
            mock_tiny.assert_called_once()

    # 2. HF with LoRA and 4-bit quantization on GPU
    cfg_hf_gpu = TrainingJobConfig(
        model_name="meta-llama/Llama-2-7b",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_gpu_hf"),
        model_architecture="transformer",
        use_lora=True,
        quantization="4bit",
        dry_run=True,
    )
    mock_gpu_model = MagicMock()
    mock_gpu_model.return_value = MagicMock(logits=torch.zeros((1, 5, 100)))
    with patch.dict(os.environ, {"T1D_MOCK_GPU": "1"}):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                with patch.object(
                    HuggingFaceCausalLMFactory,
                    "create_model",
                    return_value=mock_gpu_model,
                ) as mock_hf:
                    res_gpu = runner.run_training(cfg_hf_gpu)
                    assert res_gpu["status"] == "dry_run_completed"
                    mock_hf.assert_called_once_with(
                        model_name="meta-llama/Llama-2-7b",
                        device="cpu",
                        quantization="4bit",
                        use_lora=True,
                    )

    # 3. Full training loop on GPU with HF model and save_pretrained
    cfg_hf_gpu_train = TrainingJobConfig(
        model_name="meta-llama/Llama-2-7b",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_gpu_hf_train"),
        model_architecture="transformer",
        use_lora=True,
        dry_run=False,
        hyperparameters=TrainingHyperparameters(num_epochs=1, batch_size=1),
    )
    mock_train_model = MagicMock()
    mock_train_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
    grad_gpu_logits = torch.randn(1, 127, 1000, requires_grad=True)
    mock_train_model.return_value = MagicMock(logits=grad_gpu_logits)
    mock_train_model.save_pretrained = MagicMock()

    with patch.dict(os.environ, {"T1D_MOCK_GPU": "1"}):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                with patch.object(
                    HuggingFaceCausalLMFactory,
                    "create_model",
                    return_value=mock_train_model,
                ):
                    res_train = runner.run_training(cfg_hf_gpu_train)
                    assert res_train["status"] == "completed"
                    mock_train_model.save_pretrained.assert_called_once()


def test_deterministic_subword_tokenizer_operations() -> None:
    """Test DeterministicSubwordTokenizer encoding, decoding, special tokens, and unknown fallbacks."""
    tok = DeterministicSubwordTokenizer(vocab_size=500)
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3

    # Empty string
    assert tok.encode("") == []

    # Text encoding without and with special tokens
    query = "SELECT AVG(glucose) FROM cgm WHERE patient = 42"
    ids = tok.encode(query, add_special_tokens=False)
    assert len(ids) > 0
    assert 2 not in ids
    assert 3 not in ids

    ids_special = tok.encode(query, add_special_tokens=True)
    assert ids_special[0] == 2
    assert ids_special[-1] == 3

    # Reconstructed decode
    decoded = tok.decode(ids)
    assert decoded == query

    # Decode with special tokens and out-of-vocab IDs (should be filtered out)
    filtered_decode = tok.decode([tok.pad_token_id, tok.unk_token_id, 99999, ids[0]])
    assert filtered_decode == "SELECT"

    # Unknown character fallback
    unknown_text = "SELECT 🌟"
    ids_unk = tok.encode(unknown_text)
    assert tok.unk_token_id in ids_unk


def test_prepare_torch_dataset_deterministic_fallback(tmp_path: Path) -> None:
    """Test prepare_torch_dataset using DeterministicSubwordTokenizer with padding, truncation, and prompt masking."""
    ds = tmp_path / "dataset_subword.jsonl"
    ds.write_text(
        '{"prompt": "Calculate average glucose", "completion": "SELECT AVG(glucose) FROM cgm;"}\n'
    )

    # Sequence length 32 (requires padding)
    pairs = prepare_torch_dataset(ds, max_seq_length=32)
    assert len(pairs) == 1
    inp_ids, labels = pairs[0]
    assert inp_ids.shape[0] == 31
    assert labels.shape[0] == 31

    # Labels must contain -100 for prompt tokens and pad tokens
    assert -100 in labels.tolist()

    # Short sequence length 8 (forces truncation)
    pairs_trunc = prepare_torch_dataset(ds, max_seq_length=8)
    inp_ids_t, labels_t = pairs_trunc[0]
    assert inp_ids_t.shape[0] == 7
    assert labels_t.shape[0] == 7


def test_resolve_device_telemetry_branches(tmp_path: Path) -> None:
    """Test resolve_device_telemetry for CUDA, MPS, CPU, and mock accelerator environments."""
    import subprocess

    # 1. CUDA success with explicit index
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-80GB"):
            with patch(
                "torch.cuda.mem_get_info",
                return_value=(40 * 1024 * 1024 * 1024, 80 * 1024 * 1024 * 1024),
            ):
                name, mem = resolve_device_telemetry("cuda:1")
                assert name == "NVIDIA A100-SXM4-80GB"
                assert mem == 81920.0

    # 2. CUDA success without colon index
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
            with patch(
                "torch.cuda.mem_get_info",
                return_value=(10 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024),
            ):
                name_def, mem_def = resolve_device_telemetry("cuda")
                assert name_def == "NVIDIA RTX 4090"
                assert mem_def == 24576.0

    # 3. CUDA unavailable or exception during query
    with patch("torch.cuda.is_available", return_value=False):
        name_no_cuda, _ = resolve_device_telemetry("cuda:0")
        assert name_no_cuda == "NVIDIA CUDA Device"

    with patch("torch.cuda.is_available", return_value=True):
        with patch(
            "torch.cuda.get_device_name", side_effect=RuntimeError("CUDA query error")
        ):
            name_err, _ = resolve_device_telemetry("cuda:invalid")
            assert name_err == "NVIDIA CUDA Device"

    # 4. MPS with sysctl success
    mock_sysctl = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Apple M2 Max\n", stderr=""
    )
    with patch("subprocess.run", return_value=mock_sysctl):
        name_mps, mem_mps = resolve_device_telemetry("mps")
        assert name_mps == "Apple Silicon (Apple M2 Max)"
        assert mem_mps == 0.0

    # 5. MPS with sysctl failure
    mock_sysctl_fail = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr=""
    )
    with patch("subprocess.run", return_value=mock_sysctl_fail):
        name_mps_fail, _ = resolve_device_telemetry("mps")
        assert name_mps_fail == "Apple Silicon GPU (MPS)"

    # 6. MPS with subprocess exception
    with patch("subprocess.run", side_effect=RuntimeError("sysctl missing")):
        name_mps_exc, _ = resolve_device_telemetry("mps")
        assert name_mps_exc == "Apple Silicon GPU (MPS)"

    # 7. CPU with T1D_MOCK_GPU=1
    with patch.dict(os.environ, {"T1D_MOCK_GPU": "1"}):
        name_mock, mem_mock = resolve_device_telemetry("cpu")
        assert name_mock == "Mock Accelerator"
        assert mem_mock == 0.0

    # 8. CPU with platform.processor()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("T1D_MOCK_GPU", None)
        with patch("platform.processor", return_value="x86_64"):
            name_proc, _ = resolve_device_telemetry("cpu")
            assert name_proc == "Host CPU (x86_64)"

    # 9. CPU fallback reading /proc/cpuinfo with matching line
    cpuinfo_match = (
        "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 7B12\n"
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("T1D_MOCK_GPU", None)
        with patch("platform.processor", return_value=""):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value=cpuinfo_match):
                    name_cpuinfo, _ = resolve_device_telemetry("cpu")
                    assert name_cpuinfo == "AMD EPYC 7B12"

    # 10. CPU fallback reading /proc/cpuinfo without matching line
    cpuinfo_nomatch = "processor\t: 0\nvendor_id\t: Unknown\n"
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("T1D_MOCK_GPU", None)
        with patch("platform.processor", return_value=""):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value=cpuinfo_nomatch):
                    name_nomatch, _ = resolve_device_telemetry("cpu")
                    assert name_nomatch == "Host CPU"

    # 11. CPU fallback where platform.processor() is empty and /proc/cpuinfo does not exist
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("T1D_MOCK_GPU", None)
        with patch("platform.processor", return_value=""):
            with patch("pathlib.Path.exists", return_value=False):
                name_no_proc_file, _ = resolve_device_telemetry("cpu")
                assert name_no_proc_file == "Host CPU"

    # 12. CPU fallback generic exception
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("T1D_MOCK_GPU", None)
        with patch("platform.processor", side_effect=RuntimeError("platform error")):
            name_fallback, _ = resolve_device_telemetry("cpu")
            assert name_fallback == "Host CPU"


def test_training_job_manager_lifecycle(tmp_path: Path, mocker: MagicMock) -> None:
    """Test TrainingJobManager background thread execution, status tracking, cancellation, and error handling."""
    import threading
    import time

    from t1d_analytics.models import TrainingBackend, TrainingJobConfig
    from t1d_analytics.training_runner import TrainingJobManager

    mgr = TrainingJobManager()

    # 1. Non-existent job queries
    assert mgr.get_job("non-existent") is None
    assert mgr.cancel_job("non-existent") is False
    assert mgr.list_jobs() == []

    # 2. Successful job execution
    mocker.patch(
        "t1d_analytics.training_runner.LocalCpuRunner.run_training",
        return_value={"status": "completed", "loss": 0.05},
    )
    config = TrainingJobConfig(
        dataset_path=str(tmp_path / "ds.jsonl"),
        output_dir=str(tmp_path / "out"),
        backend=TrainingBackend.LOCAL_CPU,
        model_name="gemma4-sql",
        dry_run=True,
    )
    job_id = mgr.submit_job(config)
    assert job_id.startswith("job-")
    time.sleep(0.15)  # wait for thread to finish
    job = mgr.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert len(mgr.list_jobs()) == 1

    # Cancel on completed job returns True but status stays completed
    assert mgr.cancel_job(job_id) is True
    completed_job = mgr.get_job(job_id)
    assert completed_job is not None
    assert completed_job["status"] == "completed"

    # 3. Failed job execution
    mocker.patch(
        "t1d_analytics.training_runner.LocalCpuRunner.run_training",
        side_effect=RuntimeError("Training failed fatally"),
    )
    job_id_fail = mgr.submit_job(config)
    time.sleep(0.15)
    job_fail = mgr.get_job(job_id_fail)
    assert job_fail is not None
    assert job_fail["status"] == "failed"
    assert "Training failed fatally" in str(job_fail["error"])

    # 4. Cancelled job execution while running
    mgr2 = TrainingJobManager()
    event_start = threading.Event()
    event_release = threading.Event()

    def slow_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        event_start.set()
        event_release.wait(timeout=2.0)
        return {"status": "done"}

    mocker.patch(
        "t1d_analytics.training_runner.LocalCpuRunner.run_training",
        side_effect=slow_run,
    )
    job_id_cancel = mgr2.submit_job(config)
    event_start.wait(timeout=1.0)
    assert mgr2.cancel_job(job_id_cancel) is True
    event_release.set()
    time.sleep(0.15)
    job_cancel = mgr2.get_job(job_id_cancel)
    assert job_cancel is not None
    assert job_cancel["status"] == "cancelled"

    # 5. Pre-run cancellation (cancel before runner.run_training is called)
    from t1d_analytics import training_runner

    mgr3 = TrainingJobManager()
    orig_get_runner = training_runner.get_training_runner

    def cancel_on_get_runner(backend: Any) -> Any:
        for f in mgr3._cancel_flags.values():
            f.set()
        return orig_get_runner(backend)

    mocker.patch(
        "t1d_analytics.training_runner.get_training_runner",
        side_effect=cancel_on_get_runner,
    )
    job_id_pre = mgr3.submit_job(config)
    time.sleep(0.15)
    job_pre = mgr3.get_job(job_id_pre)
    assert job_pre is not None
    assert job_pre["status"] == "cancelled"
