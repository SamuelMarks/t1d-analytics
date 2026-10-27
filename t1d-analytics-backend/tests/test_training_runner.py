"""Unit tests for LocalCpuRunner, LocalGpuRunner, HuggingFaceCausalLMRunner, Gemma4SqlRunner, and TrainingJobManager."""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union
from unittest.mock import MagicMock, patch

import duckdb
import pytest
import torch

from t1d_analytics.job_manager import JobMetricValue
from t1d_analytics.models import (
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.sql_utils import SqlValue
from t1d_analytics.training_runner import (
    AcceleratorDetectorProtocol,
    BaseTrainingRunner,
    DeviceLike,
    Gemma4SqlRunner,
    HuggingFaceCausalLMFactory,
    HuggingFaceCausalLMRunner,
    InsufficientDataError,
    LocalCpuRunner,
    LocalGpuRunner,
    RemoteTpuMaxTextRunner,
    TinyCausalLMFactory,
    TrainingExecutionError,
    TrainingJobManager,
    get_training_runner,
)


@pytest.fixture(autouse=True)
def mock_huggingface_tokenizer() -> Generator[None, None, None]:
    """Provide a deterministic lightweight tokenizer for runner tests."""

    class MockTokenizer:
        """Lightweight mock tokenizer implementing TokenizerProtocol."""

        def __init__(self, *args: SqlValue, **kwargs: SqlValue) -> None:
            """Initialize pad_token_id."""
            self.pad_token_id = 0

        def encode(
            self, text: str, add_special_tokens: bool = False, **kwargs: SqlValue
        ) -> List[int]:
            """Encode text characters into integer IDs."""
            _ = add_special_tokens
            return [ord(c) % 50 for c in text] if text else []

        @classmethod
        def from_pretrained(
            cls, *args: SqlValue, **kwargs: SqlValue
        ) -> "MockTokenizer":
            """Return a MockTokenizer instance."""
            return cls()

    with patch(
        "transformers.AutoTokenizer.from_pretrained", MockTokenizer.from_pretrained
    ):
        yield


@pytest.fixture(autouse=True)
def mock_huggingface_factory() -> Generator[None, None, None]:
    """Provide a real tiny PyTorch nn.Module for HuggingFaceCausalLMFactory in unit tests."""

    class MockOutput:
        """Forward output container exposing real PyTorch logits tensor."""

        def __init__(self, logits: torch.Tensor) -> None:
            """Store logits tensor."""
            self.logits = logits

    class RealTinyModule(torch.nn.Module):
        """Real tiny PyTorch module used for fast CPU/GPU runner verification."""

        def __init__(self) -> None:
            """Initialize embedding and projection layer."""
            super().__init__()
            self.embed = torch.nn.Embedding(1000, 16)
            self.head = torch.nn.Linear(16, 1000)

        def forward(self, input_ids: torch.Tensor, **kwargs: SqlValue) -> MockOutput:
            """Compute logits on real PyTorch tensors matching input device."""
            _ = kwargs
            if next(self.parameters()).device != input_ids.device:
                self.to(input_ids.device)
            clamped = input_ids.clamp(0, 999)
            return MockOutput(self.head(self.embed(clamped)))

        def save_pretrained(self, path: Union[str, Path]) -> None:
            """Persist state_dict to path."""
            out_dir = Path(path)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), out_dir / "model.pt")

    def mock_create_model(
        self: HuggingFaceCausalLMFactory,
        model_name: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
        use_lora: bool = False,
    ) -> RealTinyModule:
        """Return a RealTinyModule instance."""
        _ = (self, model_name, device, quantization, use_lora)
        return RealTinyModule()

    with patch(
        "t1d_analytics.hf_models.HuggingFaceCausalLMFactory.create_model",
        mock_create_model,
    ):
        yield


_real_torch_device = torch.device


def MockTorchDevice(*args: SqlValue, **kwargs: SqlValue) -> torch.device:
    """Return a real CPU torch device for deterministic testing."""
    _ = (args, kwargs)
    return _real_torch_device("cpu")


class FakeAcceleratorDetector:
    """Contract-verified fake implementing AcceleratorDetectorProtocol."""

    def __init__(self, cuda: bool = False, mps: bool = False) -> None:
        """Initialize detector flags."""
        self._cuda = cuda
        self._mps = mps

    def is_cuda_available(self) -> bool:
        """Return simulated CUDA availability."""
        return self._cuda

    def is_mps_available(self) -> bool:
        """Return simulated MPS availability."""
        return self._mps


def test_local_cpu_runner_execution_and_dry_run(tmp_path: Path) -> None:
    """Test LocalCpuRunner execution paths for normal, dry-run, and empty dataset modes."""
    ds = tmp_path / "data.jsonl"
    ds.write_text(
        """{"prompt": "q1", "completion": "SELECT 1"}
{"prompt": "q2", "completion": "SELECT 2"}
"""
    )
    out = tmp_path / "cpu_out"

    runner = LocalCpuRunner()
    assert runner.validate_environment() is True

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
    assert isinstance(res["final_loss"], (int, float))
    assert res["final_loss"] > 0.0
    assert (out / "checkpoint-final" / "training_args.json").exists()

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
    """Test LocalGpuRunner environment validation with AcceleratorDetectorProtocol and training runs."""
    runner = LocalGpuRunner()

    cuda_detector: AcceleratorDetectorProtocol = FakeAcceleratorDetector(cuda=True)
    mps_detector: AcceleratorDetectorProtocol = FakeAcceleratorDetector(mps=True)
    none_detector: AcceleratorDetectorProtocol = FakeAcceleratorDetector(
        cuda=False, mps=False
    )
    assert isinstance(cuda_detector, AcceleratorDetectorProtocol)
    assert runner.validate_environment(detector=cuda_detector) is True
    assert runner.validate_environment(detector=mps_detector) is True
    with pytest.raises(RuntimeError, match="No compatible GPU device"):
        runner.validate_environment(detector=none_detector)

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
        assert runner.validate_environment() is True

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(RuntimeError, match="No compatible GPU device"):
                runner.validate_environment()

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        with patch("torch.cuda.is_available", return_value=True):
            assert runner.validate_environment() is True

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                assert runner.validate_environment() is True

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                with pytest.raises(RuntimeError, match="No compatible GPU device"):
                    runner.validate_environment()

    ds = tmp_path / "data_gpu.jsonl"
    ds.write_text(
        """{"prompt": "q1", "completion": "SELECT 1"}
{"prompt": "q2", "completion": "SELECT 2"}
"""
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


def test_get_training_runner() -> None:
    """Test get_training_runner factory returns correct instances."""
    assert isinstance(get_training_runner(TrainingBackend.LOCAL_CPU), LocalCpuRunner)
    assert isinstance(get_training_runner(TrainingBackend.GEMMA_4_SQL), Gemma4SqlRunner)
    assert isinstance(get_training_runner(TrainingBackend.LOCAL_GPU), LocalGpuRunner)
    assert isinstance(
        get_training_runner(TrainingBackend.REMOTE_TPU_MAXTEXT), RemoteTpuMaxTextRunner
    )
    assert isinstance(
        get_training_runner(TrainingBackend.HUGGINGFACE), HuggingFaceCausalLMRunner
    )
    assert isinstance(get_training_runner(TrainingBackend("local-cpu")), LocalCpuRunner)


def test_local_cpu_runner_with_real_datasets(tmp_path: Path) -> None:
    """Test LocalCpuRunner dataset inspection, step progression, and metrics saving."""
    runner = LocalCpuRunner()

    jsonl_ds = tmp_path / "train.jsonl"
    jsonl_ds.write_text(
        """{"prompt": "q1", "completion": "SELECT 1"}
{"prompt": "q2", "completion": "SELECT 2"}
{"prompt": "q3", "completion": "SELECT 3"}
{"prompt": "q4", "completion": "SELECT 4"}
"""
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
    assert res_jsonl["total_steps"] == 4
    assert isinstance(res_jsonl["loss_history"], list)
    assert len(res_jsonl["loss_history"]) == 4
    metrics_file = out_jsonl / "checkpoint-final" / "training_metrics.json"
    assert metrics_file.exists()

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
    assert res_pq["total_steps"] == 2

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
    """Test LocalGpuRunner with real datasets, CUDA telemetry, MPS device detection, and fallback."""
    runner = LocalGpuRunner()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    jsonl_ds = tmp_path / "gpu_train.jsonl"
    jsonl_ds.write_text(
        """{"prompt": "q1", "completion": "SELECT 1"}
{"prompt": "q2", "completion": "SELECT 2"}
{"prompt": "q3", "completion": "SELECT 3"}
{"prompt": "q4", "completion": "SELECT 4"}
"""
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
                        patch(
                            "torch.optim.Optimizer._accelerator_graph_capture_health_check",
                            lambda self: None,
                            create=True,
                        ),
                        patch(
                            "torch.optim.Optimizer._cuda_graph_capture_health_check",
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

    pq_ds = tmp_path / "gpu_train.parquet"
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE sft AS SELECT range AS id, 'SELECT ' || range AS sql FROM range(4)"
    )
    conn.execute(f"COPY sft TO '{pq_ds}' (FORMAT PARQUET)")
    conn.close()

    out_mps = tmp_path / "gpu_mps_out"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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


def test_local_gpu_runner_torch_import_and_cpu_fallback(tmp_path: Path) -> None:
    """Test LocalGpuRunner when torch is not importable or falls back to CPU."""
    runner = LocalGpuRunner()
    ds = tmp_path / "data.jsonl"
    ds.write_text("""{"prompt": "q1", "completion": "SELECT 1"}
""")
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

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                res = runner.run_training(config)
                assert res["device"] == "cpu"


def test_gemma_4_sql_runner_lifecycle(tmp_path: Path) -> None:
    """Test Gemma4SqlRunner environment validation, dry-run stages, and execution."""
    runner = Gemma4SqlRunner()
    with patch(
        "t1d_analytics.gemma_bridge.check_gemma_sql_installed", return_value=True
    ):
        assert runner.validate_environment() is True

    with patch(
        "t1d_analytics.gemma_bridge.check_gemma_sql_installed", return_value=False
    ):
        with pytest.raises(
            RuntimeError, match="gemma-4-sql toolchain is not installed"
        ):
            runner.validate_environment()

    ds = tmp_path / "train.parquet"
    ds.write_text("data")
    out = tmp_path / "gemma_out"

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

    out_exec = tmp_path / "gemma_exec_out"
    config_exec = TrainingJobConfig(
        model_name="gemma-4-sql",
        backend=TrainingBackend.GEMMA_4_SQL,
        dataset_path=str(ds),
        output_dir=str(out_exec),
        dry_run=False,
    )
    mock_completed = subprocess.CompletedProcess(
        args=["gemma-4-sql"],
        returncode=0,
        stdout="[MOCK] Success" + chr(10),
        stderr="",
    )
    with patch(
        "t1d_analytics.gemma_bridge.run_gemma_sql_train", return_value=mock_completed
    ) as mock_train:
        res_exec = runner.run_training(config_exec)
        assert res_exec["status"] == "completed"
        assert res_exec["returncode"] == 0
        assert "[MOCK] Success" in str(res_exec["stdout"])
        mock_train.assert_called_once()


def test_local_cpu_and_gpu_uneven_batches_and_exceptions(tmp_path: Path) -> None:
    """Test uneven batch loop termination, gradient accumulation, and error propagation."""
    runner = LocalCpuRunner()
    ds = tmp_path / "three_samples.jsonl"
    ds.write_text(
        """{"prompt": "q1", "completion": "SELECT 1"}
{"prompt": "q2", "completion": "SELECT 2"}
{"prompt": "q3", "completion": "SELECT 3"}
"""
    )
    out = tmp_path / "cpu_uneven_out"
    config = TrainingJobConfig(
        model_name="gemma-2b",
        dataset_path=str(ds),
        output_dir=str(out),
        hyperparameters=TrainingHyperparameters(
            batch_size=2, num_epochs=1, gradient_accumulation_steps=2
        ),
        dry_run=False,
    )
    res = runner.run_training(config)
    assert res["status"] == "completed"
    assert res["total_steps"] == 1

    with patch(
        "t1d_analytics.training_runner.prepare_torch_dataset",
        side_effect=RuntimeError("Mock torch failure"),
    ):
        with pytest.raises(
            TrainingExecutionError, match="CPU training execution failed"
        ):
            runner.run_training(config)

    gpu_runner = LocalGpuRunner()
    cfg_gpu = TrainingJobConfig(
        model_name="gemma-7b",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "gpu_err_out"),
        dry_run=False,
    )
    with patch(
        "t1d_analytics.training_runner.prepare_torch_dataset",
        side_effect=RuntimeError("GPU tensor error"),
    ):
        with pytest.raises(
            TrainingExecutionError, match="GPU training execution failed"
        ):
            gpu_runner.run_training(cfg_gpu)


def test_huggingface_causal_lm_runner_lifecycle(tmp_path: Path) -> None:
    """Test HuggingFaceCausalLMRunner validation, dry-run, evaluation, and training run with real tiny PyTorch modules."""
    import torch.nn as nn

    runner = HuggingFaceCausalLMRunner()
    assert runner.validate_environment() is True

    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(RuntimeError, match="HuggingFace execution requires"):
            runner.validate_environment()

    ds = tmp_path / "hf_train.jsonl"
    ds.write_text(
        """{"prompt": "q1", "completion": "SELECT 1"}
{"prompt": "q2", "completion": "SELECT 2"}
"""
    )
    val_ds = tmp_path / "hf_val.jsonl"
    val_ds.write_text("""{"prompt": "qv", "completion": "SELECT v"}
""")
    out_dir = tmp_path / "hf_out"

    cfg_dry = TrainingJobConfig(
        model_name="gemma-2b",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "hf_dry"),
        dry_run=True,
    )
    res_dry = runner.run_training(cfg_dry)
    assert res_dry["status"] == "dry_run_completed"
    assert res_dry["final_loss"] == 0.0

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

    class RealHFModel(nn.Module):
        """Real PyTorch causal LM module returning raw tensor logits."""

        def __init__(self) -> None:
            """Initialize embedding and projection."""
            super().__init__()
            self.embed = nn.Embedding(1000, 32)
            self.head = nn.Linear(32, 1000)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Forward pass returning real logits tensor."""
            x = self.embed(input_ids.clamp(0, 999))
            logits: torch.Tensor = self.head(x)
            return logits

        def save_pretrained(self, save_dir: str) -> None:
            """Write config.json to save_dir."""
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / "config.json").write_text("{}")

    real_model = RealHFModel()
    with patch(
        "t1d_analytics.training_runner.HuggingFaceCausalLMFactory.create_model",
        return_value=real_model,
    ):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                res_full = runner.run_training(cfg_full)
                assert res_full["status"] == "completed"
                assert isinstance(res_full["eval_metrics"], dict)
                assert "eval_loss" in res_full["eval_metrics"]
                assert "perplexity" in res_full["eval_metrics"]
                assert (out_dir / "checkpoint-hf-final" / "config.json").exists()

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

    with patch(
        "t1d_analytics.training_runner.HuggingFaceCausalLMFactory.create_model",
        side_effect=RuntimeError("Instantiation error"),
    ):
        with pytest.raises(
            TrainingExecutionError, match="Hugging Face training execution failed"
        ):
            runner.run_training(cfg_full)


def test_runner_eval_scheduler_and_cancellation(tmp_path: Path) -> None:
    """Test CosineAnnealingLR scheduler, cooperative cancel_event termination, and fallback checkpointing across CPU, GPU, and HF runners."""
    ds = tmp_path / "train.jsonl"
    ds.write_text(
        """{"prompt": "q", "completion": "a"}
{"prompt": "q2", "completion": "a2"}
"""
    )
    val_ds = tmp_path / "val.jsonl"
    val_ds.write_text("""{"prompt": "vq", "completion": "va"}
""")

    hparams = TrainingHyperparameters(
        batch_size=1, num_epochs=2, warmup_steps=1, gradient_accumulation_steps=2
    )
    config = TrainingJobConfig(
        model_name="mock",
        backend=TrainingBackend.LOCAL_CPU,
        dataset_path=str(ds),
        validation_dataset_path=str(val_ds),
        output_dir=str(tmp_path / "out_hf"),
        hyperparameters=hparams,
        dry_run=False,
    )

    class OutContainer:
        """Output container."""

        def __init__(self, logits: torch.Tensor) -> None:
            """Store logits."""
            self.logits = logits

    class ModelWithoutSavePretrained(torch.nn.Module):
        """Real PyTorch module without save_pretrained to exercise torch.save fallback."""

        def __init__(self) -> None:
            """Initialize linear layer."""
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x: torch.Tensor, **kwargs: SqlValue) -> OutContainer:
            """Return logits container."""
            _ = kwargs
            return OutContainer(
                torch.randn(
                    x.size(0), x.size(1), 10, device=x.device, requires_grad=True
                )
            )

    model_no_save = ModelWithoutSavePretrained()

    with (
        patch(
            "t1d_analytics.training_runner.HuggingFaceCausalLMFactory.create_model",
            return_value=model_no_save,
        ),
        patch(
            "t1d_analytics.hf_models.HuggingFaceCausalLMFactory.create_model",
            return_value=model_no_save,
        ),
        patch("torch.save"),
        patch(
            "t1d_analytics.training_runner.prepare_torch_dataset",
            return_value=[
                (torch.tensor([1, 2]), torch.tensor([1, 2])),
                (torch.tensor([2, 3]), torch.tensor([2, 3])),
            ],
        ),
    ):
        for backend in (
            TrainingBackend.LOCAL_CPU,
            TrainingBackend.LOCAL_GPU,
            TrainingBackend.HUGGINGFACE,
        ):
            config.backend = backend
            runner = get_training_runner(backend)
            res = runner.run_training(config)
            assert res["status"] == "completed"
            assert res["total_steps"] == 4

        cancel_ev = threading.Event()
        cancel_ev.set()
        res_cancelled = get_training_runner(TrainingBackend.LOCAL_CPU).run_training(
            config, cancel_event=cancel_ev
        )
        assert res_cancelled["final_loss"] == 0.0

        with patch("torch.optim.lr_scheduler.CosineAnnealingLR", None, create=True):
            res_no_sched = get_training_runner(TrainingBackend.LOCAL_CPU).run_training(
                config
            )
            assert res_no_sched["status"] == "completed"

    config.dry_run = True
    config.backend = TrainingBackend.HUGGINGFACE
    runner_hf = get_training_runner(TrainingBackend.HUGGINGFACE)

    class ModelWithFailingTo:
        """Callable model whose .to() raises an error to cover fallback."""

        def to(self, device: DeviceLike) -> "ModelWithFailingTo":
            """Raise RuntimeError to exercise fallback branch."""
            _ = device
            raise RuntimeError("device move unsupported")

        def __call__(self, x: torch.Tensor) -> OutContainer:
            """Return mock output."""
            return OutContainer(torch.randn(x.size(0), x.size(1), 10))

    with (
        patch(
            "t1d_analytics.training_runner.HuggingFaceCausalLMFactory.create_model",
            return_value=ModelWithFailingTo(),
        ),
        patch(
            "t1d_analytics.training_runner.prepare_torch_dataset",
            return_value=[(torch.tensor([1, 2]), torch.tensor([1, 2]))],
        ),
    ):
        res_dry_no_to = runner_hf.run_training(config)
        assert res_dry_no_to["status"] == "dry_run_completed"


def test_huggingface_runner_validation_path_and_model_no_train(tmp_path: Path) -> None:
    """Test HuggingFaceCausalLMRunner when validation path is missing and model lacks .train()."""
    ds = tmp_path / "train.jsonl"
    ds.write_text("""{"prompt": "q", "completion": "a"}
""")

    runner = HuggingFaceCausalLMRunner()
    config = TrainingJobConfig(
        model_name="mock",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        validation_dataset_path=str(tmp_path / "missing_val.jsonl"),
        output_dir=str(tmp_path / "out_hf_val"),
        dry_run=True,
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.device", MockTorchDevice),
    ):
        res_dry = runner.run_training(config)
    assert res_dry["status"] == "dry_run_completed"
    assert res_dry["eval_metrics"] == {}

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.device", MockTorchDevice),
    ):
        res_dry_mps = runner.run_training(config)
    assert res_dry_mps["status"] == "dry_run_completed"
    assert res_dry_mps["eval_metrics"] == {}

    class MinimalOut:
        """Output container."""

        def __init__(self, logits: torch.Tensor) -> None:
            """Store logits."""
            self.logits = logits

    class MinimalCallableModel:
        """Callable model without .train() or .save_pretrained()."""

        def parameters(self) -> List[torch.nn.Parameter]:
            """Return single mock parameter."""
            return [torch.nn.Parameter(torch.randn(2, 2))]

        def state_dict(self) -> Dict[str, torch.Tensor]:
            """Return mock state_dict."""
            return {"w": torch.zeros(1)}

        def __call__(self, x: torch.Tensor) -> MinimalOut:
            """Return mock logits."""
            return MinimalOut(
                torch.randn(
                    x.size(0), x.size(1), 10, device=x.device, requires_grad=True
                )
            )

    config_run = TrainingJobConfig(
        model_name="mock",
        backend=TrainingBackend.HUGGINGFACE,
        dataset_path=str(ds),
        validation_dataset_path=str(tmp_path / "missing_val.jsonl"),
        output_dir=str(tmp_path / "out_hf_minimal"),
        dry_run=False,
    )
    with (
        patch(
            "t1d_analytics.training_runner.HuggingFaceCausalLMFactory.create_model",
            return_value=MinimalCallableModel(),
        ),
        patch(
            "t1d_analytics.training_runner.prepare_torch_dataset",
            return_value=[(torch.tensor([1, 2]), torch.tensor([1, 2]))],
        ),
    ):
        res_run = runner.run_training(config_run)
        assert res_run["status"] == "completed"
        assert (
            tmp_path / "out_hf_minimal" / "checkpoint-hf-final" / "model.pt"
        ).exists()


def test_local_cpu_and_gpu_tiny_model_execution(tmp_path: Path) -> None:
    """Test LocalCpuRunner and LocalGpuRunner with use_tiny_model=True using real TinyCausalLMFactory."""
    ds = tmp_path / "dataset_tiny.jsonl"
    ds.write_text("""{"prompt": "question", "completion": "SELECT 1"}
""")

    cpu_runner = LocalCpuRunner()
    cfg_tiny_cpu = TrainingJobConfig(
        model_name="test-tiny",
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_tiny_cpu"),
        use_tiny_model=True,
        dry_run=False,
        hyperparameters=TrainingHyperparameters(num_epochs=1, batch_size=1),
    )
    res_cpu = cpu_runner.run_training(cfg_tiny_cpu)
    assert res_cpu["status"] == "completed"
    assert (tmp_path / "out_tiny_cpu" / "checkpoint-final" / "model.pt").exists()

    gpu_runner = LocalGpuRunner()
    cfg_tiny_gpu = TrainingJobConfig(
        model_name="test-tiny",
        backend=TrainingBackend.LOCAL_GPU,
        dataset_path=str(ds),
        output_dir=str(tmp_path / "out_tiny_gpu"),
        use_tiny_model=True,
        dry_run=True,
    )
    with patch(
        "t1d_analytics.hf_models.TinyCausalLMFactory.create_model",
        wraps=TinyCausalLMFactory().create_model,
    ) as mock_tiny:
        res_gpu = gpu_runner.run_training(cfg_tiny_gpu)
        assert res_gpu["status"] == "dry_run_completed"
        mock_tiny.assert_called_once()


def test_training_job_manager_lifecycle(tmp_path: Path, mocker: MagicMock) -> None:
    """Test TrainingJobManager background thread execution, status tracking, cancellation, and error handling."""
    mgr = TrainingJobManager()

    assert mgr.get_job("non-existent") is None
    assert mgr.cancel_job("non-existent") is False
    assert mgr.list_jobs() == []

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
    time.sleep(0.15)
    job = mgr.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert len(mgr.list_jobs()) == 1

    assert mgr.cancel_job(job_id) is True
    completed_job = mgr.get_job(job_id)
    assert completed_job is not None
    assert completed_job["status"] == "completed"

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

    mgr2 = TrainingJobManager()
    event_start = threading.Event()
    event_release = threading.Event()

    def slow_run(
        cfg: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """Signal start and wait for release event."""
        _ = (cfg, cancel_event)
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

    from t1d_analytics import training_runner

    mgr3 = TrainingJobManager()
    orig_get_runner = training_runner.get_training_runner

    def cancel_on_get_runner(backend: TrainingBackend) -> BaseTrainingRunner:
        """Set all cancel flags before returning training runner."""
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
