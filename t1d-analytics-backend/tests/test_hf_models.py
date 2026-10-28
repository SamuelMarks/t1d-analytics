"""Unit tests for Hugging Face and PyTorch causal LM factories, protocols, and device telemetry."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from t1d_analytics.dataset_loader import prepare_torch_dataset
from t1d_analytics.hf_models import (
    HuggingFaceCausalLMFactory,
    TinyCausalLMFactory,
    _extract_output_logits,
    resolve_causal_lm_factory,
    resolve_device_telemetry,
)
from t1d_analytics.models import (
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.training_runner import LocalCpuRunner

ORIGINAL_CREATE_MODEL = HuggingFaceCausalLMFactory.create_model


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


def test_hf_factory_direct_coverage() -> None:
    """Test HuggingFaceCausalLMFactory model loading, quantization, LoRA, and error branches."""
    factory = HuggingFaceCausalLMFactory()

    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(RuntimeError, match="transformers"):
            ORIGINAL_CREATE_MODEL(factory, "test")

    mock_transformers = MagicMock()
    mock_model = MagicMock()
    mock_model.to = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    with patch.dict("sys.modules", {"transformers": mock_transformers, "peft": None}):
        with pytest.raises(RuntimeError, match="peft"):
            ORIGINAL_CREATE_MODEL(factory, "test", use_lora=True)

    mock_peft = MagicMock()
    mock_peft.get_peft_model.return_value = mock_model

    with patch.dict(
        "sys.modules", {"transformers": mock_transformers, "peft": mock_peft}
    ):
        mock_model.to.side_effect = None
        res = ORIGINAL_CREATE_MODEL(factory, "test", device="cuda:0")
        assert res is not None

        mock_transformers.BitsAndBytesConfig = MagicMock()
        res_q8 = ORIGINAL_CREATE_MODEL(factory, "test", quantization="8bit")
        assert res_q8 is not None

        res_q4 = ORIGINAL_CREATE_MODEL(factory, "test", quantization="4bit")
        assert res_q4 is not None

        res_lora = ORIGINAL_CREATE_MODEL(factory, "test", use_lora=True)
        assert res_lora is not None

        mock_transformers.BitsAndBytesConfig.side_effect = Exception("BNB err")
        res_bnb_err = ORIGINAL_CREATE_MODEL(factory, "test", quantization="8bit")
        assert res_bnb_err is not None
        mock_transformers.BitsAndBytesConfig.side_effect = None

        mock_model.to.side_effect = Exception("to err")
        res_to_err = ORIGINAL_CREATE_MODEL(factory, "test", device="cpu")
        assert res_to_err is not None
        mock_model.to.side_effect = None

        mock_peft.get_peft_model.side_effect = Exception("LoRA err")
        with pytest.raises(RuntimeError, match="Failed to apply LoRA"):
            ORIGINAL_CREATE_MODEL(factory, "test", use_lora=True)
        mock_peft.get_peft_model.side_effect = None

    mock_transformers.AutoModelForCausalLM.from_pretrained.side_effect = Exception(
        "Load err"
    )
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with pytest.raises(RuntimeError, match="Failed to load Hugging Face model"):
            ORIGINAL_CREATE_MODEL(factory, "test")


def test_tiny_factory_and_resolve_causal_lm_factory(tmp_path: Path) -> None:
    """Test TinyCausalLMFactory model creation, save_pretrained, and resolve_causal_lm_factory."""
    cfg_tiny = TrainingJobConfig(
        model_name="test-tiny",
        dataset_path=str(tmp_path / "ds.jsonl"),
        output_dir=str(tmp_path / "out_tiny"),
        use_tiny_model=True,
        hyperparameters=TrainingHyperparameters(num_epochs=1, batch_size=1),
    )
    assert isinstance(resolve_causal_lm_factory(cfg_tiny), TinyCausalLMFactory)

    cfg_hf = TrainingJobConfig(
        model_name="google/gemma-2b",
        dataset_path=str(tmp_path / "ds.jsonl"),
        output_dir=str(tmp_path / "out_hf"),
        model_architecture="transformer",
    )
    assert isinstance(resolve_causal_lm_factory(cfg_hf), HuggingFaceCausalLMFactory)

    with patch("torch.device", side_effect=RuntimeError("bad device")):
        tiny_m = TinyCausalLMFactory().create_model("test-tiny", device="invalid")
        assert tiny_m is not None
    ckpt_dir = tmp_path / "tiny_ckpt"
    tiny_m.save_pretrained(str(ckpt_dir))
    assert (ckpt_dir / "model.pt").exists()


def test_resolve_device_telemetry_branches() -> None:
    """Test resolve_device_telemetry for CUDA, MPS, and CPU environments."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-80GB"):
            with patch(
                "torch.cuda.mem_get_info",
                return_value=(40 * 1024 * 1024 * 1024, 80 * 1024 * 1024 * 1024),
            ):
                name, mem = resolve_device_telemetry("cuda:1")
                assert name == "NVIDIA A100-SXM4-80GB"
                assert mem == 81920.0

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
            with patch(
                "torch.cuda.mem_get_info",
                return_value=(10 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024),
            ):
                name_def, mem_def = resolve_device_telemetry("cuda")
                assert name_def == "NVIDIA RTX 4090"
                assert mem_def == 24576.0

    with patch("torch.cuda.is_available", return_value=False):
        name_no_cuda, _ = resolve_device_telemetry("cuda:0")
        assert name_no_cuda == "NVIDIA CUDA Device"

    with patch("torch.cuda.is_available", return_value=True):
        with patch(
            "torch.cuda.get_device_name", side_effect=RuntimeError("CUDA query error")
        ):
            name_err, _ = resolve_device_telemetry("cuda:invalid")
            assert name_err == "NVIDIA CUDA Device"

    mock_sysctl = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Apple M2 Max" + chr(10), stderr=""
    )
    with patch("subprocess.run", return_value=mock_sysctl):
        name_mps, mem_mps = resolve_device_telemetry("mps")
        assert name_mps == "Apple Silicon (Apple M2 Max)"
        assert mem_mps == 0.0

    mock_sysctl_fail = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr=""
    )
    with patch("subprocess.run", return_value=mock_sysctl_fail):
        name_mps_fail, _ = resolve_device_telemetry("mps")
        assert name_mps_fail == "Apple Silicon GPU (MPS)"

    with patch("subprocess.run", side_effect=RuntimeError("sysctl missing")):
        name_mps_exc, _ = resolve_device_telemetry("mps")
        assert name_mps_exc == "Apple Silicon GPU (MPS)"

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with patch("platform.processor", return_value="x86_64"):
            name_proc, _ = resolve_device_telemetry("cpu")
            assert name_proc == "Host CPU (x86_64)"

    cpuinfo_match = """processor	: 0
vendor_id	: AuthenticAMD
model name	: AMD EPYC 7B12
"""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with patch("platform.processor", return_value=""):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value=cpuinfo_match):
                    name_cpuinfo, _ = resolve_device_telemetry("cpu")
                    assert name_cpuinfo == "AMD EPYC 7B12"

    cpuinfo_nomatch = """processor	: 0
vendor_id	: Unknown
"""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with patch("platform.processor", return_value=""):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value=cpuinfo_nomatch):
                    name_nomatch, _ = resolve_device_telemetry("cpu")
                    assert name_nomatch == "Host CPU"

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with patch("platform.processor", return_value=""):
            with patch("pathlib.Path.exists", return_value=False):
                name_no_proc_file, _ = resolve_device_telemetry("cpu")
                assert name_no_proc_file == "Host CPU"

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with patch("platform.processor", side_effect=RuntimeError("platform error")):
            name_fallback, _ = resolve_device_telemetry("cpu")
            assert name_fallback == "Host CPU"


def test_domain_invariants_prompt_masking_and_lora_freezing(tmp_path: Path) -> None:
    """Verify mathematical domain invariants: -100 prompt masking zeroes prompt loss and gradient updates modify model weights."""
    ds_file = tmp_path / "mask_test.jsonl"
    ds_file.write_text("""{"prompt": "How many?", "sql": "SELECT COUNT(*) FROM t;"}
""")

    class _CharTokenizer:
        """Simple character-level tokenizer for invariant verification."""

        pad_token_id: int = 0

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            """Encode characters as non-zero token IDs."""
            _ = add_special_tokens
            return [min(ord(c), 120) for c in text]

    pairs = prepare_torch_dataset(
        ds_file,
        max_seq_length=128,
        vocab_size=128,
        tokenizer=_CharTokenizer(),
    )
    assert len(pairs) == 1
    input_ids, labels = pairs[0]
    assert int(labels[0].item()) == -100
    assert any(int(v.item()) > 0 for v in labels)

    factory = TinyCausalLMFactory()
    model = factory.create_model(
        model_name="tiny",
        device="cpu",
        vocab_size=128,
        embed_dim=16,
    )
    params_list = list(model.parameters())
    assert len(params_list) >= 2
    params_list[0].requires_grad = False
    frozen_before = params_list[0].detach().clone()
    trainable_before = [p.detach().clone() for p in params_list[1:]]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    out = model(input_ids.unsqueeze(0))
    logits = _extract_output_logits(out)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()

    assert torch.equal(params_list[0].detach(), frozen_before)
    assert any(
        not torch.equal(p.detach(), before)
        for p, before in zip(params_list[1:], trainable_before)
    )


def test_hf_models_protocol_and_abstract_stubs(tmp_path: Path) -> None:
    """Test protocol and abstract method/property stubs in t1d_analytics.hf_models."""
    from contextlib import nullcontext
    from typing import ContextManager, Optional, Sequence, Union

    from t1d_analytics.hf_models import (
        AcceleratorDetectorProtocol,
        AutocastTorchProtocol,
        AutoModelFactoryProtocol,
        AutoTokenizerFactoryProtocol,
        BaseCausalLMFactory,
        BaseTrainingRunner,
        CosineSchedulerNamespaceProtocol,
        DeviceProtocol,
        DtypeLike,
        LrSchedulerProtocol,
        ModelOutputWithLogitsProtocol,
        MpsBackendProtocol,
        MpsBackendsHolderProtocol,
        PaddedTokenizerProtocol,
        PretrainedSavableProtocol,
        QuantizationConfigProtocol,
        TensorLike,
        TensorProtocol,
        TokenizerProtocol,
        TorchCausalLMProtocol,
        TorchDeviceFactoryProtocol,
        TrainableModelProtocol,
    )

    dummy_obj = object()
    for proto_cls, prop_name in (
        (DeviceProtocol, "type"),
        (TensorProtocol, "shape"),
        (PaddedTokenizerProtocol, "pad_token_id"),
        (ModelOutputWithLogitsProtocol, "logits"),
        (CosineSchedulerNamespaceProtocol, "CosineAnnealingLR"),
        (MpsBackendProtocol, "is_available"),
        (MpsBackendsHolderProtocol, "mps"),
    ):
        prop = getattr(proto_cls, prop_name)
        assert isinstance(prop, property)
        assert prop.fget is not None
        with pytest.raises(NotImplementedError):
            prop.fget(dummy_obj)

    class _ConcreteTensor:
        """Helper satisfying TensorProtocol for direct stub invocation."""

        @property
        def shape(self) -> Sequence[int]:
            """Return shape."""
            return (1,)

        def size(self, dim: int = -1) -> int:
            """Return size."""
            return 1

        def view(self, *shape: int) -> TensorProtocol:
            """Return view."""
            return self

        def unsqueeze(self, dim: int) -> TensorProtocol:
            """Return unsqueezed tensor."""
            return self

        def to(self, device: Union[str, DeviceProtocol]) -> TensorProtocol:
            """Return device-moved tensor."""
            return self

        def clamp(self, min_val: int, max_val: int) -> TensorProtocol:
            """Return clamped tensor."""
            return self

        def item(self) -> float:
            """Return scalar float."""
            return 0.0

        def backward(self) -> None:
            """No-op backward."""

        def __truediv__(self, other: Union[int, float]) -> TensorProtocol:
            """Return divided tensor."""
            return self

    fake_tensor = _ConcreteTensor()
    assert fake_tensor.shape == (1,)
    assert fake_tensor.size() == 1
    assert fake_tensor.view(1) is fake_tensor
    assert fake_tensor.unsqueeze(0) is fake_tensor
    assert fake_tensor.to("cpu") is fake_tensor
    assert fake_tensor.clamp(0, 1) is fake_tensor
    assert fake_tensor.item() == 0.0
    fake_tensor.backward()
    assert (fake_tensor / 2) is fake_tensor
    with pytest.raises(NotImplementedError):
        TensorProtocol.size(fake_tensor)
    with pytest.raises(NotImplementedError):
        TensorProtocol.view(fake_tensor, 1)
    with pytest.raises(NotImplementedError):
        TensorProtocol.unsqueeze(fake_tensor, 0)
    with pytest.raises(NotImplementedError):
        TensorProtocol.to(fake_tensor, "cpu")
    with pytest.raises(NotImplementedError):
        TensorProtocol.clamp(fake_tensor, 0, 1)
    with pytest.raises(NotImplementedError):
        TensorProtocol.item(fake_tensor)
    with pytest.raises(NotImplementedError):
        TensorProtocol.backward(fake_tensor)
    with pytest.raises(NotImplementedError):
        TensorProtocol.__truediv__(fake_tensor, 2)

    tensor = torch.zeros(1)
    tiny_factory = TinyCausalLMFactory()
    tiny_model = tiny_factory.create_model("tiny")

    class _ConcreteExtras:
        """Helper satisfying remaining protocol signatures for direct stub invocation."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            """No-op encode."""
            return []

        def from_pretrained(
            self,
            pretrained_model_name_or_path: str,
            torch_dtype: Optional[DtypeLike] = None,
            quantization_config: Optional[QuantizationConfigProtocol] = None,
        ) -> TorchCausalLMProtocol:
            """No-op from_pretrained."""
            return tiny_model

        def step(self) -> None:
            """No-op step."""

        def autocast(self, device_type: str, dtype: TensorLike) -> ContextManager[None]:
            """No-op autocast."""
            return nullcontext()

        def device(self, dev: str) -> str:
            """No-op device."""
            return dev

        def is_cuda_available(self) -> bool:
            """No-op cuda check."""
            return False

        def is_mps_available(self) -> bool:
            """No-op mps check."""
            return False

    class _ConcreteAutoTok:
        """Helper satisfying AutoTokenizerFactoryProtocol."""

        def from_pretrained(
            self, pretrained_model_name_or_path: str
        ) -> TokenizerProtocol:
            """Return concrete tokenizer."""
            return _ConcreteExtras()

    extras = _ConcreteExtras()
    auto_tok = _ConcreteAutoTok()
    assert extras.encode("hi") == []
    assert extras.from_pretrained("m") is tiny_model
    extras.step()
    with extras.autocast("cpu", tensor):
        pass
    assert extras.device("cpu") == "cpu"
    assert extras.is_cuda_available() is False
    assert extras.is_mps_available() is False
    assert isinstance(auto_tok.from_pretrained("m"), _ConcreteExtras)
    with pytest.raises(NotImplementedError):
        TokenizerProtocol.encode(extras, "hi")
    with pytest.raises(NotImplementedError):
        AutoTokenizerFactoryProtocol.from_pretrained(auto_tok, "m")
    with pytest.raises(NotImplementedError):
        AutoModelFactoryProtocol.from_pretrained(extras, "m")

    with pytest.raises(NotImplementedError):
        TorchCausalLMProtocol.__call__(tiny_model, tensor)
    with pytest.raises(NotImplementedError):
        TorchCausalLMProtocol.parameters(tiny_model)
    with pytest.raises(NotImplementedError):
        TorchCausalLMProtocol.train(tiny_model)
    with pytest.raises(NotImplementedError):
        TorchCausalLMProtocol.eval(tiny_model)
    with pytest.raises(NotImplementedError):
        TorchCausalLMProtocol.to(tiny_model, "cpu")
    with pytest.raises(NotImplementedError):
        TorchCausalLMProtocol.state_dict(tiny_model)
    with pytest.raises(NotImplementedError):
        TorchCausalLMProtocol.save_pretrained(tiny_model, "out")

    with pytest.raises(NotImplementedError):
        TrainableModelProtocol.train(tiny_model)
    with pytest.raises(NotImplementedError):
        PretrainedSavableProtocol.save_pretrained(tiny_model, "out")
    with pytest.raises(NotImplementedError):
        LrSchedulerProtocol.step(extras)
    with pytest.raises(NotImplementedError):
        AutocastTorchProtocol.autocast(extras, "cpu", tensor)
    with pytest.raises(NotImplementedError):
        TorchDeviceFactoryProtocol.device(extras, "cpu")
    with pytest.raises(NotImplementedError):
        AcceleratorDetectorProtocol.is_cuda_available(extras)
    with pytest.raises(NotImplementedError):
        AcceleratorDetectorProtocol.is_mps_available(extras)

    runner = LocalCpuRunner()
    cfg = TrainingJobConfig(
        model_name="tiny",
        dataset_path=str(tmp_path / "ds.jsonl"),
    )
    with pytest.raises(NotImplementedError):
        BaseTrainingRunner.validate_environment(runner)
    with pytest.raises(NotImplementedError):
        BaseTrainingRunner.run_training(runner, cfg)
    with pytest.raises(NotImplementedError):
        BaseCausalLMFactory.create_model(tiny_factory, "tiny")
