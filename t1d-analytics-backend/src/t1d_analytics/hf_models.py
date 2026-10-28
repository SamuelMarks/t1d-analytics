"""PyTorch and Hugging Face model factories, protocols, and hardware device telemetry."""

import importlib
import logging
import platform
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from t1d_analytics.errors import (
    MissingSdkError,
    TrainingExecutionError,
)
from t1d_analytics.job_manager import JobMetricValue
from t1d_analytics.models import TrainingJobConfig

logger = logging.getLogger(__name__)


class DeviceProtocol(Protocol):
    """Protocol representing a PyTorch compute device."""

    @property
    def type(self) -> str:
        """Return device type string (e.g., 'cpu', 'cuda', 'mps')."""
        raise NotImplementedError


class DtypeProtocol(Protocol):
    """Protocol representing a PyTorch tensor data type."""


class TensorProtocol(Protocol):
    """Structural protocol representing a PyTorch Tensor at type-check and runtime."""

    @property
    def shape(self) -> Sequence[int]:
        """Return tensor dimensions."""
        raise NotImplementedError

    def size(self, dim: int = -1) -> int:
        """Return size along the specified dimension."""
        raise NotImplementedError

    def view(self, *shape: int) -> "TensorProtocol":
        """Return reshaped tensor view."""
        raise NotImplementedError

    def unsqueeze(self, dim: int) -> "TensorProtocol":
        """Return tensor with singleton dimension inserted at dim."""
        raise NotImplementedError

    def to(self, device: Union[str, DeviceProtocol]) -> "TensorProtocol":
        """Move tensor to target device."""
        raise NotImplementedError

    def clamp(self, min_val: int, max_val: int) -> "TensorProtocol":
        """Clamp tensor elements to the range [min_val, max_val]."""
        raise NotImplementedError

    def item(self) -> float:
        """Extract scalar float value from 1-element tensor."""
        raise NotImplementedError

    def backward(self) -> None:
        """Compute gradients via backpropagation."""
        raise NotImplementedError

    def __truediv__(self, other: Union[int, float]) -> "TensorProtocol":
        """Divide tensor by scalar."""
        raise NotImplementedError


if TYPE_CHECKING:
    import threading

    import torch

    TensorLike = torch.Tensor
    DeviceLike = Union[str, torch.device]
    DtypeLike = Union[str, torch.dtype]
else:
    import threading

    TensorLike = TensorProtocol
    DeviceLike = Union[str, DeviceProtocol]
    DtypeLike = Union[str, DtypeProtocol]


class TokenizerProtocol(Protocol):
    """Protocol representing a pretrained tokenizer with encode."""

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text into integer token identifiers."""
        raise NotImplementedError


class AutoTokenizerFactoryProtocol(Protocol):
    """Protocol for Hugging Face AutoTokenizer class."""

    def from_pretrained(self, pretrained_model_name_or_path: str) -> TokenizerProtocol:
        """Load pretrained tokenizer by name or path."""
        raise NotImplementedError


class QuantizationConfigProtocol(Protocol):
    """Marker protocol for Hugging Face quantization configuration objects."""


class AutoModelFactoryProtocol(Protocol):
    """Protocol for Hugging Face AutoModelForCausalLM class."""

    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        torch_dtype: Optional[DtypeLike] = None,
        quantization_config: Optional[QuantizationConfigProtocol] = None,
    ) -> "TorchCausalLMProtocol":
        """Load pretrained causal LM by name or path."""
        raise NotImplementedError


class TransformersModuleProtocol(Protocol):
    """Typed protocol for the `transformers` module."""

    AutoTokenizer: AutoTokenizerFactoryProtocol
    AutoModelForCausalLM: AutoModelFactoryProtocol
    BitsAndBytesConfig: Callable[..., QuantizationConfigProtocol]


@runtime_checkable
class PaddedTokenizerProtocol(Protocol):
    """Runtime-checkable protocol for tokenizers exposing pad_token_id."""

    @property
    def pad_token_id(self) -> Optional[int]:
        """Return pad token identifier."""
        raise NotImplementedError


@runtime_checkable
class ModelOutputWithLogitsProtocol(Protocol):
    """Protocol representing a model forward output exposing logits."""

    @property
    def logits(self) -> TensorLike:
        """Return output logits tensor."""
        raise NotImplementedError


ModelForwardOutput = Union[TensorLike, ModelOutputWithLogitsProtocol]


class TorchCausalLMProtocol(Protocol):
    """Protocol representing a PyTorch or Hugging Face causal language model."""

    def __call__(self, input_ids: TensorLike) -> ModelForwardOutput:
        """Execute forward pass."""
        raise NotImplementedError

    def parameters(self) -> Iterable[TensorLike]:
        """Return trainable parameters."""
        raise NotImplementedError

    def train(self, mode: bool = True) -> "TorchCausalLMProtocol":
        """Set training mode."""
        raise NotImplementedError

    def eval(self) -> "TorchCausalLMProtocol":
        """Set evaluation mode."""
        raise NotImplementedError

    def to(self, device: DeviceLike) -> "TorchCausalLMProtocol":
        """Move model to target compute device."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, TensorLike]:
        """Return state dictionary."""
        raise NotImplementedError

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights to directory."""
        raise NotImplementedError


@runtime_checkable
class TrainableModelProtocol(Protocol):
    """Runtime-checkable protocol for models exposing `.train()`."""

    def train(self, mode: bool = True) -> TorchCausalLMProtocol:
        """Set training mode."""
        raise NotImplementedError


@runtime_checkable
class PretrainedSavableProtocol(Protocol):
    """Runtime-checkable protocol for models exposing `.save_pretrained()`."""

    def save_pretrained(self, save_directory: str) -> None:
        """Save pretrained weights to directory."""
        raise NotImplementedError


class LrSchedulerProtocol(Protocol):
    """Protocol for a PyTorch learning rate scheduler."""

    def step(self) -> None:
        """Advance learning rate schedule by one step."""
        raise NotImplementedError


@runtime_checkable
class CosineSchedulerNamespaceProtocol(Protocol):
    """Runtime-checkable protocol for `torch.optim.lr_scheduler` exposing `CosineAnnealingLR`."""

    @property
    def CosineAnnealingLR(self) -> Callable[..., LrSchedulerProtocol]:
        """Return CosineAnnealingLR scheduler constructor."""
        raise NotImplementedError


@runtime_checkable
class AutocastTorchProtocol(Protocol):
    """Runtime-checkable protocol for `torch` module exposing `.autocast()`."""

    def autocast(self, device_type: str, dtype: TensorLike) -> ContextManager[None]:
        """Return an autocast context manager."""
        raise NotImplementedError


@runtime_checkable
class TorchDeviceFactoryProtocol(Protocol):
    """Runtime-checkable protocol for `torch` module exposing `.device()`."""

    def device(self, dev: str) -> DeviceLike:
        """Construct a torch device object."""
        raise NotImplementedError


@runtime_checkable
class MpsBackendProtocol(Protocol):
    """Runtime-checkable protocol for `torch.backends.mps`."""

    @property
    def is_available(self) -> Callable[[], bool]:
        """Return callable checking whether Apple Silicon MPS acceleration is available."""
        raise NotImplementedError


@runtime_checkable
class MpsBackendsHolderProtocol(Protocol):
    """Runtime-checkable protocol for `torch.backends` exposing `.mps`."""

    @property
    def mps(self) -> MpsBackendProtocol:
        """Return MPS backend namespace."""
        raise NotImplementedError


@runtime_checkable
class AcceleratorDetectorProtocol(Protocol):
    """Injectable protocol for detecting local hardware accelerator availability."""

    def is_cuda_available(self) -> bool:
        """Return True if an NVIDIA CUDA accelerator is available."""
        raise NotImplementedError

    def is_mps_available(self) -> bool:
        """Return True if an Apple Silicon MPS accelerator is available."""
        raise NotImplementedError


def _is_mps_available(backends_obj: MpsBackendsHolderProtocol) -> bool:
    """
    Check whether Apple Silicon MPS acceleration is available on `torch.backends`.

    Args:
    ----
        backends_obj: The `torch.backends` namespace or mock object.

    Returns:
    -------
        bool: True if `backends_obj.mps.is_available()` returns True.

    """
    return bool(backends_obj.mps.is_available())


def _extract_output_logits(outputs: ModelForwardOutput) -> TensorLike:
    """
    Extract the logits tensor from a model forward output.

    Args:
    ----
        outputs: Model forward output (either a raw Tensor or an output container with `.logits`).

    Returns:
    -------
        TensorLike: Extracted logits tensor.

    """
    if isinstance(outputs, ModelOutputWithLogitsProtocol):
        return outputs.logits
    return outputs


class BaseTrainingRunner(ABC):
    """Abstract base class for model fine-tuning and pretraining execution engines."""

    @abstractmethod
    def validate_environment(self) -> bool:
        """
        Validate that required drivers, libraries, and hardware devices are present.

        Returns
        -------
            bool: True if execution environment is validated.

        Raises
        ------
            RuntimeError: If mandatory execution dependencies are missing.

        """
        raise NotImplementedError

    def prepare_dataset(self, config: TrainingJobConfig) -> Path:
        """
        Validate and prepare input dataset file for training execution.

        Args:
        ----
            config: Structured training job configuration.

        Returns:
        -------
            Path: Verified path to dataset.

        Raises:
        ------
            FileNotFoundError: If dataset path does not exist on disk.

        """
        ds_path = Path(config.dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"Training dataset not found at '{ds_path}'.")
        return ds_path

    def _count_dataset_samples(self, dataset_path: Path) -> int:
        """
        Count readable samples in dataset file (.parquet, .jsonl, .csv).

        Args:
        ----
            dataset_path: Path to dataset.

        Returns:
        -------
            int: Total sample count, or 0 if unreadable or empty.

        """
        if dataset_path.suffix in (".jsonl", ".json", ".csv"):
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())
            except Exception as e:
                logger.debug(f"Failed to count jsonl samples: {e}")
                return 0
        elif dataset_path.suffix == ".parquet":
            try:
                import duckdb

                escaped = str(dataset_path.resolve()).replace("'", "''")
                cnt = duckdb.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escaped}')"
                ).fetchone()
                return int(cnt[0]) if cnt else 0
            except Exception as e:
                logger.debug(f"Failed to count parquet samples: {e}")
                return 0
        return 0

    @abstractmethod
    def run_training(
        self,
        config: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Execute training run according to configuration parameters.

        Args:
        ----
            config: Training job configuration.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Dict[str, JobMetricValue]: Execution results containing training loss, steps, and duration.

        """
        raise NotImplementedError


class BaseCausalLMFactory(ABC):
    """Abstract factory interface for creating causal language models."""

    @abstractmethod
    def create_model(
        self,
        model_name: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> TorchCausalLMProtocol:
        """
        Create and return a causal language model instance configured for training.

        Args:
        ----
            model_name: Name or local path of base pretrained model.
            device: Target device string ('cpu', 'cuda:0', 'mps').
            quantization: Quantization mode ('4bit', '8bit', or None).
            use_lora: Whether to apply LoRA adapter weights.
            lora_r: LoRA rank dimension.
            lora_alpha: LoRA scaling alpha factor.
            lora_dropout: LoRA dropout probability.

        Returns:
        -------
            TorchCausalLMProtocol: Configured PyTorch nn.Module ready for training.

        Raises:
        ------
            RuntimeError: If dependencies for model creation are missing or fail.

        """
        raise NotImplementedError


class TinyCausalLMFactory(BaseCausalLMFactory):
    """Factory producing a lightweight pure-PyTorch causal language model for fast local CPU/GPU training."""

    def create_model(
        self,
        model_name: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        vocab_size: int = 1000,
        embed_dim: int = 32,
    ) -> TorchCausalLMProtocol:
        """
        Create a lightweight PyTorch Embedding + Linear causal language model.

        Args:
        ----
            model_name: Identifier for the tiny model.
            device: Target compute device ('cpu', 'cuda:0', 'mps').
            quantization: Optional quantization flag (ignored for tiny model).
            use_lora: Optional LoRA flag (ignored for tiny model).
            lora_r: LoRA rank dimension.
            lora_alpha: LoRA scaling alpha factor.
            lora_dropout: LoRA dropout probability.
            vocab_size: Vocabulary size for token embeddings.
            embed_dim: Hidden embedding dimension.

        Returns:
        -------
            TorchCausalLMProtocol: Configured lightweight PyTorch causal model.

        """
        import torch
        import torch.nn as nn

        _ = (model_name, quantization, use_lora, lora_r, lora_alpha, lora_dropout)

        class _TinyCausalLM(nn.Module):
            """Lightweight causal language model with token embedding and projection head."""

            def __init__(self, v_size: int, h_dim: int) -> None:
                """Initialize embedding and linear output projection layers."""
                super().__init__()
                self.embed = nn.Embedding(v_size, h_dim)
                self.head = nn.Linear(h_dim, v_size)

            def forward(self, input_ids: TensorLike) -> TensorLike:
                """Compute token logits from input token IDs."""
                clamped = input_ids.clamp(0, self.embed.num_embeddings - 1)
                embedded: TensorLike = self.embed(clamped)
                logits: TensorLike = self.head(embedded)
                return logits

            def save_pretrained(self, save_directory: str) -> None:
                """Save model state dictionary to save_directory/model.pt."""
                out_dir = Path(save_directory)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), out_dir / "model.pt")

        tiny_instance = _TinyCausalLM(vocab_size, embed_dim)
        try:
            tiny_instance.to(torch.device(device))
        except Exception as e:
            logger.debug(f"Could not move tiny model to {device}: {e}")
        result_model: TorchCausalLMProtocol = tiny_instance
        return result_model


class HuggingFaceCausalLMFactory(BaseCausalLMFactory):
    """Factory producing production Hugging Face transformers causal models with LoRA/quantization."""

    def create_model(
        self,
        model_name: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> TorchCausalLMProtocol:
        """
        Create a Hugging Face AutoModelForCausalLM instance with optional LoRA and quantization.

        Args:
        ----
            model_name: Pretrained model name or path (e.g., 'google/gemma-4-2b').
            device: Target compute device.
            quantization: Quantization mode ('4bit' or '8bit').
            use_lora: Whether to apply LoRA adapter weights.
            lora_r: LoRA rank dimension.
            lora_alpha: LoRA scaling alpha factor.
            lora_dropout: LoRA dropout probability.

        Returns:
        -------
            TorchCausalLMProtocol: Configured causal LM model.

        Raises:
        ------
            RuntimeError: If transformers or peft are missing or model loading fails.

        """
        try:
            import torch

            tf_mod: TransformersModuleProtocol = importlib.import_module("transformers")
            auto_model_cls = tf_mod.AutoModelForCausalLM
        except (ImportError, AttributeError) as e:
            raise MissingSdkError(
                "The 'transformers' and 'torch' libraries are required for HuggingFace models. "
                "Install via 'pip install transformers torch'."
            ) from e

        bnb_config: Optional[QuantizationConfigProtocol] = None
        if quantization in ("4bit", "8bit"):
            try:
                bnb_cls = tf_mod.BitsAndBytesConfig
                if quantization == "4bit":
                    bnb_config = bnb_cls(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                else:
                    bnb_config = bnb_cls(load_in_8bit=True)
            except (ImportError, AttributeError, Exception) as e:
                logger.warning(
                    f"BitsAndBytesConfig failed ({e}); falling back to standard precision."
                )

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        try:
            if bnb_config is not None:
                model: TorchCausalLMProtocol = auto_model_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    quantization_config=bnb_config,
                )
            else:
                model = auto_model_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                )
        except Exception as e:
            raise TrainingExecutionError(
                f"Failed to load Hugging Face model '{model_name}': {e}"
            ) from e

        if bnb_config is None:
            try:
                model.to(torch.device(device))
            except Exception as e:
                logger.debug(f"Failed to move model to {device}: {e}")

        if use_lora:
            try:
                import peft

                peft_config = peft.LoraConfig(
                    task_type=peft.TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                )
                model = peft.get_peft_model(model, peft_config)
            except ImportError as e:
                raise MissingSdkError(
                    "The 'peft' library is required for LoRA fine-tuning. "
                    "Install via 'pip install peft'."
                ) from e
            except Exception as e:
                raise TrainingExecutionError(
                    f"Failed to apply LoRA configuration to model '{model_name}': {e}"
                ) from e

        return model


def resolve_causal_lm_factory(config: TrainingJobConfig) -> BaseCausalLMFactory:
    """
    Select the appropriate causal LM factory based on TrainingJobConfig flags.

    Args:
    ----
        config: Training job configuration.

    Returns:
    -------
        BaseCausalLMFactory: TinyCausalLMFactory when use_tiny_model is True or
            model_architecture is 'tiny', otherwise HuggingFaceCausalLMFactory.

    """
    if config.use_tiny_model or config.model_architecture.lower() == "tiny":
        return TinyCausalLMFactory()
    return HuggingFaceCausalLMFactory()


def resolve_device_telemetry(device: str) -> Tuple[str, float]:
    """
    Resolve descriptive hardware identifier and total memory for the active execution device.

    Args:
    ----
        device: Target hardware device string ('cuda:0', 'mps', 'cpu', etc.).

    Returns:
    -------
        Tuple[str, float]: (device_name, total_memory_mb).

    """
    if device.startswith("cuda"):
        try:
            import torch

            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            if torch.cuda.is_available():
                dev_name = str(torch.cuda.get_device_name(idx))
                _, total_vram = torch.cuda.mem_get_info(idx)
                return dev_name, round(total_vram / (1024 * 1024), 2)
        except Exception as e:
            logger.debug(f"Failed to gather CUDA telemetry: {e}")
        return "NVIDIA CUDA Device", 0.0

    if device == "mps":
        dev_name = "Apple Silicon GPU (MPS)"
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                dev_name = f"Apple Silicon ({res.stdout.strip()})"
        except Exception as e:
            logger.debug(f"Failed to gather MPS telemetry: {e}")
        return dev_name, 0.0

    cpu_name = "Host CPU"
    try:
        proc = platform.processor()
        if proc:
            cpu_name = f"Host CPU ({proc})"
        elif Path("/proc/cpuinfo").exists():
            content = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in content.splitlines():
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception as e:
        logger.debug(f"Failed to gather CPU telemetry: {e}")

    return cpu_name, 0.0
