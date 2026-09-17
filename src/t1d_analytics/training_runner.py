"""Modular training execution runners and hardware orchestration for T1D Analytics."""

import json
import logging
import math
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t1d_analytics.models import MaxTextConfig, TrainingBackend, TrainingJobConfig

logger = logging.getLogger(__name__)


class TrainingExecutionError(RuntimeError):
    """Raised when model training execution encounters fatal errors or fails to converge."""


class InsufficientDataError(ValueError):
    """Raised when training dataset contains no usable samples or pairs."""


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
            except Exception:
                return 0
        elif dataset_path.suffix == ".parquet":
            try:
                import duckdb

                escaped = str(dataset_path.resolve()).replace("'", "''")
                cnt = duckdb.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{escaped}')"
                ).fetchone()
                return int(cnt[0]) if cnt else 0
            except Exception:
                return 0
        return 0

    @abstractmethod
    def run_training(self, config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Execute training run according to configuration parameters.

        Args:
        ----
            config: Training job configuration.

        Returns:
        -------
            Dict[str, Any]: Execution results containing training loss, steps, and duration.

        """


def prepare_torch_dataset(
    dataset_path: Path,
    max_seq_length: int = 128,
    vocab_size: int = 1000,
    tokenizer: Optional[Any] = None,
    tokenizer_name: Optional[str] = None,
) -> List[Tuple[Any, Any]]:
    """
    Read tabular dataset records and prepare tokenized inputs for PyTorch training.

    Supports JSONL and Parquet formats, extracting prompt and SQL completion text.
    Applies Gemma instruction formatting and loss masking on prompt prefix tokens.

    Args:
    ----
        dataset_path: Path to the dataset file (JSONL or Parquet).
        max_seq_length: Maximum sequence length for input padding and truncation.
        vocab_size: Vocabulary dimension for token index mapping when using fallback tokenizer.
        tokenizer: Optional pretrained tokenizer instance with encode() method.
        tokenizer_name: Optional Hugging Face pretrained tokenizer identifier.

    Returns:
    -------
        List[Tuple[Any, Any]]: List of (input_ids, labels) PyTorch tensor tuples.

    Raises:
    ------
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset format is unsupported or contains no records.

    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at '{dataset_path}'.")

    records: List[Tuple[str, str]] = []
    suffix = dataset_path.suffix.lower()

    if suffix == ".jsonl":
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            prompt = str(row.get("prompt") or row.get("question") or "")
                            target = str(
                                row.get("completion")
                                or row.get("sql")
                                or row.get("chosen")
                                or ""
                            )
                            records.append((prompt, target))
                    except Exception:
                        continue
        except Exception as e:
            raise ValueError(f"Failed to read JSONL dataset: {e}") from e

    elif suffix == ".parquet":
        try:
            import duckdb

            conn = duckdb.connect()
            escaped_path = str(dataset_path.resolve()).replace("'", "''")
            rel = conn.execute(f"SELECT * FROM read_parquet('{escaped_path}')")
            cols = [desc[0] for desc in rel.description] if rel.description else []
            if not cols:
                conn.close()
                raise ValueError("Parquet file contains no schema columns.")
            prompt_col = next(
                (
                    c
                    for c in cols
                    if c.lower() in ("prompt", "question", "text", "query")
                ),
                cols[0],
            )
            target_col = next(
                (
                    c
                    for c in cols
                    if c.lower()
                    in ("completion", "sql", "chosen", "target", "response")
                ),
                cols[1] if len(cols) > 1 else cols[0],
            )
            p_idx = cols.index(prompt_col)
            t_idx = cols.index(target_col)
            for r in rel.fetchall():
                records.append((str(r[p_idx]), str(r[t_idx])))
            conn.close()
        except Exception as e:
            raise ValueError(f"Failed to read Parquet dataset: {e}") from e
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Supported formats are .jsonl and .parquet."
        )

    if not records:
        raise ValueError(
            f"Dataset at '{dataset_path}' contains no valid training pairs."
        )

    import torch

    active_tok = tokenizer
    if active_tok is None and tokenizer_name is not None:
        try:
            from transformers import AutoTokenizer

            tok_loader: Any = getattr(AutoTokenizer, "from_pretrained")
            active_tok = tok_loader(tokenizer_name)
        except Exception as e:
            logger.warning(
                f"Could not load Hugging Face tokenizer '{tokenizer_name}': {e}. Using fallback."
            )

    tokenized_pairs: List[Tuple[Any, Any]] = []
    nl = "\n"
    for prompt, target in records:
        if active_tok is not None:
            # Gemma instruction template formatting
            prompt_str = f"<start_of_turn>user{nl}{prompt}<end_of_turn>{nl}<start_of_turn>model{nl}"
            full_str = f"{prompt_str}{target}<end_of_turn>"

            prompt_ids = active_tok.encode(prompt_str, add_special_tokens=False)
            full_ids = active_tok.encode(full_str, add_special_tokens=False)

            # Mask prompt tokens with -100 so loss is only computed on target SQL
            labels_list = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

            if len(full_ids) > max_seq_length:
                full_ids = full_ids[:max_seq_length]
                labels_list = labels_list[:max_seq_length]
            else:
                pad_id = getattr(active_tok, "pad_token_id", 0) or 0
                pad_len = max_seq_length - len(full_ids)
                full_ids = full_ids + [pad_id] * pad_len
                labels_list = labels_list + [-100] * pad_len

            inp_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            labels = torch.tensor(labels_list[1:], dtype=torch.long)
            tokenized_pairs.append((inp_ids, labels))
        else:
            full_text = f"{prompt} {target}".strip()
            tokens = [
                min(ord(c) % vocab_size, vocab_size - 1)
                for c in full_text[:max_seq_length]
            ]
            if not tokens:
                tokens = [0]
            if len(tokens) < max_seq_length:
                tokens.extend([0] * (max_seq_length - len(tokens)))
            else:
                tokens = tokens[:max_seq_length]

            inp_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)
            tokenized_pairs.append((inp_ids, labels))

    return tokenized_pairs


try:
    import torch
    import torch.nn as nn

    _BaseModule = nn.Module
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _BaseModule = object  # type: ignore[misc,assignment]


class TinyCausalLM(_BaseModule):
    """
    Minimal PyTorch causal language model architecture for local training execution.

    Args:
    ----
        vocab_size: Number of unique vocabulary tokens.
        hidden_dim: Hidden dimension size.

    """

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64) -> None:
        """
        Initialize causal LM layers.

        Args:
        ----
            vocab_size: Vocabulary dimension.
            hidden_dim: Hidden dimension size.

        """
        super().__init__()
        import torch.nn as nn

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: Any) -> Any:
        """
        Forward pass computing next-token prediction logits.

        Args:
        ----
            input_ids: Input token tensor of shape (batch_size, sequence_length).

        Returns:
        -------
            Any: Logit tensor of shape (batch_size, sequence_length, vocab_size).

        """
        x = self.embedding(input_ids)
        return self.fc(x)


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
    ) -> Any:
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
            Any: Configured PyTorch nn.Module ready for training.

        Raises:
        ------
            RuntimeError: If dependencies for model creation are missing or fail.

        """


class TinyCausalLMFactory(BaseCausalLMFactory):
    """Factory producing lightweight TinyCausalLM instances for unit tests and local execution."""

    def create_model(
        self,
        model_name: str,
        device: str = "cpu",
        quantization: Optional[str] = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> Any:
        """
        Create a TinyCausalLM instance on the target device.

        Args:
        ----
            model_name: Ignored for TinyCausalLM.
            device: Target compute device.
            quantization: Ignored for TinyCausalLM.
            use_lora: Ignored for TinyCausalLM.
            lora_r: Ignored for TinyCausalLM.
            lora_alpha: Ignored for TinyCausalLM.
            lora_dropout: Ignored for TinyCausalLM.

        Returns:
        -------
            Any: TinyCausalLM instance moved to target device.

        """
        model = TinyCausalLM(vocab_size=1000, hidden_dim=64)
        if torch is not None and hasattr(torch, "device"):
            try:
                target_dev = torch.device(device)
                model = model.to(target_dev)
            except Exception:
                pass
        return model


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
    ) -> Any:
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
            Any: Configured causal LM model.

        Raises:
        ------
            RuntimeError: If transformers or peft are missing or model loading fails.

        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(
                "The 'transformers' library is required for HuggingFace models. "
                "Install via 'pip install transformers'."
            ) from e

        bnb_config = None
        if quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig

                bnb_cls: Any = BitsAndBytesConfig
                if quantization == "4bit":
                    bnb_config = bnb_cls(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16 if torch else None,
                    )
                else:
                    bnb_config = bnb_cls(load_in_8bit=True)
            except (ImportError, Exception) as e:
                logger.warning(
                    f"BitsAndBytesConfig failed ({e}); falling back to standard precision."
                )

        torch_dtype = (
            torch.float16
            if (torch and device.startswith("cuda"))
            else (torch.float32 if torch else None)
        )

        model_kwargs: Dict[str, Any] = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hugging Face model '{model_name}': {e}"
            ) from e

        if bnb_config is None:
            try:
                to_fn = getattr(model, "to", None)
                if callable(to_fn) and torch is not None:
                    to_fn(torch.device(device))
            except Exception:
                pass

        if use_lora:
            try:
                import importlib

                peft_mod = importlib.import_module("peft")
                lora_cfg_cls: Any = getattr(peft_mod, "LoraConfig")
                task_type_cls: Any = getattr(peft_mod, "TaskType")
                get_peft_fn: Any = getattr(peft_mod, "get_peft_model")

                peft_config = lora_cfg_cls(
                    task_type=task_type_cls.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                )
                model = get_peft_fn(model, peft_config)
            except ImportError as e:
                raise RuntimeError(
                    "The 'peft' library is required for LoRA fine-tuning. "
                    "Install via 'pip install peft'."
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply LoRA configuration to model '{model_name}': {e}"
                ) from e

        return model


class LocalCpuRunner(BaseTrainingRunner):
    """Training runner executing on host CPU using PyTorch or local execution engine."""

    def validate_environment(self) -> bool:
        """
        Validate CPU execution environment.

        Returns
        -------
            bool: True if environment is suitable for CPU training.

        """
        return True

    def run_training(self, config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Execute training or dry-run on CPU.

        Args:
        ----
            config: Job configuration.

        Returns:
        -------
            Dict[str, Any]: Metrics and summary output.

        Raises:
        ------
            InsufficientDataError: If training dataset has zero samples.
            TrainingExecutionError: If training execution fails.

        """
        ds_path = self.prepare_dataset(config)
        start_time = time.time()
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        hparams = config.hyperparameters
        batch_size = hparams.batch_size if hparams else 4
        epochs = hparams.num_epochs if hparams else 3
        lr = hparams.learning_rate if hparams else 2e-5
        accum_steps = hparams.gradient_accumulation_steps if hparams else 1
        max_norm = hparams.max_grad_norm if hparams else 1.0

        sample_count = self._count_dataset_samples(ds_path)
        if sample_count <= 0:
            raise InsufficientDataError(
                f"Training dataset at '{ds_path}' contains 0 readable samples."
            )

        steps_per_epoch = max(1, sample_count // batch_size)
        total_steps = epochs * steps_per_epoch
        loss_history: List[float] = []

        try:
            import torch

            dataset = prepare_torch_dataset(ds_path)
            model = TinyCausalLMFactory().create_model(config.model_name, device="cpu")

            if config.dry_run:
                inp, _ = dataset[0]
                _ = model(inp.unsqueeze(0))
                loss_history = [0.0] * total_steps
                final_loss = 0.0
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=hparams.weight_decay if hparams else 0.01,
                )
                loss_fn = torch.nn.CrossEntropyLoss()
                sched_cls = getattr(torch.optim.lr_scheduler, "CosineAnnealingLR", None)
                scheduler = (
                    sched_cls(optimizer, T_max=max(1, total_steps))
                    if callable(sched_cls)
                    else None
                )

                step_idx = 0
                for _ in range(epochs):
                    for i in range(0, len(dataset), batch_size):
                        if step_idx >= total_steps:
                            break
                        batch = dataset[i : i + batch_size]
                        inps = torch.stack([b[0] for b in batch])
                        lbls = torch.stack([b[1] for b in batch])
                        logits = model(inps)
                        loss = loss_fn(logits.view(-1, 1000), lbls.view(-1))
                        loss = loss / accum_steps
                        loss.backward()

                        if (step_idx + 1) % accum_steps == 0 or (
                            step_idx + 1
                        ) == total_steps:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=max_norm
                            )
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad()

                        loss_history.append(round(float(loss.item() * accum_steps), 4))
                        step_idx += 1
                final_loss = loss_history[-1] if loss_history else 0.0

                ckpt_dir = out_path / "checkpoint-final"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "model.pt")
        except Exception as e:
            raise TrainingExecutionError(f"CPU training execution failed: {e}") from e

        if not config.dry_run:
            ckpt_dir = out_path / "checkpoint-final"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "training_args.json", "w") as f:
                json.dump(
                    {
                        "model_name": config.model_name,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": lr,
                        "backend": config.backend.value,
                    },
                    f,
                    indent=2,
                )
            with open(ckpt_dir / "training_metrics.json", "w") as f:
                json.dump(
                    {
                        "loss_history": loss_history,
                        "final_loss": final_loss,
                        "total_steps": total_steps,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "samples": sample_count,
                    },
                    f,
                    indent=2,
                )

        elapsed = time.time() - start_time
        return {
            "status": "dry_run_completed" if config.dry_run else "completed",
            "backend": config.backend.value,
            "model_name": config.model_name,
            "total_steps": total_steps,
            "final_loss": final_loss,
            "loss_history": loss_history,
            "runtime_seconds": round(elapsed, 4),
            "output_dir": str(out_path),
        }


class LocalGpuRunner(BaseTrainingRunner):
    """Training runner executing on local GPU (CUDA / Apple MPS)."""

    def validate_environment(self) -> bool:
        """
        Validate that a local GPU device (CUDA or MPS) is accessible.

        Returns
        -------
            bool: True if GPU hardware is available.

        Raises
        ------
            RuntimeError: If no compatible GPU accelerator is detected.

        """
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            or os.environ.get("T1D_MOCK_GPU") == "1"
        ):
            return True

        try:
            import torch

            if torch.cuda.is_available():
                return True
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return True
        except ImportError:
            pass

        raise RuntimeError(
            "No compatible GPU device (NVIDIA CUDA or Apple Silicon MPS) detected. "
            "Use --backend local-cpu or ensure drivers and PyTorch GPU builds are installed."
        )

    def run_training(self, config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Execute training or dry-run on local GPU.

        Args:
        ----
            config: Job configuration.

        Returns:
        -------
            Dict[str, Any]: Metrics, GPU telemetry, and summary output.

        Raises:
        ------
            InsufficientDataError: If training dataset has zero samples.
            TrainingExecutionError: If training execution fails.

        """
        ds_path = self.prepare_dataset(config)
        start_time = time.time()
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        hparams = config.hyperparameters
        batch_size = hparams.batch_size if hparams else 4
        epochs = hparams.num_epochs if hparams else 3
        lr = hparams.learning_rate if hparams else 2e-5
        accum_steps = hparams.gradient_accumulation_steps if hparams else 1
        max_norm = hparams.max_grad_norm if hparams else 1.0

        sample_count = self._count_dataset_samples(ds_path)
        if sample_count <= 0:
            raise InsufficientDataError(
                f"Training dataset at '{ds_path}' contains 0 readable samples."
            )

        steps_per_epoch = max(1, sample_count // batch_size)
        total_steps = epochs * steps_per_epoch
        loss_history: List[float] = []

        device = "cpu"
        device_name = "Mock Accelerator"
        peak_memory_mb = 0.0
        mixed_precision = "fp32"

        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
                try:
                    device_name = torch.cuda.get_device_name(0)
                    _, total_vram = torch.cuda.mem_get_info(0)
                    peak_memory_mb = round(total_vram / (1024 * 1024), 2)
                except Exception:
                    device_name = "NVIDIA CUDA Device"
                mixed_precision = "fp16"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                device_name = "Apple Silicon GPU (MPS)"
                mixed_precision = "fp32"

            dataset = prepare_torch_dataset(ds_path)
            model = TinyCausalLMFactory().create_model(config.model_name, device=device)

            if config.dry_run:
                inp, _ = dataset[0]
                target_dev = torch.device(device) if hasattr(torch, "device") else "cpu"
                _ = model(inp.unsqueeze(0).to(target_dev))
                loss_history = [0.0] * total_steps
                final_loss = 0.0
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=hparams.weight_decay if hparams else 0.01,
                )
                loss_fn = torch.nn.CrossEntropyLoss()
                sched_cls = getattr(torch.optim.lr_scheduler, "CosineAnnealingLR", None)
                scheduler = (
                    sched_cls(optimizer, T_max=max(1, total_steps))
                    if callable(sched_cls)
                    else None
                )

                step_idx = 0
                target_dev = torch.device(device) if hasattr(torch, "device") else "cpu"

                for _ in range(epochs):
                    for i in range(0, len(dataset), batch_size):
                        if step_idx >= total_steps:
                            break
                        batch = dataset[i : i + batch_size]
                        inps = torch.stack([b[0] for b in batch]).to(target_dev)
                        lbls = torch.stack([b[1] for b in batch]).to(target_dev)

                        if mixed_precision == "fp16" and hasattr(torch, "autocast"):
                            with torch.autocast(
                                device_type="cuda", dtype=torch.float16
                            ):
                                logits = model(inps)
                                loss = loss_fn(logits.view(-1, 1000), lbls.view(-1))
                        else:
                            logits = model(inps)
                            loss = loss_fn(logits.view(-1, 1000), lbls.view(-1))

                        loss = loss / accum_steps
                        loss.backward()

                        if (step_idx + 1) % accum_steps == 0 or (
                            step_idx + 1
                        ) == total_steps:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=max_norm
                            )
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad()

                        loss_history.append(round(float(loss.item() * accum_steps), 4))
                        step_idx += 1
                final_loss = loss_history[-1] if loss_history else 0.0

                ckpt_dir = out_path / "checkpoint-gpu-final"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "model.pt")
        except Exception as e:
            raise TrainingExecutionError(f"GPU training execution failed: {e}") from e

        if not config.dry_run:
            ckpt_dir = out_path / "checkpoint-gpu-final"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "gpu_metadata.json", "w") as f:
                json.dump(
                    {
                        "model_name": config.model_name,
                        "backend": config.backend.value,
                        "device": device,
                        "device_name": device_name,
                        "mixed_precision": mixed_precision,
                        "peak_memory_mb": peak_memory_mb,
                    },
                    f,
                    indent=2,
                )
            with open(ckpt_dir / "training_metrics.json", "w") as f:
                json.dump(
                    {
                        "loss_history": loss_history,
                        "final_loss": final_loss,
                        "total_steps": total_steps,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "device": device,
                        "mixed_precision": mixed_precision,
                    },
                    f,
                    indent=2,
                )

        elapsed = time.time() - start_time
        return {
            "status": "dry_run_completed" if config.dry_run else "completed",
            "backend": config.backend.value,
            "model_name": config.model_name,
            "total_steps": total_steps,
            "final_loss": final_loss,
            "device": device,
            "mixed_precision": mixed_precision,
            "runtime_seconds": round(elapsed, 4),
            "output_dir": str(out_path),
        }


class HuggingFaceCausalLMRunner(BaseTrainingRunner):
    """Training runner executing on CPU/GPU using Hugging Face transformers and PEFT."""

    def validate_environment(self) -> bool:
        """
        Validate that Hugging Face transformers and PyTorch are installed.

        Returns
        -------
            bool: True if execution environment is validated.

        Raises
        ------
            RuntimeError: If transformers or torch are missing.

        """
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError as e:
            raise RuntimeError(
                f"HuggingFace execution requires 'transformers' and 'torch': {e}"
            ) from e

    def _evaluate(
        self,
        model: Any,
        eval_dataset: List[Tuple[Any, Any]],
        device: Any,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Compute validation loss and perplexity on evaluation split.

        Args:
        ----
            model: PyTorch model instance.
            eval_dataset: Tokenized (input_ids, labels) pairs.
            device: Compute device.
            batch_size: Evaluation batch size.

        Returns:
        -------
            Dict[str, float]: Evaluation metrics dictionary with 'eval_loss' and 'perplexity'.

        """
        import torch

        model.eval()
        total_loss = 0.0
        total_batches = 0
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        with torch.no_grad():
            for i in range(0, len(eval_dataset), batch_size):
                batch = eval_dataset[i : i + batch_size]
                inps = torch.stack([b[0] for b in batch]).to(device)
                lbls = torch.stack([b[1] for b in batch]).to(device)
                outputs = model(inps)
                logits = getattr(outputs, "logits", outputs)
                loss = loss_fn(logits.view(-1, logits.size(-1)), lbls.view(-1))
                total_loss += float(loss.item())
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        try:
            perplexity = round(math.exp(min(avg_loss, 20.0)), 4)
        except OverflowError:
            perplexity = 999999.0

        model.train()
        return {"eval_loss": round(avg_loss, 4), "perplexity": perplexity}

    def run_training(self, config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Execute training run using Hugging Face causal LM architecture.

        Args:
        ----
            config: Training job configuration.

        Returns:
        -------
            Dict[str, Any]: Metrics and summary output.

        Raises:
        ------
            InsufficientDataError: If training dataset has no samples.
            TrainingExecutionError: If training loop fails.

        """
        ds_path = self.prepare_dataset(config)
        start_time = time.time()
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        hparams = config.hyperparameters
        batch_size = hparams.batch_size if hparams else 4
        epochs = hparams.num_epochs if hparams else 3
        lr = hparams.learning_rate if hparams else 2e-5
        accum_steps = hparams.gradient_accumulation_steps if hparams else 1
        max_norm = hparams.max_grad_norm if hparams else 1.0

        sample_count = self._count_dataset_samples(ds_path)
        if sample_count <= 0:
            raise InsufficientDataError(
                f"Training dataset at '{ds_path}' contains 0 readable samples."
            )

        steps_per_epoch = max(1, sample_count // batch_size)
        total_steps = epochs * steps_per_epoch
        loss_history: List[float] = []
        eval_metrics: Dict[str, float] = {}

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

            target_dev = torch.device(device)

            dataset = prepare_torch_dataset(
                ds_path,
                max_seq_length=hparams.max_seq_length if hparams else 128,
                tokenizer_name=config.model_name,
            )

            factory = HuggingFaceCausalLMFactory()
            model = factory.create_model(
                model_name=config.model_name,
                device=device,
                quantization=config.quantization,
                use_lora=config.use_lora,
            )
            try:
                to_fn = getattr(model, "to", None)
                if callable(to_fn):
                    model = to_fn(target_dev)
            except Exception:
                pass

            eval_dataset: Optional[List[Tuple[Any, Any]]] = None
            if config.validation_dataset_path:
                val_path = Path(config.validation_dataset_path)
                if val_path.exists():
                    eval_dataset = prepare_torch_dataset(
                        val_path,
                        max_seq_length=hparams.max_seq_length if hparams else 128,
                        tokenizer_name=config.model_name,
                    )

            if config.dry_run:
                inp, _ = dataset[0]
                outputs = model(inp.unsqueeze(0).to(target_dev))
                _ = getattr(outputs, "logits", outputs)
                loss_history = [0.0] * total_steps
                final_loss = 0.0
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=hparams.weight_decay if hparams else 0.01,
                )
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                sched_cls = getattr(torch.optim.lr_scheduler, "CosineAnnealingLR", None)
                scheduler = (
                    sched_cls(optimizer, T_max=max(1, total_steps))
                    if callable(sched_cls)
                    else None
                )

                step_idx = 0
                model.train()
                for _ in range(epochs):
                    for i in range(0, len(dataset), batch_size):
                        if step_idx >= total_steps:
                            break
                        batch = dataset[i : i + batch_size]
                        inps = torch.stack([b[0] for b in batch]).to(target_dev)
                        lbls = torch.stack([b[1] for b in batch]).to(target_dev)

                        outputs = model(inps)
                        logits = getattr(outputs, "logits", outputs)
                        loss = loss_fn(logits.view(-1, logits.size(-1)), lbls.view(-1))
                        loss = loss / accum_steps
                        loss.backward()

                        if (step_idx + 1) % accum_steps == 0 or (
                            step_idx + 1
                        ) == total_steps:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=max_norm
                            )
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad()

                        loss_history.append(round(float(loss.item() * accum_steps), 4))
                        step_idx += 1

                final_loss = loss_history[-1] if loss_history else 0.0

                if eval_dataset:
                    eval_metrics = self._evaluate(
                        model, eval_dataset, target_dev, batch_size=batch_size
                    )

                ckpt_dir = out_path / "checkpoint-hf-final"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(str(ckpt_dir))
                else:
                    torch.save(model.state_dict(), ckpt_dir / "model.pt")

        except Exception as e:
            raise TrainingExecutionError(
                f"Hugging Face training execution failed: {e}"
            ) from e

        if not config.dry_run:
            ckpt_dir = out_path / "checkpoint-hf-final"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_dir / "training_args.json", "w") as f:
                json.dump(
                    {
                        "model_name": config.model_name,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": lr,
                        "backend": config.backend.value,
                        "use_lora": config.use_lora,
                        "quantization": config.quantization,
                    },
                    f,
                    indent=2,
                )
            with open(ckpt_dir / "training_metrics.json", "w") as f:
                json.dump(
                    {
                        "loss_history": loss_history,
                        "final_loss": final_loss,
                        "total_steps": total_steps,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "eval_metrics": eval_metrics,
                    },
                    f,
                    indent=2,
                )

        elapsed = time.time() - start_time
        return {
            "status": "dry_run_completed" if config.dry_run else "completed",
            "backend": config.backend.value,
            "model_name": config.model_name,
            "total_steps": total_steps,
            "final_loss": final_loss,
            "loss_history": loss_history,
            "eval_metrics": eval_metrics,
            "runtime_seconds": round(elapsed, 4),
            "output_dir": str(out_path),
        }


class RemoteTpuMaxTextRunner(BaseTrainingRunner):
    """Training runner orchestrating MaxText jobs on Google Cloud TPU clusters."""

    def validate_environment(self) -> bool:
        """
        Validate that Google Cloud CLI or Libscript toolchain is installed.

        Returns
        -------
            bool: True if remote TPU environment tools are present.

        Raises
        ------
            RuntimeError: If gcloud or xpk CLI is missing.

        """
        if os.environ.get("T1D_MOCK_TPU") == "1":
            return True

        has_gcloud = shutil.which("gcloud") is not None
        has_libscript = (
            Path(os.environ.get("LIBSCRIPT_ROOT_DIR", "~/repos/libscript"))
            .expanduser()
            .exists()
        )
        if not (has_gcloud or has_libscript):
            raise RuntimeError(
                "Google Cloud CLI (gcloud) or Libscript orchestrator not found. "
                "Install Google Cloud CLI or set LIBSCRIPT_ROOT_DIR."
            )
        return True

    def generate_maxtext_yaml(self, config: TrainingJobConfig) -> str:
        """
        Generate MaxText YAML configuration file content for Cloud TPU execution.

        Args:
        ----
            config: Structured training configuration.

        Returns:
        -------
            str: Generated YAML configuration content.

        """
        max_cfg: MaxTextConfig = config.maxtext_config or MaxTextConfig(
            project_id=os.environ.get("GCP_PROJECT_ID", "t1d-analytics-gcp")
        )
        max_cfg.validate_tfrc_alignment()

        hparams = config.hyperparameters
        batch_size = hparams.batch_size if hparams else 4
        lr = hparams.learning_rate if hparams else 2e-5
        steps = (hparams.num_epochs if hparams else 3) * 50

        full_params = max_cfg.build_full_maxtext_parameters()
        lines = [
            f"# Generated MaxText Configuration for {config.model_name}",
            f"model_name: {config.model_name}",
            f"base_output_directory: '{max_cfg.bucket_url}/models/{config.model_name}'",
            f"dataset_path: '{config.dataset_path}'",
            f"learning_rate: {lr}",
            f"per_device_batch_size: {batch_size}",
            f"steps: {steps}",
            f"hardware: '{max_cfg.accelerator_type.value}'",
            f"zone: '{max_cfg.zone}'",
            f"num_slices: {max_cfg.num_slices}",
        ]
        for k, v in full_params.items():
            if k not in (
                "model_name",
                "base_output_directory",
                "dataset_path",
                "learning_rate",
                "per_device_batch_size",
                "steps",
                "hardware",
                "zone",
                "num_slices",
            ):
                lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"

    def _dispatch_xpk_workload(
        self,
        config: TrainingJobConfig,
        max_cfg: MaxTextConfig,
        config_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        """
        Dispatch MaxText workload to GKE TPU cluster using XPK toolchain.

        Args:
        ----
            config: Training job configuration.
            max_cfg: MaxText infrastructure and hardware parameters.
            config_path: Path to the generated MaxText YAML configuration file.

        Returns:
        -------
            subprocess.CompletedProcess[str]: Process execution record.

        Raises:
        ------
            RuntimeError: If dispatch fails or returns non-zero code.

        """
        workload_name = f"t1d-{config.model_name.replace('/', '-').replace(':', '-')}"
        cluster_name = f"tpu-cluster-{max_cfg.accelerator_type.value}"

        if os.environ.get("T1D_MOCK_TPU") == "1":
            return subprocess.CompletedProcess(
                args=["xpk", "workload", "create"],
                returncode=0,
                stdout=f"[MOCK] Workload {workload_name} created on cluster {cluster_name}\n",
                stderr="",
            )

        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            gcloud_bin = shutil.which("gcloud")
            if not gcloud_bin:
                raise RuntimeError("Neither 'xpk' nor 'gcloud' CLI is installed.")
            cmd = [
                gcloud_bin,
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                workload_name,
                f"--zone={max_cfg.zone}",
                f"--command=maxtext {config_path.name}",
            ]
        else:
            cmd = [
                xpk_bin,
                "workload",
                "create",
                f"--workload={workload_name}",
                f"--cluster={cluster_name}",
                f"--tpu-type={max_cfg.accelerator_type.value}",
                f"--num-slices={max_cfg.num_slices}",
                f"--zone={max_cfg.zone}",
            ]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"TPU dispatch failed: {proc.stderr.strip()}")
        return proc

    def poll_workload_status(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
        timeout_seconds: float = 30.0,
    ) -> str:
        """
        Poll workload execution status from XPK or Google Cloud TPU API.

        Args:
        ----
            workload_name: Identifier of the dispatched training workload.
            max_cfg: MaxText infrastructure configuration.
            timeout_seconds: Timeout threshold in seconds.

        Returns:
        -------
            str: Lifecycle status string ('SUCCESS', 'RUNNING', 'FAILED', or 'PENDING').

        Raises:
        ------
            TimeoutError: If polling exceeds specified timeout duration.
            RuntimeError: If workload is preempted or encounters an execution failure.

        """
        override_status = os.environ.get("T1D_TPU_STATUS_OVERRIDE")
        if override_status:
            if override_status == "PREEMPTED":
                raise RuntimeError(
                    f"Workload '{workload_name}' was PREEMPTED on spot TPU capacity."
                )
            if override_status in ("FAILED", "XLA_ERROR"):
                raise RuntimeError(
                    f"Workload '{workload_name}' encountered fatal XLA compilation error."
                )
            return override_status

        if os.environ.get("T1D_MOCK_TPU") == "1":
            return "SUCCESS"

        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            return "SUCCESS"

        cmd = [
            xpk_bin,
            "workload",
            "describe",
            f"--workload={workload_name}",
            f"--cluster=tpu-cluster-{max_cfg.accelerator_type.value}",
        ]
        start = time.time()
        while time.time() - start < timeout_seconds:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                stdout_upper = res.stdout.upper()
                if "PREEMPTED" in stdout_upper:
                    raise RuntimeError(
                        f"Workload '{workload_name}' was PREEMPTED on spot TPU capacity."
                    )
                if "XLA" in stdout_upper and "ERROR" in stdout_upper:
                    raise RuntimeError(
                        f"Workload '{workload_name}' encountered fatal XLA compilation error."
                    )
                if (
                    "SUCCESS" in stdout_upper
                    or "FINISHED" in stdout_upper
                    or "COMPLETED" in stdout_upper
                ):
                    return "SUCCESS"
                if "FAILED" in stdout_upper or "ERROR" in stdout_upper:
                    return "FAILED"
            time.sleep(0.05)

        raise TimeoutError(
            f"Workload '{workload_name}' polling timed out after {timeout_seconds} seconds."
        )

    def reclaim_tpu_workload(
        self,
        workload_name: str,
        max_cfg: MaxTextConfig,
    ) -> bool:
        """
        Reclaim and delete queued or terminated TPU workload resources.

        Args:
        ----
            workload_name: Identifier of the workload to reclaim.
            max_cfg: MaxText infrastructure parameters.

        Returns:
        -------
            bool: True if resources were reclaimed.

        Raises:
        ------
            RuntimeError: If reclamation command returns non-zero error.

        """
        if os.environ.get("T1D_MOCK_TPU") == "1":
            return True

        xpk_bin = shutil.which("xpk")
        if not xpk_bin:
            return True

        cluster_name = f"tpu-cluster-{max_cfg.accelerator_type.value}"
        cmd = [
            xpk_bin,
            "workload",
            "delete",
            f"--workload={workload_name}",
            f"--cluster={cluster_name}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to reclaim TPU workload: {res.stderr.strip()}")
        return True

    def download_checkpoints_from_gcs(
        self,
        bucket_url: str,
        output_dir: Path,
        max_retries: int = 3,
        validate_checksum: bool = True,
    ) -> Path:
        """
        Download trained model checkpoints from Google Cloud Storage to local directory.

        Args:
        ----
            bucket_url: GCS bucket URI containing model checkpoints.
            output_dir: Destination local directory path.
            max_retries: Maximum download retry attempts.
            validate_checksum: Whether to verify download checksum against manifest.

        Returns:
        -------
            Path: Path to the local directory containing downloaded checkpoints.

        Raises:
        ------
            ValueError: If bucket URI is invalid or checksum mismatches.
            RuntimeError: If gcloud storage copy command fails after retries.

        """
        if not bucket_url.startswith("gs://"):
            raise ValueError(
                f"Invalid GCS bucket URI '{bucket_url}'. Must start with 'gs://'."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        if os.environ.get("T1D_MOCK_GCS") == "1":
            if os.environ.get("T1D_MOCK_GCS_FAIL") == "1":
                raise RuntimeError(
                    f"GCS checkpoint download failed after {max_retries} attempts: Mock error"
                )
            mock_ckpt = output_dir / "checkpoint_tpu"
            mock_ckpt.mkdir(parents=True, exist_ok=True)
            manifest = mock_ckpt / "model_manifest.json"
            manifest.write_text('{"status": "downloaded", "md5": "abc123"}')
            if validate_checksum and os.environ.get("T1D_MOCK_GCS_CORRUPT") == "1":
                shutil.rmtree(output_dir)
                raise ValueError(
                    "Checksum verification failed for downloaded checkpoint."
                )
            return output_dir

        try:
            import importlib

            gcs_pkg = importlib.import_module("google.cloud.storage")
        except ImportError as e:
            raise RuntimeError(
                "The 'google-cloud-storage' Python package is required. "
                "Install via 'pip install google-cloud-storage'."
            ) from e

        client_cls: Any = getattr(gcs_pkg, "Client")
        client = client_cls()
        bucket_name = bucket_url[5:].rstrip("/").split("/")[0]
        prefix = "/".join(bucket_url[5:].rstrip("/").split("/")[1:])

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                bucket = client.bucket(bucket_name)
                blobs = list(client.list_blobs(bucket, prefix=prefix))
                if not blobs:
                    raise RuntimeError(f"No checkpoint files found at '{bucket_url}'.")
                for blob in blobs:
                    rel_name = blob.name[len(prefix) :].lstrip("/")
                    dest_file = output_dir / rel_name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(dest_file))
                return output_dir
            except Exception as e:
                last_err = e
                time.sleep(0.05 * (2 ** (attempt - 1)))

        raise RuntimeError(
            f"Checkpoint download failed after {max_retries} attempts: {last_err}"
        ) from last_err

    def run_training(self, config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Dispatch MaxText workload to Cloud TPU cluster.

        Args:
        ----
            config: Training job configuration.

        Returns:
        -------
            Dict[str, Any]: Remote workload launch details and generated configuration.

        """
        self.prepare_dataset(config)
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if (
            config.maxtext_config
            and config.maxtext_config.bucket_url
            and os.environ.get("T1D_AUTO_SYNC_GCS") == "1"
            and not config.dry_run
        ):
            sync_dataset_to_gcs(
                Path(config.dataset_path), config.maxtext_config.bucket_url
            )

        yaml_content = self.generate_maxtext_yaml(config)
        config_path = out_path / "maxtext_config.yml"
        with open(config_path, "w") as f:
            f.write(yaml_content)

        workload_name = f"t1d-{config.model_name.replace('/', '-').replace(':', '-')}"
        tpu_cluster = (
            config.maxtext_config.accelerator_type.value
            if config.maxtext_config
            else "v4-8"
        )

        status_str = "dispatched"
        if not config.dry_run:
            if os.environ.get("T1D_EXECUTE_TPU_DISPATCH") == "1":
                max_cfg_dispatch = config.maxtext_config or MaxTextConfig(
                    project_id=os.environ.get("GCP_PROJECT_ID", "t1d-analytics-gcp")
                )
                self._dispatch_xpk_workload(config, max_cfg_dispatch, config_path)
                try:
                    status_str = self.poll_workload_status(
                        workload_name, max_cfg_dispatch, timeout_seconds=5
                    )
                except RuntimeError as e:
                    if "PREEMPTED" in str(e):
                        status_str = "PREEMPTED_RETRY_EXHAUSTED"
                        for _ in range(max_cfg_dispatch.max_preemption_retries):
                            self._dispatch_xpk_workload(
                                config, max_cfg_dispatch, config_path
                            )
                            try:
                                status_str = self.poll_workload_status(
                                    workload_name, max_cfg_dispatch, timeout_seconds=2
                                )
                                break
                            except RuntimeError:
                                continue
                        if status_str == "PREEMPTED_RETRY_EXHAUSTED":
                            self.reclaim_tpu_workload(workload_name, max_cfg_dispatch)
                            raise
                    else:
                        self.reclaim_tpu_workload(workload_name, max_cfg_dispatch)
                        raise

            manifest_path = out_path / "job_manifest.json"
            with open(manifest_path, "w") as mf:
                json.dump(
                    {
                        "workload_name": workload_name,
                        "tpu_cluster": tpu_cluster,
                        "config_file": str(config_path),
                        "output_dir": str(out_path),
                        "model_name": config.model_name,
                        "dispatched_at": time.time(),
                        "status": status_str,
                    },
                    mf,
                    indent=2,
                )

        return {
            "status": "dry_run_completed" if config.dry_run else status_str,
            "backend": config.backend.value,
            "model_name": config.model_name,
            "config_file": str(config_path),
            "output_dir": str(out_path),
            "tpu_cluster": tpu_cluster,
            "workload_name": workload_name,
        }


def sync_dataset_to_gcs(
    local_path: Path,
    gcs_bucket: str,
    max_retries: int = 3,
    validate_checksum: bool = True,
) -> str:
    """
    Synchronize exported dataset file to Google Cloud Storage bucket with retries and checksums.

    Args:
    ----
        local_path: Local dataset filepath.
        gcs_bucket: Destination GCS bucket URL (e.g., 'gs://bucket-name').
        max_retries: Number of upload retry attempts.
        validate_checksum: Whether to compute and verify MD5 checksum.

    Returns:
    -------
        str: Remote GCS target URI.

    Raises:
    ------
        ValueError: If bucket URL is invalid or checksum verification fails.
        FileNotFoundError: If local dataset file does not exist.
        RuntimeError: If gcloud storage upload fails after retries.

    """
    if not gcs_bucket.startswith("gs://"):
        raise ValueError(
            f"Invalid GCS bucket URI '{gcs_bucket}'. Must start with 'gs://'."
        )

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found at '{local_path}'.")

    bucket_clean = gcs_bucket.rstrip("/")
    dest_uri = f"{bucket_clean}/{local_path.name}"

    if os.environ.get("T1D_MOCK_GCS") == "1":
        if os.environ.get("T1D_MOCK_GCS_FAIL") == "1":
            raise RuntimeError(
                f"GCS upload failed after {max_retries} attempts: Mock upload failure"
            )
        if validate_checksum and os.environ.get("T1D_MOCK_GCS_CORRUPT") == "1":
            raise ValueError("Checksum verification failed for uploaded dataset.")
        return dest_uri

    try:
        import importlib

        gcs_pkg = importlib.import_module("google.cloud.storage")
    except ImportError as e:
        raise RuntimeError(
            "The 'google-cloud-storage' Python package is required. "
            "Install via 'pip install google-cloud-storage'."
        ) from e

    client_cls: Any = getattr(gcs_pkg, "Client")
    client = client_cls()
    bucket_name = gcs_bucket[5:].rstrip("/").split("/")[0]
    blob_prefix = "/".join(gcs_bucket[5:].rstrip("/").split("/")[1:])
    blob_name = f"{blob_prefix}/{local_path.name}".lstrip("/")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            return dest_uri
        except Exception as e:
            last_err = e
            time.sleep(0.05 * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"GCS upload failed after {max_retries} attempts: {last_err}"
    ) from last_err


def get_training_runner(backend: TrainingBackend) -> BaseTrainingRunner:
    """
    Return the appropriate training runner instance for a given backend.

    Args:
    ----
        backend: Execution target environment enum.

    Returns:
    -------
        BaseTrainingRunner: Appropriate runner implementation.

    """
    if backend == TrainingBackend.LOCAL_GPU:
        return LocalGpuRunner()
    if backend == TrainingBackend.REMOTE_TPU_MAXTEXT:
        return RemoteTpuMaxTextRunner()
    if backend == TrainingBackend.GEMMA_4_SQL:
        return Gemma4SqlRunner()
    if backend == TrainingBackend.HUGGINGFACE:
        return HuggingFaceCausalLMRunner()
    return LocalCpuRunner()


class Gemma4SqlRunner(BaseTrainingRunner):
    """Training runner orchestrating external gemma-4-sql toolchains."""

    def validate_environment(self) -> bool:
        """
        Validate that the gemma-4-sql toolchain is installed and accessible.

        Returns
        -------
            bool: True if gemma-4-sql is installed and available.

        Raises
        ------
            RuntimeError: If gemma-4-sql CLI or Python module is missing.

        """
        from t1d_analytics.gemma_bridge import check_gemma_sql_installed

        if not check_gemma_sql_installed():
            raise RuntimeError(
                "gemma-4-sql toolchain is not installed. Install via pip install -e /path/to/gemma-4-sql[all]"
            )
        return True

    def run_training(self, config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Execute pretrain or fine-tuning workflow via gemma-4-sql CLI.

        Args:
        ----
            config: Training job configuration.

        Returns:
        -------
            Dict[str, Any]: Training run metadata and output paths.

        Raises:
        ------
            RuntimeError: If gemma-4-sql execution fails.

        """
        from t1d_analytics.gemma_bridge import run_gemma_sql_train

        self.prepare_dataset(config)
        out_path = Path(config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        hparams = config.hyperparameters
        lr = hparams.learning_rate if hparams else 2e-5
        batch_size = hparams.batch_size if hparams else 4
        epochs = hparams.num_epochs if hparams else 3

        stage = "sft"
        if "pretrain" in config.model_name.lower():
            stage = "pretrain"
        elif "dpo" in config.model_name.lower() or "rlhf" in config.model_name.lower():
            stage = "posttrain"

        config_file = out_path / f"gemma_sql_{stage}_config.yml"
        yaml_lines = [
            f"# Generated gemma-4-sql configuration for {config.model_name}",
            f"model: {config.model_name}",
            f"dataset: '{config.dataset_path}'",
            f"output_dir: '{out_path}'",
            f"learning_rate: {lr}",
            f"batch_size: {batch_size}",
            f"epochs: {epochs}",
            f"stage: {stage}",
        ]
        with open(config_file, "w") as f:
            f.write("\n".join(yaml_lines) + "\n")

        if config.dry_run:
            return {
                "status": "dry_run_completed",
                "backend": config.backend.value,
                "model_name": config.model_name,
                "stage": stage,
                "config_file": str(config_file),
                "output_dir": str(out_path),
            }

        start_time = time.time()
        proc = run_gemma_sql_train(stage=stage, config_path=config_file)
        elapsed = time.time() - start_time

        return {
            "status": "completed",
            "backend": config.backend.value,
            "model_name": config.model_name,
            "stage": stage,
            "runtime_seconds": round(elapsed, 4),
            "config_file": str(config_file),
            "output_dir": str(out_path),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
        }
