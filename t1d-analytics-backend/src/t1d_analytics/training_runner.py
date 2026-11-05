"""Training runner execution engines for CPU, GPU, Hugging Face, Gemma-4-SQL, and Cloud TPU."""

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from t1d_analytics.dataset_loader import (
    _tokenize_pair_with_masking,
    prepare_torch_dataset,
)
from t1d_analytics.errors import (
    GemmaToolchainError,
    HardwareUnavailableError,
    InsufficientDataError,
    MissingSdkError,
    TrainingExecutionError,
)
from t1d_analytics.hf_models import (
    AcceleratorDetectorProtocol,
    AutocastTorchProtocol,
    BaseCausalLMFactory,
    BaseTrainingRunner,
    DeviceLike,
    HuggingFaceCausalLMFactory,
    PretrainedSavableProtocol,
    TensorLike,
    TinyCausalLMFactory,
    TokenizerProtocol,
    TorchCausalLMProtocol,
    TorchDeviceFactoryProtocol,
    TrainableModelProtocol,
    _extract_output_logits,
    _is_mps_available,
    resolve_causal_lm_factory,
    resolve_device_telemetry,
)
from t1d_analytics.job_manager import (
    JobMetricValue,
)
from t1d_analytics.job_manager import (
    TrainingJobManager as _SharedTrainingJobManager,
)
from t1d_analytics.models import (
    TrainingBackend,
    TrainingJobConfig,
)
from t1d_analytics.sql_utils import SqlValue
from t1d_analytics.tpu_orchestrator import RemoteTpuMaxTextRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TorchRunnerProfile:
    """Configuration profile for a local PyTorch causal LM runner variant."""

    device: str
    checkpoint_subdir: str
    metadata_filename: str
    error_prefix: str
    mixed_precision: str = "fp32"
    device_name: Optional[str] = None
    peak_memory_mb: Optional[float] = None
    include_loss_history_in_result: bool = True
    include_device_in_result: bool = False
    include_eval_metrics_in_result: bool = False
    custom_factory: Optional[BaseCausalLMFactory] = None
    ignore_index: Optional[int] = None
    move_model_to_device: bool = False


class BaseTorchCausalLMRunner(BaseTrainingRunner):
    """Shared base class for PyTorch causal LM training runners on CPU and GPU."""

    def _evaluate(
        self,
        model: TorchCausalLMProtocol,
        eval_dataset: Sequence[Tuple[TensorLike, TensorLike]],
        device: DeviceLike,
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
                logits = _extract_output_logits(outputs)
                loss = loss_fn(logits.view(-1, logits.size(-1)), lbls.view(-1))
                total_loss += float(loss.item())
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = round(math.exp(min(avg_loss, 20.0)), 4)

        model.train()
        return {"eval_loss": round(avg_loss, 4), "perplexity": perplexity}

    _evaluate_validation_dataset = _evaluate

    @staticmethod
    def _compute_batch_causal_lm_loss(
        model: TorchCausalLMProtocol,
        batch: Sequence[Tuple[TensorLike, TensorLike]],
        loss_fn: Callable[[TensorLike, TensorLike], TensorLike],
        target_dev: Optional[DeviceLike],
        mixed_precision: str,
        accum_steps: int = 1,
    ) -> float:
        """
        Collate a batch of `(input_ids, labels)` pairs, compute causal LM loss, backpropagate, and return scalar step loss.

        Args:
        ----
            model: PyTorch causal LM model instance.
            batch: Sequence of `(input_ids, labels)` tensor pairs.
            loss_fn: CrossEntropyLoss callable.
            target_dev: Optional target compute device.
            mixed_precision: Precision mode (`'fp32'` or `'fp16'`).
            accum_steps: Gradient accumulation steps divisor.

        Returns:
        -------
            float: Unscaled scalar loss float for the step.

        """
        import torch

        inps = torch.stack([b[0] for b in batch])
        lbls = torch.stack([b[1] for b in batch])
        if target_dev is not None:
            inps = inps.to(target_dev)
            lbls = lbls.to(target_dev)

        if mixed_precision == "fp16" and isinstance(torch, AutocastTorchProtocol):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inps)
                logits = _extract_output_logits(outputs)
                raw_loss = loss_fn(logits.view(-1, logits.size(-1)), lbls.view(-1))
        else:
            outputs = model(inps)
            logits = _extract_output_logits(outputs)
            raw_loss = loss_fn(logits.view(-1, logits.size(-1)), lbls.view(-1))

        scaled_loss = raw_loss / accum_steps
        backward_fn: Callable[[], None] = scaled_loss.backward
        backward_fn()
        return round(float(scaled_loss.item() * accum_steps), 4)

    def _run_torch_optimizer_loop(
        self,
        model: TorchCausalLMProtocol,
        dataset: Sequence[Tuple[TensorLike, TensorLike]],
        epochs: int,
        batch_size: int,
        total_steps: int,
        lr: float,
        weight_decay: float,
        accum_steps: int,
        max_norm: float,
        target_dev: Optional[DeviceLike] = None,
        mixed_precision: str = "fp32",
        ignore_index: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[List[float], float]:
        """
        Execute the shared PyTorch AdamW + CosineAnnealingLR training loop.

        Args:
        ----
            model: PyTorch causal LM model instance.
            dataset: Tokenized list of (input_ids, labels) tensor pairs.
            epochs: Number of training epochs.
            batch_size: Per-step batch size.
            total_steps: Total optimization steps across all epochs.
            lr: Learning rate.
            weight_decay: AdamW weight decay coefficient.
            accum_steps: Gradient accumulation steps.
            max_norm: Maximum gradient norm for clipping.
            target_dev: Optional torch.device to move batch tensors onto.
            mixed_precision: Precision mode ('fp32' or 'fp16').
            ignore_index: Optional label index to ignore in CrossEntropyLoss.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Tuple[List[float], float]: (loss_history, final_loss).

        """
        import torch

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        loss_fn = (
            torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            if ignore_index is not None
            else torch.nn.CrossEntropyLoss()
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps)
        )

        loss_history: List[float] = []
        step_idx = 0
        if isinstance(model, TrainableModelProtocol):
            model.train()

        for _ in range(epochs):
            for i in range(0, len(dataset), batch_size):
                if step_idx >= total_steps or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    break
                batch = dataset[i : i + batch_size]
                step_loss = self._compute_batch_causal_lm_loss(
                    model=model,
                    batch=batch,
                    loss_fn=loss_fn,
                    target_dev=target_dev,
                    mixed_precision=mixed_precision,
                    accum_steps=accum_steps,
                )

                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == total_steps:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_history.append(step_loss)
                step_idx += 1

        final_loss = loss_history[-1] if loss_history else 0.0
        return loss_history, final_loss

    _execute_pytorch_causal_lm_loop = _run_torch_optimizer_loop

    def _prepare_execution_context(
        self, config: TrainingJobConfig
    ) -> Tuple[Path, Path, int, int, float, int, float, int, int]:
        """
        Validate dataset, create output directory, and extract hyperparameters and step counts.

        Args:
        ----
            config: Training job configuration.

        Returns:
        -------
            Tuple[Path, Path, int, int, float, int, float, int, int]:
                (ds_path, out_path, batch_size, epochs, lr, accum_steps, max_norm, sample_count, total_steps).

        Raises:
        ------
            InsufficientDataError: If training dataset contains zero readable samples.

        """
        ds_path = self.prepare_dataset(config)
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
        return (
            ds_path,
            out_path,
            batch_size,
            epochs,
            lr,
            accum_steps,
            max_norm,
            sample_count,
            total_steps,
        )

    def _save_model_checkpoint(
        self, model: TorchCausalLMProtocol, ckpt_dir: Path
    ) -> None:
        """
        Save model weights to a checkpoint directory using save_pretrained or torch.save.

        Args:
        ----
            model: Trained PyTorch or Hugging Face model.
            ckpt_dir: Destination checkpoint directory.

        """
        import torch

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(model, PretrainedSavableProtocol):
            model.save_pretrained(str(ckpt_dir))
        else:
            torch.save(model.state_dict(), ckpt_dir / "model.pt")

    def _write_training_artifacts(
        self,
        ckpt_dir: Path,
        metadata_filename: str,
        metadata_payload: Dict[str, JobMetricValue],
        metrics_payload: Dict[str, JobMetricValue],
    ) -> None:
        """
        Write training metadata and metrics JSON files to the checkpoint directory.

        Args:
        ----
            ckpt_dir: Destination checkpoint directory.
            metadata_filename: Filename for configuration/device metadata JSON.
            metadata_payload: Metadata dictionary to serialize.
            metrics_payload: Training metrics dictionary to serialize.

        """
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / metadata_filename, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)
        with open(ckpt_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

    def _build_checkpoint_payloads(
        self,
        config: TrainingJobConfig,
        profile: TorchRunnerProfile,
        batch_size: int,
        epochs: int,
        lr: float,
        total_steps: int,
        sample_count: int,
        final_loss: float,
        loss_vals: List[SqlValue],
        eval_metrics_vals: Dict[str, SqlValue],
    ) -> Tuple[Dict[str, JobMetricValue], Dict[str, JobMetricValue]]:
        """
        Construct the metadata and metrics dictionaries persisted to the checkpoint directory.

        Args:
        ----
            config: Training job configuration.
            profile: Runner execution and artifact reporting profile.
            batch_size: Effective batch size.
            epochs: Effective epoch count.
            lr: Effective learning rate.
            total_steps: Total optimization steps.
            sample_count: Number of training samples.
            final_loss: Final scalar loss value.
            loss_vals: Per-step loss history list.
            eval_metrics_vals: Validation evaluation metrics dictionary.

        Returns:
        -------
            Tuple[Dict[str, JobMetricValue], Dict[str, JobMetricValue]]: `(metadata_payload, metrics_payload)`.

        """
        if profile.include_device_in_result:
            metadata_payload: Dict[str, JobMetricValue] = {
                "model_name": config.model_name,
                "backend": config.backend.value,
                "device": profile.device,
                "device_name": profile.device_name or profile.device,
                "mixed_precision": profile.mixed_precision,
                "peak_memory_mb": profile.peak_memory_mb or 0.0,
            }
            metrics_payload: Dict[str, JobMetricValue] = {
                "loss_history": loss_vals,
                "final_loss": final_loss,
                "total_steps": total_steps,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "device": profile.device,
                "mixed_precision": profile.mixed_precision,
            }
            return metadata_payload, metrics_payload

        if profile.include_eval_metrics_in_result:
            metadata_payload = {
                "model_name": config.model_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "backend": config.backend.value,
                "use_lora": config.use_lora,
                "quantization": config.quantization,
            }
            metrics_payload = {
                "loss_history": loss_vals,
                "final_loss": final_loss,
                "total_steps": total_steps,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "eval_metrics": eval_metrics_vals,
            }
            return metadata_payload, metrics_payload

        metadata_payload = {
            "model_name": config.model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "backend": config.backend.value,
        }
        metrics_payload = {
            "loss_history": loss_vals,
            "final_loss": final_loss,
            "total_steps": total_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "samples": sample_count,
        }
        return metadata_payload, metrics_payload

    _save_training_checkpoint = _save_model_checkpoint

    def _prepare_model_and_datasets(
        self,
        config: TrainingJobConfig,
        profile: TorchRunnerProfile,
        ds_path: Path,
    ) -> Tuple[
        TorchCausalLMProtocol,
        List[Tuple[TensorLike, TensorLike]],
        Optional[List[Tuple[TensorLike, TensorLike]]],
        Optional[DeviceLike],
    ]:
        """
        Load training and optional validation datasets and instantiate the target causal LM model.

        Args:
        ----
            config: Training job configuration.
            profile: Runner execution profile.
            ds_path: Validated training dataset path.

        Returns:
        -------
            Tuple[TorchCausalLMProtocol, List[Tuple[TensorLike, TensorLike]], Optional[List[Tuple[TensorLike, TensorLike]]], Optional[DeviceLike]]:
                `(model, train_dataset, eval_dataset, target_device)`.

        """
        import torch

        hparams = config.hyperparameters
        max_seq_len = (
            (hparams.max_seq_length if hparams else 128)
            if profile.include_eval_metrics_in_result
            else 128
        )
        dataset = prepare_torch_dataset(
            ds_path,
            max_seq_length=max_seq_len,
            tokenizer_name=config.model_name,
        )
        factory = profile.custom_factory or resolve_causal_lm_factory(config)
        model = factory.create_model(
            model_name=config.model_name,
            device=profile.device,
            quantization=config.quantization,
            use_lora=config.use_lora,
        )

        target_dev: Optional[DeviceLike] = None
        if profile.include_device_in_result or profile.move_model_to_device:
            target_dev = (
                torch.device(profile.device)
                if isinstance(torch, TorchDeviceFactoryProtocol)
                else "cpu"
            )
        if profile.move_model_to_device and target_dev is not None:
            try:
                model = model.to(target_dev)
            except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
                logger.debug(f"Failed to move HF model to device: {e}")

        eval_dataset: Optional[List[Tuple[TensorLike, TensorLike]]] = None
        if (
            profile.include_eval_metrics_in_result
            and config.validation_dataset_path
            and Path(config.validation_dataset_path).exists()
        ):
            eval_dataset = prepare_torch_dataset(
                Path(config.validation_dataset_path),
                max_seq_length=max_seq_len,
                tokenizer_name=config.model_name,
            )

        return model, dataset, eval_dataset, target_dev

    def _run_local_torch_training(
        self,
        config: TrainingJobConfig,
        profile: TorchRunnerProfile,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Execute the shared local PyTorch dataset preparation, model creation, optimization, and artifact persistence flow.

        Args:
        ----
            config: Training job configuration.
            profile: Runner execution and artifact reporting profile.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Dict[str, JobMetricValue]: Summary dictionary of training execution metrics.

        Raises:
        ------
            InsufficientDataError: If training dataset has zero samples.
            TrainingExecutionError: If dataset loading, model creation, or training fails.

        """
        start_time = time.time()
        (
            ds_path,
            out_path,
            batch_size,
            epochs,
            lr,
            accum_steps,
            max_norm,
            sample_count,
            total_steps,
        ) = self._prepare_execution_context(config)
        hparams = config.hyperparameters
        loss_history: List[float] = []
        eval_metrics: Dict[str, float] = {}

        try:
            model, dataset, eval_dataset, target_dev = self._prepare_model_and_datasets(
                config, profile, ds_path
            )
            if config.dry_run:
                inp, _ = dataset[0]
                dry_input = (
                    inp.unsqueeze(0).to(target_dev)
                    if target_dev is not None
                    else inp.unsqueeze(0)
                )
                outputs = model(dry_input)
                _ = _extract_output_logits(outputs)
                loss_history = [0.0] * total_steps
                final_loss = 0.0
            else:
                loss_history, final_loss = self._run_torch_optimizer_loop(
                    model=model,
                    dataset=dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    total_steps=total_steps,
                    lr=lr,
                    weight_decay=hparams.weight_decay if hparams else 0.01,
                    accum_steps=accum_steps,
                    max_norm=max_norm,
                    target_dev=target_dev,
                    mixed_precision=profile.mixed_precision,
                    ignore_index=profile.ignore_index,
                    cancel_event=cancel_event,
                )
                if eval_dataset and target_dev is not None:
                    eval_metrics = self._evaluate_validation_dataset(
                        model,
                        eval_dataset,
                        target_dev,
                        batch_size=batch_size,
                    )
                self._save_training_checkpoint(
                    model, out_path / profile.checkpoint_subdir
                )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as e:
            raise TrainingExecutionError(f"{profile.error_prefix}: {e}") from e

        loss_vals: List[SqlValue] = list(loss_history)
        eval_metrics_vals: Dict[str, SqlValue] = dict(eval_metrics)
        if not config.dry_run:
            metadata_payload, metrics_payload = self._build_checkpoint_payloads(
                config=config,
                profile=profile,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                total_steps=total_steps,
                sample_count=sample_count,
                final_loss=final_loss,
                loss_vals=loss_vals,
                eval_metrics_vals=eval_metrics_vals,
            )
            self._write_training_artifacts(
                ckpt_dir=out_path / profile.checkpoint_subdir,
                metadata_filename=profile.metadata_filename,
                metadata_payload=metadata_payload,
                metrics_payload=metrics_payload,
            )

        elapsed = time.time() - start_time
        result: Dict[str, JobMetricValue] = {
            "status": "dry_run_completed" if config.dry_run else "completed",
            "backend": config.backend.value,
            "model_name": config.model_name,
            "total_steps": total_steps,
            "final_loss": final_loss,
            "runtime_seconds": round(elapsed, 4),
            "output_dir": str(out_path),
        }
        if profile.include_loss_history_in_result:
            result["loss_history"] = loss_vals
        if profile.include_eval_metrics_in_result:
            result["eval_metrics"] = eval_metrics_vals
        if profile.include_device_in_result:
            result["device"] = profile.device
            result["mixed_precision"] = profile.mixed_precision
        return result


class LocalCpuRunner(BaseTorchCausalLMRunner):
    """Training runner executing on host CPU using PyTorch or local execution engine."""

    def validate_environment(self) -> bool:
        """
        Validate CPU execution environment.

        Returns
        -------
            bool: True if environment is suitable for CPU training.

        """
        return True

    def run_training(
        self,
        config: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Execute training or dry-run on CPU.

        Args:
        ----
            config: Job configuration.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Dict[str, JobMetricValue]: Metrics and summary output.

        Raises:
        ------
            InsufficientDataError: If training dataset has zero samples.
            TrainingExecutionError: If training execution fails.

        """
        return self._run_local_torch_training(
            config=config,
            profile=TorchRunnerProfile(
                device="cpu",
                checkpoint_subdir="checkpoint-final",
                metadata_filename="training_args.json",
                error_prefix="CPU training execution failed",
                include_loss_history_in_result=True,
                include_device_in_result=False,
            ),
            cancel_event=cancel_event,
        )


class LocalGpuRunner(BaseTorchCausalLMRunner):
    """Training runner executing on local GPU (CUDA / Apple MPS)."""

    def validate_environment(
        self,
        detector: Optional[AcceleratorDetectorProtocol] = None,
    ) -> bool:
        """
        Validate that a local GPU device (CUDA or MPS) is accessible.

        Args:
        ----
            detector: Optional injectable accelerator detector.

        Returns:
        -------
            bool: True if GPU hardware is available.

        Raises:
        ------
            HardwareUnavailableError: If no compatible GPU accelerator is detected.

        """
        if detector is not None:
            if detector.is_cuda_available() or detector.is_mps_available():
                return True
            raise HardwareUnavailableError(
                "No compatible GPU device (NVIDIA CUDA or Apple Silicon MPS) detected. "
                "Use --backend local-cpu or ensure drivers and PyTorch GPU builds are installed."
            )

        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            return True

        try:
            import torch

            if torch.cuda.is_available():
                return True
            if _is_mps_available(torch.backends):
                return True
        except ImportError as e:
            logger.debug(f"PyTorch is not installed or available: {e}")

        raise HardwareUnavailableError(
            "No compatible GPU device (NVIDIA CUDA or Apple Silicon MPS) detected. "
            "Use --backend local-cpu or ensure drivers and PyTorch GPU builds are installed."
        )

    def run_training(
        self,
        config: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Execute training or dry-run on local GPU.

        Args:
        ----
            config: Job configuration.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Dict[str, JobMetricValue]: Metrics, GPU telemetry, and summary output.

        Raises:
        ------
            InsufficientDataError: If training dataset has zero samples.
            TrainingExecutionError: If training execution fails.

        """
        device = "cpu"
        mixed_precision = "fp32"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
                mixed_precision = "fp16"
            elif _is_mps_available(torch.backends):
                device = "mps"
                mixed_precision = "fp32"
            device_name, peak_memory_mb = resolve_device_telemetry(device)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as e:
            raise TrainingExecutionError(f"GPU training execution failed: {e}") from e

        return self._run_local_torch_training(
            config=config,
            profile=TorchRunnerProfile(
                device=device,
                checkpoint_subdir="checkpoint-gpu-final",
                metadata_filename="gpu_metadata.json",
                error_prefix="GPU training execution failed",
                mixed_precision=mixed_precision,
                device_name=device_name,
                peak_memory_mb=peak_memory_mb,
                include_loss_history_in_result=False,
                include_device_in_result=True,
            ),
            cancel_event=cancel_event,
        )


class HuggingFaceCausalLMRunner(BaseTorchCausalLMRunner):
    """Training runner executing on CPU/GPU using Hugging Face transformers and PEFT."""

    def validate_environment(self) -> bool:
        """
        Validate that Hugging Face transformers and PyTorch are installed.

        Returns
        -------
            bool: True if execution environment is validated.

        Raises
        ------
            MissingSdkError: If transformers or torch are missing.

        """
        try:
            import torch

            _ = torch
            import transformers

            _ = transformers

            return True
        except ImportError as e:
            raise MissingSdkError(
                f"HuggingFace execution requires 'transformers' and 'torch': {e}"
            ) from e

    def run_training(
        self,
        config: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Execute training run using Hugging Face causal LM architecture.

        Args:
        ----
            config: Training job configuration.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Dict[str, JobMetricValue]: Metrics and summary output.

        Raises:
        ------
            InsufficientDataError: If training dataset has no samples.
            TrainingExecutionError: If training loop fails.

        """
        import torch

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        elif _is_mps_available(torch.backends):
            device = "mps"

        return self._run_local_torch_training(
            config=config,
            profile=TorchRunnerProfile(
                device=device,
                checkpoint_subdir="checkpoint-hf-final",
                metadata_filename="training_args.json",
                error_prefix="Hugging Face training execution failed",
                include_loss_history_in_result=True,
                include_device_in_result=False,
                include_eval_metrics_in_result=True,
                custom_factory=HuggingFaceCausalLMFactory(),
                ignore_index=-100,
                move_model_to_device=True,
            ),
            cancel_event=cancel_event,
        )


@dataclass(frozen=True)
class DpoOptimizationResult:
    """
    Structured result from a Bradley-Terry DPO preference optimization run.

    Attributes
    ----------
        losses: Per-step Bradley-Terry DPO loss trajectory.
        margins: Per-step reward margin trajectory `beta * (logp_chosen - logp_rejected)`.
        checkpoint_path: Optional path to the saved checkpoint directory if `checkpoint_dir` was provided.

    """

    losses: List[float]
    margins: List[float]
    checkpoint_path: Optional[str] = None


def optimize_dpo_preference_pairs(
    pairs: Sequence[Dict[str, str]],
    beta: float,
    num_steps: int = 1,
    seed: int = 42,
    model: Optional[TorchCausalLMProtocol] = None,
    tokenizer: Optional[TokenizerProtocol] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> DpoOptimizationResult:
    """
    Execute a PyTorch Bradley-Terry preference optimization pass over `(prompt, chosen_sql, rejected_sql)`.

    Computes token-level log-probabilities for chosen and rejected SQL completions under a
    causal LM and minimizes `-log(sigmoid(beta * (logp_chosen - logp_rejected)))`.

    Args:
    ----
        pairs: Preference dataset triplets with `'prompt'`, `'chosen_sql'`, and `'rejected_sql'`.
        beta: DPO temperature hyperparameter.
        num_steps: Number of optimization epochs over the preference pairs.
        seed: Deterministic random seed for PyTorch parameter initialization.
        model: Optional pre-instantiated causal language model implementing `TorchCausalLMProtocol`.
        tokenizer: Optional pre-instantiated tokenizer implementing `TokenizerProtocol`.
        checkpoint_dir: Optional output directory where `model.pt` and `training_metrics.json` are written.

    Returns:
    -------
        DpoOptimizationResult: Structured DPO result containing losses, margins, and checkpoint path.

    """
    import torch

    class _ByteTokenizer(TokenizerProtocol):
        """Lightweight byte-level tokenizer for in-process DPO optimization."""

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            """
            Encode UTF-8 text into byte token IDs in `[1, 999]`.

            Args:
            ----
                text: Input string to tokenize.
                add_special_tokens: Whether to append special boundary tokens.

            Returns:
            -------
                List[int]: Token ID list.

            """
            _ = add_special_tokens
            return [min(999, max(1, b)) for b in text.encode("utf-8")]

    torch.manual_seed(seed)
    active_tokenizer: TokenizerProtocol = (
        tokenizer if tokenizer is not None else _ByteTokenizer()
    )
    active_model: TorchCausalLMProtocol = (
        model
        if model is not None
        else TinyCausalLMFactory().create_model(
            "tiny-dpo", vocab_size=1000, embed_dim=32
        )
    )
    active_model.train()
    optimizer = torch.optim.AdamW(active_model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    losses: List[float] = []
    margins: List[float] = []
    for _ in range(max(1, num_steps)):
        for pair in pairs:
            c_in, c_lbl = _tokenize_pair_with_masking(
                pair["prompt"],
                pair["chosen_sql"],
                active_tokenizer,
                max_seq_length=256,
            )
            r_in, r_lbl = _tokenize_pair_with_masking(
                pair["prompt"],
                pair["rejected_sql"] or "SELECT 0",
                active_tokenizer,
                max_seq_length=256,
            )

            optimizer.zero_grad()
            c_logits = _extract_output_logits(active_model(c_in.unsqueeze(0))).view(
                -1, 1000
            )
            r_logits = _extract_output_logits(active_model(r_in.unsqueeze(0))).view(
                -1, 1000
            )

            c_token_nll = loss_fn(c_logits, c_lbl.view(-1))
            r_token_nll = loss_fn(r_logits, r_lbl.view(-1))
            margin_tensor = beta * (-c_token_nll - (-r_token_nll))
            dpo_loss = -torch.nn.functional.logsigmoid(margin_tensor)
            backward_fn: Callable[[], None] = dpo_loss.backward
            backward_fn()
            optimizer.step()

            dpo_loss_val = float(dpo_loss.item())
            margin_val = float(margin_tensor.item())

            losses.append(round(dpo_loss_val, 4))
            margins.append(round(margin_val, 4))

    saved_ckpt: Optional[str] = None
    if checkpoint_dir is not None:
        ckpt_p = Path(checkpoint_dir)
        ckpt_p.mkdir(parents=True, exist_ok=True)
        torch.save(active_model.state_dict(), ckpt_p / "model.pt")
        avg_loss = round(sum(losses) / len(losses), 4) if losses else 0.0
        avg_margin = round(sum(margins) / len(margins), 4) if margins else 0.0
        with open(ckpt_p / "training_metrics.json", "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "dpo_losses": losses,
                    "reward_margins": margins,
                    "dpo_loss": avg_loss,
                    "reward_margin": avg_margin,
                    "beta": beta,
                },
                mf,
                indent=2,
            )
        saved_ckpt = str(ckpt_p.resolve())

    return DpoOptimizationResult(
        losses=losses,
        margins=margins,
        checkpoint_path=saved_ckpt,
    )


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
            GemmaToolchainError: If gemma-4-sql CLI or Python module is missing.

        """
        from t1d_analytics.gemma_bridge import check_gemma_sql_installed

        if not check_gemma_sql_installed():
            raise GemmaToolchainError(
                "gemma-4-sql toolchain is not installed. Install via pip install -e /path/to/gemma-4-sql[all]"
            )
        return True

    def run_training(
        self,
        config: TrainingJobConfig,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Execute pretrain or fine-tuning workflow via gemma-4-sql CLI.

        Args:
        ----
            config: Training job configuration.
            cancel_event: Optional threading event for cooperative cancellation.

        Returns:
        -------
            Dict[str, JobMetricValue]: Training run metadata and output paths.

        Raises:
        ------
            TrainingExecutionError: If gemma-4-sql execution fails.

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
        nl = chr(10)
        with open(config_file, "w") as f:
            f.write(nl.join(yaml_lines) + nl)

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
        proc = run_gemma_sql_train(
            stage=stage,
            config_path=config_file,
            cancel_event=cancel_event,
        )
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


class TrainingJobManager(_SharedTrainingJobManager):
    """Thread-safe asynchronous background manager for training jobs."""

    def __init__(self) -> None:
        """Initialize the job manager bound to get_training_runner."""
        super().__init__(runner_resolver=lambda backend: get_training_runner(backend))


global_training_job_manager = TrainingJobManager()


__all__ = [
    "AcceleratorDetectorProtocol",
    "BaseTorchCausalLMRunner",
    "BaseTrainingRunner",
    "DeviceLike",
    "DpoOptimizationResult",
    "Gemma4SqlRunner",
    "HuggingFaceCausalLMFactory",
    "HuggingFaceCausalLMRunner",
    "InsufficientDataError",
    "LocalCpuRunner",
    "LocalGpuRunner",
    "RemoteTpuMaxTextRunner",
    "TinyCausalLMFactory",
    "TorchRunnerProfile",
    "TrainingExecutionError",
    "TrainingJobManager",
    "get_training_runner",
    "global_training_job_manager",
    "optimize_dpo_preference_pairs",
    "prepare_torch_dataset",
    "resolve_causal_lm_factory",
    "resolve_device_telemetry",
]
