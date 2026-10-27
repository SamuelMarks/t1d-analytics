"""Training runner execution engines for CPU, GPU, Hugging Face, Gemma-4-SQL, and Cloud TPU."""

import json
import logging
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from t1d_analytics.dataset_loader import (
    StreamingJsonlDataset,
    TokenizedPairDataset,
    _load_raw_pairs_from_file,
    _resolve_tokenizer,
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
    AutoModelFactoryProtocol,
    AutoTokenizerFactoryProtocol,
    BaseCausalLMFactory,
    BaseTrainingRunner,
    CosineSchedulerNamespaceProtocol,
    DeviceLike,
    DeviceProtocol,
    DtypeLike,
    DtypeProtocol,
    HuggingFaceCausalLMFactory,
    LrSchedulerProtocol,
    ModelForwardOutput,
    ModelOutputWithLogitsProtocol,
    MpsBackendProtocol,
    MpsBackendsHolderProtocol,
    PaddedTokenizerProtocol,
    PretrainedSavableProtocol,
    QuantizationConfigProtocol,
    TensorLike,
    TensorProtocol,
    TinyCausalLMFactory,
    TokenizerProtocol,
    TorchCausalLMProtocol,
    TorchDeviceFactoryProtocol,
    TrainableModelProtocol,
    TransformersModuleProtocol,
    _extract_output_logits,
    _is_mps_available,
    resolve_causal_lm_factory,
    resolve_device_telemetry,
)
from t1d_analytics.job_manager import (
    JobMetricValue,
    TrainingJobState,
)
from t1d_analytics.job_manager import (
    TrainingJobManager as _SharedTrainingJobManager,
)
from t1d_analytics.models import (
    TrainingBackend,
    TrainingJobConfig,
)
from t1d_analytics.sql_utils import SqlValue
from t1d_analytics.tpu_orchestrator import (
    GcsBlobProtocol,
    GcsBucketProtocol,
    GcsClientProtocol,
    Md5BlobProtocol,
    RemoteTpuMaxTextRunner,
    SubprocessExecutorProtocol,
    _build_gcloud_ssh_command,
    _build_xpk_create_command,
    _compute_md5_base64,
    sync_dataset_to_gcs,
)

logger = logging.getLogger(__name__)
_ = subprocess


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
        sched_cls = (
            torch.optim.lr_scheduler.CosineAnnealingLR
            if isinstance(torch.optim.lr_scheduler, CosineSchedulerNamespaceProtocol)
            else None
        )
        scheduler = (
            sched_cls(optimizer, T_max=max(1, total_steps))
            if callable(sched_cls)
            else None
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
                inps = torch.stack([b[0] for b in batch])
                lbls = torch.stack([b[1] for b in batch])
                if target_dev is not None:
                    inps = inps.to(target_dev)
                    lbls = lbls.to(target_dev)

                if mixed_precision == "fp16" and isinstance(
                    torch, AutocastTorchProtocol
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(inps)
                        logits = _extract_output_logits(outputs)
                        loss = loss_fn(logits.view(-1, logits.size(-1)), lbls.view(-1))
                else:
                    outputs = model(inps)
                    logits = _extract_output_logits(outputs)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), lbls.view(-1))

                loss = loss / accum_steps
                loss.backward()

                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == total_steps:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_norm,
                    )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                loss_history.append(round(float(loss.item() * accum_steps), 4))
                step_idx += 1

        final_loss = loss_history[-1] if loss_history else 0.0
        return loss_history, final_loss

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

    def _run_local_torch_training(
        self,
        config: TrainingJobConfig,
        device: str,
        checkpoint_subdir: str,
        metadata_filename: str,
        error_prefix: str,
        mixed_precision: str = "fp32",
        device_name: Optional[str] = None,
        peak_memory_mb: Optional[float] = None,
        include_loss_history_in_result: bool = True,
        include_device_in_result: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, JobMetricValue]:
        """
        Execute the shared local PyTorch dataset preparation, model creation, optimization, and artifact persistence flow.

        Args:
        ----
            config: Training job configuration.
            device: Target PyTorch device string (e.g., `'cpu'`, `'cuda:0'`, `'mps'`).
            checkpoint_subdir: Subdirectory name under `output_dir` for the final checkpoint.
            metadata_filename: Filename for the metadata JSON written inside the checkpoint directory.
            error_prefix: Prefix for `TrainingExecutionError` messages on failure.
            mixed_precision: Precision mode (`'fp32'` or `'fp16'`).
            device_name: Optional hardware device name for GPU metadata.
            peak_memory_mb: Optional peak memory telemetry in MB for GPU metadata.
            include_loss_history_in_result: Whether to include `loss_history` in the returned dictionary.
            include_device_in_result: Whether to include `device` and `mixed_precision` in the returned dictionary.
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

        try:
            import torch

            dataset = prepare_torch_dataset(
                ds_path,
                tokenizer_name=config.model_name,
            )
            factory = resolve_causal_lm_factory(config)
            model = factory.create_model(
                model_name=config.model_name,
                device=device,
                quantization=config.quantization,
                use_lora=config.use_lora,
            )

            target_dev: Optional[DeviceLike] = None
            if include_device_in_result:
                target_dev = (
                    torch.device(device)
                    if isinstance(torch, TorchDeviceFactoryProtocol)
                    else "cpu"
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
                    mixed_precision=mixed_precision,
                    cancel_event=cancel_event,
                )
                self._save_model_checkpoint(model, out_path / checkpoint_subdir)
        except Exception as e:
            raise TrainingExecutionError(f"{error_prefix}: {e}") from e

        loss_vals: List[SqlValue] = list(loss_history)
        if not config.dry_run:
            if include_device_in_result:
                metadata_payload: Dict[str, JobMetricValue] = {
                    "model_name": config.model_name,
                    "backend": config.backend.value,
                    "device": device,
                    "device_name": device_name or device,
                    "mixed_precision": mixed_precision,
                    "peak_memory_mb": peak_memory_mb or 0.0,
                }
                metrics_payload: Dict[str, JobMetricValue] = {
                    "loss_history": loss_vals,
                    "final_loss": final_loss,
                    "total_steps": total_steps,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "device": device,
                    "mixed_precision": mixed_precision,
                }
            else:
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
            self._write_training_artifacts(
                ckpt_dir=out_path / checkpoint_subdir,
                metadata_filename=metadata_filename,
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
        if include_loss_history_in_result:
            result["loss_history"] = loss_vals
        if include_device_in_result:
            result["device"] = device
            result["mixed_precision"] = mixed_precision
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
            device="cpu",
            checkpoint_subdir="checkpoint-final",
            metadata_filename="training_args.json",
            error_prefix="CPU training execution failed",
            include_loss_history_in_result=True,
            include_device_in_result=False,
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
        except Exception as e:
            raise TrainingExecutionError(f"GPU training execution failed: {e}") from e

        return self._run_local_torch_training(
            config=config,
            device=device,
            checkpoint_subdir="checkpoint-gpu-final",
            metadata_filename="gpu_metadata.json",
            error_prefix="GPU training execution failed",
            mixed_precision=mixed_precision,
            device_name=device_name,
            peak_memory_mb=peak_memory_mb,
            include_loss_history_in_result=False,
            include_device_in_result=True,
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
            RuntimeError: If transformers or torch are missing.

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
        start_time = time.time()
        (
            ds_path,
            out_path,
            batch_size,
            epochs,
            lr,
            accum_steps,
            max_norm,
            _,
            total_steps,
        ) = self._prepare_execution_context(config)
        hparams = config.hyperparameters
        loss_history: List[float] = []
        eval_metrics: Dict[str, float] = {}

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
            elif _is_mps_available(torch.backends):
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
                model = model.to(target_dev)
            except Exception as e:
                logger.debug(f"Failed to move HF model to device: {e}")

            eval_dataset: Optional[List[Tuple[TensorLike, TensorLike]]] = None
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
                    ignore_index=-100,
                    cancel_event=cancel_event,
                )
                if eval_dataset:
                    eval_metrics = self._evaluate(
                        model,
                        eval_dataset,
                        target_dev,
                        batch_size=batch_size,
                    )
                self._save_model_checkpoint(model, out_path / "checkpoint-hf-final")

        except Exception as e:
            raise TrainingExecutionError(
                f"Hugging Face training execution failed: {e}"
            ) from e

        loss_history_vals: List[SqlValue] = list(loss_history)
        eval_metrics_vals: Dict[str, SqlValue] = dict(eval_metrics)
        if not config.dry_run:
            self._write_training_artifacts(
                ckpt_dir=out_path / "checkpoint-hf-final",
                metadata_filename="training_args.json",
                metadata_payload={
                    "model_name": config.model_name,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "backend": config.backend.value,
                    "use_lora": config.use_lora,
                    "quantization": config.quantization,
                },
                metrics_payload={
                    "loss_history": loss_history_vals,
                    "final_loss": final_loss,
                    "total_steps": total_steps,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "eval_metrics": eval_metrics_vals,
                },
            )

        elapsed = time.time() - start_time
        return {
            "status": "dry_run_completed" if config.dry_run else "completed",
            "backend": config.backend.value,
            "model_name": config.model_name,
            "total_steps": total_steps,
            "final_loss": final_loss,
            "loss_history": loss_history_vals,
            "eval_metrics": eval_metrics_vals,
            "runtime_seconds": round(elapsed, 4),
            "output_dir": str(out_path),
        }


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
            RuntimeError: If gemma-4-sql execution fails.

        """
        _ = cancel_event
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


class TrainingJobManager(_SharedTrainingJobManager):
    """Thread-safe asynchronous background manager for training jobs."""

    def __init__(self) -> None:
        """Initialize the job manager bound to get_training_runner."""
        super().__init__(runner_resolver=lambda backend: get_training_runner(backend))


global_training_job_manager = TrainingJobManager()


__all__ = [
    "AcceleratorDetectorProtocol",
    "AutocastTorchProtocol",
    "AutoModelFactoryProtocol",
    "AutoTokenizerFactoryProtocol",
    "BaseCausalLMFactory",
    "BaseTorchCausalLMRunner",
    "BaseTrainingRunner",
    "CosineSchedulerNamespaceProtocol",
    "DeviceLike",
    "DeviceProtocol",
    "DtypeLike",
    "DtypeProtocol",
    "GcsBlobProtocol",
    "GcsBucketProtocol",
    "GcsClientProtocol",
    "Gemma4SqlRunner",
    "HuggingFaceCausalLMFactory",
    "HuggingFaceCausalLMRunner",
    "InsufficientDataError",
    "LocalCpuRunner",
    "LocalGpuRunner",
    "LrSchedulerProtocol",
    "Md5BlobProtocol",
    "ModelForwardOutput",
    "ModelOutputWithLogitsProtocol",
    "MpsBackendProtocol",
    "MpsBackendsHolderProtocol",
    "PaddedTokenizerProtocol",
    "PretrainedSavableProtocol",
    "QuantizationConfigProtocol",
    "RemoteTpuMaxTextRunner",
    "StreamingJsonlDataset",
    "SubprocessExecutorProtocol",
    "TensorLike",
    "TensorProtocol",
    "TinyCausalLMFactory",
    "TokenizedPairDataset",
    "TokenizerProtocol",
    "TorchCausalLMProtocol",
    "TorchDeviceFactoryProtocol",
    "TrainableModelProtocol",
    "TrainingExecutionError",
    "TrainingJobManager",
    "TrainingJobState",
    "TransformersModuleProtocol",
    "_build_gcloud_ssh_command",
    "_build_xpk_create_command",
    "_compute_md5_base64",
    "_extract_output_logits",
    "_is_mps_available",
    "_load_raw_pairs_from_file",
    "_resolve_tokenizer",
    "_tokenize_pair_with_masking",
    "get_training_runner",
    "global_training_job_manager",
    "prepare_torch_dataset",
    "resolve_causal_lm_factory",
    "resolve_device_telemetry",
    "sync_dataset_to_gcs",
]
