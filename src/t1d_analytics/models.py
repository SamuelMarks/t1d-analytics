"""Data models for T1D analytics, downloader, and training orchestration."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set


@dataclass
class DatasetInfo:
    """
    Information about a dataset parsed from the website.

    Attributes
    ----------
        protocol: Clinical protocol name or identifier.
        dataset_url: Remote URL to download tabular dataset.
        document_url: Remote URL for protocol documentation or study design.
        expected_sha256: Optional SHA-256 checksum for download integrity verification.
        postback_target: Optional ASP.NET __EVENTTARGET control ID for simulated postbacks.
        postback_argument: Optional ASP.NET __EVENTARGUMENT parameter for simulated postbacks.

    """

    protocol: str
    dataset_url: Optional[str]
    document_url: Optional[str]
    expected_sha256: Optional[str] = None
    postback_target: Optional[str] = None
    postback_argument: Optional[str] = None


class TrainingBackend(str, Enum):
    """Supported execution backends for model training."""

    LOCAL_CPU = "local-cpu"
    LOCAL_GPU = "local-gpu"
    REMOTE_TPU_MAXTEXT = "remote-tpu-maxtext"
    GEMMA_4_SQL = "gemma-4-sql"
    HUGGINGFACE = "huggingface"


class TpuAcceleratorType(str, Enum):
    """Supported Cloud TPU hardware accelerator configurations."""

    V4_8 = "v4-8"
    V4_16 = "v4-16"
    V4_32 = "v4-32"
    V5LITEPOD_8 = "v5litepod-8"
    V5LITEPOD_64 = "v5litepod-64"
    V6E_64 = "v6e-64"


class TpuSchedulingType(str, Enum):
    """Scheduling strategy for Cloud TPU allocations."""

    SPOT = "spot"
    ON_DEMAND = "on-demand"


@dataclass
class JaxFlaxModelConfig:
    """
    Structured JAX/Flax architecture and XLA compilation parameters for MaxText.

    Attributes
    ----------
        model_name: Base architecture identifier (e.g., 'gemma-7b', 'gemma-2b').
        dtype: Numerical precision for weights and activations ('bfloat16', 'float32').
        scan_layers: Whether to apply JAX scan optimization across transformer layers.
        ici_fsdp_parallelism: Fully Sharded Data Parallel dimension along ICI mesh.
        ici_tensor_parallelism: Megatron tensor parallel dimension along ICI mesh.
        dcn_data_parallelism: Data parallel dimension across Data Center Network slices.
        remat_policy: Activation gradient rematerialization strategy ('minimal', 'full').

    """

    model_name: str = "gemma-7b"
    dtype: str = "bfloat16"
    scan_layers: bool = True
    ici_fsdp_parallelism: int = 1
    ici_tensor_parallelism: int = 1
    dcn_data_parallelism: int = 1
    remat_policy: str = "minimal"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert JAX/Flax configuration to dictionary mapping.

        Returns
        -------
            Dict[str, Any]: Key-value configuration mapping for MaxText.

        """
        return {
            "model_name": self.model_name,
            "dtype": self.dtype,
            "scan_layers": self.scan_layers,
            "ici_fsdp_parallelism": self.ici_fsdp_parallelism,
            "ici_tensor_parallelism": self.ici_tensor_parallelism,
            "dcn_data_parallelism": self.dcn_data_parallelism,
            "remat_policy": self.remat_policy,
        }


@dataclass
class GrainDatasetConfig:
    """
    Configuration for Google Grain dataset streaming pipeline to Cloud TPU nodes.

    Attributes
    ----------
        dataset_path: Source filepath or GCS URI to DuckDB parquet records.
        worker_count: Number of independent data loader worker threads per host.
        batch_size: Global batch size across all TPU cores.
        max_seq_length: Maximum token length for packing and padding.
        shuffle_buffer_size: Number of records to buffer for in-memory shuffling.
        seed: Random seed for reproducible dataset sampling.

    """

    dataset_path: str
    worker_count: int = 8
    batch_size: int = 32
    max_seq_length: int = 2048
    shuffle_buffer_size: int = 10000
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Grain configuration to dictionary mapping.

        Returns
        -------
            Dict[str, Any]: MaxText dataset dictionary structure.

        """
        return {
            "dataset_type": "grain",
            "grain_train_files": self.dataset_path,
            "grain_worker_count": self.worker_count,
            "global_batch_size": self.batch_size,
            "max_target_length": self.max_seq_length,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "seed": self.seed,
        }


@dataclass
class MaxTextConfig:
    """
    Structured configuration for distributed MaxText training on Google Cloud TPUs.

    Attributes
    ----------
        project_id: Google Cloud project identifier.
        zone: Compute Engine zone where TPU resources reside.
        bucket_url: Google Cloud Storage bucket URL (gs://<bucket>).
        accelerator_type: TPU hardware accelerator specification.
        scheduling_type: Spot or on-demand instance scheduling.
        num_slices: Number of TPU slices for multi-slice distributed jobs.
        model_name: Base model architecture configuration in MaxText.
        base_output_directory: Optional GCS directory for checkpoints and tensorboard logs.
        jax_config: Optional detailed JAX/Flax model architecture parameters.
        grain_config: Optional Google Grain streaming dataset pipeline parameters.
        max_preemption_retries: Maximum automated retry attempts when spot instance is preempted.

    """

    project_id: str
    zone: str = "us-central2-b"
    bucket_url: str = "gs://t1d-analytics-artifacts"
    accelerator_type: TpuAcceleratorType = TpuAcceleratorType.V4_8
    scheduling_type: TpuSchedulingType = TpuSchedulingType.SPOT
    num_slices: int = 1
    model_name: str = "gemma-7b"
    base_output_directory: Optional[str] = None
    jax_config: Optional[JaxFlaxModelConfig] = None
    grain_config: Optional[GrainDatasetConfig] = None
    max_preemption_retries: int = 3

    def build_full_maxtext_parameters(self) -> Dict[str, Any]:
        """
        Assemble comprehensive MaxText YAML dictionary combining cluster and model parameters.

        Returns
        -------
            Dict[str, Any]: Full MaxText parameter dictionary ready for export.

        """
        params: Dict[str, Any] = {
            "project_id": self.project_id,
            "zone": self.zone,
            "base_output_directory": self.base_output_directory
            or f"{self.bucket_url}/models/{self.model_name}",
            "hardware": self.accelerator_type.value,
            "num_slices": self.num_slices,
            "model_name": self.model_name,
        }
        if self.jax_config:
            params.update(self.jax_config.to_dict())
        if self.grain_config:
            params.update(self.grain_config.to_dict())
        return params

    def validate_tfrc_alignment(self) -> bool:
        """
        Validate that the configuration aligns with TPU Research Cloud (TFRC) authorized capacity.

        Validates against:
        - Cloud TPU v4 (us-central2-b): v4-8, v4-16, v4-32 (spot or on-demand)
        - Cloud TPU v5e (europe-west4-b & us-central1-a): v5litepod-8, v5litepod-64 (spot)
        - Cloud TPU v6e (europe-west4-a & us-east1-d): v6e-64 (spot)

        Returns
        -------
            bool: True if configuration matches authorized region and hardware slice.

        Raises
        ------
            ValueError: If zone and accelerator type combination violates TFRC grant.

        """
        valid_allocations: Dict[str, Dict[str, Set[Any]]] = {
            "us-central2-b": {
                "accelerators": {
                    TpuAcceleratorType.V4_8,
                    TpuAcceleratorType.V4_16,
                    TpuAcceleratorType.V4_32,
                },
                "schedulings": {TpuSchedulingType.SPOT, TpuSchedulingType.ON_DEMAND},
            },
            "europe-west4-b": {
                "accelerators": {
                    TpuAcceleratorType.V5LITEPOD_8,
                    TpuAcceleratorType.V5LITEPOD_64,
                },
                "schedulings": {TpuSchedulingType.SPOT},
            },
            "us-central1-a": {
                "accelerators": {
                    TpuAcceleratorType.V5LITEPOD_8,
                    TpuAcceleratorType.V5LITEPOD_64,
                },
                "schedulings": {TpuSchedulingType.SPOT},
            },
            "europe-west4-a": {
                "accelerators": {TpuAcceleratorType.V6E_64},
                "schedulings": {TpuSchedulingType.SPOT},
            },
            "us-east1-d": {
                "accelerators": {TpuAcceleratorType.V6E_64},
                "schedulings": {TpuSchedulingType.SPOT},
            },
        }

        if self.zone not in valid_allocations:
            raise ValueError(
                f"Zone '{self.zone}' is not an authorized TFRC allocation zone."
            )

        spec = valid_allocations[self.zone]
        if self.accelerator_type not in spec["accelerators"]:
            raise ValueError(
                f"Accelerator '{self.accelerator_type.value}' is not authorized in zone '{self.zone}'."
            )

        if self.scheduling_type not in spec["schedulings"]:
            raise ValueError(
                f"Scheduling '{self.scheduling_type.value}' is not authorized in zone '{self.zone}'."
            )

        return True


@dataclass
class TrainingHyperparameters:
    """
    Hyperparameters for fine-tuning or pretraining runs.

    Attributes
    ----------
        learning_rate: Learning rate for gradient descent.
        batch_size: Number of training samples per gradient batch.
        num_epochs: Total number of full passes through the training set.
        warmup_steps: Number of initial linear learning rate warmup steps.
        weight_decay: L2 regularization penalty factor.
        max_seq_length: Maximum sequence context token length.
        checkpoint_interval: Number of training steps between saving model checkpoints.
        eval_steps: Number of training steps between validation evaluations.
        gradient_accumulation_steps: Number of update steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for gradient clipping.

    """

    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    checkpoint_interval: int = 500
    eval_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class TrainingJobConfig:
    """
    Structured configuration for training orchestration jobs.

    Attributes
    ----------
        model_name: Identifier or path of the base neural model.
        backend: Execution target environment for model training.
        dataset_path: Filepath to the formatted training dataset.
        output_dir: Destination directory for model checkpoints and logs.
        hyperparameters: Optional hyperparameters configuration.
        dry_run: Whether to simulate training without executing gradient updates.
        maxtext_config: Optional configuration when running under remote TPU MaxText.
        use_lora: Whether to attach LoRA parameter-efficient fine-tuning adapters.
        quantization: Optional quantization mode ('4bit', '8bit', or None).
        validation_dataset_path: Optional filepath to validation dataset split.
        use_tiny_model: Whether to use lightweight 2-layer TinyCausalLM for unit tests.
        model_architecture: Architecture selection strategy ('auto', 'tiny', or 'transformer').

    """

    model_name: str
    backend: TrainingBackend = TrainingBackend.LOCAL_CPU
    dataset_path: str = "./training_data.parquet"
    output_dir: str = "./checkpoints"
    hyperparameters: Optional[TrainingHyperparameters] = None
    dry_run: bool = False
    maxtext_config: Optional[MaxTextConfig] = None
    use_lora: bool = False
    quantization: Optional[str] = None
    validation_dataset_path: Optional[str] = None
    use_tiny_model: bool = False
    model_architecture: str = "auto"


class MockEnvironmentController:
    """
    Centralized controller governing simulation and test mock flags.

    Enforces structured interrogation of environment variable overrides
    (e.g., T1D_MOCK_TPU, T1D_MOCK_GCS, T1D_MOCK_GPU, T1D_MOCK_GEMMA_SQL).
    """

    @staticmethod
    def is_tpu_mocked() -> bool:
        """
        Check if TPU execution and XPK workloads are mocked.

        Returns
        -------
            bool: True if T1D_MOCK_TPU environment variable is set to '1'.

        """
        import os

        return os.environ.get("T1D_MOCK_TPU") == "1"

    @staticmethod
    def is_gcs_mocked() -> bool:
        """
        Check if Google Cloud Storage operations are mocked.

        Returns
        -------
            bool: True if T1D_MOCK_GCS environment variable is set to '1'.

        """
        import os

        return os.environ.get("T1D_MOCK_GCS") == "1"

    @staticmethod
    def is_gcs_failure_simulated() -> bool:
        """
        Check if simulated GCS operation failure is requested.

        Returns
        -------
            bool: True if T1D_MOCK_GCS_FAIL environment variable is set to '1'.

        """
        import os

        return os.environ.get("T1D_MOCK_GCS_FAIL") == "1"

    @staticmethod
    def is_gcs_corruption_simulated() -> bool:
        """
        Check if simulated GCS checksum corruption is requested.

        Returns
        -------
            bool: True if T1D_MOCK_GCS_CORRUPT environment variable is set to '1'.

        """
        import os

        return os.environ.get("T1D_MOCK_GCS_CORRUPT") == "1"

    @staticmethod
    def is_gpu_mocked() -> bool:
        """
        Check if GPU hardware availability is mocked.

        Returns
        -------
            bool: True if T1D_MOCK_GPU environment variable is set to '1'.

        """
        import os

        return os.environ.get("T1D_MOCK_GPU") == "1"

    @staticmethod
    def is_gemma_sql_mocked() -> bool:
        """
        Check if gemma-4-sql CLI and toolchain execution is mocked.

        Returns
        -------
            bool: True if T1D_MOCK_GEMMA_SQL environment variable is set to '1'.

        """
        import os

        return os.environ.get("T1D_MOCK_GEMMA_SQL") == "1"
