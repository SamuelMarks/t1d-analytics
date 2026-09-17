"""Tests for the models."""

import pytest
from t1d_analytics.models import (
    DatasetInfo,
    GrainDatasetConfig,
    JaxFlaxModelConfig,
    MaxTextConfig,
    TpuAcceleratorType,
    TpuSchedulingType,
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)


def test_dataset_info_creation() -> None:
    """Test that DatasetInfo can be instantiated correctly."""
    info = DatasetInfo(protocol="Test", dataset_url="http://a", document_url=None)
    assert info.protocol == "Test"
    assert info.dataset_url == "http://a"
    assert info.document_url is None
    assert info.expected_sha256 is None


def test_training_backend_enum() -> None:
    """Test TrainingBackend enum values and string conversion."""
    assert TrainingBackend.LOCAL_CPU.value == "local-cpu"
    assert TrainingBackend.LOCAL_GPU.value == "local-gpu"
    assert TrainingBackend.REMOTE_TPU_MAXTEXT.value == "remote-tpu-maxtext"
    assert TrainingBackend.GEMMA_4_SQL.value == "gemma-4-sql"
    assert TrainingBackend.HUGGINGFACE.value == "huggingface"


def test_training_hyperparameters_defaults_and_custom() -> None:
    """Test TrainingHyperparameters default values and custom overrides."""
    params = TrainingHyperparameters()
    assert params.learning_rate == 2e-5
    assert params.batch_size == 4
    assert params.num_epochs == 3
    assert params.warmup_steps == 50
    assert params.weight_decay == 0.01
    assert params.max_seq_length == 2048
    assert params.checkpoint_interval == 500
    assert params.eval_steps == 100
    assert params.gradient_accumulation_steps == 1
    assert params.max_grad_norm == 1.0

    custom = TrainingHyperparameters(
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=5,
        warmup_steps=100,
        weight_decay=0.05,
        max_seq_length=4096,
        checkpoint_interval=250,
        eval_steps=50,
        gradient_accumulation_steps=2,
        max_grad_norm=0.5,
    )
    assert custom.learning_rate == 1e-4
    assert custom.batch_size == 8
    assert custom.num_epochs == 5
    assert custom.checkpoint_interval == 250
    assert custom.eval_steps == 50
    assert custom.gradient_accumulation_steps == 2
    assert custom.max_grad_norm == 0.5


def test_training_job_config_creation() -> None:
    """Test TrainingJobConfig instantiation and field assignment."""
    maxtext_cfg = MaxTextConfig(
        project_id="test-proj",
        zone="us-central2-b",
        accelerator_type=TpuAcceleratorType.V4_8,
    )
    config = TrainingJobConfig(
        model_name="gemma-4",
        backend=TrainingBackend.REMOTE_TPU_MAXTEXT,
        dataset_path="./data.parquet",
        output_dir="./out",
        dry_run=True,
        maxtext_config=maxtext_cfg,
    )
    assert config.model_name == "gemma-4"
    assert config.backend == TrainingBackend.REMOTE_TPU_MAXTEXT
    assert config.dataset_path == "./data.parquet"
    assert config.output_dir == "./out"
    assert config.hyperparameters is None
    assert config.dry_run is True
    assert config.maxtext_config == maxtext_cfg
    assert config.use_lora is False
    assert config.quantization is None
    assert config.validation_dataset_path is None

    config_custom = TrainingJobConfig(
        model_name="gemma-4",
        use_lora=True,
        quantization="4bit",
        validation_dataset_path="./val.parquet",
    )
    assert config_custom.use_lora is True
    assert config_custom.quantization == "4bit"
    assert config_custom.validation_dataset_path == "./val.parquet"


def test_maxtext_config_validation() -> None:
    """Test MaxTextConfig TFRC alignment validation for all zones and error branches."""
    cfg = MaxTextConfig(
        project_id="test-proj",
        zone="us-central2-b",
        accelerator_type=TpuAcceleratorType.V4_8,
        scheduling_type=TpuSchedulingType.SPOT,
    )
    assert cfg.validate_tfrc_alignment() is True

    # Valid europe-west4-b spot
    cfg_eu = MaxTextConfig(
        project_id="test-proj",
        zone="europe-west4-b",
        accelerator_type=TpuAcceleratorType.V5LITEPOD_8,
        scheduling_type=TpuSchedulingType.SPOT,
    )
    assert cfg_eu.validate_tfrc_alignment() is True

    # Valid us-central1-a spot
    cfg_us = MaxTextConfig(
        project_id="test-proj",
        zone="us-central1-a",
        accelerator_type=TpuAcceleratorType.V5LITEPOD_64,
        scheduling_type=TpuSchedulingType.SPOT,
    )
    assert cfg_us.validate_tfrc_alignment() is True

    # Valid europe-west4-a spot v6e
    cfg_v6e = MaxTextConfig(
        project_id="test-proj",
        zone="europe-west4-a",
        accelerator_type=TpuAcceleratorType.V6E_64,
        scheduling_type=TpuSchedulingType.SPOT,
    )
    assert cfg_v6e.validate_tfrc_alignment() is True

    # Valid us-east1-d spot v6e
    cfg_v6e_us = MaxTextConfig(
        project_id="test-proj",
        zone="us-east1-d",
        accelerator_type=TpuAcceleratorType.V6E_64,
        scheduling_type=TpuSchedulingType.SPOT,
    )
    assert cfg_v6e_us.validate_tfrc_alignment() is True

    # Invalid zone
    with pytest.raises(ValueError, match="not an authorized TFRC allocation zone"):
        MaxTextConfig(
            project_id="test-proj", zone="asia-east1-a"
        ).validate_tfrc_alignment()

    # Invalid accelerator in zone
    with pytest.raises(ValueError, match="is not authorized in zone"):
        MaxTextConfig(
            project_id="test-proj",
            zone="us-central2-b",
            accelerator_type=TpuAcceleratorType.V6E_64,
        ).validate_tfrc_alignment()

    # Invalid scheduling type in zone (e.g. on-demand in v5e spot-only zone)
    with pytest.raises(ValueError, match="is not authorized in zone"):
        MaxTextConfig(
            project_id="test-proj",
            zone="europe-west4-b",
            accelerator_type=TpuAcceleratorType.V5LITEPOD_8,
            scheduling_type=TpuSchedulingType.ON_DEMAND,
        ).validate_tfrc_alignment()


def test_jax_flax_model_config() -> None:
    """Test JaxFlaxModelConfig defaults, overrides, and to_dict."""
    cfg = JaxFlaxModelConfig()
    d = cfg.to_dict()
    assert d["model_name"] == "gemma-7b"
    assert d["dtype"] == "bfloat16"
    assert d["scan_layers"] is True
    assert d["ici_fsdp_parallelism"] == 1
    assert d["remat_policy"] == "minimal"

    custom = JaxFlaxModelConfig(
        model_name="gemma-2b",
        dtype="float32",
        scan_layers=False,
        ici_fsdp_parallelism=4,
        ici_tensor_parallelism=2,
        dcn_data_parallelism=2,
        remat_policy="full",
    )
    d_custom = custom.to_dict()
    assert d_custom["model_name"] == "gemma-2b"
    assert d_custom["dtype"] == "float32"
    assert d_custom["scan_layers"] is False
    assert d_custom["ici_fsdp_parallelism"] == 4
    assert d_custom["remat_policy"] == "full"


def test_grain_dataset_config() -> None:
    """Test GrainDatasetConfig defaults, overrides, and to_dict."""
    cfg = GrainDatasetConfig(dataset_path="gs://bucket/data.parquet")
    d = cfg.to_dict()
    assert d["dataset_type"] == "grain"
    assert d["grain_train_files"] == "gs://bucket/data.parquet"
    assert d["grain_worker_count"] == 8
    assert d["global_batch_size"] == 32
    assert d["max_target_length"] == 2048
    assert d["seed"] == 42


def test_maxtext_config_build_full_parameters() -> None:
    """Test MaxTextConfig.build_full_maxtext_parameters with jax and grain configurations."""
    jax_cfg = JaxFlaxModelConfig(model_name="gemma-7b")
    grain_cfg = GrainDatasetConfig(dataset_path="gs://my-bucket/train.parquet")
    cfg = MaxTextConfig(
        project_id="test-proj",
        zone="us-central2-b",
        bucket_url="gs://my-bucket",
        jax_config=jax_cfg,
        grain_config=grain_cfg,
    )
    params = cfg.build_full_maxtext_parameters()
    assert params["project_id"] == "test-proj"
    assert params["zone"] == "us-central2-b"
    assert params["dataset_type"] == "grain"
    assert params["dtype"] == "bfloat16"
    assert params["hardware"] == "v4-8"
    assert params["base_output_directory"] == "gs://my-bucket/models/gemma-7b"

    # Test without jax and grain config
    cfg_minimal = MaxTextConfig(project_id="p1")
    params_min = cfg_minimal.build_full_maxtext_parameters()
    assert params_min["project_id"] == "p1"
    assert "dtype" not in params_min
