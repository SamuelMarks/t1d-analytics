"""CLI argument parser construction and strongly-typed option models for T1D Analytics."""

import argparse
from typing import List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class ManifestStageDict(TypedDict):
    """Metadata for an exported synthetic training stage."""

    rows: int
    files: List[str]


class CliNamespace(argparse.Namespace):
    """Strongly-typed argparse Namespace with explicit default attributes for all CLI subcommands."""

    lang: Optional[str] = None
    command: str = ""
    url: str = "https://public.t1d.org/datasets/diabetes"
    output: str = "./data"
    concurrency: int = 4
    token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    headers: Optional[str] = None
    auth_plugin: Optional[str] = None
    data_dir: str = "./data"
    db: str = "t1d_analytics.duckdb"
    prefix_subdirs: bool = False
    include_sas: bool = False
    include_spss: bool = False
    model: str = "gemma4"
    provider: str = "ollama"
    num_pairs: int = 5
    split_ratios: Optional[str] = None
    test_set: str = ""
    allow_empty: bool = False
    min_tables: int = 1
    min_rows: int = 1
    json: bool = False
    sessions_db: Optional[str] = None
    overwrite: bool = False
    max_decompressed_gb: float = 200.0
    tabular_only: bool = False
    table: str = "sft_data"
    format: str = "jsonl"
    output_file: str = ""
    action: str = "check"
    stage: str = "sft"
    output_dir: Optional[str] = None
    config: Optional[str] = None
    interval: float = 2.0
    iterations: int = 0
    samples_per_table: int = 5
    export_formats: str = "jsonl,parquet"
    backend: str = "local-cpu"
    use_lora: bool = False
    quantization: Optional[str] = None
    execute_training: bool = False
    split_validation: bool = False
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    checkpoint_interval: int = 500
    eval_steps: int = 100
    dry_run: bool = False
    gcs_bucket: str = "gs://t1d-analytics-artifacts"
    tpu_zone: str = "us-central2-b"
    tpu_type: str = "v4-8"
    tpu_slices: int = 1


class CliOptionsModel(BaseModel):
    """Strongly-typed immutable configuration representing parsed CLI command-line options."""

    model_config = ConfigDict(from_attributes=True, extra="ignore", frozen=True)

    lang: Optional[str] = None
    command: Optional[str] = None
    url: str = "https://public.t1d.org/datasets/diabetes"
    output: Optional[str] = None
    concurrency: int = 4
    token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    headers: Optional[str] = None
    auth_plugin: Optional[str] = None
    data_dir: str = "./data"
    db: str = "t1d_analytics.duckdb"
    prefix_subdirs: bool = False
    include_sas: bool = False
    include_spss: bool = False
    model: str = "gemma4"
    provider: str = "ollama"
    num_pairs: int = 5
    split_ratios: Optional[str] = None
    test_set: str = ""
    allow_empty: bool = False
    min_tables: int = 1
    min_rows: int = 1
    json_output: bool = Field(
        default=False,
        validation_alias=AliasChoices("json_output", "json"),
    )
    sessions_db: Optional[str] = None
    overwrite: bool = False
    max_decompressed_gb: float = 200.0
    tabular_only: bool = False
    table: str = "sft_data"
    format: str = "jsonl"
    output_file: str = ""
    action: str = "check"
    stage: str = "sft"
    output_dir: Optional[str] = None
    config: Optional[str] = None
    interval: float = 2.0
    iterations: int = 0
    samples_per_table: int = 5
    export_formats: str = "jsonl,parquet"
    backend: str = "local-cpu"
    use_lora: bool = False
    quantization: Optional[str] = None
    execute_training: bool = False
    split_validation: bool = False
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    checkpoint_interval: int = 500
    eval_steps: int = 100
    dry_run: bool = False
    gcs_bucket: str = "gs://t1d-analytics-artifacts"
    tpu_zone: str = "us-central2-b"
    tpu_type: str = "v4-8"
    tpu_slices: int = 1


def parse_cli_options(args: argparse.Namespace) -> CliOptionsModel:
    """
    Convert an `argparse.Namespace` into a strongly-typed `CliOptionsModel`.

    Args:
    ----
        args: Parsed CLI namespace.

    Returns:
    -------
        CliOptionsModel: Immutable CLI options structure.

    """
    return CliOptionsModel.model_validate(args)


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Build and return the root ArgumentParser for the T1D Analytics CLI.

    Returns
    -------
        argparse.ArgumentParser: Configured CLI argument parser with all subcommands.

    """
    parser = argparse.ArgumentParser(
        description="Downloader and analytics tool for T1D public datasets."
    )
    parser.add_argument(
        "-l",
        "--lang",
        choices=["en", "ja", "ar", "he"],
        default=None,
        help="Language code for CLI localization (en, ja, ar, he).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    download_parser = subparsers.add_parser(
        "download", help="Download datasets from the T1D public repository."
    )
    download_parser.add_argument(
        "-u",
        "--url",
        default="https://public.t1d.org/datasets/diabetes",
        help="The URL of the T1D dataset page to parse.",
    )
    download_parser.add_argument(
        "-o",
        "--output",
        default="./data",
        help="The directory to save downloaded datasets.",
    )
    download_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent downloads (default: 4).",
    )
    download_parser.add_argument(
        "--token",
        help="Bearer token for authenticated downloads (or set T1D_AUTH_TOKEN).",
    )
    download_parser.add_argument(
        "--user",
        help="Username for HTTP Basic Authentication.",
    )
    download_parser.add_argument(
        "--password",
        help="Password for HTTP Basic Authentication.",
    )
    download_parser.add_argument(
        "--headers",
        help="Path to JSON or key-value file containing custom HTTP headers.",
    )
    download_parser.add_argument(
        "--auth-plugin",
        help="Custom authentication plugin in module:function format (e.g. auth:get_token).",
    )

    extract_parser = subparsers.add_parser(
        "extract", help="Extract downloaded zip files."
    )
    extract_parser.add_argument(
        "-d",
        "--data-dir",
        default="./data",
        help="The directory containing downloaded zip files.",
    )
    extract_parser.add_argument(
        "--max-decompressed-gb",
        type=float,
        default=200.0,
        help="Maximum cumulative decompressed size in GiB across archives (0 for unlimited, default: 200.0).",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-extraction of archives even if target files already exist.",
    )
    extract_parser.add_argument(
        "--tabular-only",
        action="store_true",
        help="Only extract tabular clinical data files and nested archives, skipping PDFs and docs.",
    )

    load_parser = subparsers.add_parser(
        "load", help="Parse and load CSV data into a DuckDB file."
    )
    load_parser.add_argument(
        "-d",
        "--data-dir",
        default="./data",
        help="The directory containing extracted CSV datasets.",
    )
    load_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    load_parser.add_argument(
        "--prefix-subdirs",
        action="store_true",
        help="Prefix table names with parent subdirectories to prevent collisions.",
    )
    load_parser.add_argument(
        "--include-sas",
        action="store_true",
        help="Include SAS clinical files (.sas7bdat, .xpt) during ingestion.",
    )
    load_parser.add_argument(
        "--include-spss",
        action="store_true",
        help="Include SPSS clinical files (.sav) during ingestion.",
    )

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Inspect ingestion manifest and audit history in the database.",
    )
    manifest_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )

    query_parser = subparsers.add_parser(
        "query", help="Open a REPL to query the populated DuckDB database."
    )
    query_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    query_parser.add_argument(
        "--model",
        default="gemma4",
        help="The model name (e.g. 'sql', 'hf.co/google/gemma-4-E4B', 'openai/gpt-4o') for query execution/translation.",
    )
    query_parser.add_argument(
        "--provider",
        default="ollama",
        choices=["sql", "ollama", "huggingface", "openai", "anthropic", "google"],
        help="The LLM/SQL provider to use for query translation (default: 'ollama').",
    )

    generate_parser = subparsers.add_parser(
        "generate-training-data",
        help="Generate synthetic Text-to-SQL training pairs from the database schema.",
    )
    generate_parser.add_argument(
        "--db",
        type=str,
        default="t1d_analytics.duckdb",
        help="Path to the target DuckDB file.",
    )
    generate_parser.add_argument(
        "--num-pairs",
        type=int,
        required=True,
        help="The number of synthetic SFT/DPO pairs to generate per table.",
    )
    generate_parser.add_argument(
        "--model",
        type=str,
        default="gemma4",
        help="The local Ollama model to use for generation.",
    )
    generate_parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        help="The LLM provider for any-llm (default: 'ollama').",
    )
    generate_parser.add_argument(
        "--split-ratios",
        type=str,
        help="Customizable train,val,test ratios (e.g. '0.8,0.1,0.1').",
    )

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate fine-tuned Text-to-SQL model accuracy against holdout benchmark questions.",
    )
    eval_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    eval_parser.add_argument(
        "--test-set",
        required=True,
        help="Path to JSON or JSONL file containing test benchmark questions.",
    )
    eval_parser.add_argument(
        "--model",
        default="gemma4",
        help="The LLM model to evaluate.",
    )
    eval_parser.add_argument(
        "--output",
        help="Optional path to write the markdown or CSV evaluation report.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect database configuration, initial data, and LLM readiness.",
    )
    doctor_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    doctor_parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 even if database has 0 tables or 0 rows.",
    )
    doctor_parser.add_argument(
        "--min-tables",
        type=int,
        default=1,
        help="Minimum number of clinical tables required (default: 1).",
    )
    doctor_parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum total row count required across tables (default: 1).",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit system health report as a JSON object.",
    )

    seed_parser = subparsers.add_parser(
        "seed",
        help="Seed baseline T1D clinical datasets into data directory and load into DuckDB.",
    )
    seed_parser.add_argument(
        "-d",
        "--data-dir",
        default="./data",
        help="Directory to write baseline clinical CSVs and ingest datasets from.",
    )
    seed_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    seed_parser.add_argument(
        "--sessions-db",
        default=None,
        help="Optional path to the chat sessions DuckDB database file.",
    )
    seed_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing baseline clinical CSV files.",
    )

    export_parser = subparsers.add_parser(
        "export-training-data",
        help="Export generated training data from DuckDB to JSONL or Parquet.",
    )
    export_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    export_parser.add_argument(
        "--table",
        default="sft_data",
        choices=["sft_data", "dpo_data", "pretrain_data"],
        help="The database table containing training data to export.",
    )
    export_parser.add_argument(
        "--format",
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Output export format.",
    )
    export_parser.add_argument(
        "--output-file",
        required=True,
        help="Destination path for exported file.",
    )

    generic_export_parser = subparsers.add_parser(
        "export",
        help="Export any database table to Excel (.xlsx), Parquet, or CSV format.",
    )
    generic_export_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    generic_export_parser.add_argument(
        "--table",
        required=True,
        help="The database table to export.",
    )
    generic_export_parser.add_argument(
        "--format",
        default="excel",
        choices=["excel", "parquet", "csv"],
        help="Output export format (excel, parquet, csv).",
    )
    generic_export_parser.add_argument(
        "--output-file",
        required=True,
        help="Destination path for exported file.",
    )

    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile a database table to assess data quality and completeness.",
    )
    profile_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    profile_parser.add_argument(
        "--table",
        required=True,
        help="Name of the table to profile.",
    )

    bridge_parser = subparsers.add_parser(
        "bridge-gemma-sql",
        help="Bridge interface connecting T1D datasets to external gemma-4-sql training toolchains.",
    )
    bridge_parser.add_argument(
        "action",
        choices=["check", "etl", "train"],
        help="Action to perform: 'check' installation, 'etl' dataset ingestion, or 'train' model.",
    )
    bridge_parser.add_argument(
        "--stage",
        choices=["pretrain", "sft", "posttrain"],
        default="sft",
        help="Pipeline stage: 'pretrain', 'sft', or 'posttrain' (default: 'sft').",
    )
    bridge_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to DuckDB database file for ETL ingestion.",
    )
    bridge_parser.add_argument(
        "--table",
        default="sft_data",
        help="DuckDB table to ingest during ETL.",
    )
    bridge_parser.add_argument(
        "--output-dir",
        help="Destination directory for output dataset shards.",
    )
    bridge_parser.add_argument(
        "--config",
        help="Path to YAML training configuration file.",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Continuously monitor a directory for new tabular files and ingest them.",
    )
    watch_parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory to watch for incoming data files.",
    )
    watch_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds between file system checks.",
    )
    watch_parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of iterations to run (0 for infinite loop).",
    )
    watch_parser.add_argument(
        "--prefix-subdirs",
        action="store_true",
        help="Prefix table names with subdirectory name.",
    )
    watch_parser.add_argument(
        "--include-sas",
        action="store_true",
        help="Include SAS dataset files (*.sas7bdat, *.xpt) in watch directory discovery.",
    )
    watch_parser.add_argument(
        "--include-spss",
        action="store_true",
        help="Include SPSS dataset files (*.sav) in watch directory discovery.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Orchestrate end-to-end dataset schema extraction, synthetic pair generation, and export.",
    )
    train_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )
    train_parser.add_argument(
        "--output-dir",
        default="training_output",
        help="Output directory for generated dataset files.",
    )
    train_parser.add_argument(
        "--samples-per-table",
        type=int,
        default=5,
        help="Number of question-SQL pairs to generate per table.",
    )
    train_parser.add_argument(
        "--provider",
        default="ollama",
        help="LLM provider to use for generation (e.g., 'ollama').",
    )
    train_parser.add_argument(
        "--model",
        default="gemma4",
        help="LLM model identifier to use.",
    )
    train_parser.add_argument(
        "--export-formats",
        default="jsonl,parquet",
        help="Comma-separated list of formats to export (e.g., 'jsonl,parquet').",
    )
    train_parser.add_argument(
        "--backend",
        default="local-cpu",
        choices=[
            "local-cpu",
            "local-gpu",
            "remote-tpu-maxtext",
            "gemma-4-sql",
            "huggingface",
        ],
        help="Target training backend execution environment (default: 'local-cpu').",
    )
    train_parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Attach LoRA parameter-efficient fine-tuning adapters.",
    )
    train_parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit"],
        default=None,
        help="Model weight quantization mode ('4bit' or '8bit').",
    )
    train_parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Trigger downstream training execution runner after data export.",
    )
    train_parser.add_argument(
        "--split-validation",
        action="store_true",
        help="Validate that train, validation, and test splits have no prompt leakage.",
    )
    train_parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=2e-5,
        help="Learning rate for model fine-tuning (default: 2e-5).",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4).",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of full training epochs (default: 3).",
    )
    train_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Linear warmup steps (default: 50).",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="L2 weight decay factor (default: 0.01).",
    )
    train_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length context (default: 2048).",
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Steps between checkpoint snapshots (default: 500).",
    )
    train_parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Steps between evaluation passes (default: 100).",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate training pipeline configuration without running gradient updates.",
    )
    train_parser.add_argument(
        "--gcs-bucket",
        default="gs://t1d-analytics-artifacts",
        help="GCS bucket URI for remote TPU training runs (default: 'gs://t1d-analytics-artifacts').",
    )
    train_parser.add_argument(
        "--tpu-zone",
        default="us-central2-b",
        help="Compute zone for Cloud TPU allocation (default: 'us-central2-b').",
    )
    train_parser.add_argument(
        "--tpu-type",
        default="v4-8",
        choices=["v4-8", "v4-16", "v4-32", "v5litepod-8", "v5litepod-64", "v6e-64"],
        help="Cloud TPU hardware accelerator specification (default: 'v4-8').",
    )
    train_parser.add_argument(
        "--tpu-slices",
        type=int,
        default=1,
        help="Number of TPU slices for multi-slice orchestration (default: 1).",
    )
    return parser
