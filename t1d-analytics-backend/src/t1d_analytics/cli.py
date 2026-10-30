"""Command Line Interface for the T1D Analytics Suite."""

import argparse
import datetime
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import duckdb
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from t1d_analytics import (
    analytics,
    api_exports,
    diagnostics,
    gemma_bridge,
    seed,
    training_data,
)
from t1d_analytics.downloader import parse_headers_file, process_datasets
from t1d_analytics.errors import (
    AuthPluginError,
    CliValidationError,
    DownloadError,
)
from t1d_analytics.i18n import get_translator
from t1d_analytics.models import (
    MaxTextConfig,
    TpuAcceleratorType,
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.parser import fetch_html, parse_datasets


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

    # Download subcommand
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

    # Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract", help="Extract downloaded zip files."
    )
    extract_parser.add_argument(
        "-d",
        "--data-dir",
        default="./data",
        help="The directory containing downloaded zip files.",
    )

    # Load subcommand
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

    # Manifest subcommand
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Inspect ingestion manifest and audit history in the database.",
    )
    manifest_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
    )

    # Query subcommand
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
        help="The model name (or 'provider/model' e.g. openai/gpt-4o) for query translation.",
    )
    query_parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai", "anthropic", "google"],
        help="The LLM provider to use for query translation (default: 'ollama').",
    )

    # Generate Training Data subcommand
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

    # Evaluate subcommand
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

    # Doctor subcommand
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

    # Seed subcommand
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

    # Export Training Data subcommand
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

    # Generic Table Export subcommand
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

    # Profile subcommand
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

    # Gemma-4-SQL Bridge subcommand
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

    # Watch subcommand
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

    # Train subcommand
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


def main() -> None:
    """Execute the CLI application."""
    _ = get_translator()
    parser = build_cli_parser()
    args = parser.parse_args(namespace=CliNamespace())
    opts = parse_cli_options(args)

    if opts.lang:
        os.environ["LANG"] = opts.lang
        _ = get_translator(opts.lang)

    command_handlers: dict[str, Callable[[CliOptionsModel], None]] = {
        "download": handle_download,
        "extract": handle_extract,
        "load": handle_load,
        "seed": handle_seed,
        "manifest": handle_manifest,
        "query": handle_query,
        "generate-training-data": handle_generate_training_data,
        "export-training-data": handle_export_training_data,
        "export": handle_export,
        "evaluate": handle_evaluate,
        "doctor": handle_doctor,
        "profile": handle_profile,
        "watch": handle_watch,
        "train": handle_train,
        "bridge-gemma-sql": handle_bridge_gemma_sql,
    }

    try:
        handler = command_handlers.get(opts.command or "")
        if handler is not None:
            handler(opts)
    except Exception as e:
        print(_("An error occurred: {}", e), file=sys.stderr)
        sys.exit(1)


def load_auth_plugin(plugin_spec: str) -> Callable[[], str]:
    """
    Dynamically load an authentication plugin callable from a module specifier.

    Args:
    ----
        plugin_spec: Specifier formatted as 'module:function' or 'pkg.module:func'.

    Returns:
    -------
        Callable[[], str]: The loaded callable function returning a Bearer token or credentials.

    Raises:
    ------
        AuthPluginError: If format is invalid or target function cannot be found or invoked.

    """
    if ":" not in plugin_spec:
        raise AuthPluginError(
            f"Invalid auth plugin specifier '{plugin_spec}'. Expected 'module:function'."
        )
    mod_name, func_name = plugin_spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        raise AuthPluginError(
            f"Failed to import auth plugin module '{mod_name}': {e}"
        ) from e

    members = dict(inspect.getmembers(mod))
    if not func_name.isidentifier() or func_name not in members:
        raise AuthPluginError(
            f"Auth plugin module '{mod_name}' has no attribute '{func_name}'."
        )

    extracted = members[func_name]
    if not callable(extracted):
        raise AuthPluginError(
            f"Auth plugin attribute '{func_name}' in module '{mod_name}' is not callable."
        )

    def _invoke_plugin() -> str:
        """Invoke loaded auth plugin and return string token."""
        return str(extracted())

    return _invoke_plugin


def handle_download(opts: CliOptionsModel) -> None:
    """
    Handle the download subcommand.

    Args:
    ----
        opts: Validated CLI options model.

    Raises:
    ------
        DownloadError: If one or more dataset files fail to download.

    """
    _ = get_translator()
    print(_("Fetching HTML from {}...", opts.url))
    html = fetch_html(opts.url)

    print(_("Parsing datasets..."))
    datasets = parse_datasets(html, base_url=opts.url)
    print(_("Found {} protocols.", len(datasets)))

    if not datasets:
        print(_("No datasets found. Exiting."))
        return

    out_target = str(opts.output)
    print(_("Starting downloads to {}...", out_target))

    auth: Optional[tuple[str, str]] = None
    if opts.user and opts.password:
        auth = (opts.user, opts.password)

    custom_headers: Optional[dict[str, str]] = None
    if opts.headers:
        custom_headers = parse_headers_file(Path(opts.headers))

    token = opts.token
    auth_plugin_cb: Optional[Callable[[], str]] = None
    if opts.auth_plugin:
        auth_plugin_cb = load_auth_plugin(opts.auth_plugin)
        if not token:
            token = auth_plugin_cb()

    summary = process_datasets(
        datasets,
        out_target,
        concurrency=opts.concurrency,
        token=token,
        auth=auth,
        headers=custom_headers,
        page_url=opts.url,
        token_refresh_callback=auth_plugin_cb,
    )
    failed = summary.get("failed", 0) if isinstance(summary, dict) else 0
    if isinstance(failed, int) and failed > 0:
        raise DownloadError(
            _("Download encountered errors: {} file(s) failed.", failed)
        )
    print(_("Done!"))


def handle_extract(opts: CliOptionsModel) -> None:
    """
    Handle the extract subcommand.

    Args:
    ----
        opts: Validated CLI options model.

    """
    analytics.extract_zips(opts.data_dir)


def handle_load(opts: CliOptionsModel) -> None:
    """
    Handle the load subcommand.

    Args:
    ----
        opts: Validated CLI options model.

    """
    analytics.load_data_to_duckdb(
        opts.data_dir,
        opts.db,
        prefix_subdirs=opts.prefix_subdirs,
        include_sas=opts.include_sas,
        include_spss=opts.include_spss,
    )


def handle_seed(opts: CliOptionsModel) -> None:
    """
    Handle the seed subcommand to generate baseline clinical CSVs and populate DuckDB.

    Args:
    ----
        opts: Validated CLI options model containing data_dir, db, sessions_db, and overwrite.

    """
    table_counts = seed.seed_database(
        data_dir=opts.data_dir,
        db_path=opts.db,
        sessions_db_path=opts.sessions_db,
        overwrite_csvs=opts.overwrite,
    )
    total_rows = sum(table_counts.values())
    print(
        f"Seeded {len(table_counts)} clinical tables ({total_rows} total rows) into {opts.db}."
    )


def handle_manifest(opts: CliOptionsModel) -> None:
    """
    Handle the manifest subcommand.

    Args:
    ----
        opts: Validated CLI options model containing db path.

    """
    records = analytics.get_ingestion_manifest(opts.db)
    if not records:
        print("No ingestion manifest records found.")
        return

    print(
        f"{'Table Name':<25} | {'Row Count':<10} | {'File Size':<10} | {'Loaded At':<25} | Source File"
    )
    print("-" * 100)
    for r in records:
        print(
            f"{r['table_name']!s:<25} | {r['row_count']!s:<10} | {r['file_size']!s:<10} | {r['loaded_at']!s:<25} | {r['source_file']}"
        )


def handle_query(opts: CliOptionsModel) -> None:
    """
    Handle the query subcommand.

    Args:
    ----
        opts: Validated CLI options model containing db, model, and optional provider.

    """
    analytics.run_query_repl(opts.db, model=opts.model, provider=opts.provider)


def handle_generate_training_data(opts: CliOptionsModel) -> None:
    """
    Handle the generate-training-data subcommand.

    This function initializes a connection to DuckDB, creates the TrainingDataGenerator,
    and runs the generation pipeline for the specified number of pairs per table using
    the provided local LLM model and provider.

    Args:
    ----
        opts: Validated CLI options model containing db, num_pairs, model, provider, and split_ratios.

    """
    _ = get_translator()

    print(_("Connecting to DuckDB at {}...", opts.db))
    conn = duckdb.connect(opts.db)

    provider = opts.provider
    print(
        _(
            "Initializing TrainingDataGenerator with model '{}' (provider: '{}')...",
            opts.model,
            provider,
        )
    )
    generator = training_data.TrainingDataGenerator(conn, opts.model, provider=provider)
    try:
        ratios = None
        if opts.split_ratios:
            parts = [float(p.strip()) for p in opts.split_ratios.split(",")]
            if len(parts) == 3:
                ratios = (parts[0], parts[1], parts[2])

        # Extract schema
        schema = generator.extract_schema()

        # Generate pairs and write to db
        total_tables = len(schema)
        print(_("Found {} tables in the schema.", total_tables))

        for table_name, table_schema in schema.items():
            print(_("Generating {} pairs for table: {}...", opts.num_pairs, table_name))
            pairs = generator.generate_pairs(table_schema, opts.num_pairs)
            generator.write_to_db(pairs, split_ratios=ratios)
    finally:
        conn.close()

    print(_("Training data generation complete!"))


def handle_evaluate(opts: CliOptionsModel) -> None:
    """
    Handle the evaluate subcommand.

    Args:
    ----
        opts: Validated CLI options model containing db, test_set, model, and output.

    """
    test_file = Path(opts.test_set)
    if not test_file.exists():
        print(f"Test set file {test_file} does not exist.")
        return

    test_data: list[dict[str, str]] = []
    content = test_file.read_text().strip()
    if content.startswith("["):
        test_data = json.loads(content)
    else:
        for line in content.splitlines():
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    metrics = training_data.evaluate_text_to_sql(opts.db, test_data, model=opts.model)
    print(metrics["markdown_report"])

    if opts.output:
        out_p = Path(opts.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".csv":
            out_p.write_text(metrics["csv_report"])
        else:
            out_p.write_text(metrics["markdown_report"])
        print(f"Report saved to {out_p}")


def handle_export_training_data(opts: CliOptionsModel) -> None:
    """
    Handle exporting training data to JSONL or Parquet format.

    Args:
    ----
        opts: Validated CLI options model containing db, table, format, and output_file.

    """
    _ = get_translator()
    print(_("Connecting to DuckDB at {}...", opts.db))
    conn = duckdb.connect(opts.db, read_only=True)
    try:
        generator = training_data.TrainingDataGenerator(conn)
        print(_("Exporting table '{}' to {}...", opts.table, opts.output_file))
        if opts.format == "jsonl":
            generator.export_to_jsonl(opts.table, Path(opts.output_file))
        else:
            generator.export_to_parquet(opts.table, Path(opts.output_file))
    finally:
        conn.close()

    print(_("Export complete!"))


def handle_export(opts: CliOptionsModel) -> None:
    """
    Handle exporting database tables to Excel (.xlsx), Parquet, or CSV format.

    Args:
    ----
        opts: Validated CLI options model containing db, table, format, and output_file.

    """
    _ = get_translator(opts.lang)
    print(_("Connecting to DuckDB at {}...", opts.db))
    print(_("Exporting table '{}' to {}...", opts.table, opts.output_file))
    api_exports.export_table(
        db_path=opts.db,
        table_name=opts.table,
        output_path=opts.output_file,
        export_format=opts.format,
    )
    print(_("Export complete!"))


def handle_doctor(opts: CliOptionsModel) -> None:
    """
    Handle the doctor subcommand to inspect system health.

    Args:
    ----
        opts: Validated CLI options model containing db path and threshold flags.

    """
    health = diagnostics.get_system_health(db_path=opts.db)
    db = health.database
    ollama = health.ollama

    if opts.json_output:
        print(json.dumps(health.to_dict(), indent=2))
    else:
        print("=" * 60)
        print("T1D Analytics System Health Report")
        print("=" * 60)
        print(f"Overall Status: {health.status.upper()}")
        print("-" * 60)
        print(f"Database ({db.configured_path}):")
        print(f"  Connected: {db.connected}")
        print(f"  Status:    {db.status_code.value}")
        print(f"  Tables:    {db.table_count}")
        print(f"  Rows:      {db.total_rows}")
        print(f"  Message:   {db.message}")
        if db.remediation:
            print(f"  Action:    {db.remediation}")
        print("-" * 60)
        print("Ollama LLM Service:")
        print(f"  Online:    {ollama.accessible}")
        models_str = (
            ", ".join(ollama.available_models) if ollama.available_models else "None"
        )
        print(f"  Models:    {models_str}")
        print(f"  Message:   {ollama.message}")
        if ollama.remediation:
            print(f"  Action:    {ollama.remediation}")
        print("-" * 60)
        print("Cloud LLM Providers:")
        for prov_name, prov_stat in health.providers.items():
            print(f"  {prov_name.capitalize()}:")
            print(f"    Configured:    {prov_stat.configured}")
            print(f"    SDK Installed: {prov_stat.sdk_installed}")
            print(f"    API Key Set:   {prov_stat.api_key_set}")
            if prov_stat.remediation:
                print(f"    Action:        {prov_stat.remediation}")
        print("=" * 60)

    if not db.connected or db.status_code in (
        diagnostics.DatabaseStatusCode.MISSING_FILE,
        diagnostics.DatabaseStatusCode.EMPTY_FILE,
        diagnostics.DatabaseStatusCode.UNREADABLE,
    ):
        sys.exit(1)

    if not opts.allow_empty and (
        db.table_count < opts.min_tables or db.total_rows < opts.min_rows
    ):
        sys.exit(1)


def handle_profile(opts: CliOptionsModel) -> None:
    """
    Handle the profile subcommand to analyze data quality and completeness of a table.

    Args:
    ----
        opts: Validated CLI options model containing db path and table name.

    """
    report = analytics.profile_table(opts.db, opts.table)
    print("=" * 60)
    print(f"Table Profile: {report['table_name']}")
    print(f"Total Rows: {report['total_rows']} | Columns: {report['column_count']}")
    print("=" * 60)
    for col in report["columns"]:
        details = (
            f"{col['column_name']} ({col['data_type']}): nulls={col['null_count']} "
            f"({col['null_percentage']}%), distinct={col['distinct_count']}"
        )
        if "min" in col:
            details += f", min={col['min']}, max={col['max']}, mean={col['mean']}"
        print(f"  - {details}")
    print("=" * 60)


def handle_watch(opts: CliOptionsModel) -> None:
    """
    Handle the watch subcommand to monitor and incrementally ingest files into DuckDB.

    Args:
    ----
        opts: Validated CLI options model containing data_dir, db, interval, iterations, and prefix_subdirs.

    """
    data_path = Path(opts.data_dir)
    print(
        f"Watching directory '{data_path}' for changes (interval: {opts.interval}s)..."
    )
    seen_hashes: dict[str, str] = {}

    count = 0
    while True:
        count += 1
        exts = ["*.csv", "*.parquet", "*.txt", "*.csv.gz"]
        if opts.include_sas:
            exts.extend(["*.sas7bdat", "*.xpt"])
        if opts.include_spss:
            exts.append("*.sav")

        current_files: list[Path] = []
        if data_path.exists():
            for ext in exts:
                current_files.extend(data_path.rglob(ext))

        # Detect deleted files and prune manifest records
        current_paths_set = {str(f.resolve()) for f in current_files}
        deleted_files = [
            f_str for f_str in seen_hashes if f_str not in current_paths_set
        ]
        for del_f in deleted_files:
            del seen_hashes[del_f]
            print(f"Detected file deletion: '{del_f}'. Pruning manifest...")
            if Path(opts.db).exists():
                conn_del: Optional[duckdb.DuckDBPyConnection] = None
                try:
                    conn_del = duckdb.connect(opts.db)
                    del_name = Path(del_f).name
                    conn_del.execute(
                        "DELETE FROM _t1d_ingestion_manifest WHERE source_file = ? OR source_file = ?",
                        (del_f, del_name),
                    )
                except duckdb.Error as e:
                    print(
                        f"Warning: Failed to update manifest after deletion: {e}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"Warning: Unexpected error updating manifest: {e}",
                        file=sys.stderr,
                    )
                finally:
                    if conn_del is not None:
                        conn_del.close()

        changed = False
        for f in current_files:
            f_str = str(f.resolve())
            h = analytics.compute_file_hash(f)
            if seen_hashes.get(f_str) != h:
                seen_hashes[f_str] = h
                changed = True

        if changed:
            print(f"Detected file changes or new files in '{data_path}'. Ingesting...")
            analytics.load_data_to_duckdb(
                str(data_path),
                opts.db,
                prefix_subdirs=opts.prefix_subdirs,
            )
            print("Ingestion complete.")

        if opts.iterations > 0 and count >= opts.iterations:
            break
        time.sleep(opts.interval)


def validate_cli_hyperparameters(opts: CliOptionsModel) -> TrainingHyperparameters:
    """
    Validate CLI training hyperparameter bounds and return a `TrainingHyperparameters` instance.

    Args:
    ----
        opts: Validated CLI options model.

    Returns:
    -------
        TrainingHyperparameters: Validated hyperparameter configuration.

    Raises:
    ------
        CliValidationError: If any hyperparameter violates domain constraints.

    """
    if opts.learning_rate <= 0.0:
        raise CliValidationError("Learning rate must be positive.")
    if opts.batch_size < 1:
        raise CliValidationError("Batch size must be at least 1.")
    if opts.num_epochs < 1:
        raise CliValidationError("Number of epochs must be at least 1.")
    if opts.warmup_steps < 0:
        raise CliValidationError("Warmup steps cannot be negative.")
    if opts.weight_decay < 0.0:
        raise CliValidationError("Weight decay cannot be negative.")
    if opts.max_seq_length < 64:
        raise CliValidationError("Max sequence length must be at least 64.")

    return TrainingHyperparameters(
        learning_rate=opts.learning_rate,
        batch_size=opts.batch_size,
        num_epochs=opts.num_epochs,
        warmup_steps=opts.warmup_steps,
        weight_decay=opts.weight_decay,
        max_seq_length=opts.max_seq_length,
        checkpoint_interval=opts.checkpoint_interval,
        eval_steps=opts.eval_steps,
    )


def export_training_stages(
    conn: duckdb.DuckDBPyConnection,
    generator: training_data.TrainingDataGenerator,
    out_dir: Path,
    formats: List[str],
    split_val_flag: bool,
) -> Dict[str, ManifestStageDict]:
    """
    Export generated training tables from DuckDB into JSONL and/or Parquet files.

    Args:
    ----
        conn: Active DuckDB connection.
        generator: Initialized `TrainingDataGenerator`.
        out_dir: Target output directory.
        formats: Requested export formats (e.g., `['jsonl', 'parquet']`).
        split_val_flag: Whether stratified split tables should also be exported.

    Returns:
    -------
        Dict[str, ManifestStageDict]: Stage-to-metadata mapping for the training manifest.

    """
    import shutil

    stages_to_export = ["sft_data", "pretrain_data", "dpo_data"]
    if split_val_flag:
        stages_to_export.extend(["sft_data_train", "sft_data_val", "sft_data_test"])

    all_tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    manifest_stages: Dict[str, ManifestStageDict] = {}

    for stage in stages_to_export:
        if stage in all_tables:
            count_res = conn.execute(f'SELECT COUNT(*) FROM "{stage}"').fetchone()
            stage_count = int(count_res[0]) if count_res else 0
            manifest_stages[stage] = {"rows": stage_count, "files": []}

            if "jsonl" in formats:
                p = out_dir / f"{stage}.jsonl"
                print(f"Exporting JSONL to {p}...")
                generator.export_to_jsonl(stage, p)
                manifest_stages[stage]["files"].append(str(p.name))

            if "parquet" in formats:
                p = out_dir / f"{stage}.parquet"
                print(f"Exporting Parquet to {p}...")
                generator.export_to_parquet(stage, p)
                manifest_stages[stage]["files"].append(str(p.name))

    if "jsonl" in formats and (out_dir / "sft_data.jsonl").exists():
        shutil.copyfile(out_dir / "sft_data.jsonl", out_dir / "training_data.jsonl")

    if "parquet" in formats and (out_dir / "sft_data.parquet").exists():
        shutil.copyfile(out_dir / "sft_data.parquet", out_dir / "training_data.parquet")

    return manifest_stages


verify_split_prompt_leakage = diagnostics.verify_split_prompt_leakage


def handle_train(opts: CliOptionsModel) -> None:
    """
    Handle the train subcommand to orchestrate synthetic data generation and export.

    Args:
    ----
        opts: Validated CLI options model containing db, output_dir, samples_per_table, provider, model, export_formats, and optional flags.

    """
    print("=" * 60)
    print("T1D Synthetic Training Orchestrator")
    print(f"Database: {opts.db}")
    print(f"Model: {opts.provider}/{opts.model}")
    print(f"Samples per table: {opts.samples_per_table}")
    print("=" * 60)

    conn = duckdb.connect(opts.db)
    try:
        generator = training_data.TrainingDataGenerator(
            conn,
            model=opts.model,
            provider=opts.provider,
        )

        print("Generating synthetic pairs and persisting to DuckDB...")
        schema = generator.extract_schema()
        split_val_flag = opts.split_validation
        split_r = (0.8, 0.1, 0.1) if split_val_flag else None
        for table_name, table_schema in schema.items():
            print(
                f"Generating {opts.samples_per_table} pairs for table: {table_name}..."
            )
            pairs = generator.generate_pairs(table_schema, opts.samples_per_table)
            generator.write_to_db(pairs, split_ratios=split_r)

        out_dir = Path(str(opts.output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        formats = [f.strip().lower() for f in opts.export_formats.split(",")]
        manifest_stages = export_training_stages(
            conn=conn,
            generator=generator,
            out_dir=out_dir,
            formats=formats,
            split_val_flag=split_val_flag,
        )

        split_validation_passed = (
            verify_split_prompt_leakage(conn) if split_val_flag else True
        )
        hparams = validate_cli_hyperparameters(opts)

        backend_val = opts.backend
        manifest: Dict[
            str,
            Union[str, int, bool, List[str], Dict[str, ManifestStageDict]],
        ] = {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": f"{opts.provider}/{opts.model}",
            "database": str(opts.db),
            "samples_per_table": int(opts.samples_per_table),
            "backend": backend_val,
            "tables_processed": list(schema.keys()),
            "export_formats": formats,
            "stages": manifest_stages,
            "split_validation_passed": split_validation_passed,
        }
        manifest_path = out_dir / "training_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"Exported training manifest to {manifest_path}...")

        if opts.execute_training:
            maxtext_cfg = None
            if backend_val == "remote-tpu-maxtext":
                maxtext_cfg = MaxTextConfig(
                    project_id=os.environ.get("GCP_PROJECT_ID", "t1d-analytics-gcp"),
                    zone=opts.tpu_zone,
                    bucket_url=opts.gcs_bucket,
                    accelerator_type=TpuAcceleratorType(opts.tpu_type),
                    num_slices=opts.tpu_slices,
                    model_name=opts.model,
                )

            target_parquet = out_dir / "sft_data.parquet"
            target_jsonl = out_dir / "sft_data.jsonl"
            ds_target = (
                str(target_parquet)
                if "parquet" in formats
                else (
                    str(target_jsonl)
                    if "jsonl" in formats
                    else str(out_dir / "training_data.parquet")
                )
            )
            config = TrainingJobConfig(
                model_name=opts.model,
                backend=TrainingBackend(backend_val),
                dataset_path=ds_target,
                output_dir=str(out_dir / "checkpoints"),
                hyperparameters=hparams,
                dry_run=opts.dry_run,
                maxtext_config=maxtext_cfg,
                use_lora=opts.use_lora,
                quantization=opts.quantization,
            )
            print(
                f"Initiating training runner with backend '{config.backend.value}'..."
            )
            from t1d_analytics import training_runner

            runner = training_runner.get_training_runner(config.backend)
            runner.validate_environment()
            res = runner.run_training(config)
            results_file = out_dir / "training_results.json"
            with open(results_file, "w") as rf:
                json.dump(res, rf, indent=2)
            print(
                f"Training run completed for model '{config.model_name}'. Results saved to {results_file}."
            )
    finally:
        conn.close()
    print("Training orchestration completed successfully.")


def handle_bridge_gemma_sql(opts: CliOptionsModel) -> None:
    """
    Handle the bridge-gemma-sql command to interact with external gemma-4-sql toolchains.

    Args:
    ----
        opts: Validated CLI options model containing action, stage, db, table, output_dir, and config.

    Raises:
    ------
        CliValidationError: If `--config` is omitted for the `'train'` action.

    """
    if opts.action == "check":
        installed = gemma_bridge.check_gemma_sql_installed()
        print(f"gemma-4-sql installed: {installed}")
        if not installed:
            print("Action: Install via 'pip install -e /path/to/gemma-4-sql[all]'.")
    elif opts.action == "etl":
        print(
            f"Running gemma-4-sql ETL for stage '{opts.stage}' from {opts.db}:{opts.table}..."
        )
        res = gemma_bridge.run_gemma_sql_etl(
            stage=opts.stage,
            duckdb_path=opts.db,
            duckdb_table=opts.table,
            output_dir=opts.output_dir,
        )
        print(res.stdout)
        print("gemma-4-sql ETL completed successfully.")
    else:  # train
        if not opts.config:
            raise CliValidationError("--config path is required for 'train' action.")
        print(
            f"Running gemma-4-sql training for stage '{opts.stage}' with config '{opts.config}'..."
        )
        res = gemma_bridge.run_gemma_sql_train(
            stage=opts.stage,
            config_path=opts.config,
        )
        print(res.stdout)
        print("gemma-4-sql training completed successfully.")


if __name__ == "__main__":
    main()
