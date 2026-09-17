"""Command Line Interface for the T1D Analytics Suite."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, cast

from t1d_analytics.downloader import process_datasets
from t1d_analytics.i18n import get_translator
from t1d_analytics.parser import fetch_html, parse_datasets


def main() -> None:
    """Execute the CLI application."""
    # Initialize translator based on LANG env var
    _ = get_translator()

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

    args = parser.parse_args()

    if isinstance(getattr(args, "lang", None), str):
        os.environ["LANG"] = args.lang
        _ = get_translator(args.lang)

    try:
        if args.command == "download":
            handle_download(args)
        elif args.command == "extract":
            handle_extract(args)
        elif args.command == "load":
            handle_load(args)
        elif args.command == "manifest":
            handle_manifest(args)
        elif args.command == "query":
            handle_query(args)
        elif args.command == "generate-training-data":
            handle_generate_training_data(args)
        elif args.command == "export-training-data":
            handle_export_training_data(args)
        elif args.command == "export":
            handle_export(args)
        elif args.command == "evaluate":
            handle_evaluate(args)
        elif args.command == "doctor":
            handle_doctor(args)
        elif args.command == "profile":
            handle_profile(args)
        elif args.command == "watch":
            handle_watch(args)
        elif args.command == "train":
            handle_train(args)
        elif args.command == "bridge-gemma-sql":
            handle_bridge_gemma_sql(args)
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
        ValueError: If format is invalid or target function cannot be found or invoked.

    """
    import importlib

    if ":" not in plugin_spec:
        raise ValueError(
            f"Invalid auth plugin specifier '{plugin_spec}'. Expected 'module:function'."
        )
    mod_name, func_name = plugin_spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        raise ValueError(
            f"Failed to import auth plugin module '{mod_name}': {e}"
        ) from e

    if not hasattr(mod, func_name):
        raise ValueError(
            f"Auth plugin module '{mod_name}' has no attribute '{func_name}'."
        )

    target_fn = getattr(mod, func_name)
    if not callable(target_fn):
        raise ValueError(
            f"Auth plugin attribute '{func_name}' in module '{mod_name}' is not callable."
        )
    return cast(Callable[[], str], target_fn)


def handle_download(args: argparse.Namespace) -> None:
    """
    Handle the download subcommand.

    Args:
    ----
        args: Arguments.

    """
    _ = get_translator()
    print(_("Fetching HTML from {}...", args.url))
    html = fetch_html(args.url)

    print(_("Parsing datasets..."))
    datasets = parse_datasets(html, base_url=args.url)
    print(_("Found {} protocols.", len(datasets)))

    if not datasets:
        print(_("No datasets found. Exiting."))
        return

    print(_("Starting downloads to {}...", args.output))
    from t1d_analytics.downloader import parse_headers_file

    auth: Optional[tuple[str, str]] = None
    if getattr(args, "user", None) and getattr(args, "password", None):
        auth = (args.user, args.password)

    custom_headers: Optional[dict[str, str]] = None
    if getattr(args, "headers", None):
        custom_headers = parse_headers_file(Path(args.headers))

    concurrency = getattr(args, "concurrency", 4)
    token = getattr(args, "token", None)

    auth_plugin_cb: Optional[Callable[[], str]] = None
    if getattr(args, "auth_plugin", None):
        auth_plugin_cb = load_auth_plugin(args.auth_plugin)
        if not token:
            token = auth_plugin_cb()

    summary = process_datasets(
        datasets,
        args.output,
        concurrency=concurrency,
        token=token,
        auth=auth,
        headers=custom_headers,
        page_url=args.url,
        token_refresh_callback=auth_plugin_cb,
    )
    failed = summary.get("failed", 0) if isinstance(summary, dict) else 0
    if isinstance(failed, int) and failed > 0:
        raise RuntimeError(_("Download encountered errors: {} file(s) failed.", failed))
    print(_("Done!"))


def handle_extract(args: argparse.Namespace) -> None:
    """
    Handle the extract subcommand.

    Args:
    ----
        args: Command-line arguments.

    """
    from t1d_analytics.analytics import extract_zips

    extract_zips(args.data_dir)


def handle_load(args: argparse.Namespace) -> None:
    """
    Handle the load subcommand.

    Args:
    ----
        args: Arguments.

    """
    from t1d_analytics.analytics import load_data_to_duckdb

    load_data_to_duckdb(
        args.data_dir,
        args.db,
        prefix_subdirs=getattr(args, "prefix_subdirs", False),
        include_sas=getattr(args, "include_sas", False),
        include_spss=getattr(args, "include_spss", False),
    )


def handle_manifest(args: argparse.Namespace) -> None:
    """
    Handle the manifest subcommand.

    Args:
    ----
        args: Arguments containing db path.

    """
    from t1d_analytics.analytics import get_ingestion_manifest

    records = get_ingestion_manifest(args.db)
    if not records:
        print("No ingestion manifest records found.")
        return

    print(
        f"{'Table Name':<25} | {'Row Count':<10} | {'File Size':<10} | {'Loaded At':<25} | Source File"
    )
    print("-" * 100)
    for r in records:
        print(
            f"{str(r['table_name']):<25} | {str(r['row_count']):<10} | {str(r['file_size']):<10} | {str(r['loaded_at']):<25} | {r['source_file']}"
        )


def handle_query(args: argparse.Namespace) -> None:
    """
    Handle the query subcommand.

    Args:
    ----
        args: Arguments containing db, model, and optional provider.

    """
    from t1d_analytics.analytics import run_query_repl

    provider = getattr(args, "provider", "ollama")
    run_query_repl(args.db, model=args.model, provider=provider)


def handle_generate_training_data(args: argparse.Namespace) -> None:
    """
    Handle the generate-training-data subcommand.

    This function initializes a connection to DuckDB, creates the TrainingDataGenerator,
    and runs the generation pipeline for the specified number of pairs per table using
    the provided local LLM model and provider.

    Args:
    ----
        args: Command-line arguments containing db, num_pairs, model, provider, and split_ratios.

    """
    import duckdb

    from t1d_analytics.training_data import TrainingDataGenerator

    _ = get_translator()

    print(_("Connecting to DuckDB at {}...", args.db))
    conn = duckdb.connect(args.db)

    provider = getattr(args, "provider", "ollama")
    print(
        _(
            "Initializing TrainingDataGenerator with model '{}' (provider: '{}')...",
            args.model,
            provider,
        )
    )
    generator = TrainingDataGenerator(conn, args.model, provider=provider)

    ratios = None
    if getattr(args, "split_ratios", None):
        parts = [float(p.strip()) for p in args.split_ratios.split(",")]
        if len(parts) == 3:
            ratios = (parts[0], parts[1], parts[2])

    # Extract schema
    schema = generator._extract_schema()

    # Generate pairs and write to db
    total_tables = len(schema)
    print(_("Found {} tables in the schema.", total_tables))

    for table_name, table_schema in schema.items():
        print(_("Generating {} pairs for table: {}...", args.num_pairs, table_name))
        pairs = generator._generate_pairs(table_schema, args.num_pairs)
        generator.write_to_db(pairs, split_ratios=ratios)

    conn.close()
    print(_("Training data generation complete!"))


def handle_evaluate(args: argparse.Namespace) -> None:
    """
    Handle the evaluate subcommand.

    Args:
    ----
        args: Command-line arguments containing db, test_set, model, and output.

    """
    import json

    from t1d_analytics.training_data import evaluate_text_to_sql

    test_file = Path(args.test_set)
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

    metrics = evaluate_text_to_sql(args.db, test_data, model=args.model)
    print(metrics["markdown_report"])

    if getattr(args, "output", None):
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".csv":
            out_p.write_text(metrics["csv_report"])
        else:
            out_p.write_text(metrics["markdown_report"])
        print(f"Report saved to {out_p}")


def handle_export_training_data(args: argparse.Namespace) -> None:
    """
    Handle exporting training data to JSONL or Parquet format.

    Args:
    ----
        args: Command-line arguments containing db, table, format, and output-file.

    """
    import duckdb

    from t1d_analytics.training_data import TrainingDataGenerator

    _ = get_translator()
    print(_("Connecting to DuckDB at {}...", args.db))
    conn = duckdb.connect(args.db, read_only=True)
    generator = TrainingDataGenerator(conn)

    print(_("Exporting table '{}' to {}...", args.table, args.output_file))
    if args.format == "jsonl":
        generator.export_to_jsonl(args.table, Path(args.output_file))
    else:
        generator.export_to_parquet(args.table, Path(args.output_file))

    conn.close()
    print(_("Export complete!"))


def handle_export(args: argparse.Namespace) -> None:
    """
    Handle exporting database tables to Excel (.xlsx), Parquet, or CSV format.

    Args:
    ----
        args: Command-line arguments containing db, table, format, and output_file.

    """
    from t1d_analytics.analytics import export_table

    _ = get_translator()
    print(_("Connecting to DuckDB at {}...", args.db))
    print(_("Exporting table '{}' to {}...", args.table, args.output_file))
    export_table(
        db_path=args.db,
        table_name=args.table,
        output_path=args.output_file,
        export_format=args.format,
    )
    print(_("Export complete!"))


def handle_doctor(args: argparse.Namespace) -> None:
    """
    Handle the doctor subcommand to inspect system health.

    Args:
    ----
        args: Arguments containing db path.

    """
    from t1d_analytics.diagnostics import get_system_health

    health = get_system_health(db_path=args.db)
    db = health.database
    ollama = health.ollama

    print("=" * 60)
    print("T1D Analytics System Health Report")
    print("=" * 60)
    print(f"Overall Status: {health.status.upper()}")
    print("-" * 60)
    print(f"Database ({db.configured_path}):")
    print(f"  Connected: {db.connected}")
    print(f"  Status:    {db.status_code.value}")
    print(f"  Tables:    {db.table_count}")
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


def handle_profile(args: argparse.Namespace) -> None:
    """
    Handle the profile subcommand to analyze data quality and completeness of a table.

    Args:
    ----
        args: Arguments containing db path and table name.

    """
    from t1d_analytics.analytics import profile_table

    report = profile_table(args.db, args.table)
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


def handle_watch(args: argparse.Namespace) -> None:
    """
    Handle the watch subcommand to monitor and incrementally ingest files into DuckDB.

    Args:
    ----
        args: Arguments containing data_dir, db, interval, iterations, and prefix_subdirs.

    """
    from t1d_analytics.analytics import compute_file_hash, load_data_to_duckdb

    data_path = Path(args.data_dir)
    print(
        f"Watching directory '{data_path}' for changes (interval: {args.interval}s)..."
    )
    seen_hashes: dict[str, str] = {}

    count = 0
    while True:
        count += 1
        current_files: list[Path] = []
        if data_path.exists():
            for ext in ("*.csv", "*.parquet", "*.txt"):
                current_files.extend(data_path.rglob(ext))

        changed = False
        for f in current_files:
            f_str = str(f)
            h = compute_file_hash(f)
            if seen_hashes.get(f_str) != h:
                seen_hashes[f_str] = h
                changed = True

        if changed:
            print(f"Detected file changes or new files in '{data_path}'. Ingesting...")
            load_data_to_duckdb(
                str(data_path),
                args.db,
                prefix_subdirs=getattr(args, "prefix_subdirs", False),
            )
            print("Ingestion complete.")

        if args.iterations > 0 and count >= args.iterations:
            break
        time.sleep(args.interval)


def handle_train(args: argparse.Namespace) -> None:
    """
    Handle the train subcommand to orchestrate synthetic data generation and export.

    Args:
    ----
        args: Arguments containing db, output_dir, samples_per_table, provider, model, export_formats, and optional flags.

    """
    import json

    import duckdb

    from t1d_analytics.models import (
        TrainingBackend,
        TrainingHyperparameters,
        TrainingJobConfig,
    )
    from t1d_analytics.training_data import TrainingDataGenerator

    print("=" * 60)
    print("T1D Synthetic Training Orchestrator")
    print(f"Database: {args.db}")
    print(f"Model: {args.provider}/{args.model}")
    print(f"Samples per table: {args.samples_per_table}")
    print("=" * 60)

    conn = duckdb.connect(args.db)
    generator = TrainingDataGenerator(
        conn,
        model=args.model,
        provider=args.provider,
    )

    print("Generating synthetic pairs and persisting to DuckDB...")
    schema = generator._extract_schema()
    split_r = (0.8, 0.1, 0.1) if getattr(args, "split_validation", False) else None
    for table_name, table_schema in schema.items():
        print(f"Generating {args.samples_per_table} pairs for table: {table_name}...")
        pairs = generator._generate_pairs(table_schema, args.samples_per_table)
        generator.write_to_db(pairs, split_ratios=split_r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = [f.strip().lower() for f in args.export_formats.split(",")]
    if "jsonl" in formats:
        jsonl_path = out_dir / "training_data.jsonl"
        print(f"Exporting JSONL to {jsonl_path}...")
        generator.export_to_jsonl("sft_data", jsonl_path)

    if "parquet" in formats:
        parquet_path = out_dir / "training_data.parquet"
        print(f"Exporting Parquet to {parquet_path}...")
        generator.export_to_parquet("sft_data", parquet_path)

    split_validation_passed = True
    if getattr(args, "split_validation", False):
        print("Validating dataset splits for prompt leakage...")
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        train_tbl = (
            "sft_train"
            if "sft_train" in tables
            else ("sft_data_train" if "sft_data_train" in tables else None)
        )
        val_tbl = (
            "sft_val"
            if "sft_val" in tables
            else ("sft_data_val" if "sft_data_val" in tables else None)
        )
        if train_tbl and val_tbl:
            train_prompts = set(
                r[0]
                for r in conn.execute(f'SELECT prompt FROM "{train_tbl}"').fetchall()
            )
            val_prompts = set(
                r[0] for r in conn.execute(f'SELECT prompt FROM "{val_tbl}"').fetchall()
            )
            overlap = train_prompts.intersection(val_prompts)
            if overlap:
                print(
                    f"Warning: Found {len(overlap)} overlapping prompts between train and val splits."
                )
                split_validation_passed = False
            else:
                print("Split validation passed: 0 prompt overlap detected.")
        else:
            print("Split validation skipped: split tables not found.")

    lr = getattr(args, "learning_rate", 2e-5)
    if lr <= 0.0:
        raise ValueError("Learning rate must be positive.")
    batch_sz = getattr(args, "batch_size", 4)
    if batch_sz < 1:
        raise ValueError("Batch size must be at least 1.")
    epochs = getattr(args, "num_epochs", 3)
    if epochs < 1:
        raise ValueError("Number of epochs must be at least 1.")
    warmup = getattr(args, "warmup_steps", 50)
    if warmup < 0:
        raise ValueError("Warmup steps cannot be negative.")
    w_decay = getattr(args, "weight_decay", 0.01)
    if w_decay < 0.0:
        raise ValueError("Weight decay cannot be negative.")
    max_len = getattr(args, "max_seq_length", 2048)
    if max_len < 64:
        raise ValueError("Max sequence length must be at least 64.")

    hparams = TrainingHyperparameters(
        learning_rate=lr,
        batch_size=batch_sz,
        num_epochs=epochs,
        warmup_steps=warmup,
        weight_decay=w_decay,
        max_seq_length=max_len,
        checkpoint_interval=getattr(args, "checkpoint_interval", 500),
        eval_steps=getattr(args, "eval_steps", 100),
    )

    manifest = {
        "model": f"{args.provider}/{args.model}",
        "database": str(args.db),
        "samples_per_table": args.samples_per_table,
        "backend": getattr(args, "backend", "local-cpu"),
        "tables_processed": list(schema.keys()),
        "export_formats": formats,
        "split_validation_passed": split_validation_passed,
    }
    manifest_path = out_dir / "training_manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"Exported training manifest to {manifest_path}...")

    if getattr(args, "execute_training", False):
        backend_val = getattr(args, "backend", "local-cpu")
        maxtext_cfg = None
        if backend_val == "remote-tpu-maxtext":
            from t1d_analytics.models import MaxTextConfig, TpuAcceleratorType

            maxtext_cfg = MaxTextConfig(
                project_id=os.environ.get("GCP_PROJECT_ID", "t1d-analytics-gcp"),
                zone=getattr(args, "tpu_zone", "us-central2-b"),
                bucket_url=getattr(args, "gcs_bucket", "gs://t1d-analytics-artifacts"),
                accelerator_type=TpuAcceleratorType(getattr(args, "tpu_type", "v4-8")),
                num_slices=getattr(args, "tpu_slices", 1),
                model_name=args.model,
            )

        ds_target = (
            str(parquet_path)
            if "parquet" in formats
            else (
                str(jsonl_path)
                if "jsonl" in formats
                else str(out_dir / "training_data.parquet")
            )
        )
        config = TrainingJobConfig(
            model_name=args.model,
            backend=TrainingBackend(backend_val),
            dataset_path=ds_target,
            output_dir=str(out_dir / "checkpoints"),
            hyperparameters=hparams,
            dry_run=getattr(args, "dry_run", False),
            maxtext_config=maxtext_cfg,
            use_lora=getattr(args, "use_lora", False),
            quantization=getattr(args, "quantization", None),
        )
        print(f"Initiating training runner with backend '{config.backend.value}'...")
        from t1d_analytics.training_runner import get_training_runner

        runner = get_training_runner(config.backend)
        runner.validate_environment()
        res = runner.run_training(config)
        results_file = out_dir / "training_results.json"
        with open(results_file, "w") as rf:
            json.dump(res, rf, indent=2)
        print(
            f"Training run completed for model '{config.model_name}'. Results saved to {results_file}."
        )

    conn.close()
    print("Training orchestration completed successfully.")


def handle_bridge_gemma_sql(args: argparse.Namespace) -> None:
    """
    Handle the bridge-gemma-sql command to interact with external gemma-4-sql toolchains.

    Args:
    ----
        args: Command-line arguments containing action, stage, db, table, output_dir, and config.

    """
    from t1d_analytics.gemma_bridge import (
        check_gemma_sql_installed,
        run_gemma_sql_etl,
        run_gemma_sql_train,
    )

    if args.action == "check":
        installed = check_gemma_sql_installed()
        print(f"gemma-4-sql installed: {installed}")
        if not installed:
            print("Action: Install via 'pip install -e /path/to/gemma-4-sql[all]'.")
    elif args.action == "etl":
        print(
            f"Running gemma-4-sql ETL for stage '{args.stage}' from {args.db}:{args.table}..."
        )
        res = run_gemma_sql_etl(
            stage=args.stage,
            duckdb_path=args.db,
            duckdb_table=args.table,
            output_dir=args.output_dir,
        )
        print(res.stdout)
        print("gemma-4-sql ETL completed successfully.")
    else:  # train
        if not args.config:
            raise ValueError("--config path is required for 'train' action.")
        print(
            f"Running gemma-4-sql training for stage '{args.stage}' with config '{args.config}'..."
        )
        res = run_gemma_sql_train(
            stage=args.stage,
            config_path=args.config,
        )
        print(res.stdout)
        print("gemma-4-sql training completed successfully.")


if __name__ == "__main__":  # pragma: no cover
    main()
