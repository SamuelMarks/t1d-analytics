"""Subcommand handler implementations for the T1D Analytics CLI."""

import datetime
import importlib
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import duckdb

from t1d_analytics import (
    analytics,
    api_exports,
    diagnostics,
    gemma_bridge,
    ingestion,
    seed,
    training_data,
)
from t1d_analytics.cli_parser import CliOptionsModel, ManifestStageDict
from t1d_analytics.downloader import (
    DownloadSummaryDict,
    parse_headers_file,
    process_datasets,
)
from t1d_analytics.errors import (
    AuthPluginError,
    CliValidationError,
    DownloadError,
)
from t1d_analytics.i18n import get_translator
from t1d_analytics.models import (
    DatasetInfo,
    MaxTextConfig,
    TpuAcceleratorType,
    TrainingBackend,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from t1d_analytics.parser import fetch_html, parse_datasets
from t1d_analytics.sql_utils import connect_duckdb_with_retry


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


def handle_download(
    opts: CliOptionsModel,
    fetch_html_fn: Callable[..., str] = fetch_html,
    parse_datasets_fn: Callable[..., List[DatasetInfo]] = parse_datasets,
    process_datasets_fn: Callable[..., DownloadSummaryDict] = process_datasets,
    load_auth_plugin_fn: Callable[[str], Callable[[], str]] = load_auth_plugin,
) -> None:
    """
    Handle the download subcommand.

    Args:
    ----
        opts: Validated CLI options model.
        fetch_html_fn: Callable to fetch HTML from the dataset index URL.
        parse_datasets_fn: Callable to parse dataset links from HTML.
        process_datasets_fn: Callable to download discovered datasets.
        load_auth_plugin_fn: Callable to load an authentication plugin specifier.

    Raises:
    ------
        DownloadError: If one or more dataset files fail to download.

    """
    _ = get_translator()
    print(_("Fetching HTML from {}...", opts.url))
    html = fetch_html_fn(opts.url)

    print(_("Parsing datasets..."))
    datasets = parse_datasets_fn(html, base_url=opts.url)
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
        auth_plugin_cb = load_auth_plugin_fn(opts.auth_plugin)
        if not token:
            token = auth_plugin_cb()

    summary = process_datasets_fn(
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
    ingestion.extract_zips(opts.data_dir)


def handle_load(opts: CliOptionsModel) -> None:
    """
    Handle the load subcommand.

    Args:
    ----
        opts: Validated CLI options model.

    """
    ingestion.load_data_to_duckdb(
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
    records = ingestion.get_ingestion_manifest(opts.db)
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

        schema = generator.extract_schema()
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

    Raises:
    ------
        CliValidationError: If the specified `--test-set` file does not exist.

    """
    test_file = Path(opts.test_set)
    if not test_file.exists():
        msg = f"Test set file {test_file} does not exist."
        print(msg)
        raise CliValidationError(msg)

    test_data: list[dict[str, str]] = []
    content = test_file.read_text(encoding="utf-8").strip()
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
            out_p.write_text(metrics["csv_report"], encoding="utf-8")
        else:
            out_p.write_text(metrics["markdown_report"], encoding="utf-8")
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
        exts = ingestion.get_supported_tabular_globs(
            include_sas=opts.include_sas,
            include_spss=opts.include_spss,
        )

        current_files: list[Path] = []
        if data_path.exists():
            for ext in exts:
                current_files.extend(data_path.rglob(ext))

        current_paths_set = {str(f.resolve()) for f in current_files}
        deleted_files = [
            f_str for f_str in seen_hashes if f_str not in current_paths_set
        ]
        for del_f in deleted_files:
            del seen_hashes[del_f]
            print(f"Detected file deletion: '{del_f}'. Pruning manifest...")
            if Path(opts.db).exists():
                try:
                    with connect_duckdb_with_retry(
                        opts.db,
                        read_only=False,
                        max_attempts=3,
                    ) as conn_del:
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

        changed = False
        for f in current_files:
            f_str = str(f.resolve())
            h = ingestion.compute_file_hash(f)
            if seen_hashes.get(f_str) != h:
                seen_hashes[f_str] = h
                changed = True

        if changed:
            print(f"Detected file changes or new files in '{data_path}'. Ingesting...")
            ingestion.load_data_to_duckdb(
                str(data_path),
                opts.db,
                prefix_subdirs=opts.prefix_subdirs,
                include_sas=opts.include_sas,
                include_spss=opts.include_spss,
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
            with open(results_file, "w", encoding="utf-8") as rf:
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
    else:
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
