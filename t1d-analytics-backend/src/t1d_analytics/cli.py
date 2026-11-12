"""Command Line Interface entrypoint for the T1D Analytics Suite."""

import sys

import duckdb
from pydantic import ValidationError

from t1d_analytics.api_dependencies import configure_api_logging
from t1d_analytics.cli_handlers import (
    export_training_stages as export_training_stages,
)
from t1d_analytics.cli_handlers import (
    handle_bridge_gemma_sql as handle_bridge_gemma_sql,
)
from t1d_analytics.cli_handlers import (
    handle_doctor as handle_doctor,
)
from t1d_analytics.cli_handlers import (
    handle_download as _handle_download_impl,
)
from t1d_analytics.cli_handlers import (
    handle_evaluate as handle_evaluate,
)
from t1d_analytics.cli_handlers import (
    handle_export as handle_export,
)
from t1d_analytics.cli_handlers import (
    handle_export_training_data as handle_export_training_data,
)
from t1d_analytics.cli_handlers import (
    handle_extract as handle_extract,
)
from t1d_analytics.cli_handlers import (
    handle_generate_training_data as handle_generate_training_data,
)
from t1d_analytics.cli_handlers import (
    handle_load as handle_load,
)
from t1d_analytics.cli_handlers import (
    handle_manifest as handle_manifest,
)
from t1d_analytics.cli_handlers import (
    handle_profile as handle_profile,
)
from t1d_analytics.cli_handlers import (
    handle_query as handle_query,
)
from t1d_analytics.cli_handlers import (
    handle_seed as handle_seed,
)
from t1d_analytics.cli_handlers import (
    handle_train as handle_train,
)
from t1d_analytics.cli_handlers import (
    handle_watch as handle_watch,
)
from t1d_analytics.cli_handlers import (
    load_auth_plugin as load_auth_plugin,
)
from t1d_analytics.cli_handlers import (
    validate_cli_hyperparameters as validate_cli_hyperparameters,
)
from t1d_analytics.cli_handlers import (
    verify_split_prompt_leakage as verify_split_prompt_leakage,
)
from t1d_analytics.cli_parser import (
    CliSubcommandOptions,
    DoctorCliOpts,
    DownloadCliOpts,
    EvaluateCliOpts,
    ExportCliOpts,
    ExportTrainingDataCliOpts,
    ExtractCliOpts,
    GenerateTrainingDataCliOpts,
    LoadCliOpts,
    ManifestCliOpts,
    ProfileCliOpts,
    QueryCliOpts,
    SeedCliOpts,
    TrainCliOpts,
    WatchCliOpts,
)
from t1d_analytics.cli_parser import (
    ManifestStageDict as ManifestStageDict,
)
from t1d_analytics.cli_parser import (
    build_cli_parser as build_cli_parser,
)
from t1d_analytics.cli_parser import (
    parse_cli_options as parse_cli_options,
)
from t1d_analytics.downloader import process_datasets as process_datasets
from t1d_analytics.errors import (
    T1DError,
)
from t1d_analytics.errors import (
    TrainingExecutionError as TrainingExecutionError,
)
from t1d_analytics.i18n import get_translator
from t1d_analytics.parser import (
    fetch_html as fetch_html,
)
from t1d_analytics.parser import (
    parse_datasets as parse_datasets,
)


def handle_download(opts: DownloadCliOpts) -> None:
    """
    Handle the download subcommand using module-level fetch/parse/process callables.

    Args:
        opts: Validated `DownloadCliOpts` model.

    """
    _handle_download_impl(
        opts,
        fetch_html_fn=fetch_html,
        parse_datasets_fn=parse_datasets,
        process_datasets_fn=process_datasets,
        load_auth_plugin_fn=load_auth_plugin,
    )


_DATA_CLI_OPT_TYPES = (
    DownloadCliOpts,
    ExtractCliOpts,
    LoadCliOpts,
    SeedCliOpts,
    ManifestCliOpts,
    QueryCliOpts,
    ExportCliOpts,
    ProfileCliOpts,
    WatchCliOpts,
)

DataCliSubcommandOptions = (
    DownloadCliOpts
    | ExtractCliOpts
    | LoadCliOpts
    | SeedCliOpts
    | ManifestCliOpts
    | QueryCliOpts
    | ExportCliOpts
    | ProfileCliOpts
    | WatchCliOpts
)


def _dispatch_data_cli_subcommand(opts: DataCliSubcommandOptions) -> None:
    """
    Dispatch data-ingestion, query, and export CLI subcommands.

    Args:
        opts: Per-subcommand typed data CLI options model.

    """
    if isinstance(opts, DownloadCliOpts):
        handle_download(opts)
    elif isinstance(opts, ExtractCliOpts):
        handle_extract(opts)
    elif isinstance(opts, LoadCliOpts):
        handle_load(opts)
    elif isinstance(opts, SeedCliOpts):
        handle_seed(opts)
    elif isinstance(opts, ManifestCliOpts):
        handle_manifest(opts)
    elif isinstance(opts, QueryCliOpts):
        handle_query(opts)
    elif isinstance(opts, ExportCliOpts):
        handle_export(opts)
    elif isinstance(opts, ProfileCliOpts):
        handle_profile(opts)
    else:
        handle_watch(opts)


def dispatch_cli_subcommand(opts: CliSubcommandOptions) -> None:
    """
    Dispatch a validated `CliSubcommandOptions` discriminated-union model to its typed handler.

    Args:
        opts: Per-subcommand typed CLI options model.

    """
    if isinstance(opts, _DATA_CLI_OPT_TYPES):
        _dispatch_data_cli_subcommand(opts)
    elif isinstance(opts, GenerateTrainingDataCliOpts):
        handle_generate_training_data(opts)
    elif isinstance(opts, ExportTrainingDataCliOpts):
        handle_export_training_data(opts)
    elif isinstance(opts, EvaluateCliOpts):
        handle_evaluate(opts)
    elif isinstance(opts, DoctorCliOpts):
        handle_doctor(opts)
    elif isinstance(opts, TrainCliOpts):
        handle_train(opts)
    else:
        handle_bridge_gemma_sql(opts)


def main() -> None:
    """Execute the CLI application."""
    configure_api_logging()
    parser = build_cli_parser()
    args = parser.parse_args()
    try:
        opts = parse_cli_options(args)
    except ValidationError:
        return

    _ = get_translator(opts.lang)

    try:
        dispatch_cli_subcommand(opts)
    except (
        T1DError,
        duckdb.Error,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        ImportError,
    ) as e:
        print(_("An error occurred: {}", e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
