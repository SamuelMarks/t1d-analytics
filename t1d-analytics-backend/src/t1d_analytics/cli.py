"""Command Line Interface entrypoint for the T1D Analytics Suite."""

import os
import sys
from typing import Callable

import duckdb

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
    CliNamespace as CliNamespace,
)
from t1d_analytics.cli_parser import (
    CliOptionsModel as CliOptionsModel,
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


def handle_download(opts: CliOptionsModel) -> None:
    """
    Handle the download subcommand using module-level fetch/parse/process callables.

    Args:
    ----
        opts: Validated CLI options model.

    """
    _handle_download_impl(
        opts,
        fetch_html_fn=fetch_html,
        parse_datasets_fn=parse_datasets,
        process_datasets_fn=process_datasets,
        load_auth_plugin_fn=load_auth_plugin,
    )


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
