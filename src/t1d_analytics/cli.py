"""Command Line Interface for the T1D Analytics Suite."""

import argparse
import sys

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

    # Query subcommand
    query_parser = subparsers.add_parser(
        "query", help="Open a REPL to query the populated DuckDB database."
    )
    query_parser.add_argument(
        "--db",
        default="t1d_analytics.duckdb",
        help="Path to the DuckDB database file.",
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

    args = parser.parse_args()

    try:
        if args.command == "download":
            handle_download(args)
        elif args.command == "extract":
            handle_extract(args)
        elif args.command == "load":
            handle_load(args)
        elif args.command == "query":
            handle_query(args)
        elif args.command == "generate-training-data":
            handle_generate_training_data(args)
    except Exception as e:
        print(_("An error occurred: {}", e), file=sys.stderr)
        sys.exit(1)


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
    datasets = parse_datasets(html)
    print(_("Found {} protocols.", len(datasets)))

    if not datasets:
        print(_("No datasets found. Exiting."))
        return

    print(_("Starting downloads to {}...", args.output))
    process_datasets(datasets, args.output)
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

    load_data_to_duckdb(args.data_dir, args.db)


def handle_query(args: argparse.Namespace) -> None:
    """
    Handle the query subcommand.

    Args:
    ----
        args: Arguments.

    """
    from t1d_analytics.analytics import run_query_repl

    run_query_repl(args.db)


def handle_generate_training_data(args: argparse.Namespace) -> None:
    """
    Handle the generate-training-data subcommand.

    This function initializes a connection to DuckDB, creates the TrainingDataGenerator,
    and runs the generation pipeline for the specified number of pairs per table using
    the provided local LLM model.

    Args:
    ----
        args: Command-line arguments containing db (DuckDB path), num_pairs (count),
            and model (LLM name).

    """
    import duckdb

    from t1d_analytics.training_data import TrainingDataGenerator

    _ = get_translator()

    print(_("Connecting to DuckDB at {}...", args.db))
    conn = duckdb.connect(args.db)

    print(_("Initializing TrainingDataGenerator with model '{}'...", args.model))
    generator = TrainingDataGenerator(conn, args.model)

    # Extract schema
    schema = generator._extract_schema()

    # Generate pairs and write to db
    total_tables = len(schema)
    print(_("Found {} tables in the schema.", total_tables))

    for table_name, table_schema in schema.items():
        print(_("Generating {} pairs for table: {}...", args.num_pairs, table_name))
        pairs = generator._generate_pairs(table_schema, args.num_pairs)
        generator.write_to_db(pairs)

    conn.close()
    print(_("Training data generation complete!"))


if __name__ == "__main__":  # pragma: no cover
    main()
