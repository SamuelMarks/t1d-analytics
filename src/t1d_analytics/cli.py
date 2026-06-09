"""Command Line Interface for the T1D Analytics Suite."""

import argparse
import sys

from t1d_analytics.downloader import process_datasets
from t1d_analytics.parser import fetch_html, parse_datasets


def main() -> None:
    """Execute the CLI application."""
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
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def handle_download(args: argparse.Namespace) -> None:
    """
    Handle the download subcommand.

    Args:
        args: Arguments.

    """
    print(f"Fetching HTML from {args.url}...")
    html = fetch_html(args.url)

    print("Parsing datasets...")
    datasets = parse_datasets(html)
    print(f"Found {len(datasets)} protocols.")

    if not datasets:
        print("No datasets found. Exiting.")
        return

    print(f"Starting downloads to {args.output}...")
    process_datasets(datasets, args.output)
    print("Done!")


def handle_extract(args: argparse.Namespace) -> None:
    """
    Handle the extract subcommand.

    Args:
        args: Command-line arguments.

    """
    from t1d_analytics.analytics import extract_zips

    extract_zips(args.data_dir)


def handle_load(args: argparse.Namespace) -> None:
    """
    Handle the load subcommand.

    Args:
        args: Arguments.

    """
    from t1d_analytics.analytics import load_data_to_duckdb

    load_data_to_duckdb(args.data_dir, args.db)


def handle_query(args: argparse.Namespace) -> None:
    """
    Handle the query subcommand.

    Args:
        args: Arguments.

    """
    from t1d_analytics.analytics import run_query_repl

    run_query_repl(args.db)


if __name__ == "__main__":  # pragma: no cover
    main()
