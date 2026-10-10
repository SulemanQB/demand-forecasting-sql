"""Command line entry points: download, build-features, backtest, all."""

import argparse

from demandforecast.backtest import run_backtest
from demandforecast.download import download_raw
from demandforecast.pipeline import build_features


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retail demand forecasting pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download UCI Online Retail II to a local parquet file")
    download.add_argument("--force", action="store_true", help="Download again even if the parquet exists")
    sub.add_parser("build-features", help="Load raw invoices and build DuckDB feature tables")
    sub.add_parser("backtest", help="Run the rolling-origin backtest and write reports")
    run_all = sub.add_parser("all", help="Download, build features, and backtest")
    run_all.add_argument("--force", action="store_true", help="Download again even if the parquet exists")

    args = parser.parse_args(argv)
    if args.command == "download":
        download_raw(force=args.force)
    elif args.command == "build-features":
        build_features()
    elif args.command == "backtest":
        run_backtest()
    elif args.command == "all":
        download_raw(force=args.force)
        build_features()
        run_backtest()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
