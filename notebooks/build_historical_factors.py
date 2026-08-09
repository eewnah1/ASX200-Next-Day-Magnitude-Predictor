"""Build the historical daily factor dataset for the ASX200 predictor.

Usage:
    python notebooks/build_historical_factors.py \
        --period max \
        --output data/historical_factors.parquet

The script downloads the longest available daily history for every input proxy,
reconstructs the exact FeatureVector that the scoring engine consumes, and saves
the table as Parquet + CSV.  It also stores the realised next-day ASX 200 return
so backtests can be run without re-fetching prices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.logging_config import setup_logging
from asx200_mag_predictor.scoring.ml import HistoricalFeatureBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical ASX200 factor dataset")
    parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="yfinance period string (e.g. 10y, max). Default: max",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=60,
        help="Minimum ASX 200 history days before producing the first row.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/historical_factors.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/historical_factors.csv",
        help="Optional CSV output path",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)

    print(f"Building historical factor table (period={args.period}) ...")
    builder = HistoricalFeatureBuilder(period=args.period, min_history=args.min_history)
    df = builder.build()

    if df.empty:
        print("No historical factor rows were produced.")
        sys.exit(1)

    # Sort and normalise index.
    df = df.sort_values("date").reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved Parquet: {out_path} ({len(df)} rows, {len(df.columns)} columns)")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV:     {csv_path}")

    print("\nDate range:")
    print(f"  Start: {df['date'].min()}")
    print(f"  End:   {df['date'].max()}")
    print("\nPrimary label distribution:")
    print(df["primary_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
