"""Out-of-sample style historical backtest using the saved factor table.

Usage:
    # Use the current ScoringEngine recommendation (high conviction)
    python notebooks/historical_backtest.py \
        --factors data/historical_factors.parquet \
        --strategy engine

    # Use a configurable rule-score threshold (default, more signals)
    python notebooks/historical_backtest.py \
        --factors data/historical_factors.parquet \
        --strategy score \
        --primary-threshold 0.6

The script replays the scoring engine day-by-day, records the primary/secondary
scores, and compares the daily GO LONG / STAY IN CASH decision to the next-day
ASX 200 return.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.logging_config import setup_logging
from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scoring.engine import ScoringEngine, bucket_from_return

DROP_COLUMNS = {
    "date",
    "next_return_pct",
    "primary_label",
    "secondary_label",
}


def _feature_vector_from_row(row: pd.Series) -> FeatureVector:
    """Convert a factor-table row into a FeatureVector."""
    data = {k: v for k, v in row.items() if k not in DROP_COLUMNS}
    return FeatureVector(**data)


def _annualised_simple(daily_returns: pd.Series) -> float:
    """Annualised return from mean daily percentage return, assuming 252 trading days."""
    if daily_returns.empty:
        return 0.0
    return daily_returns.mean() * 252.0


def _sharpe(daily_returns: pd.Series) -> float:
    """Daily-return Sharpe assuming 252 trading days and zero risk-free rate."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() * 252.0) / (daily_returns.std() * np.sqrt(252.0))


def _go_long_from_score(
    row: pd.Series,
    primary_threshold: float,
    rsi_cap: float,
    ath_cap: float,
    require_technicals_ok: bool,
) -> bool:
    """A simple, configurable rule-score filter for the backtest.

    GO LONG when the primary rule score is sufficiently bullish and technical
    conditions are not strongly bearish/overbought.
    """
    if row["primary_score"] < primary_threshold:
        return False
    if require_technicals_ok:
        rsi = row.get("rsi_14")
        ath = row.get("ath_distance_pct")
        if rsi is not None and rsi > rsi_cap:
            return False
        if ath is not None and ath > ath_cap:
            return False
    return True


def run_backtest(
    factors_path: str,
    strategy: str = "score",
    use_ml: bool = False,
    primary_threshold: float = 0.6,
    rsi_cap: float = 70.0,
    ath_cap: float = -0.5,
    start: str | None = None,
    end: str | None = None,
    require_technicals_ok: bool = True,
) -> None:
    settings = get_settings()
    setup_logging(settings)

    df = pd.read_parquet(factors_path)
    df["date"] = pd.to_datetime(df["date"])

    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    if df.empty:
        print("Factor table is empty or the date filter excludes all rows.")
        sys.exit(1)

    engine = ScoringEngine(settings)
    if not use_ml:
        engine.hybrid_ml.available = False

    rows: list[dict] = []
    for _, factor_row in df.iterrows():
        fv = _feature_vector_from_row(factor_row)
        pred = engine.predict(fv, prediction_for=factor_row["date"])

        rows.append(
            {
                "date": factor_row["date"],
                "recommendation": pred.recommendation,
                "primary_bucket": pred.primary_bucket,
                "secondary_bucket": pred.secondary_bucket,
                "primary_score": pred.primary_score,
                "secondary_score": pred.secondary_score,
                "confidence": pred.recommendation_confidence,
                "actual_return_pct": factor_row["next_return_pct"],
                "actual_bucket": bucket_from_return(factor_row["next_return_pct"]),
                "model": pred.model,
                "rsi_14": factor_row.get("rsi_14"),
                "ath_distance_pct": factor_row.get("ath_distance_pct"),
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values("date").reset_index(drop=True)

    # Apply the chosen decision strategy.
    if strategy == "engine":
        results["signal"] = results["recommendation"] == "GO LONG"
        results["strategy_label"] = results["recommendation"]
    else:
        results["signal"] = results.apply(
            lambda r: _go_long_from_score(
                r, primary_threshold, rsi_cap, ath_cap, require_technicals_ok
            ),
            axis=1,
        )
        results["strategy_label"] = results["signal"].map({True: "GO LONG", False: "STAY IN CASH"})

    go_long = results[results["signal"]]
    all_returns = results["actual_return_pct"]

    # Strategy daily return series: 0% on non-signal days, next-day ASX return on signal days.
    strategy_returns = pd.Series(0.0, index=results.index)
    strategy_returns[results["signal"]] = results.loc[results["signal"], "actual_return_pct"]

    print("\n=== ASX200 Next-Day Predictor — Historical Backtest ===\n")
    print(f"Factor source:      {factors_path}")
    print(f"Strategy:           {strategy}")
    print(f"Period:             {results['date'].min().date()} to {results['date'].max().date()}")
    print(f"Total days:         {len(results)}")
    print(f"ML enabled:         {use_ml}")

    if strategy == "score":
        print(f"Primary threshold:  {primary_threshold}")
        if require_technicals_ok:
            print(f"RSI cap:            {rsi_cap}")
            print(f"ATH distance cap:   {ath_cap}%")

    print("\n--- GO LONG signal performance ---")
    if go_long.empty:
        print("No GO LONG signals were generated over the backtest period.")
    else:
        signal_returns = go_long["actual_return_pct"]
        hit_rate = (signal_returns > 0.0).mean()
        avg_return = signal_returns.mean()
        median_return = signal_returns.median()
        win_days = (signal_returns > 0.0).sum()
        lose_days = (signal_returns <= 0.0).sum()

        print(f"Signal days:        {len(go_long)} ({len(go_long) / len(results):.1%} of all days)")
        print(f"Hit rate:           {hit_rate:.1%}  ({win_days} up / {lose_days} down)")
        print(f"Avg return:         {avg_return:+.3f}%")
        print(f"Median return:      {median_return:+.3f}%")
        print(f"Simple total return: {signal_returns.sum():+.2f}%")
        print(f"Annualised (1-day trades, 252d): {_annualised_simple(strategy_returns):+.2f}%")
        print(f"Signal Sharpe:      {_sharpe(strategy_returns):.2f}")
        print(f"Best signal day:    {signal_returns.max():+.2f}%")
        print(f"Worst signal day:   {signal_returns.min():+.2f}%")

    print("\n--- Buy & hold comparison ---")
    print(f"Avg return:         {all_returns.mean():+.3f}%")
    print(f"Median return:      {all_returns.median():+.3f}%")
    print(f"Annualised (B&H):   {_annualised_simple(all_returns):+.2f}%")
    print(f"B&H Sharpe:         {_sharpe(all_returns):.2f}")
    print(f"Up days:            {(all_returns > 0).sum()} ({(all_returns > 0).mean():.1%})")

    if not go_long.empty:
        print("\n--- Primary bucket accuracy (when engine took a large-move stance) ---")
        stanced = results[results["primary_bucket"] != "Neutral"]
        if not stanced.empty:
            correct = sum(
                (r["primary_bucket"] == "Large Up" and r["actual_return_pct"] >= 0.6)
                or (r["primary_bucket"] == "Large Down" and r["actual_return_pct"] <= -0.6)
                for _, r in stanced.iterrows()
            )
            print(f"Large-move accuracy: {correct}/{len(stanced)} = {correct / len(stanced):.1%}")

    print("\n--- Recommendation / strategy distribution ---")
    print(results["strategy_label"].value_counts().to_string())

    print("\n--- Primary bucket distribution ---")
    print(results["primary_bucket"].value_counts().to_string())

    # Save detailed results
    out_path = Path("data/historical_backtest_results.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_path, index=False)
    print(f"\nSaved detailed results: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical backtest from factor table")
    parser.add_argument(
        "--factors",
        type=str,
        default="data/historical_factors.parquet",
        help="Path to the factor table Parquet file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["engine", "score"],
        default="score",
        help=(
            "Decision strategy: 'engine' = full ScoringEngine logic, "
            "'score' = configurable primary-score threshold"
        ),
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Load and use the trained hybrid ML models if available",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.6,
        help="Primary rule-score threshold for GO LONG in 'score' mode",
    )
    parser.add_argument(
        "--rsi-cap",
        type=float,
        default=70.0,
        help="RSI cap for GO LONG in 'score' mode",
    )
    parser.add_argument(
        "--ath-cap",
        type=float,
        default=-0.5,
        help="ATH distance cap for GO LONG in 'score' mode (must be <= this value)",
    )
    parser.add_argument(
        "--ignore-technicals",
        action="store_true",
        help="Ignore RSI/ATH technical filters in 'score' mode",
    )
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    args = parser.parse_args()

    run_backtest(
        factors_path=args.factors,
        strategy=args.strategy,
        use_ml=args.ml,
        primary_threshold=args.primary_threshold,
        rsi_cap=args.rsi_cap,
        ath_cap=args.ath_cap,
        start=args.start,
        end=args.end,
        require_technicals_ok=not args.ignore_technicals,
    )


if __name__ == "__main__":
    main()
