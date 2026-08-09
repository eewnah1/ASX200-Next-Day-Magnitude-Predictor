"""Backtest the ASX200 high-conviction overlay against the Daily Rates CSV.

The CSV columns are daily percentage returns for a set of investment options.
We use the same-day `International Shares`, `High Growth`, `Balanced` and
`Conservative Balanced` columns as inputs to the empirical high-conviction
rules and the next-day `Australian Shares` column as the target outcome.

Usage:
    python notebooks/daily_rates_backtest.py \
        --csv /path/to/Daily_Rates_01_Jul_2008_-_08_Aug_2026.csv

The script writes `data/daily_rates_backtest_summary.json` and a detailed
`data/daily_rates_backtest_results.parquet`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scoring.daily_rates_overlay import HIGH_CONVICTION_RULES
from asx200_mag_predictor.scoring.engine import ScoringEngine, bucket_from_return

COLUMN_MAP = {
    "International Shares": "international_shares",
    "High Growth": "high_growth",
    "Balanced": "balanced",
    "Conservative Balanced": "conservative_balanced",
    "Socially Aware": "socially_aware",
    "Indexed Diversified": "indexed_diversified",
    "Stable": "stable",
    "Diversified Fixed Interest": "diversified_fixed_interest",
    "Australian Shares": "australian_shares",
    "Cash": "cash",
}


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Rate Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Drop rows where all return columns are zero / missing.
    ret_cols = [c for c in df.columns if c in COLUMN_MAP]
    df = df[(df[ret_cols].abs() > 1e-6).any(axis=1)]
    return df


def _feature_vector(row: pd.Series, daily_rates: dict[str, float]) -> FeatureVector:
    return FeatureVector(daily_rates=daily_rates, data_as_of=row.name)


def _accuracy_for_bucket(
    mask: pd.Series, target: pd.Series, bucket: str
) -> tuple[int, float, float]:
    n = int(mask.sum())
    if n == 0:
        return n, 0.0, 0.0
    if bucket == "Large Up":
        direction_acc = (target[mask] > 0).mean()
        bucket_acc = (target[mask] >= 0.6).mean()
    else:
        direction_acc = (target[mask] < 0).mean()
        bucket_acc = (target[mask] <= -0.6).mean()
    return n, float(direction_acc), float(bucket_acc)


def run_backtest(csv_path: Path) -> dict:
    settings = get_settings()
    engine = ScoringEngine(settings)
    # Force the model into rule-only mode so the overlay is fully in control.
    engine.hybrid_ml.available = False

    df = _load_csv(csv_path)
    target = df["Australian Shares"].shift(-1)
    valid = target.notna()
    df = df[valid]
    target = target[valid]

    records: list[dict] = []
    for idx, row in df.iterrows():
        daily_rates = {
            v: float(row[k])
            for k, v in COLUMN_MAP.items()
            if k in row and pd.notna(row[k])
        }
        fv = _feature_vector(row, daily_rates)
        pred = engine.predict(fv, prediction_for=idx)

        actual = float(target.loc[idx])
        records.append(
            {
                "date": idx,
                "predicted_bucket": pred.primary_bucket,
                "predicted_recommendation": pred.recommendation,
                "high_conviction": pred.high_conviction,
                "high_conviction_historical_accuracy": pred.high_conviction_historical_accuracy,
                "actual_return_pct": actual,
                "actual_bucket": bucket_from_return(actual),
            }
        )

    results = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    hc_results = results[results["high_conviction"]]
    if hc_results.empty:
        summary = {
            "total_days": len(results),
            "high_conviction_signals": 0,
            "directional_accuracy": None,
            "bucket_accuracy": None,
            "by_bucket": {},
        }
    else:
        by_bucket: dict[str, dict] = {}
        for bucket in ["Large Up", "Large Down"]:
            subset = hc_results[hc_results["predicted_bucket"] == bucket]
            if subset.empty:
                continue
            n = len(subset)
            if bucket == "Large Up":
                direction_acc = (subset["actual_return_pct"] > 0).mean()
                bucket_acc = (subset["actual_return_pct"] >= 0.6).mean()
            else:
                direction_acc = (subset["actual_return_pct"] < 0).mean()
                bucket_acc = (subset["actual_return_pct"] <= -0.6).mean()
            by_bucket[bucket] = {
                "signals": n,
                "directional_accuracy": round(float(direction_acc), 4),
                "bucket_accuracy": round(float(bucket_acc), 4),
                "mean_return_pct": round(float(subset["actual_return_pct"].mean()), 4),
            }

        up_mask = hc_results["predicted_bucket"] == "Large Up"
        down_mask = hc_results["predicted_bucket"] == "Large Down"
        overall_correct = int(
            (up_mask & (hc_results["actual_return_pct"] > 0)).sum()
            + (down_mask & (hc_results["actual_return_pct"] < 0)).sum()
        )
        overall_bucket_correct = int(
            (up_mask & (hc_results["actual_return_pct"] >= 0.6)).sum()
            + (down_mask & (hc_results["actual_return_pct"] <= -0.6)).sum()
        )
        summary = {
            "total_days": len(results),
            "high_conviction_signals": len(hc_results),
            "signal_coverage": round(len(hc_results) / len(results), 4),
            "directional_accuracy": round(overall_correct / len(hc_results), 4),
            "bucket_accuracy": round(overall_bucket_correct / len(hc_results), 4),
            "by_bucket": by_bucket,
        }

    # Recompute per-rule stats from the CSV for an accurate rule card.
    rule_rows = []
    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    for rule in HIGH_CONVICTION_RULES:
        mask = pd.Series(True, index=df.index)
        for field, (op, threshold) in rule["conditions"].items():
            col = reverse_map.get(field)
            if col is None or col not in df.columns:
                mask = pd.Series(False, index=df.index)
                break
            series = pd.to_numeric(df[col], errors="coerce")
            if op == ">":
                mask = mask & (series > threshold)
            elif op == ">=":
                mask = mask & (series >= threshold)
            elif op == "<":
                mask = mask & (series < threshold)
            elif op == "<=":
                mask = mask & (series <= threshold)
        mask = mask & target.notna()
        n = int(mask.sum())
        if n == 0:
            continue
        if rule["bucket"] == "Large Up":
            direction_acc = (target[mask] > 0).mean()
        else:
            direction_acc = (target[mask] < 0).mean()
        rule_rows.append(
            {
                "bucket": rule["bucket"],
                "conditions": [
                    [field, op, threshold]
                    for field, (op, threshold) in rule["conditions"].items()
                ],
                "signals": n,
                "directional_accuracy": round(float(direction_acc), 4),
            }
        )
    summary["rules"] = rule_rows

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "daily_rates_backtest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    results_path = out_dir / "daily_rates_backtest_results.parquet"
    results.to_parquet(results_path, index=False)

    print("\n=== ASX200 Daily Rates High-Conviction Backtest ===\n")
    print(f"CSV source:         {csv_path}")
    print(f"Period:             {results['date'].min().date()} to {results['date'].max().date()}")
    print(f"Total trading days: {summary['total_days']}")
    coverage = summary.get('signal_coverage', 0)
    print(f"HC signals:         {summary['high_conviction_signals']} ({coverage:.1%})")
    if summary["high_conviction_signals"]:
        print(f"Directional accuracy: {summary['directional_accuracy']:.1%}")
        print(f"Bucket accuracy:      {summary['bucket_accuracy']:.1%}")
        for bucket, stats in summary["by_bucket"].items():
            print(
                f"  {bucket}: {stats['signals']} signals, "
                f"direction {stats['directional_accuracy']:.1%}, "
                f"bucket {stats['bucket_accuracy']:.1%}, "
                f"mean {stats['mean_return_pct']:+.3f}%"
            )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved results: {results_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest ASX200 daily-rates high-conviction overlay"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "/home/ubuntu/attachments/baeeabd4-344e-442d-866d-337439cbd057/Daily_Rates_01_Jul_2008_-_08_Aug_2026.csv"
        ),
        help="Path to the Daily Rates CSV file",
    )
    args = parser.parse_args()
    run_backtest(args.csv)


if __name__ == "__main__":
    main()
