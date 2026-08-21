"""Empirical high-conviction overlay mined from the user's Daily Rates CSV.

The CSV contains daily percentage returns for several investment options.
Backtests against the `Australian Shares` column found two high-conviction
next-day rules that fire on the subset of days where same-day International
Shares (and related option) moves are extreme:

- Large Up: International Shares > 2.0%, High Growth > -0.5%, Balanced < 1.0%
  -> 94.7% positive next-day direction for Australian Shares (19 signals)
- Large Down: International Shares < -1.5%, High Growth < -1.0%, Balanced > -1.0%
  -> 100.0% negative next-day direction for Australian Shares, 75.0% large down
     (12 signals)

These rules are used as a transparent overlay: when same-day daily-rates
conditions are available before the 2 PM fund-switch cutoff, the engine can
override the primary bucket with the historically high-accuracy bucket and
expose the signal in the dashboard. If daily-rates data is not available in
time, the overlay is skipped and the ML/primary model remains the active signal.
"""

from __future__ import annotations

import math
from typing import Any

from asx200_mag_predictor.models import FeatureVector


class HighConvictionSignal:
    """Triggered daily-rates high-conviction signal."""

    def __init__(
        self,
        bucket: str,
        historical_accuracy: float,
        mean_return_pct: float | None,
        reason: str,
        signals: list[str],
    ) -> None:
        self.bucket = bucket
        self.historical_accuracy = historical_accuracy
        self.mean_return_pct = mean_return_pct
        self.reason = reason
        self.signals = signals


def _safe(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _check(value: float | None, op: str, threshold: float) -> bool:
    v = _safe(value)
    if v is None:
        return False
    if op == ">":
        return v > threshold
    if op == ">=":
        return v >= threshold
    if op == "<":
        return v < threshold
    if op == "<=":
        return v <= threshold
    return False


# Ordered list: first match wins.
HIGH_CONVICTION_RULES: list[dict[str, Any]] = [
    {
        "bucket": "Large Up",
        "conditions": {
            "international_shares": (">", 2.0),
            "high_growth": (">", -0.5),
            "balanced": ("<", 1.0),
        },
        "historical_accuracy": 0.9474,
        "mean_return_pct": 1.10,
        "signals_n": 19,
        "description": (
            "International Shares up >2.0%, High Growth >-0.5%, Balanced <1.0% "
            "→ next-day ASX 200 up (94.7% historical directional hit rate, 19 signals)"
        ),
    },
    {
        "bucket": "Large Down",
        "conditions": {
            "international_shares": ("<", -1.5),
            "high_growth": ("<", -1.0),
            "balanced": (">", -1.0),
        },
        "historical_accuracy": 1.0,
        "mean_return_pct": -1.74,
        "signals_n": 12,
        "description": (
            "International Shares down <-1.5%, High Growth <-1.0%, Balanced >-1.0% "
            "→ next-day ASX 200 down (100% historical directional hit rate, 12 signals, "
            "75.0% large down)"
        ),
    },
]


def _rule_matches(rule: dict[str, Any], daily_rates: dict[str, float] | None) -> bool:
    if not daily_rates:
        return False
    for field, (op, threshold) in rule["conditions"].items():
        if not _check(daily_rates.get(field), op, threshold):
            return False
    return True


def evaluate_high_conviction(fv: FeatureVector) -> HighConvictionSignal | None:
    """Return a high-conviction signal if the current daily-rates snapshot matches a rule."""
    if not fv.daily_rates:
        return None
    for rule in HIGH_CONVICTION_RULES:
        if _rule_matches(rule, fv.daily_rates):
            signals = []
            for field, (op, threshold) in rule["conditions"].items():
                value = fv.daily_rates.get(field)
                value_str = f"{value:+.2f}%" if value is not None else "n/a"
                signals.append(f"{field.replace('_', ' ').title()} ({value_str}) {op} {threshold}%")
            return HighConvictionSignal(
                bucket=rule["bucket"],
                historical_accuracy=rule["historical_accuracy"],
                mean_return_pct=rule.get("mean_return_pct"),
                reason=rule["description"],
                signals=signals,
            )
    return None
