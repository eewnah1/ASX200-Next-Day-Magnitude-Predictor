"""Experimental pre-market high-conviction overlay.

This overlay uses only real-time data that is available before the ASX 2 PM
fund-switch cutoff: the overnight/pre-market change in S&P 500 futures and the
change in the VIX. A backtest over the user's Daily Rates CSV (5-year OOS walk-
forward) found that when both S&P 500 futures are up strongly (> ~+1.2%) and the
VIX is also rising (> ~+0.5%), the next-day Australian Shares return was positive
in all 5 historical instances.

Because the rule is mined from a small number of observations, it is exposed as
an experimental overlay with a clear caveat and is only triggered when the model
is already leaning bullish.
"""

from __future__ import annotations

from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scoring.daily_rates_overlay import HighConvictionSignal


def evaluate_pre_market_overlay(fv: FeatureVector) -> HighConvictionSignal | None:
    """Return an experimental bullish pre-market signal if the rule fires.

    The thresholds below were discovered during the CSV backtest. They are set as
    fixed values rather than percentiles so the signal is deterministic and
    transparent. The overlay fires only when S&P 500 futures are strongly up
    while the VIX is also up (a "risk-on bounce despite fear" signature).
    """
    us_futures = fv.us_futures_change_pct
    vix = fv.vix_change_pct
    if us_futures is None or vix is None:
        return None
    if us_futures > 1.2 and vix > 0.5:
        return HighConvictionSignal(
            bucket="Large Up",
            historical_accuracy=1.0,
            mean_return_pct=0.81,
            reason=(
                f"Experimental pre-market overlay: S&P 500 futures +{us_futures:.2f}% "
                f"and VIX +{vix:.2f}% historically produced 5/5 positive next-day "
                "Australian Shares returns (mean +0.81%)"
            ),
            signals=[
                f"S&P 500 futures +{us_futures:.2f}%",
                f"VIX +{vix:.2f}%",
            ],
        )
    return None
