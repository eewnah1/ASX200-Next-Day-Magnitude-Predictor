"""Historical backtest for the rule-based scoring engine.

Fetches roughly 12 months of daily data and walks forward one day at a time,
simulating the 13:30 AEST feature set with the information that would have
been available at that point.  It stores predictions/actuals and prints a
simple calibration report.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

sys.path.insert(0, "src")

from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.logging_config import setup_logging
from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scoring.engine import ScoringEngine, bucket_from_return
from asx200_mag_predictor.scoring.features import (
    classify_session,
    compute_catalyst_score,
    compute_cross_asset_alignment,
    compute_vol_regime,
)
from asx200_mag_predictor.storage.models import init_db
from asx200_mag_predictor.storage.repository import Repository
from asx200_mag_predictor.timezone import to_sydney


def _close_series(df: pd.DataFrame | pd.Series) -> pd.Series:
    """Return the closing-price series from a yfinance download."""
    if isinstance(df, pd.Series):
        return df.dropna()
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def _to_float(value) -> float | None:
    """Convert a scalar/pandas scalar to float without deprecation warnings."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _value_asof(series: pd.Series, dt: datetime) -> float | None:
    """Return the value at or before `dt`, or None."""
    try:
        val = series.asof(dt)
        if pd.isna(val):
            return None
        return _to_float(val)
    except Exception:  # noqa: S112
        return None


def _pct_change_at(series: pd.Series, dt: datetime, lookback_days: int = 1) -> float | None:
    """Percent change between `dt` and `dt - lookback_days` business days."""
    try:
        from pandas.tseries.offsets import BDay

        bday = BDay
    except Exception:  # noqa: S112
        bday = None
    if bday is None:
        prev_dt = dt - timedelta(days=lookback_days)
    else:
        prev_dt = dt - bday(lookback_days)
    cur = _value_asof(series, dt)
    prev = _value_asof(series, prev_dt)
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / prev * 100.0


def _realized_vol(series: pd.Series, dt: datetime, window: int = 5) -> float | None:
    """Annualised daily stdev over the `window` business days prior to `dt`."""
    from pandas.tseries.offsets import BDay

    start = dt - BDay(window * 2)  # grab enough to cover holidays
    try:
        slice_ = series.loc[start:dt].iloc[:-1]  # exclude current day
        if len(slice_) < window:
            return None
        rets = slice_.tail(window).pct_change().dropna() * 100
        if len(rets) < 2:
            return None
        return _to_float(rets.std() * 16.0)  # rule of 16
    except Exception:  # noqa: S112
        return None


def fetch_universe(end: datetime) -> dict[str, pd.Series]:
    """Download daily closing price histories for the factor proxies."""
    start = end - timedelta(days=420)
    symbols = {
        "axjo": "^AXJO",
        "vix": "^VIX",
        "gspc": "^GSPC",
        "ixic": "^IXIC",
        "dji": "^DJI",
        "aud": "AUDUSD=X",
        "iron": "FE=F",
    }
    data = {}
    for name, sym in symbols.items():
        try:
            df = yf.download(sym, start=start, end=end, progress=False, threads=False)
            if not df.empty:
                data[name] = _close_series(df)
        except Exception:  # noqa: S112
            pass
    return data


def run_backtest(months: int = 12, dry_run: bool = False) -> None:
    settings = get_settings()
    setup_logging(settings)
    init_db(settings)
    repo = Repository()
    engine = ScoringEngine(settings)

    end = datetime.now() + timedelta(days=1)
    data = fetch_universe(end)
    axjo = data.get("axjo")
    if axjo is None or len(axjo) < 30:
        print("Could not fetch enough ^AXJO history for backtest.")
        return

    cutoff = end - timedelta(days=30 * months)
    test_dates = [d for d in axjo.index if d >= cutoff]

    results: list[dict] = []
    for current_date in test_dates:
        current_date = pd.Timestamp(current_date)
        axjo_price = _value_asof(axjo, current_date)
        if axjo_price is None:
            continue

        # Realised vol and regime
        realized = _realized_vol(axjo, current_date, window=5)
        vix_val = _value_asof(data.get("vix", pd.Series(dtype=float)), current_date)
        vol_regime = compute_vol_regime(vix_val, None, realized)

        # Cross-asset proxies (signed changes)
        gspc_chg = _pct_change_at(data.get("gspc"), current_date, 1)
        ixic_chg = _pct_change_at(data.get("ixic"), current_date, 1)
        dji_chg = _pct_change_at(data.get("dji"), current_date, 1)
        aud_chg = _pct_change_at(data.get("aud"), current_date, 1)
        iron_chg = _pct_change_at(data.get("iron"), current_date, 1)

        alignment, magnitude = compute_cross_asset_alignment(
            us_futures_change_pct=gspc_chg,
            iron_ore_change_pct=iron_chg,
            aud_usd_change_pct=aud_chg,
            sp500_change_pct=gspc_chg,
            nasdaq_change_pct=ixic_chg,
            dow_change_pct=dji_chg,
            us_10y_change_bps=None,
            vix_change_pct=None,
        )

        # Session character from today's ASX move and a proxy ATR
        today_return = _pct_change_at(axjo, current_date, 1) or 0.0
        atr_window_end = current_date
        atr_window_start = current_date - timedelta(days=15)
        atr_window = axjo.loc[atr_window_start:atr_window_end]
        if len(atr_window) > 1 and axjo_price:
            atr = (
                _to_float(atr_window.max()) - _to_float(atr_window.min())
            ) / axjo_price * 100
        else:
            atr = None
        session_char = classify_session(
            today_return, 1.0, (abs(today_return) / atr) if atr else 1.0
        )

        # Catalyst score proxy from VIX / realised vol
        catalyst = compute_catalyst_score(
            int(3 if (realized and realized > 18) else 1),
            int(1 if (realized and realized > 22) else 0),
        )

        features = FeatureVector(
            a_vix=vix_val,
            atr_5d_pct=atr,
            realized_vol_annual=realized,
            vol_regime=vol_regime,
            catalyst_score=catalyst,
            high_impact_events_next_24h=3 if catalyst >= 3 else 1,
            high_impact_events_next_48h=1 if catalyst >= 4 else 0,
            us_futures_change_pct=gspc_chg,
            sp500_change_pct=gspc_chg,
            nasdaq_change_pct=ixic_chg,
            dow_change_pct=dji_chg,
            aud_usd_change_pct=aud_chg,
            iron_ore_change_pct=iron_chg,
            vix_change_pct=None,
            cross_asset_alignment_score=alignment,
            cross_asset_magnitude=magnitude,
            asx_open_to_now_return_pct=today_return,
            asx_session_character=session_char,
            spi_basis_pct=None,
            spi_momentum_pct=None,
        )

        prediction_for = to_sydney(current_date + timedelta(days=1))
        pred = engine.predict(features, prediction_for=prediction_for)

        if not dry_run:
            pred_id = repo.save_prediction(pred)
            actual_return = _pct_change_at(axjo, current_date + timedelta(days=1), 1) or 0.0
            repo.save_actual(pred_id, actual_return)
            results.append(
                {
                    "date": current_date.date().isoformat(),
                    "predicted": pred.bucket,
                    "actual_bucket": bucket_from_return(actual_return),
                    "probs": pred.probabilities.model_dump(),
                    "actual_return": round(actual_return, 4),
                }
            )
        else:
            results.append(
                {
                    "date": current_date.date().isoformat(),
                    "predicted": pred.bucket,
                    "probs": pred.probabilities.model_dump(),
                }
            )

    if not results:
        print("No backtest results produced.")
        return

    print(f"Backtest complete: {len(results)} days")
    if not dry_run:
        metrics = repo.calibration_metrics()
        print(metrics.model_dump_json(indent=2))
    else:
        print("Dry run; no actuals recorded.")


def main():
    parser = argparse.ArgumentParser(description="Run historical backtest")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_backtest(months=args.months, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
