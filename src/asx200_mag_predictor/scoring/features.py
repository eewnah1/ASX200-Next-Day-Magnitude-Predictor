"""Feature engineering helpers.

These functions turn raw market snapshots into the numeric inputs the
rule-based scoring engine expects.  They are deliberately defensive:
missing fields fall back to neutral defaults and raise data-quality flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from asx200_mag_predictor.logging_config import get_logger
from asx200_mag_predictor.models import DataQualityFlags, FeatureVector
from asx200_mag_predictor.timezone import now_sydney

logger = get_logger(__name__)


def _safe_pct_change(series: list[float] | None) -> float | None:
    """Return percent change between first and last value, or None."""
    if not series or len(series) < 2:
        return None
    first, last = series[0], series[-1]
    if first is None or last is None or first == 0:
        return None
    return (last - first) / first * 100.0


def _atr(
    closes: list[float], highs: list[float], lows: list[float], period: int = 5
) -> float | None:
    """Simple average true range over the last `period` bars."""
    if not closes or len(closes) < period or len(highs) < period or len(lows) < period:
        return None
    trs: list[float] = []
    for i in range(-period, 0):
        if i == -len(closes):
            tr = highs[i] - lows[i]
        else:
            prev_close = closes[i - 1]
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev_close),
                abs(lows[i] - prev_close),
            )
        trs.append(tr)
    return mean(trs) if trs else None


def _rule_of_16(realized_daily_stdev_pct: float) -> float:
    """Annualise daily stdev using the rule of 16.

    A daily stdev of X% annualises to roughly 16 * X%.
    This mirrors the VIX convention (annualised vol expectation).
    """
    return realized_daily_stdev_pct * 16.0


def compute_vol_regime(
    a_vix: float | None, atr_5d_pct: float | None, realized_vol_annual: float | None
) -> int:
    """Map realised/implied volatility into a 0-4 regime.

    Regime 0 = calm (< 12% annualised)
    Regime 1 = normal (12-16%)
    Regime 2 = elevated (16-22%)
    Regime 3 = high (22-30%)
    Regime 4 = extreme (> 30%)
    """
    vol = a_vix
    if vol is None:
        vol = realized_vol_annual
    if vol is None and atr_5d_pct is not None:
        # ATR as % of price is a proxy for daily range vol.
        vol = _rule_of_16(atr_5d_pct)
    if vol is None:
        return 1  # neutral default
    if vol < 12.0:
        return 0
    if vol < 16.0:
        return 1
    if vol < 22.0:
        return 2
    if vol < 30.0:
        return 3
    return 4


def compute_catalyst_score(events_24h: int, events_48h: int) -> int:
    """Score scheduled high-impact events 0-5.

    Weight events in the next 24h more heavily.
    """
    score = min(events_24h * 2, 4) + min(events_48h, 1)
    return min(max(int(score), 0), 5)


def compute_cross_asset_alignment(
    *,
    us_futures_change_pct: float | None,
    iron_ore_change_pct: float | None,
    aud_usd_change_pct: float | None,
    sp500_change_pct: float | None,
    nasdaq_change_pct: float | None,
    dow_change_pct: float | None,
    us_10y_change_bps: float | None,
    vix_change_pct: float | None,
) -> tuple[float, float]:
    """Return (alignment_score, magnitude).

    alignment_score ranges -1 (strongly risk-off/negative for ASX) to +1 (strongly pro-ASX).
    magnitude is the average absolute move across the key drivers, in percent-equivalent units.
    """
    drivers: list[float] = []
    if us_futures_change_pct is not None:
        drivers.append(us_futures_change_pct)
    if iron_ore_change_pct is not None:
        drivers.append(iron_ore_change_pct)
    if aud_usd_change_pct is not None:
        drivers.append(aud_usd_change_pct)
    if sp500_change_pct is not None:
        drivers.append(sp500_change_pct)
    if nasdaq_change_pct is not None:
        drivers.append(nasdaq_change_pct)
    if dow_change_pct is not None:
        drivers.append(dow_change_pct)
    if us_10y_change_bps is not None:
        drivers.append(us_10y_change_bps / 10.0)  # rough conversion to %-equivalent
    if vix_change_pct is not None:
        drivers.append(-vix_change_pct)  # VIX rise is risk-off

    if not drivers:
        return 0.0, 0.0

    alignment = mean(drivers) / 2.0  # scale so +/-2% average -> ~1.0
    alignment = max(-1.0, min(1.0, alignment))
    magnitude = mean(abs(d) for d in drivers)
    return alignment, magnitude


def classify_session(
    asx_open_to_now_return_pct: float | None,
    current_volume_vs_20d_avg: float | None,
    current_range_vs_atr: float | None,
) -> str:
    """Classify the current ASX session character."""
    if asx_open_to_now_return_pct is None:
        return "unknown"
    abs_ret = abs(asx_open_to_now_return_pct)
    vol_ratio = current_volume_vs_20d_avg or 1.0
    range_ratio = current_range_vs_atr or 1.0
    if abs_ret > 0.25 and (vol_ratio > 1.1 or range_ratio > 1.0):
        return "trend"
    if abs_ret < 0.15 and range_ratio < 0.8 and vol_ratio < 1.0:
        return "range"
    return "mixed"


@dataclass
class RawMarketData:
    """Container produced by data fetchers."""

    asx_cash: dict[str, Any] | None = None
    spi_futures: dict[str, Any] | None = None
    a_vix: dict[str, Any] | None = None
    commodities: dict[str, Any] | None = None
    fx: dict[str, Any] | None = None
    us_assets: dict[str, Any] | None = None
    calendar: dict[str, Any] | None = None
    volume: dict[str, Any] | None = None


def build_features(raw: RawMarketData) -> tuple[FeatureVector, DataQualityFlags]:
    """Transform raw snapshots into a FeatureVector and data-quality flags."""
    flags = DataQualityFlags()
    feats: dict[str, Any] = {"fetched_at": now_sydney(), "sources": {}}

    def _hist(key: str, field: str) -> list[float] | None:
        obj = getattr(raw, key)
        if obj is None:
            return None
        if field in obj:
            return obj.get(field)
        series = obj.get("series")
        if isinstance(series, dict):
            return series.get(field)
        return None

    def _last(key: str, field: str) -> float | None:
        hist = _hist(key, field)
        if hist is None:
            return None
        if isinstance(hist, (list, tuple)):
            if not hist:
                return None
            if field == "prev_close" and len(hist) >= 2:
                return hist[-2]
            return hist[-1]
        return hist

    # A-VIX and realised vol
    a_vix_close = _last("a_vix", "close")
    if a_vix_close is None:
        flags.a_vix = "missing or stale"

    asx_close = _hist("asx_cash", "close") or []
    asx_high = _hist("asx_cash", "high") or []
    asx_low = _hist("asx_cash", "low") or []

    atr_5d = None
    atr_5d_pct = None
    realized_vol_annual = None
    if len(asx_close) >= 5:
        atr_5d = _atr(asx_close, asx_high, asx_low, period=5)
        last_price = asx_close[-1]
        if atr_5d and last_price:
            atr_5d_pct = atr_5d / last_price * 100.0
            # Realised vol estimate: daily stdev of last 5 returns, annualised.
            returns = [
                (asx_close[i] - asx_close[i - 1]) / asx_close[i - 1] * 100
                for i in range(-4, 0)
                if asx_close[i - 1] != 0
            ]
            if len(returns) >= 4:

                stdev = (sum((r - mean(returns)) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                realized_vol_annual = _rule_of_16(stdev)
    else:
        flags.asx_cash = "missing or stale"

    feats["a_vix"] = a_vix_close
    feats["atr_5d_pct"] = atr_5d_pct
    feats["realized_vol_annual"] = realized_vol_annual
    feats["vol_regime"] = compute_vol_regime(a_vix_close, atr_5d_pct, realized_vol_annual)

    # Catalyst
    events_24h = 0
    events_48h = 0
    if raw.calendar:
        events_24h = raw.calendar.get("high_impact_24h", 0) or 0
        events_48h = raw.calendar.get("high_impact_48h", 0) or 0
    else:
        flags.calendar = "missing or stale"
    feats["catalyst_score"] = compute_catalyst_score(events_24h, events_48h)
    feats["high_impact_events_next_24h"] = events_24h
    feats["high_impact_events_next_48h"] = events_48h

    # Cross-asset
    def _change(raw_key: str, field: str = "change_pct") -> float | None:
        obj = getattr(raw, raw_key) or {}
        return obj.get(field)

    us_futures = _change("us_assets", "us_futures_change_pct")
    iron_ore = _change("commodities", "iron_ore_change_pct")
    aud = _change("fx", "aud_usd_change_pct")
    sp500 = _change("us_assets", "sp500_change_pct")
    nasdaq = _change("us_assets", "nasdaq_change_pct")
    dow = _change("us_assets", "dow_change_pct")
    us10y = _change("us_assets", "us_10y_change_bps")
    vix = _change("us_assets", "vix_change_pct")

    if all(v is None for v in [us_futures, iron_ore, aud, sp500, nasdaq, dow, us10y, vix]):
        flags.us_assets = "missing or stale"
        flags.commodities = "missing or stale"
        flags.fx = "missing or stale"

    alignment, magnitude = compute_cross_asset_alignment(
        us_futures_change_pct=us_futures,
        iron_ore_change_pct=iron_ore,
        aud_usd_change_pct=aud,
        sp500_change_pct=sp500,
        nasdaq_change_pct=nasdaq,
        dow_change_pct=dow,
        us_10y_change_bps=us10y,
        vix_change_pct=vix,
    )
    feats["us_futures_change_pct"] = us_futures
    feats["iron_ore_change_pct"] = iron_ore
    feats["aud_usd_change_pct"] = aud
    feats["sp500_change_pct"] = sp500
    feats["nasdaq_change_pct"] = nasdaq
    feats["dow_change_pct"] = dow
    feats["us_10y_change_bps"] = us10y
    feats["vix_change_pct"] = vix
    feats["cross_asset_alignment_score"] = alignment
    feats["cross_asset_magnitude"] = magnitude

    # Session character
    if raw.volume:
        open_to_now = raw.volume.get("asx_open_to_now_return_pct")
        vol_ratio = raw.volume.get("current_volume_vs_20d_avg")
        range_ratio = raw.volume.get("current_range_vs_atr")
        feats["asx_open_to_now_return_pct"] = open_to_now
        feats["current_volume_vs_20d_avg"] = vol_ratio
        feats["current_range_vs_atr"] = range_ratio
        feats["asx_session_character"] = classify_session(open_to_now, vol_ratio, range_ratio)
    else:
        flags.volume = "missing or stale"
        feats["asx_session_character"] = "unknown"

    # SPI basis
    if raw.spi_futures and raw.asx_cash:
        spi_series = _hist("spi_futures", "close") or []
        asx_last = _last("asx_cash", "close")
        if isinstance(spi_series, (list, tuple)) and len(spi_series) >= 1 and asx_last:
            spi_last = spi_series[-1]
            feats["spi_basis_pct"] = (spi_last - asx_last) / asx_last * 100.0
            if len(spi_series) >= 2 and spi_series[-2] != 0:
                feats["spi_momentum_pct"] = (spi_last - spi_series[-2]) / spi_series[-2] * 100.0
    else:
        flags.spi_futures = "missing or stale"

    return FeatureVector(**feats), flags
