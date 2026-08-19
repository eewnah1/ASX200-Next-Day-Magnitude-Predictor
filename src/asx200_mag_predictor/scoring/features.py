"""Feature engineering helpers.

These functions turn raw market snapshots into the numeric inputs the
rule-based scoring engine expects.  They are deliberately defensive:
missing fields fall back to neutral defaults and raise data-quality flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
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


def _clamp(value: float | None, low: float, high: float, default: float = 0.0) -> float:
    if value is None or math.isnan(value):
        return default
    return max(low, min(high, value))


def _score_tv_sector_spread(value: float | None) -> float:
    """Map Financials - Materials daily % change into a directional score.

    Strong financials outperformance is bullish for the index; materials
    outperformance is mixed because resource profits are often priced in USD.
    """
    if value is None or math.isnan(value):
        return 0.0
    return _clamp(value * 0.5, -1.5, 1.5)


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


def score_financials_vs_materials(weighted_diff_pct: float | None) -> float | None:
    """Map Financials vs Materials weighted diff to a score."""
    if weighted_diff_pct is None:
        return None
    if weighted_diff_pct > 0.8:
        return 1.5
    if weighted_diff_pct > 0.3:
        return 0.7
    if weighted_diff_pct > -0.3:
        return 0.0
    if weighted_diff_pct > -0.8:
        return -0.7
    return -1.5


def score_housing_credit_pulse(pulse_score: float | None) -> float | None:
    """Map 0-10 housing/credit pulse score to an ASX directional score."""
    if pulse_score is None:
        return None
    if pulse_score >= 8:
        return 1.5
    if pulse_score >= 5:
        return 0.6
    if pulse_score >= 3:
        return 0.0
    return -1.2


def score_china_steel_property(composite_return_pct: float | None) -> float | None:
    """Map China steel/property proxy composite return to a score."""
    if composite_return_pct is None:
        return None
    if composite_return_pct > 0.8:
        return 1.2
    if composite_return_pct > 0.3:
        return 0.6
    if composite_return_pct > -0.3:
        return 0.0
    if composite_return_pct > -0.8:
        return -0.6
    if composite_return_pct > -1.2:
        return -1.2
    return -1.8


def score_heavyweight_idio(
    weighted_return_pct: float | None, news_boost: float = 0.0
) -> float | None:
    """Map CBA+BHP weighted return to an idiosyncratic score, with optional news boost."""
    if weighted_return_pct is None:
        return None
    if weighted_return_pct > 1.2:
        score = 1.3
    elif weighted_return_pct > 0.5:
        score = 0.6
    elif weighted_return_pct > -0.5:
        score = 0.0
    elif weighted_return_pct > -1.2:
        score = -0.6
    else:
        score = -1.3
    if news_boost and score != 0.0:
        score = score * (1.0 + news_boost)
    return score


def compute_rsi(closes: list[float], window: int = 14) -> float | None:
    """Simple RSI from a list of closes."""
    if len(closes) < window + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-window, 0):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)
    avg_gain = gains / window
    avg_loss = losses / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def score_rsi(rsi: float | None) -> float | None:
    """Map RSI 14 to a directional score (positive = oversold/bullish)."""
    if rsi is None:
        return None
    if rsi > 70:
        return -1.5
    if rsi > 60:
        return -0.5
    if rsi >= 40:
        return 0.0
    if rsi >= 30:
        return 0.5
    return 1.5


def compute_distance_from_high(closes: list[float], days: int | None = None) -> float | None:
    """Return percent distance from the most recent close to a trailing high.

    days=None uses the all-time high in the supplied series.
    """
    if not closes:
        return None
    if days is not None and len(closes) < days:
        return None
    window = closes[-days:] if days else closes
    high = max(window)
    last = closes[-1]
    if high == 0:
        return None
    return (last - high) / high * 100.0


def score_ath_distance(distance_pct: float | None) -> float | None:
    """Profit-taking risk based on proximity to all-time / trailing highs."""
    if distance_pct is None:
        return None
    if distance_pct > -0.5:
        return -1.5
    if distance_pct > -1.5:
        return -0.8
    if distance_pct > -3.0:
        return -0.3
    return 0.0


def compute_bollinger_position(
    closes: list[float], window: int = 20, num_std: float = 2.0
) -> float | None:
    """Z-score of the last close within a 20-day, 2-std Bollinger Band."""
    if len(closes) < window:
        return None
    window_closes = closes[-window:]
    sma = mean(window_closes)
    variance = sum((x - sma) ** 2 for x in window_closes) / window
    std = variance**0.5
    if std == 0:
        return None
    return (closes[-1] - sma) / std


def score_bollinger(position: float | None) -> float | None:
    """Map Bollinger position to a mean-reversion score."""
    if position is None:
        return None
    if position > 2.0:
        return -0.7
    if position < -2.0:
        return 0.7
    return 0.0


def compute_momentum_exhaustion(
    rsi: float | None, index_5d_return_pct: float | None
) -> float | None:
    """Extra profit-taking or bounce potential when a run coincides with RSI extremes."""
    if rsi is None or index_5d_return_pct is None:
        return None
    if index_5d_return_pct > 2.5 and rsi > 65:
        return -1.0
    if index_5d_return_pct < -2.5 and rsi < 35:
        return 1.0
    return 0.0


def compute_profit_taking_combo(
    rsi: float | None,
    ath_distance_pct: float | None,
    index_5d_return_pct: float | None,
) -> float:
    """Significant mean-reversion / profit-taking surge when overbought + near ATH + strong run."""
    if rsi is None or ath_distance_pct is None or index_5d_return_pct is None:
        return 0.0
    if rsi > 70 and ath_distance_pct > -0.5 and index_5d_return_pct > 2.5:
        return -1.5
    return 0.0


@dataclass
class RawMarketData:
    """Container produced by data fetchers."""

    asx_cash: dict[str, Any] | None = None
    spi_futures: dict[str, Any] | None = None
    a_vix: dict[str, Any] | None = None
    commodities: dict[str, Any] | None = None
    fx: dict[str, Any] | None = None
    us_assets: dict[str, Any] | None = None
    financials_vs_materials: dict[str, Any] | None = None
    housing_credit: dict[str, Any] | None = None
    china_pulse: dict[str, Any] | None = None
    heavyweight_idio: dict[str, Any] | None = None
    calendar: dict[str, Any] | None = None
    volume: dict[str, Any] | None = None
    tradingview: dict[str, Any] | None = None
    alpha_vantage: dict[str, Any] | None = None
    source_status: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _status_map(raw: RawMarketData) -> dict[str, dict[str, Any]]:
    """Map source_status list to a dict keyed by source name."""
    return {s.get("name", "unknown"): s for s in raw.source_status or []}


def _source_flag(status: dict[str, Any] | None) -> str:
    if not status:
        return "failed"
    return status.get("status", "failed")


def _latest_success_timestamp(raw: RawMarketData) -> datetime | None:
    """Return the timestamp of the latest available market data (calendar excluded)."""
    from zoneinfo import ZoneInfo

    latest: datetime | None = None
    for s in raw.source_status or []:
        if s.get("name") == "calendar":
            continue
        ts = s.get("last_success_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            # Ignore future fetch timestamps so stale/last-available data is shown truthfully.
            if latest is None or (dt > latest and dt < now_sydney()):
                latest = dt
        except Exception:  # noqa: BLE001
            continue
    return latest


def build_features(raw: RawMarketData) -> tuple[FeatureVector, DataQualityFlags]:
    """Transform raw snapshots into a FeatureVector and data-quality flags."""
    flags = DataQualityFlags()
    statuses = _status_map(raw)
    feats: dict[str, Any] = {
        "fetched_at": now_sydney(),
        "data_as_of": _latest_success_timestamp(raw),
        "sources": {},
        "source_status": raw.source_status or [],
        "errors": raw.errors or [],
    }

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
        flags.a_vix = _source_flag(statuses.get("a_vix"))

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
            returns = [
                (asx_close[i] - asx_close[i - 1]) / asx_close[i - 1] * 100
                for i in range(-4, 0)
                if asx_close[i - 1] != 0
            ]
            if len(returns) >= 4:
                stdev = (sum((r - mean(returns)) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                realized_vol_annual = _rule_of_16(stdev)
    else:
        flags.asx_cash = _source_flag(statuses.get("asx_cash"))

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
        flags.calendar = _source_flag(statuses.get("calendar"))
    feats["catalyst_score"] = compute_catalyst_score(events_24h, events_48h)
    feats["high_impact_events_next_24h"] = events_24h
    feats["high_impact_events_next_48h"] = events_48h

    # Cross-asset
    def _change(raw_key: str, field: str = "change_pct") -> float | None:
        obj = getattr(raw, raw_key) or {}
        return obj.get(field)

    us_futures = _change("us_assets", "us_futures_change_pct")
    iron_ore = _change("commodities", "iron_ore_change_pct")
    gold = _change("commodities", "gold_change_pct")
    silver = _change("commodities", "silver_change_pct")
    oil = _change("commodities", "oil_change_pct")
    copper = _change("commodities", "copper_change_pct")
    aud = _change("fx", "aud_usd_change_pct")
    sp500 = _change("us_assets", "sp500_change_pct")
    nasdaq = _change("us_assets", "nasdaq_change_pct")
    dow = _change("us_assets", "dow_change_pct")
    us10y = _change("us_assets", "us_10y_change_bps")
    vix = _change("us_assets", "vix_change_pct")

    core = [us_futures, iron_ore, aud, sp500, nasdaq, dow, us10y, vix]
    if all(v is None for v in core):
        flags.us_assets = _source_flag(statuses.get("us_assets"))
        flags.commodities = _source_flag(statuses.get("commodities"))
        flags.fx = _source_flag(statuses.get("fx"))

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
    feats["gold_change_pct"] = gold
    feats["silver_change_pct"] = silver
    feats["oil_change_pct"] = oil
    feats["copper_change_pct"] = copper
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
        session_date = raw.volume.get("session_date")
        fallback = raw.volume.get("fallback")
        feats["asx_open_to_now_return_pct"] = open_to_now
        feats["current_volume_vs_20d_avg"] = vol_ratio
        feats["current_range_vs_atr"] = range_ratio
        feats["asx_session_character"] = classify_session(open_to_now, vol_ratio, range_ratio)
        if session_date:
            feats["sources"]["asx_session_date"] = session_date
        if fallback:
            feats["sources"]["asx_session_fallback"] = fallback
    else:
        flags.volume = _source_flag(statuses.get("volume"))
        feats["asx_session_character"] = "unknown"

    # New high-priority factors
    fvm = raw.financials_vs_materials or {}
    feats["financials_minus_materials_1d_pct"] = fvm.get("diff_1d_pct")
    feats["financials_minus_materials_2d_pct"] = fvm.get("diff_2d_pct")
    feats["financials_minus_materials_3d_pct"] = fvm.get("diff_3d_pct")
    feats["financials_minus_materials_5d_pct"] = fvm.get("diff_5d_pct")
    feats["financials_minus_materials_weighted_pct"] = fvm.get("weighted_diff_pct")
    feats["financials_vs_materials_score"] = score_financials_vs_materials(
        fvm.get("weighted_diff_pct")
    )
    if not fvm:
        flags.financials_vs_materials = _source_flag(statuses.get("financials_vs_materials"))

    hc = raw.housing_credit or {}
    feats["housing_credit_pulse_score"] = hc.get("pulse_score")
    feats["housing_credit_pulse_sources"] = hc.get("sources", [])
    if not hc:
        flags.housing_credit = _source_flag(statuses.get("housing_credit"))

    cp = raw.china_pulse or {}
    feats["china_steel_property_score"] = score_china_steel_property(cp.get("composite_return_pct"))
    feats["china_steel_property_return_pct"] = cp.get("composite_return_pct")
    feats["china_steel_property_sources"] = list((cp.get("per_ticker_1d") or {}).keys())
    if not cp:
        flags.china_steel_property = _source_flag(statuses.get("china_pulse"))

    hw = raw.heavyweight_idio or {}
    feats["heavyweight_idio_return_pct"] = hw.get("weighted_change_pct")
    feats["heavyweight_idio_news_boost"] = hw.get("news_boost", 0.0)
    feats["heavyweight_idio_score"] = score_heavyweight_idio(
        hw.get("weighted_change_pct"), hw.get("news_boost", 0.0)
    )
    if not hw:
        flags.heavyweight_idio = _source_flag(statuses.get("heavyweight_idio"))

    asx_open = _hist("asx_cash", "open") or []

    # Technical indicators (derived from ASX cash closes)
    rsi = compute_rsi(asx_close) if len(asx_close) >= 15 else None
    rsi_previous = compute_rsi(asx_close[:-1]) if len(asx_close) >= 16 else None
    rsi_slope = rsi - rsi_previous if rsi is not None and rsi_previous is not None else None
    rsi_score = score_rsi(rsi)
    ath_distance = compute_distance_from_high(asx_close) if len(asx_close) >= 2 else None
    high_20d_distance = (
        compute_distance_from_high(asx_close, days=20) if len(asx_close) >= 20 else None
    )
    high_50d_distance = (
        compute_distance_from_high(asx_close, days=50) if len(asx_close) >= 50 else None
    )
    ath_score = score_ath_distance(ath_distance)
    asx_1d_return = None
    asx_2d_return = None
    asx_3d_return = None
    if len(asx_close) >= 2 and asx_close[-2] != 0:
        asx_1d_return = (asx_close[-1] - asx_close[-2]) / asx_close[-2] * 100.0
    if len(asx_close) >= 3 and asx_close[-3] != 0:
        asx_2d_return = (asx_close[-1] - asx_close[-3]) / asx_close[-3] * 100.0
    if len(asx_close) >= 4 and asx_close[-4] != 0:
        asx_3d_return = (asx_close[-1] - asx_close[-4]) / asx_close[-4] * 100.0
    index_5d_return = None
    if len(asx_close) >= 6 and asx_close[-6] != 0:
        index_5d_return = (asx_close[-1] - asx_close[-6]) / asx_close[-6] * 100.0
    momentum_exhaustion = compute_momentum_exhaustion(rsi, index_5d_return)
    bollinger_position = compute_bollinger_position(asx_close) if len(asx_close) >= 20 else None
    bollinger_score = score_bollinger(bollinger_position)
    profit_taking_combo = compute_profit_taking_combo(rsi, ath_distance, index_5d_return)

    feats["rsi_14"] = rsi
    feats["rsi_previous_14"] = rsi_previous
    feats["rsi_slope"] = rsi_slope
    feats["rsi_score"] = rsi_score
    feats["ath_distance_pct"] = ath_distance
    feats["high_20d_distance_pct"] = high_20d_distance
    feats["high_50d_distance_pct"] = high_50d_distance
    feats["ath_score"] = ath_score
    feats["asx_1d_return_pct"] = asx_1d_return
    feats["asx_2d_return_pct"] = asx_2d_return
    feats["asx_3d_return_pct"] = asx_3d_return
    feats["index_5d_return_pct"] = index_5d_return
    feats["momentum_exhaustion_score"] = momentum_exhaustion
    feats["bollinger_position"] = bollinger_position
    feats["bollinger_score"] = bollinger_score
    feats["profit_taking_combo_score"] = profit_taking_combo

    # TradingView MCP enrichment (live real-time multi-timeframe consensus + market snapshots)
    tv = raw.tradingview or {}
    tv_data = tv if not isinstance(tv, dict) or "data" not in tv else tv.get("data", {})
    xjo_daily = tv_data.get("xjo_daily") or {}
    xjo_weekly = tv_data.get("xjo_weekly") or {}
    feats["tv_xjo_daily_score"] = xjo_daily.get("net_score")
    feats["tv_xjo_weekly_score"] = xjo_weekly.get("net_score")
    daily_score = xjo_daily.get("net_score")
    weekly_score = xjo_weekly.get("net_score")
    if daily_score is not None and weekly_score is not None:
        feats["tv_xjo_trend_score"] = _clamp((daily_score + weekly_score) / 2.0, -3.0, 3.0)
    elif daily_score is not None:
        feats["tv_xjo_trend_score"] = _clamp(daily_score, -3.0, 3.0)
    elif weekly_score is not None:
        feats["tv_xjo_trend_score"] = _clamp(weekly_score, -3.0, 3.0)
    else:
        feats["tv_xjo_trend_score"] = None
    feats["tv_xjo_decision"] = (
        xjo_daily.get("decision") if isinstance(xjo_daily, dict) else None
    )

    sectors = tv_data.get("sectors") or {}
    feats["tv_financials_minus_materials_pct"] = sectors.get(
        "financials_minus_materials_pct"
    )
    feats["tv_financials_vs_materials_score"] = _score_tv_sector_spread(
        sectors.get("financials_minus_materials_pct")
    )

    hw = tv_data.get("heavyweights") or {}
    feats["tv_heavyweight_avg_score"] = hw.get("avg_score")

    asian = tv_data.get("asian") or {}
    feats["tv_asian_session_change_pct"] = asian.get("avg_change_pct")

    comm = tv_data.get("commodities") or {}
    feats["tv_commodity_basket_change_pct"] = comm.get("basket_change_pct")
    feats["tv_commodity_basket_ex_gold_change_pct"] = comm.get(
        "basket_ex_gold_change_pct"
    )
    feats["tv_commodity_vs_gold_change_pct"] = comm.get("basket_vs_gold_change_pct")

    # TradingView is optional; flag as degraded if the snapshot is missing or
    # the fetcher reported degraded health.
    tv_status = _source_flag(statuses.get("tradingview"))
    if not tv_data or tv_status != "ok":
        flags.tradingview = tv_status

    # Alpha Vantage MCP enrichment (cross-asset feeds + macro rates)
    av = raw.alpha_vantage or {}
    av_data = av if not isinstance(av, dict) or "data" not in av else av.get("data", {})
    feats["av_aud_usd_change_pct"] = av_data.get("aud_usd_change_pct")
    feats["av_spy_change_pct"] = av_data.get("spy_change_pct")
    feats["av_qqq_change_pct"] = av_data.get("qqq_change_pct")
    feats["av_gld_change_pct"] = av_data.get("gld_change_pct")
    feats["av_vixy_change_pct"] = av_data.get("vixy_change_pct")
    feats["av_us_10y_yield_change_bps"] = av_data.get("us_10y_yield_change_bps")
    feats["av_us_10y_yield_level"] = av_data.get("us_10y_yield_level")
    av_status = _source_flag(statuses.get("alpha_vantage"))
    if not av_data or av_status != "ok":
        flags.alpha_vantage = av_status

    # Secondary-model short-term features
    if len(asx_open) >= 1 and len(asx_close) >= 2:
        last_open = asx_open[-1]
        prev_close = asx_close[-2]
        if prev_close != 0:
            feats["overnight_gap_pct"] = (last_open - prev_close) / prev_close * 100.0

    gap = feats.get("overnight_gap_pct")
    session_ret = feats.get("asx_open_to_now_return_pct")
    if gap is not None and session_ret is not None:
        if gap > 0 and session_ret < 0:
            feats["gap_filled_score"] = -1.0
        elif gap < 0 and session_ret > 0:
            feats["gap_filled_score"] = 1.0
        else:
            feats["gap_filled_score"] = 0.0

    feats["vwap_distance_pct"] = session_ret

    market_breadth = 0.0
    if session_ret is not None:
        market_breadth = _clamp(session_ret * 2.0, -1.5, 1.5)
    feats["market_breadth_score"] = market_breadth

    if not asx_close:
        flags.asx_cash = _source_flag(statuses.get("asx_cash"))

    # SPI basis / momentum — only when the SPI source is fresh.
    spi_status = _source_flag(statuses.get("spi_futures"))
    if raw.spi_futures and raw.asx_cash and spi_status == "ok":
        spi_series = _hist("spi_futures", "close") or []
        asx_last = _last("asx_cash", "close")
        cash_proxy = bool(raw.spi_futures.get("cash_proxy"))
        if isinstance(spi_series, (list, tuple)) and len(spi_series) >= 1 and asx_last:
            spi_last = spi_series[-1]
            if not cash_proxy:
                feats["spi_basis_pct"] = (spi_last - asx_last) / asx_last * 100.0
            if len(spi_series) >= 2 and spi_series[-2] != 0:
                spi_momentum = (spi_last - spi_series[-2]) / spi_series[-2] * 100.0
                feats["spi_momentum_pct"] = spi_momentum
                feats["spi_short_term_momentum_pct"] = spi_momentum
    if spi_status != "ok":
        flags.spi_futures = spi_status

    # Map source status to flags for downstream consumers.
    flag_map = {
        "asx_cash": "asx_cash",
        "spi_futures": "spi_futures",
        "a_vix": "a_vix",
        "commodities": "commodities",
        "fx": "fx",
        "us_assets": "us_assets",
        "calendar": "calendar",
        "volume": "volume",
        "financials_vs_materials": "financials_vs_materials",
        "housing_credit": "housing_credit",
        "china_pulse": "china_steel_property",
        "heavyweight_idio": "heavyweight_idio",
    }
    for source, flag_key in flag_map.items():
        st = _source_flag(statuses.get(source))
        if st != "ok":
            setattr(flags, flag_key, st)

    return FeatureVector(**feats), flags
