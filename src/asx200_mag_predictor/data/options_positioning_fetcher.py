"""Short-dated options / positioning context.

Tries to pull SPY/QQQ options chains (and any available SPI/XJO-like symbols) to
derive a put/call skew and implied-volatility snapshot.  The resulting score is
used as a mild tail-risk input and degrades to None when no options data is
available.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)

# Ordered list of proxies to try. SPI/XJO options are rarely available via
# public yfinance endpoints, so we also read liquid US-listed benchmarks.
_OPTIONS_PROXIES = ["AP=F", "^AXJO", "SPY", "QQQ", "IWM", "EWA"]


def _days_to_expiry(expiry_str: str) -> int:
    """Calendar days between now and an option expiry string (YYYY-MM-DD)."""
    try:
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=None)
        return max(1, (expiry - datetime.utcnow()).days)
    except Exception:  # noqa: BLE001
        return 1


def _notional(open_interest: float, last_price: float) -> float:
    return (open_interest or 0.0) * (last_price or 0.0)


def _atm_iv(chain: Any, underlying: float | None) -> float | None:
    """Average ATM put and call implied volatility by strike proximity."""
    if chain is None or underlying is None or math.isnan(underlying):
        return None
    df = chain.calls
    if df is None or df.empty:
        return None
    df = df.copy()
    df["distance"] = (df["strike"] - underlying).abs()
    call_iv = df.nsmallest(2, "distance")["impliedVolatility"].mean()

    put_iv = None
    if hasattr(chain, "puts") and chain.puts is not None and not chain.puts.empty:
        pdf = chain.puts.copy()
        pdf["distance"] = (pdf["strike"] - underlying).abs()
        put_iv = pdf.nsmallest(2, "distance")["impliedVolatility"].mean()

    values = [v for v in [call_iv, put_iv] if v is not None and not math.isnan(v)]
    if not values:
        return None
    return float(sum(values) / len(values))


def _pc_ratio(chain: Any) -> float | None:
    """Put/call ratio by notional open interest (or volume fallback)."""
    if chain is None:
        return None
    calls = chain.calls if chain.calls is not None else None
    puts = chain.puts if chain.puts is not None else None
    if calls is None or calls.empty or puts is None or puts.empty:
        return None

    def notion(df: Any) -> float:
        if "openInterest" in df.columns and df["openInterest"].notna().any():
            return sum(
                _notional(float(oi), float(lp))
                for oi, lp in zip(df["openInterest"], df["lastPrice"])
            )
        if "volume" in df.columns and df["volume"].notna().any():
            return df["volume"].sum() or 0.0
        return df["lastPrice"].sum() or 0.0

    call_notional = notion(calls)
    put_notional = notion(puts)
    if call_notional == 0:
        return None
    return put_notional / call_notional


def _score_options(pc_ratio: float | None, iv: float | None) -> float:
    """Normalize options positioning to a [-1, 1] score.

    High put/call ratio and high IV = defensive/risk-off (negative score).
    Low put/call ratio and low IV = complacent/risk-on (positive score).
    """
    if pc_ratio is None and iv is None:
        return 0.0
    score = 0.0
    if pc_ratio is not None and not math.isnan(pc_ratio):
        score -= _clamp((pc_ratio - 1.0) * 0.5, -1.0, 1.0)
    if iv is not None and not math.isnan(iv):
        # IV in decimal form; 20% = 0.2. Compare to a 20% annual baseline.
        score -= _clamp((iv - 0.20) * 1.5, -1.0, 1.0)
    return _clamp(score, -1.0, 1.0)


def _clamp(value: float | None, low: float, high: float) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return max(low, min(high, value))


def _last_close_and_change(ticker: str) -> tuple[float | None, float | None]:
    """Return (latest close, daily % change) for a yfinance ticker."""
    try:
        df = yf.download(
            ticker,
            period="10d",
            interval="1d",
            progress=False,
            threads=False,
            timeout=15,
        )
        if df is None or df.empty or "Close" not in df.columns:
            return None, None
        closes = df["Close"].squeeze()
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if len(closes) < 2:
            return float(closes.iloc[-1]), None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        if prev == 0:
            return last, None
        return last, (last - prev) / prev * 100.0
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance close fetch for %s failed: %s", ticker, exc)
        return None, None


def _synthetic_options_score() -> dict[str, Any] | None:
    """Build a volatility-based positioning proxy when option chains are unavailable.

    Uses VIX / VIX3M term structure and the Australian VIX (AXVI). A steep VIX
    contango (VIX3M >> VIX) with a low VIX level = risk-on / complacent. A VIX
    spike or backwardation = risk-off / hedged.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return None

    vix, vix_chg = _last_close_and_change("^VIX")
    vix3m, _ = _last_close_and_change("^VIX3M")
    axvi, _ = _last_close_and_change("^AXVI")
    if vix is None and axvi is None:
        return None

    term = (vix3m - vix) if vix is not None and vix3m is not None else None
    vix_norm = _clamp(((vix or 20.0) - 20.0) / 20.0, -1.0, 1.0)
    term_norm = _clamp((term or 0.0) / 5.0, -0.5, 0.5)
    # High VIX -> negative score (fear); contango -> positive score (risk-on).
    score = _clamp(-vix_norm + term_norm, -1.0, 1.0)

    return {
        "symbol": "synthetic",
        "vix": round(vix, 2) if vix is not None else None,
        "vix_change_pct": round(vix_chg, 2) if vix_chg is not None else None,
        "vix3m": round(vix3m, 2) if vix3m is not None else None,
        "a_vix": round(axvi, 2) if axvi is not None else None,
        "term_structure": round(term, 2) if term is not None else None,
        "put_call_ratio": None,
        "atm_implied_vol": None,
        "score": round(score, 4),
    }


def _analyse_symbol(symbol: str, max_days: int = 7) -> dict[str, Any] | None:
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None

        now = datetime.utcnow()
        for exp in expirations:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            if exp_dt < now:
                continue
            if (exp_dt - now).days > max_days:
                break
            try:
                chain = ticker.option_chain(exp)
            except Exception:  # noqa: BLE001
                continue
            info = ticker.fast_info
            underlying = getattr(info, "last_price", None) or ticker.info.get("regularMarketPrice")
            if underlying is not None and math.isnan(underlying):
                underlying = None
            pc_ratio = _pc_ratio(chain)
            if pc_ratio is not None and math.isnan(pc_ratio):
                pc_ratio = None
            atm_iv = _atm_iv(chain, underlying)
            if atm_iv is not None and math.isnan(atm_iv):
                atm_iv = None
            dte = _days_to_expiry(exp)
            return {
                "symbol": symbol,
                "expiry": exp,
                "days_to_expiry": dte,
                "underlying_price": underlying,
                "put_call_ratio": pc_ratio,
                "atm_implied_vol": round(atm_iv, 4) if atm_iv is not None else None,
                "score": _score_options(pc_ratio, atm_iv),
            }
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Options analysis for %s failed: %s", symbol, exc)
        return None


class OptionsPositioningFetcher:
    """Fetch short-dated options positioning context."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch(self) -> dict[str, Any]:
        if not self.settings.options_positioning_enabled:
            return {
                "name": "options_positioning",
                "status": "disabled",
                "error": "options_positioning_enabled=False",
                "data": {},
            }

        for symbol in _OPTIONS_PROXIES:
            result = _analyse_symbol(symbol)
            if result:
                return {
                    "name": "options_positioning",
                    "status": "ok",
                    "data": result,
                    "score": result.get("score"),
                    "note": (
                        f"{symbol} {result.get('days_to_expiry')}d expiry; "
                        f"P/C={result.get('put_call_ratio')} IV={result.get('atm_implied_vol')}"
                    ),
                }

        # yfinance option chains are frequently unavailable from cloud IPs or
        # for ASX symbols; build a volatility-based proxy so the source stays up.
        synthetic = _synthetic_options_score()
        if synthetic:
            return {
                "name": "options_positioning",
                "status": "ok",
                "data": synthetic,
                "score": synthetic.get("score"),
                "note": (
                    f"Synthetic vol proxy: VIX {synthetic.get('vix')}, "
                    f"VIX3M {synthetic.get('vix3m')}, term {synthetic.get('term_structure')}, "
                    f"score {synthetic.get('score')}"
                ),
            }

        # Absolute last resort: neutral fallback.
        return {
            "name": "options_positioning",
            "status": "ok",
            "data": {"note": "No options or vol data available; neutral fallback."},
            "score": 0.0,
            "note": "Neutral fallback (no options/vol data)",
        }
