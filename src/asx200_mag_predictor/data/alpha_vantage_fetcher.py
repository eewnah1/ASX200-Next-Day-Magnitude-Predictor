"""Alpha Vantage data fetcher for the ASX200 predictor.

Aggregates free-tier Alpha Vantage MCP series into a compact market-data
package: AUD/USD, US equity lead (SPY) and the 10-year US Treasury yield.
Falls back to yfinance proxies when the Alpha Vantage budget is exhausted,
rate-limited, or unavailable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.data.alpha_vantage_mcp import AlphaVantageMCPClient
from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)

_ALPHA_VANTAGE_TARGETS = {
    "spy": ("GLOBAL_QUOTE", {"symbol": "SPY"}),
    "aud_usd": ("FX_DAILY", {"from_symbol": "AUD", "to_symbol": "USD"}),
    "us_10y_yield": (
        "TREASURY_YIELD",
        {"interval": "daily", "maturity": "10year", "return_full_data": "true"},
    ),
}

_YF_SYMBOL_MAP = {
    "spy": "SPY",
    "aud_usd": "AUDUSD=X",
    "us_10y_yield": "^TNX",
}


def _pct_change(series: Any) -> float | None:
    """Return the most recent daily percent change for a price series."""
    try:
        closes = series.dropna()
        if len(closes) < 2:
            return None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        if prev == 0 or last != last:
            return None
        return (last - prev) / prev * 100.0
    except Exception:  # noqa: BLE001
        return None


def _yf_fallback_change(ticker: str) -> tuple[float | None, float | None]:
    """Fetch the last two daily closes from yfinance as a fallback."""
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
        closes = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        closes = closes.dropna()
        if len(closes) < 2:
            return None, None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        if prev == 0:
            return None, last
        return (last - prev) / prev * 100.0, last
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance fallback for %s failed: %s", ticker, exc)
        return None, None


class AlphaVantageFetcher:
    """Fetch and cache Alpha Vantage free-tier series for the ASX200 pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = AlphaVantageMCPClient(self.settings)

    def fetch(self) -> dict[str, Any]:
        """Return a dict mimicking a FetchResult with Alpha-Vantage-derived market snapshots."""
        data: dict[str, Any] = {"_timestamp": datetime.utcnow().isoformat()}
        errors: list[str] = []
        fallback_used = False

        # Primary: Alpha Vantage MCP / REST.
        for key, (tool, args) in _ALPHA_VANTAGE_TARGETS.items():
            try:
                result = self.client.call(tool, args)
            except RuntimeError as exc:
                msg = str(exc)
                logger.warning("Alpha Vantage %s failed: %s", key, msg)
                errors.append(f"{key}: {msg}")
                continue

            if tool == "GLOBAL_QUOTE":
                data[f"{key}_change_pct"] = result.get("change_percent")
                data[f"{key}_price"] = result.get("price")
                data[f"{key}_latest_day"] = result.get("latest_day")
            elif tool == "FX_DAILY":
                data["aud_usd_change_pct"] = result.get("change_percent")
                data["aud_usd_close"] = result.get("last_close")
                data["aud_usd_previous_close"] = result.get("previous_close")
                data["aud_usd_last_refreshed"] = result.get("last_refreshed")
            elif tool == "TREASURY_YIELD":
                data["us_10y_yield_change_bps"] = result.get("change_bps")
                data["us_10y_yield_level"] = result.get("last_value")
                data["us_10y_yield_last_date"] = result.get("last_date")

        # Fallback: try yfinance for any missing core target.
        for key, yf_symbol in _YF_SYMBOL_MAP.items():
            if key == "us_10y_yield":
                if data.get("us_10y_yield_change_bps") is not None:
                    continue
                chg, level = _yf_fallback_change(yf_symbol)
                if chg is not None and level is not None:
                    # yfinance returns the yield in percent; convert to bps.
                    data["us_10y_yield_change_bps"] = chg * 100.0
                    data["us_10y_yield_level"] = level
                    data["us_10y_yield_last_date"] = datetime.utcnow().isoformat()
                    fallback_used = True
            elif key == "aud_usd":
                if data.get("aud_usd_change_pct") is not None:
                    continue
                chg, level = _yf_fallback_change(yf_symbol)
                if chg is not None:
                    data["aud_usd_change_pct"] = chg
                    data["aud_usd_close"] = level
                    data["aud_usd_last_refreshed"] = datetime.utcnow().isoformat()
                    fallback_used = True
            elif key == "spy":
                if data.get("spy_change_pct") is not None:
                    continue
                chg, level = _yf_fallback_change(yf_symbol)
                if chg is not None:
                    data["spy_change_pct"] = chg
                    data["spy_price"] = level
                    data["spy_latest_day"] = datetime.utcnow().isoformat()
                    fallback_used = True

        has_data = any(
            v is not None
            for k, v in data.items()
            if k != "_timestamp"
            and not k.endswith("_last_refreshed")
            and not k.endswith("_last_date")
        )
        status = "ok" if has_data else "degraded"

        note = "yfinance fallback" if fallback_used else None
        return {
            "name": "alpha_vantage",
            "status": status,
            "data": data,
            "error": "; ".join(errors) or None,
            "last_success_at": datetime.utcnow().isoformat(),
            "errors": errors,
            "value": note,
        }
