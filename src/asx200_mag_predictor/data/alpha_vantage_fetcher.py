"""Alpha Vantage data fetcher for the ASX200 predictor.

Aggregates free-tier Alpha Vantage MCP series into a compact market-data
package: AUD/USD, US equity leads (SPY, QQQ), gold (GLD), a VIX proxy (VIXY),
and the 10-year US Treasury yield.  Falls back to yfinance proxies when the
Alpha Vantage budget is exhausted, rate-limited, or unavailable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.data.alpha_vantage_mcp import AlphaVantageMCPClient
from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)

_ALPHA_VANTAGE_TARGETS = {
    "spy": ("GLOBAL_QUOTE", {"symbol": "SPY"}),
    "qqq": ("GLOBAL_QUOTE", {"symbol": "QQQ"}),
    "gld": ("GLOBAL_QUOTE", {"symbol": "GLD"}),
    "vixy": ("GLOBAL_QUOTE", {"symbol": "VIXY"}),
    "aud_usd": ("FX_DAILY", {"from_symbol": "AUD", "to_symbol": "USD"}),
    "us_10y_yield": (
        "TREASURY_YIELD",
        {"interval": "daily", "maturity": "10year", "return_full_data": "true"},
    ),
}


class AlphaVantageFetcher:
    """Fetch and cache Alpha Vantage free-tier series for the ASX200 pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = AlphaVantageMCPClient(self.settings)

    def fetch(self) -> dict[str, Any]:
        """Return a dict mimicking a FetchResult with Alpha-Vantage-derived market snapshots."""
        errors: list[str] = []
        data: dict[str, Any] = {"_timestamp": datetime.utcnow().isoformat()}
        failed = False

        try:
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

        except Exception as exc:  # noqa: BLE001
            logger.warning("Alpha Vantage fetcher failed: %s", exc)
            errors.append(str(exc))
            failed = True

        status = "failed" if failed or (not data and errors) else "ok"
        if errors and status == "ok" and len(errors) >= len(_ALPHA_VANTAGE_TARGETS) - 1:
            status = "degraded"

        return {
            "name": "alpha_vantage",
            "status": status,
            "data": data,
            "error": "; ".join(errors) or None,
            "last_success_at": datetime.utcnow().isoformat(),
            "errors": errors,
        }
