"""Alpha Vantage MCP client (standard REST endpoint fallback).

The official Alpha Vantage MCP server exposed over ``mcp.alphavantage.co``
rate-limits free keys aggressively, while the standard REST API
(``www.alphavantage.co/query``) works reliably for the three calls we need:
``GLOBAL_QUOTE``, ``FX_DAILY``, and ``TREASURY_YIELD``.  This module uses plain
``requests`` GET calls, caches results, and stays within the free tier
(5 calls / minute, 25 calls / day).
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from datetime import datetime
from typing import Any

import requests

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
FREE_TIER_DAILY_BUDGET = 20  # stay safely below the 25 call/day limit
FREE_TIER_PER_MINUTE = 5
CALL_INTERVAL_SECONDS = 60.0 / FREE_TIER_PER_MINUTE + 1.0  # ~13s between calls
CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


class AlphaVantageMCPClient:
    """Lightweight client for the Alpha Vantage standard REST API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = (self.settings.alphavantage_api_key or "").strip()
        self._cache: dict[str, Any] = {}
        self._last_call_at: float | None = None
        self._cache_file = self.settings.data_dir / "alpha_vantage_cache.json"
        self._load_cache()

    # ------------------------------------------------------------------ config

    def _load_cache(self) -> None:
        try:
            if self._cache_file.exists():
                self._cache = json.loads(self._cache_file.read_text())
        except Exception:  # noqa: BLE001
            self._cache = {}
        if "entries" not in self._cache:
            self._cache["entries"] = {}
        if "calls_today" not in self._cache:
            self._cache["calls_today"] = 0
        if "last_call_date" not in self._cache:
            self._cache["last_call_date"] = _today()

    def _save_cache(self) -> None:
        try:
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(self._cache, indent=2, default=str))
        except Exception:  # noqa: BLE001
            pass

    def _reset_daily_budget_if_new_day(self) -> None:
        today = _today()
        if self._cache.get("last_call_date") != today:
            self._cache["calls_today"] = 0
            self._cache["last_call_date"] = today

    def _cache_key(self, tool: str, arguments: dict[str, Any]) -> str:
        args = json.dumps(arguments, sort_keys=True)
        return f"{tool}:{args}"

    # ------------------------------------------------------------------ rpc

    def _rate_limit_wait(self) -> None:
        now = time.time()
        if self._last_call_at is not None:
            elapsed = now - self._last_call_at
            if elapsed < CALL_INTERVAL_SECONDS:
                time.sleep(CALL_INTERVAL_SECONDS - elapsed)
        self._last_call_at = time.time()

    def _has_budget(self) -> bool:
        self._reset_daily_budget_if_new_day()
        return self._cache.get("calls_today", 0) < FREE_TIER_DAILY_BUDGET

    def _use_budget(self) -> None:
        self._cache["calls_today"] = self._cache.get("calls_today", 0) + 1

    def _mark_budget_exhausted(self) -> None:
        self._cache["calls_today"] = FREE_TIER_DAILY_BUDGET
        self._save_cache()

    def _is_rate_limit_message(self, text: str | None) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return (
            "rate limit" in lowered
            or "premium" in lowered
            or "25 calls per day" in lowered
            or "api call frequency" in lowered
        )

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an Alpha Vantage endpoint with caching and rate limiting."""
        if not self.api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY not configured")

        key = self._cache_key(tool, arguments)
        now = datetime.utcnow()

        cached = self._cache.get("entries", {}).get(key)
        if cached:
            try:
                ts = datetime.fromisoformat(cached["ts"])
                if (now - ts).total_seconds() < CACHE_TTL_SECONDS:
                    return cached["data"]
            except Exception:  # noqa: BLE001
                pass

        if not self._has_budget():
            if cached:
                logger.warning("Alpha Vantage daily budget exhausted; using stale cache")
                return cached["data"]
            raise RuntimeError("Alpha Vantage daily API budget exhausted")

        params: dict[str, Any] = {"apikey": self.api_key}
        if tool == "GLOBAL_QUOTE":
            params["function"] = "GLOBAL_QUOTE"
            params["symbol"] = arguments.get("symbol")
        elif tool == "FX_DAILY":
            params["function"] = "FX_DAILY"
            params["from_symbol"] = arguments.get("from_symbol")
            params["to_symbol"] = arguments.get("to_symbol")
            params["outputsize"] = "compact"
        elif tool == "TREASURY_YIELD":
            params["function"] = "TREASURY_YIELD"
            params["interval"] = arguments.get("interval")
            params["maturity"] = arguments.get("maturity")
        else:
            raise RuntimeError(f"Unknown Alpha Vantage tool: {tool}")

        self._rate_limit_wait()
        try:
            response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Alpha Vantage request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Alpha Vantage {tool} returned invalid JSON")

        # The standard API returns an ``Information`` or ``Note`` key when the
        # call limit is hit or the key is invalid.
        info = data.get("Information") or data.get("Note")
        if info and self._is_rate_limit_message(str(info)):
            self._mark_budget_exhausted()
            raise RuntimeError(f"Alpha Vantage rate-limit/premium for {tool}: {info}")
        if info:
            raise RuntimeError(f"Alpha Vantage information for {tool}: {info}")
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error for {tool}: {data['Error Message']}")

        if "Global Quote" in data:
            parsed = self._parse_global_quote_json(data)
        elif "Time Series FX (Daily)" in data:
            parsed = self._parse_fx_daily(data)
        elif "data" in data:
            parsed = self._parse_treasury_yield(data)
        else:
            raise RuntimeError(
                f"Alpha Vantage {tool} response not recognised: {list(data.keys())[:5]}"
            )

        self._use_budget()
        self._cache.setdefault("entries", {})[key] = {"ts": now.isoformat(), "data": parsed}
        self._save_cache()
        return parsed

    # ------------------------------------------------------------------ parse

    def _parse_global_quote_json(self, data: dict[str, Any]) -> dict[str, Any]:
        quote = data.get("Global Quote", {})
        return {
            "symbol": quote.get("01. symbol", "").strip(),
            "open": _safe_float(quote.get("02. open")),
            "high": _safe_float(quote.get("03. high")),
            "low": _safe_float(quote.get("04. low")),
            "price": _safe_float(quote.get("05. price")),
            "volume": _safe_float(quote.get("06. volume")),
            "latest_day": quote.get("07. latest trading day", "").strip(),
            "previous_close": _safe_float(quote.get("08. previous close")),
            "change": _safe_float(quote.get("09. change")),
            "change_percent": _parse_pct(quote.get("10. change percent")),
        }

    @staticmethod
    def _parse_global_quote_csv(text: str) -> dict[str, Any]:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            return {
                "symbol": row.get("symbol", "").strip(),
                "open": _safe_float(row.get("open")),
                "high": _safe_float(row.get("high")),
                "low": _safe_float(row.get("low")),
                "price": _safe_float(row.get("price")),
                "volume": _safe_float(row.get("volume")),
                "latest_day": row.get("latestDay", "").strip(),
                "previous_close": _safe_float(row.get("previousClose")),
                "change": _safe_float(row.get("change")),
                "change_percent": _parse_pct(row.get("changePercent")),
            }
        raise RuntimeError("GLOBAL_QUOTE CSV empty")

    @staticmethod
    def _parse_fx_daily(data: dict[str, Any]) -> dict[str, Any]:
        series = data.get("Time Series FX (Daily)", {})
        if not series:
            raise RuntimeError("FX_DAILY empty")
        dates = sorted(series.keys())
        last = series[dates[-1]]
        prev = series[dates[-2]] if len(dates) >= 2 else last
        last_close = _safe_float(last.get("4. close"))
        prev_close = _safe_float(prev.get("4. close"))
        change = None
        if not math.isnan(prev_close) and prev_close != 0:
            change = (last_close - prev_close) / prev_close * 100.0
        return {
            "last_refreshed": data.get("Meta Data", {}).get("5. Last Refreshed"),
            "last_close": last_close,
            "previous_close": prev_close,
            "change_percent": change,
            "series_close": [_safe_float(series[d].get("4. close")) for d in dates[-10:]],
        }

    @staticmethod
    def _parse_treasury_yield_csv(text: str) -> dict[str, Any]:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        valid_rows = [r for r in rows if r.get("value") and r.get("value").strip() not in ("", ".")]
        if not valid_rows:
            raise RuntimeError("TREASURY_YIELD CSV empty")
        last = valid_rows[0]
        prev = valid_rows[1] if len(valid_rows) >= 2 else last
        last_val = _safe_float(last.get("value"))
        prev_val = _safe_float(prev.get("value"))
        change = None
        if not math.isnan(last_val) and not math.isnan(prev_val) and prev_val != 0:
            change = (last_val - prev_val) * 100.0  # yield in percent -> basis points
        return {
            "name": "Treasury Yield",
            "last_date": last.get("timestamp"),
            "last_value": last_val,
            "previous_value": prev_val,
            "change_bps": change,
        }

    @staticmethod
    def _parse_treasury_yield(data: dict[str, Any]) -> dict[str, Any]:
        points = data.get("data", [])
        if not points:
            raise RuntimeError("TREASURY_YIELD empty")
        sorted_points = sorted(points, key=lambda x: x.get("date", ""))
        last = sorted_points[-1]
        prev = sorted_points[-2] if len(sorted_points) >= 2 else last
        last_val = _safe_float(last.get("value"))
        prev_val = _safe_float(prev.get("value"))
        change = None
        if not math.isnan(last_val) and not math.isnan(prev_val) and prev_val != 0:
            change = (last_val - prev_val) * 100.0  # yield in percent -> basis points
        return {
            "name": data.get("name"),
            "last_date": last.get("date"),
            "last_value": last_val,
            "previous_value": prev_val,
            "change_bps": change,
        }


def _safe_float(value: Any) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _parse_pct(value: Any) -> float | None:
    if value is None:
        return None
    try:
        s = str(value).replace("%", "").strip()
        return float(s)
    except (TypeError, ValueError):
        return None
