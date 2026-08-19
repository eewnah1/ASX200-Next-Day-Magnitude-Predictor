"""Alpha Vantage MCP client (legacy key-in-url JSON-RPC endpoint).

The official Alpha Vantage MCP server exposes tools over a JSON-RPC HTTP
endpoint.  This module uses plain ``requests`` POST calls so it works without
a full MCP runtime.  Calls are cached and rate-limited to stay within the free
tier (5 calls / minute, 25 calls / day) and degrade gracefully when the key is
missing or the budget is exhausted.
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

MCP_URL = "https://mcp.alphavantage.co/mcp"
FREE_TIER_DAILY_BUDGET = 20  # stay safely below the 25 call/day limit
FREE_TIER_PER_MINUTE = 5
CALL_INTERVAL_SECONDS = 60.0 / FREE_TIER_PER_MINUTE + 1.0  # ~13s between calls
CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


class AlphaVantageMCPClient:
    """Lightweight JSON-RPC client for the Alpha Vantage MCP server."""

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
        # Sort keys for stable key.
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

    def _is_rate_limit_error(self, data: dict[str, Any]) -> bool:
        if isinstance(data, dict) and "error" in data:
            msg = str(data["error"]).lower()
            if "rate limit" in msg or "premium" in msg or "entitlement" in msg:
                return True
        return False

    def _mark_budget_exhausted(self) -> None:
        self._cache["calls_today"] = FREE_TIER_DAILY_BUDGET
        self._save_cache()

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an Alpha Vantage MCP tool with caching and rate limiting."""
        if not self.api_key:
            raise RuntimeError("ALPHA_VANTAGE_API_KEY not configured")

        key = self._cache_key(tool, arguments)
        now = datetime.utcnow()

        # Return cached response if still fresh.
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

        self._rate_limit_wait()
        url = f"{MCP_URL}?apikey={self.api_key}"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Alpha Vantage MCP request failed: {exc}") from exc

        result = data.get("result", {})
        structured = result.get("structuredContent", {})
        text = ""
        if result.get("content") and isinstance(result["content"], list):
            text = result["content"][0].get("text", "")

        # Some endpoints return an error inside the result text/structured.
        if "error" in structured or "error" in result:
            error_payload = structured.get("error") or result.get("error") or {}
            if self._is_rate_limit_error({"error": error_payload}):
                self._mark_budget_exhausted()
                raise RuntimeError(
                    f"Alpha Vantage rate-limit/premium for {tool}: {error_payload}"
                )
            raise RuntimeError(
                f"Alpha Vantage tool {tool} returned error: {error_payload}"
            )

        parsed = self._parse(tool, text)
        self._use_budget()
        self._cache.setdefault("entries", {})[key] = {"ts": now.isoformat(), "data": parsed}
        self._save_cache()
        return parsed

    # ------------------------------------------------------------------ parse

    def _parse(self, tool: str, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise RuntimeError(f"Alpha Vantage {tool} returned empty response")

        if tool == "GLOBAL_QUOTE" or text.startswith("symbol,"):
            return self._parse_global_quote_csv(text)

        if "FX" in tool or text.startswith("timestamp,open,high,low,close"):
            return self._parse_fx_csv(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Alpha Vantage {tool} returned invalid JSON: {exc}") from exc

        if "error" in data or (
            isinstance(data, dict) and "Information" in data and "Invalid API call" in str(data)
        ):
            raise RuntimeError(f"Alpha Vantage {tool} error: {data}")

        if "Realtime Currency Exchange Rate" in data:
            return self._parse_currency_exchange_rate(data)
        if "Time Series FX (Daily)" in data:
            return self._parse_fx_daily(data)
        if "data" in data and isinstance(data["data"], list):
            return self._parse_treasury_yield(data)
        if "sample_data" in data:
            # Preview/truncated response; try to parse embedded sample JSON or CSV.
            sample = data["sample_data"]
            try:
                sample_json = json.loads(sample)
                if "data" in sample_json:
                    return self._parse_treasury_yield(sample_json)
            except (json.JSONDecodeError, TypeError):
                if isinstance(sample, str) and sample.strip().startswith("timestamp,value"):
                    return self._parse_treasury_yield_csv(sample)

        return {"raw": data}

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
    def _parse_fx_csv(text: str) -> dict[str, Any]:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            raise RuntimeError("FX CSV empty")
        # The MCP FX response is sorted newest-first; row 0 is the latest observation.
        last = rows[0]
        prev = rows[1] if len(rows) >= 2 else last
        last_close = _safe_float(last.get("close"))
        prev_close = _safe_float(prev.get("close"))
        change = None
        if prev_close and prev_close != 0:
            change = (last_close - prev_close) / prev_close * 100.0
        return {
            "last_refreshed": last.get("timestamp"),
            "last_close": last_close,
            "previous_close": prev_close,
            "change_percent": change,
            "series_close": [_safe_float(r.get("close")) for r in rows[:10]],
        }

    @staticmethod
    def _parse_currency_exchange_rate(data: dict[str, Any]) -> dict[str, Any]:
        rate = data["Realtime Currency Exchange Rate"]
        return {
            "from_currency": rate.get("1. From_Currency Code"),
            "to_currency": rate.get("3. To_Currency Code"),
            "exchange_rate": _safe_float(rate.get("5. Exchange Rate")),
            "last_refreshed": rate.get("6. Last Refreshed"),
        }

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
        if prev_close and prev_close != 0:
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
        # The preview CSV is sorted newest-first; skip placeholder/missing values.
        valid_rows = [
            r for r in rows
            if r.get("value") and r.get("value").strip() not in ("", ".")
        ]
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
