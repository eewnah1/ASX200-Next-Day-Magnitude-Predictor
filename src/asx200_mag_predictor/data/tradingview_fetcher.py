"""Live TradingView MCP fetcher for ASX200 predictor enrichment.

Pulls richer structured data at prediction time:
- XJO/SPI multi-timeframe technical consensus (daily + weekly)
- Financials vs Materials relative strength (ASX sector indices)
- Heavyweight multi-agent consensus (CBA, BHP, RIO, FMG, WDS)
- Asian session leads (Nikkei, Hang Seng, STI, KOSPI)
- Commodity basket relative performance

Each call is wrapped in a short timeout and failures are swallowed so the
prediction can continue with yfinance-derived fallbacks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from asx200_mag_predictor.data.tradingview_mcp import (
    atila_symbol_analysis,
    fiale_lookup,
)
from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)

_WORKER_TIMEOUT = 18.0  # seconds per sub-call
_MAX_WORKERS = 6

_ASX_HEAVYWEIGHTS = [
    ("CBA", "ASX:CBA", "asx"),
    ("BHP", "ASX:BHP", "asx"),
    ("RIO", "ASX:RIO", "asx"),
    ("FMG", "ASX:FMG", "asx"),
    ("WDS", "ASX:WDS", "asx"),
]

_ASIAN_INDICES = [
    ("NI225", "TVC:NI225"),
    ("HSI", "TVC:HSI"),
    ("STI", "SGX:ES3"),
    ("KOSPI", "KRX:KOSPI"),
]

_COMMODITIES = [
    ("iron_ore", "COMEX:TIO1!"),
    ("copper", "COMEX:HG1!"),
    ("gold", "COMEX:GC1!"),
    ("oil", "NYMEX:CL1!"),
]

_SECTOR_PAIRS = [
    ("financials", "ASX:XFJ"),
    ("materials", "ASX:XMJ"),
]

# Australian short-end rates and treasury futures: price = 100 - yield.
_AU_RATES = [
    ("ib1", "ASX24:IB1!"),  # 30-day interbank cash rate (RBA expectations)
    ("yt1", "ASX24:YT1!"),  # 3-year treasury bond futures
    ("xt1", "ASX24:XT1!"),  # 10-year treasury bond futures
]

# Cleaner China / steel / iron-ore pulse than yfinance proxies.
_CHINA_PULSE = [
    ("iron_ore", "COMEX:TIO1!"),
    ("copper", "COMEX:HG1!"),
    ("hang_seng", "TVC:HSI"),
    ("china_a50", "SGX:CN50"),
    ("bhp", "ASX:BHP"),
    ("rio", "ASX:RIO"),
    ("fmg", "ASX:FMG"),
]


def _net_score(analysis: dict[str, Any] | None) -> float | None:
    """Extract a numeric net score from an atila symbol-analysis result."""
    if not analysis:
        return None
    consensus = analysis.get("consensus")
    if isinstance(consensus, dict):
        score = consensus.get("net_score")
        if score is not None:
            return float(score)
    return None


def _decision(analysis: dict[str, Any] | None) -> str | None:
    if not analysis:
        return None
    consensus = analysis.get("consensus")
    if isinstance(consensus, dict):
        return str(consensus.get("decision", ""))
    return None


def _parse_lookup_symbols(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Turn a fiale lookup response into a dict keyed by short name."""
    out: dict[str, dict[str, Any]] = {}
    if not response or "symbols" not in response:
        return out
    for item in response.get("symbols", []):
        sym = item.get("symbol", "")
        out[sym] = item
    return out


def _lookup_change_pct(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    change = item.get("change")
    if change is not None:
        return float(change)
    return None


class TradingViewFetcher:
    """Fetch TradingView-derived signals with bounded latency and graceful fallback."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def fetch(self) -> dict[str, Any]:
        """Return a structured TradingView snapshot or an error dict.

        Calls are run concurrently with a per-call timeout so a single
        slow MCP function cannot block the prediction pipeline.
        """
        self.errors = []
        result: dict[str, Any] = {"status": "ok", "data": {}}

        tasks = [
            ("xjo_daily", self._xjo_daily),
            ("xjo_weekly", self._xjo_weekly),
            ("heavyweights", self._heavyweights),
            ("sectors", self._sectors),
            ("asian", self._asian),
            ("commodities", self._commodities),
            ("rates", self._rates),
            ("china_pulse", self._china_pulse),
        ]

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {
                name: executor.submit(fn)
                for name, fn in tasks  # type: ignore[arg-type]
            }
            for name, future in futures.items():
                try:
                    result["data"][name] = future.result(timeout=_WORKER_TIMEOUT)
                except TimeoutError:
                    msg = f"TradingView {name} timed out"
                    logger.warning(msg)
                    self.errors.append(msg)
                    result["data"][name] = None
                except Exception as exc:  # noqa: BLE001
                    msg = f"TradingView {name} failed: {exc}"
                    logger.warning(msg)
                    self.errors.append(msg)
                    result["data"][name] = None

        if self.errors:
            result["status"] = "degraded"
            result["errors"] = self.errors
        return result

    def _xjo_daily(self) -> dict[str, Any] | None:
        analysis = atila_symbol_analysis("ASX:XJO", "asx", "1D")
        if analysis.get("error"):
            return None
        return {
            "analysis": analysis,
            "net_score": _net_score(analysis),
            "decision": _decision(analysis),
        }

    def _xjo_weekly(self) -> dict[str, Any] | None:
        analysis = atila_symbol_analysis("ASX:XJO", "asx", "1W")
        if analysis.get("error"):
            return None
        return {
            "analysis": analysis,
            "net_score": _net_score(analysis),
            "decision": _decision(analysis),
        }

    def _heavyweights(self) -> dict[str, Any] | None:
        results: dict[str, Any] = {}
        scores: list[float] = []
        for short, symbol, exchange in _ASX_HEAVYWEIGHTS:
            analysis = atila_symbol_analysis(symbol, exchange, "1D")
            if analysis.get("error"):
                continue
            score = _net_score(analysis)
            results[short] = {
                "analysis": analysis,
                "net_score": score,
                "decision": _decision(analysis),
            }
            if score is not None:
                scores.append(score)
        if not results:
            return None
        avg = sum(scores) / len(scores) if scores else None
        return {"by_symbol": results, "avg_score": avg}

    def _sectors(self) -> dict[str, Any] | None:
        symbols = [s for _, s in _SECTOR_PAIRS]
        resp = fiale_lookup(*symbols)
        if resp.get("error"):
            return None
        lookup = _parse_lookup_symbols(resp)
        changes: dict[str, float | None] = {}
        for short, symbol in _SECTOR_PAIRS:
            item = lookup.get(symbol)
            changes[short] = _lookup_change_pct(item)
        fin = changes.get("financials")
        mat = changes.get("materials")
        diff = None
        if fin is not None and mat is not None:
            diff = fin - mat
        return {"changes_pct": changes, "financials_minus_materials_pct": diff}

    def _asian(self) -> dict[str, Any] | None:
        symbols = [s for _, s in _ASIAN_INDICES]
        resp = fiale_lookup(*symbols)
        if resp.get("error"):
            return None
        lookup = _parse_lookup_symbols(resp)
        changes: dict[str, float] = {}
        for short, symbol in _ASIAN_INDICES:
            item = lookup.get(symbol)
            chg = _lookup_change_pct(item)
            if chg is not None:
                changes[short] = chg
        if not changes:
            return None
        avg = sum(changes.values()) / len(changes)
        return {"changes_pct": changes, "avg_change_pct": avg}

    def _commodities(self) -> dict[str, Any] | None:
        symbols = [s for _, s in _COMMODITIES]
        resp = fiale_lookup(*symbols)
        if resp.get("error"):
            return None
        lookup = _parse_lookup_symbols(resp)
        changes: dict[str, float] = {}
        for short, symbol in _COMMODITIES:
            item = lookup.get(symbol)
            chg = _lookup_change_pct(item)
            if chg is not None:
                changes[short] = chg
        if not changes:
            return None
        avg = sum(changes.values()) / len(changes)
        gold = changes.get("gold")
        ex_gold = sum(v for k, v in changes.items() if k != "gold") / max(
            1, len([k for k in changes if k != "gold"])
        )
        vs_gold = ex_gold - (gold or 0.0)
        return {
            "changes_pct": changes,
            "basket_change_pct": avg,
            "basket_ex_gold_change_pct": ex_gold,
            "basket_vs_gold_change_pct": vs_gold,
        }

    def _rates(self) -> dict[str, Any] | None:
        """Implied Australian yields from ASX interest-rate futures.

        Price is quoted as 100 - yield; a price change is the inverse of the
        yield change.  We return the implied yield level and the approximate
        daily yield change in basis points.
        """
        symbols = [s for _, s in _AU_RATES]
        resp = fiale_lookup(*symbols)
        if resp.get("error"):
            return None
        lookup = _parse_lookup_symbols(resp)
        by_name: dict[str, dict[str, Any]] = {}
        valid = 0
        for short, symbol in _AU_RATES:
            item = lookup.get(symbol)
            if not item:
                continue
            close = item.get("close")
            change = _lookup_change_pct(item)
            if close is None:
                continue
            yield_pct = 100.0 - float(close)
            # change is the percent change in price; convert to yield bps.
            # d(yield points) ~= - d(price points), and bps = points * 100.
            yield_change_bps = None
            if change is not None:
                yield_change_bps = -float(close) * change
            by_name[short] = {
                "price": float(close),
                "yield_pct": yield_pct,
                "yield_change_bps": yield_change_bps,
            }
            if yield_change_bps is not None:
                valid += 1
        if not by_name:
            return None
        return {
            "by_tenor": by_name,
            "rba_cash_rate_expectation_yield": by_name.get("ib1", {}).get("yield_pct"),
            "rba_cash_rate_change_bps": by_name.get("ib1", {}).get("yield_change_bps"),
            "au_3y_yield": by_name.get("yt1", {}).get("yield_pct"),
            "au_3y_yield_change_bps": by_name.get("yt1", {}).get("yield_change_bps"),
            "au_10y_yield": by_name.get("xt1", {}).get("yield_pct"),
            "au_10y_yield_change_bps": by_name.get("xt1", {}).get("yield_change_bps"),
            "valid_count": valid,
        }

    def _china_pulse(self) -> dict[str, Any] | None:
        """China / steel / iron-ore composite using cleaner futures/index proxies."""
        symbols = [s for _, s in _CHINA_PULSE]
        resp = fiale_lookup(*symbols)
        if resp.get("error"):
            return None
        lookup = _parse_lookup_symbols(resp)
        changes: dict[str, float] = {}
        closes: dict[str, float] = {}
        for short, symbol in _CHINA_PULSE:
            item = lookup.get(symbol)
            chg = _lookup_change_pct(item)
            if chg is not None:
                changes[short] = chg
            close = item.get("close") if item else None
            if close is not None:
                closes[short] = float(close)
        if not changes:
            return None
        # Metals / miners carry more signal than broad Hong Kong/China indices here.
        metal_keys = {"iron_ore", "copper", "bhp", "rio", "fmg"}
        broad_keys = {"hang_seng", "china_a50"}
        metal = [v for k, v in changes.items() if k in metal_keys]
        broad = [v for k, v in changes.items() if k in broad_keys]
        metal_avg = sum(metal) / len(metal) if metal else None
        broad_avg = sum(broad) / len(broad) if broad else None
        # Blend 70% metals/miners, 30% broad China proxies.
        if metal_avg is not None and broad_avg is not None:
            composite = 0.7 * metal_avg + 0.3 * broad_avg
        else:
            composite = metal_avg if metal_avg is not None else broad_avg
        return {
            "changes_pct": changes,
            "metal_avg_change_pct": metal_avg,
            "broad_china_avg_change_pct": broad_avg,
            "composite_change_pct": composite,
        }
