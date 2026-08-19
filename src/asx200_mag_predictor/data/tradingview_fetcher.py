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
_MAX_WORKERS = 4

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
        ]

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {
                name: executor.submit(fn) for name, fn in tasks  # type: ignore[arg-type]
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
        ex_gold = (
            sum(v for k, v in changes.items() if k != "gold")
            / max(1, len([k for k in changes if k != "gold"]))
        )
        vs_gold = ex_gold - (gold or 0.0)
        return {
            "changes_pct": changes,
            "basket_change_pct": avg,
            "basket_ex_gold_change_pct": ex_gold,
            "basket_vs_gold_change_pct": vs_gold,
        }
