"""TradingView MCP data adapters.

Wraps the installed ``tradingview-mcp-server`` (atilaahmettaner/tradingview-mcp)
Python API.  The previous fiale-plus CLI dependency is replaced with direct calls
to the ``stock_screener_service`` so the predictor works in containers without
a ``tradingview-cli`` binary.
"""

from __future__ import annotations

from typing import Any, cast

from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)


def _as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of a model/dict response to a plain dict."""
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return {"value": obj}


def atila_market_snapshot() -> dict[str, Any]:
    """Global market snapshot via atilaahmettaner/tradingview-mcp."""
    try:
        from tradingview_mcp.core.services.yahoo_finance_service import (
            get_market_snapshot,
        )

        return {"source": "atilaahmettaner/tradingview-mcp", "data": get_market_snapshot()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("atila market snapshot failed")
        return {"source": "atilaahmettaner/tradingview-mcp", "error": str(exc)}


def atila_symbol_analysis(symbol: str, exchange: str, interval: str = "1D") -> dict[str, Any]:
    """Multi-agent technical/sentiment/risk analysis for a symbol/timeframe."""
    try:
        from tradingview_mcp.core.services.multi_agent_service import (
            run_multi_agent_analysis,
        )

        result = cast(dict[str, Any], run_multi_agent_analysis(symbol, exchange, interval))
        result["source"] = "atilaahmettaner/tradingview-mcp"
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("atila symbol analysis failed")
        return {"source": "atilaahmettaner/tradingview-mcp", "error": str(exc)}


def atila_price(symbol: str) -> dict[str, Any]:
    """Latest price quote for a single symbol."""
    try:
        from tradingview_mcp.core.services.yahoo_finance_service import get_price

        return {"source": "atilaahmettaner/tradingview-mcp", "data": get_price(symbol)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("atila price failed")
        return {"source": "atilaahmettaner/tradingview-mcp", "error": str(exc)}


def fiale_screen(
    asset_type: str = "stocks",
    preset: str = "quality_stocks",
    limit: int = 10,
) -> dict[str, Any]:
    """Run a TradingView screener preset.

    ``asset_type`` and ``preset`` are retained for API compatibility but are not
    passed to the underlying service because it does not support arbitrary
    preset names.  Use ``screen_stocks`` directly if you need a custom screen.
    """
    try:
        from tradingview_mcp.core.services.stock_screener_service import screen_stocks

        # Default to a broad US common-stock screen; keep the requested limit.
        data = screen_stocks(
            country="america",
            stock_type="common",
            limit=max(1, min(limit, 1000)),
            exclude_otc=True,
            compact=True,
            sort_by="market_cap",
        )
        return {"source": "atilaahmettaner/tradingview-mcp", "data": _as_dict(data)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("fiale screen failed")
        return {"source": "atilaahmettaner/tradingview-mcp", "error": str(exc)}


def fiale_lookup(*symbols: str) -> dict[str, Any]:
    """Look up one or more TradingView symbols using the installed Python API.

    The legacy fiale-plus CLI is not required; this function calls
    ``fetch_stock_prices`` and returns a compatible ``{"symbols": [...]}``
    envelope so existing consumers do not need to change.
    """
    if not symbols:
        return {"error": "No symbols provided"}
    try:
        from tradingview_mcp.core.services.stock_screener_service import (
            fetch_stock_prices,
        )

        payload = ",".join(str(s).strip() for s in symbols)
        data = fetch_stock_prices(payload)
        rows = data.get("rows", [])
        out: list[dict[str, Any]] = []
        for row in rows:
            ticker = row.get("ticker") or row.get("symbol")
            if not ticker:
                continue
            change_pct = row.get("change_percent")
            price = row.get("price")
            out.append(
                {
                    "symbol": ticker,
                    "ticker": ticker,
                    "change": change_pct,
                    "change_percent": change_pct,
                    "close": price,
                    "price": price,
                    "currency": row.get("currency"),
                    "exchange": row.get("exchange"),
                }
            )
        if not out:
            not_found = data.get("not_found", [])
            if not_found:
                return {"error": f"No price data for {', '.join(str(s) for s in not_found)}"}
            return {"error": "No price data returned"}
        return {"symbols": out}
    except Exception as exc:  # noqa: BLE001
        logger.exception("fiale_lookup failed")
        return {"error": str(exc)}


def fiale_presets() -> dict[str, Any]:
    """List available presets.

    The underlying Python API does not expose preset discovery, so this returns
    the standard built-in categories.
    """
    return {
        "source": "atilaahmettaner/tradingview-mcp",
        "presets": ["quality_stocks", "top_gainers", "top_losers"],
    }


def get_asx200_insights() -> dict[str, Any]:
    """Combined TradingView MCP insights for the ASX 200 predictor."""
    return {
        "market_snapshot": atila_market_snapshot(),
        "asx200_analysis": atila_symbol_analysis("ASX:XJO", "asx"),
        "asx200_price": atila_price("^AXJO"),
        "quality_screener": fiale_screen("stocks", "quality_stocks", 10),
    }
