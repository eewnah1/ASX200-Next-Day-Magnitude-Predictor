"""Source / MCP catalogue for the ASX200 next-day magnitude predictor."""

from __future__ import annotations

import os
from typing import Any


def _env(*names: str) -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _has_import(module_name: str) -> bool:
    """Best-effort check whether a Python package is importable."""
    try:
        __import__(module_name)
        return True
    except Exception:  # noqa: BLE001
        return False


def get_market_sources() -> list[dict[str, Any]]:
    """Return a catalogue of data sources and MCPs used by the predictor."""
    yfinance_live = _has_import("yfinance")
    alpha_key = _env(
        "ALPHAVANTAGE_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "ALPHAVANTAGE_KEY",
    )
    fred_key = _env("FRED_API_KEY")
    tradingview_key = _env(
        "TRADINGVIEW_MCP_API_KEY",
        "TRADINGVIEW_API_KEY",
        "TV_MCP_API_KEY",
        "TV_MCP_TOKEN",
    )
    asx_key = _env("ASX_MCP_API_KEY", "ASX_API_KEY")
    finnhub_key = _env("FINNHUB_API_KEY", "FINNHUB_TOKEN")
    te_key = _env("TRADING_ECONOMICS_API_KEY", "TE_API_KEY")
    marketpsych_key = _env(
        "MARKETPSYCH_API_KEY",
        "BUZZBERG_API_KEY",
        "MARKETPSYCH_API_TOKEN",
        "BUZZBERG_API_TOKEN",
    )

    sources: list[dict[str, Any]] = [
        {
            "name": "Yahoo Finance / yfinance",
            "type": "price",
            "configured": yfinance_live,
            "note": (
                "Live feed for ^AXJO, SPY, VIX, GC=F, CL=F, AUDUSD=X, "
                "BHP.AX, CBA.AX, WDS.AX, FMG.AX, iron ore proxy, US 10Y. "
                + ("Package importable." if yfinance_live else "yfinance package not installed.")
            ),
        },
        {
            "name": "S&P/ASX 200 cash feed",
            "type": "price",
            "configured": yfinance_live,
            "note": "Primary anchor is ^AXJO via Yahoo Finance.",
        },
        {
            "name": "Alpha Vantage",
            "type": "price",
            "configured": bool(alpha_key),
            "note": (
                "Global quote, FX and Treasury yield enrichment. "
                + ("API key present." if alpha_key else "Set ALPHAVANTAGE_API_KEY.")
            ),
        },
        {
            "name": "FRED",
            "type": "macro",
            "configured": bool(fred_key),
            "note": (
                "US Treasury yields and macro series archive. "
                + ("API key present." if fred_key else "Set FRED_API_KEY.")
            ),
        },
        {
            "name": "TradingView MCP",
            "type": "MCP",
            "configured": bool(tradingview_key),
            "note": (
                "ASX technical analysis, screener, price and symbol lookup. "
                + ("Env key present." if tradingview_key else "Set TRADINGVIEW_MCP_API_KEY / TRADINGVIEW_API_KEY.")
            ),
        },
        {
            "name": "ASX MCP",
            "type": "MCP",
            "configured": bool(asx_key),
            "note": (
                "ASX-listed security data and announcements. "
                + ("Env key present." if asx_key else "Set ASX_MCP_API_KEY.")
            ),
        },
        {
            "name": "RBA official cash rate / yields",
            "type": "macro",
            "configured": bool(_env("RBA_API_KEY", "RBA_OFFICIAL_CASH_RATE_URL")),
            "note": "Reserve Bank of Australia cash rate and yield curve; typically scraped from rba.gov.au unless an API key is configured.",
        },
        {
            "name": "Australian Bureau of Statistics (ABS)",
            "type": "macro",
            "configured": bool(_env("ABS_API_KEY", "ABS_STATS_API_KEY")),
            "note": "Australian labour, inflation and housing statistics; configure ABS_API_KEY for API access.",
        },
        {
            "name": "Finnhub MCP",
            "type": "MCP",
            "configured": bool(finnhub_key),
            "note": (
                "News sentiment and earnings calendar. "
                + ("API key present." if finnhub_key else "Set FINNHUB_API_KEY.")
            ),
        },
        {
            "name": "Trading Economics",
            "type": "calendar",
            "configured": bool(te_key),
            "note": (
                "Economic calendar for AU, US, CN and APAC. "
                + ("API key present." if te_key else "Set TRADING_ECONOMICS_API_KEY / TE_API_KEY.")
            ),
        },
        {
            "name": "IRESS / Refinitiv / FactSet / Bloomberg",
            "type": "MCP",
            "configured": False,
            "note": "Institutional feeds; not configured in this deployment.",
        },
        {
            "name": "MarketPsych / Buzzberg sentiment",
            "type": "news",
            "configured": bool(marketpsych_key),
            "note": (
                "Alternative macro and equity sentiment signals. "
                + ("Env key present." if marketpsych_key else "Set MARKETPSYCH_API_KEY / BUZZBERG_API_KEY.")
            ),
        },
        {
            "name": "AUD/USD FX",
            "type": "price",
            "configured": yfinance_live or bool(alpha_key),
            "note": "Proxy for USD sensitivity; sourced from Yahoo Finance (AUDUSD=X) or Alpha Vantage FX_DAILY.",
        },
        {
            "name": "Iron ore",
            "type": "price",
            "configured": yfinance_live,
            "note": "Primary FE=F / TIO=F / MT=F with BHP/RIO/FMG equity fallback via yfinance.",
        },
        {
            "name": "Gold",
            "type": "price",
            "configured": yfinance_live,
            "note": "GC=F via Yahoo Finance.",
        },
        {
            "name": "Oil (WTI)",
            "type": "price",
            "configured": yfinance_live,
            "note": "CL=F via Yahoo Finance.",
        },
    ]

    return sources
