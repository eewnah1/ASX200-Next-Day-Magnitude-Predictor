"""Source / MCP catalogue for the ASX200 next-day magnitude predictor."""

from __future__ import annotations

import os
import importlib.util
from typing import Any


def _env(*names: str) -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _env_bool(*names: str) -> bool:
    return bool(_env(*names))


def _has_import(module_name: str) -> bool:
    """Best-effort check whether a Python package is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:  # noqa: BLE001
        return False


def get_market_sources() -> list[dict[str, Any]]:
    """Return a catalogue of data sources and MCPs used by the predictor."""
    yfinance_live = _has_import("yfinance")
    alpha_key = _env_bool(
        "ALPHAVANTAGE_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "ALPHAVANTAGE_KEY",
    )
    fred_key = _env_bool("FRED_API_KEY")
    tradingview_key = _env_bool(
        "TRADINGVIEW_MCP_API_KEY",
        "TRADINGVIEW_API_KEY",
        "TV_MCP_API_KEY",
        "TV_MCP_TOKEN",
    )
    asx_key = _env_bool("ASX_MCP_API_KEY", "ASX_API_KEY", "ASX_REALTIME_API_KEY")
    finnhub_key = _env_bool("FINNHUB_API_KEY", "FINNHUB_TOKEN")
    te_key = _env_bool("TRADING_ECONOMICS_API_KEY", "TE_API_KEY")
    marketpsych_key = _env_bool(
        "MARKETPSYCH_API_KEY",
        "BUZZBERG_API_KEY",
        "MARKETPSYCH_API_TOKEN",
        "BUZZBERG_API_TOKEN",
    )
    benzinga = _env_bool("BENZINGA_API_KEY")
    newsapi = _env_bool("NEWSAPI_API_KEY", "NEWS_API_KEY")
    stocktwits = _env_bool("STOCKTWITS_API_KEY")
    twitter = _env_bool("TWITTER_BEARER_TOKEN", "X_BEARER_TOKEN")
    unusualwhales = _env_bool("UNUSUAL_WHALES_API_KEY")
    openfigi = _env_bool("OPENFIGI_API_KEY")
    polygon = _env_bool("POLYGON_API_KEY")
    alpaca = _env_bool("ALPACA_API_KEY", "ALPACA_PAPER_API_KEY")
    fmp = _env_bool("FINANCIAL_MODELING_PREP_API_KEY", "FMP_API_KEY")

    def yf_note(label: str) -> str:
        return f"{label} via Yahoo Finance." + ("" if yfinance_live else " (yfinance not installed)")

    sources: list[dict[str, Any]] = [
        {
            "name": "Yahoo Finance / yfinance",
            "type": "price",
            "configured": yfinance_live,
            "note": (
                "Live feed for ^AXJO, SPY, VIX, GC=F, CL=F, AUDUSD=X, "
                "BHP.AX, CBA.AX, WDS.AX, FMG.AX, CSL.AX, NAB.AX, WBC.AX, ANZ.AX, GMG.AX, "
                "TLS.AX, RIO.AX, iron ore, gold, oil, copper and US 10Y. "
                "Yahoo Finance MCP toolkit: historical prices, stock info, news, stock actions, "
                "financial statements, holder info, option expirations, option chain, recommendations."
            ),
        },
        {
            "name": "S&P/ASX 200 cash feed",
            "type": "price",
            "configured": yfinance_live,
            "note": "Primary anchor is ^AXJO via Yahoo Finance; SPI 200 futures (^APAI=F) used as proxy.",
        },
        {
            "name": "ASX MCP / ASX Realtime",
            "type": "MCP",
            "configured": asx_key or yfinance_live,
            "note": (
                "Set ASX_MCP_API_KEY / ASX_API_KEY for ASX-listed prices and company announcements."
                if not asx_key
                else "ASX-listed prices and announcements active."
            ),
        },
        {
            "name": "TradingView MCP",
            "type": "MCP",
            "configured": tradingview_key,
            "note": (
                "Set TRADINGVIEW_MCP_API_KEY / TRADINGVIEW_API_KEY."
                if not tradingview_key
                else "ASX technical analysis, screener, price and symbol lookup."
            ),
        },
        {
            "name": "Alpha Vantage",
            "type": "price",
            "configured": alpha_key,
            "note": (
                "Set ALPHAVANTAGE_API_KEY."
                if not alpha_key
                else "Global quote, FX and Treasury yield enrichment."
            ),
        },
        {
            "name": "FRED",
            "type": "macro",
            "configured": fred_key,
            "note": (
                "Set FRED_API_KEY."
                if not fred_key
                else "US Treasury yields and macro series archive."
            ),
        },
        {
            "name": "RBA official cash rate / yield curve",
            "type": "macro",
            "configured": True,
            "note": "Reserve Bank of Australia cash rate and Australian government yield curve; sourced from rba.gov.au and Yahoo Finance (AUSB10Y=).",
        },
        {
            "name": "Australian Bureau of Statistics (ABS)",
            "type": "macro",
            "configured": True,
            "note": "Australian labour, inflation, GDP and housing statistics; configure ABS_API_KEY for API access.",
        },
        {
            "name": "ASX sector indices",
            "type": "price",
            "configured": yfinance_live,
            "note": yf_note("Sector proxy ETFs (Vanguard / iShares / BetaShares / VanEck)"),
        },
        {
            "name": "Australian ETF providers",
            "type": "fundamental",
            "configured": yfinance_live,
            "note": "Holdings and flows proxies from IOZ, VAS, VGS, VEU, VGE, VDHG, DHHF, VTS, VEU, A200, QOZ, MVW via yfinance.",
        },
        {
            "name": "AUD FX basket",
            "type": "price",
            "configured": yfinance_live or alpha_key,
            "note": "AUDUSD=X, AUDJPY=X, AUDCNY=X, AUDNZD=X, TWI proxies for USD/CNY/JPY/NZD sensitivity.",
        },
        {
            "name": "Iron ore",
            "type": "price",
            "configured": yfinance_live,
            "note": "Primary FE=F / TIO=F / MT=F with BHP/RIO/FMG equity fallback via yfinance.",
        },
        {
            "name": "Copper",
            "type": "price",
            "configured": yfinance_live,
            "note": "HG=F via Yahoo Finance; BHP/PMT.AX/RIO fallback.",
        },
        {
            "name": "Lithium",
            "type": "price",
            "configured": yfinance_live,
            "note": "Lithium carbonate equity proxies (Pilbara Minerals, Mineral Resources, IGO via yfinance); futures still limited on free feeds.",
        },
        {
            "name": "Coal / energy",
            "type": "price",
            "configured": yfinance_live,
            "note": "Coal futures (MTF=F, Newcastle coal index proxies) plus WDS.AX / YAL.AX equity fallback.",
        },
        {
            "name": "Gold",
            "type": "price",
            "configured": yfinance_live,
            "note": "GC=F via Yahoo Finance; NCM.AX, NST.AX fallback.",
        },
        {
            "name": "Oil (WTI)",
            "type": "price",
            "configured": yfinance_live,
            "note": "CL=F via Yahoo Finance; WDS.AX, STO.AX fallback.",
        },
        {
            "name": "Natural gas",
            "type": "price",
            "configured": yfinance_live,
            "note": "NG=F via Yahoo Finance; WDS.AX, STO.AX fallback.",
        },
        {
            "name": "Finnhub MCP",
            "type": "MCP",
            "configured": finnhub_key,
            "note": (
                "Set FINNHUB_API_KEY."
                if not finnhub_key
                else "News sentiment and earnings calendar."
            ),
        },
        {
            "name": "Trading Economics",
            "type": "calendar",
            "configured": te_key,
            "note": (
                "Set TRADING_ECONOMICS_API_KEY / TE_API_KEY."
                if not te_key
                else "Economic calendar for AU, US, CN and APAC."
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
            "configured": marketpsych_key,
            "note": (
                "Set MARKETPSYCH_API_KEY / BUZZBERG_API_KEY."
                if not marketpsych_key
                else "Alternative macro and equity sentiment signals."
            ),
        },
        {
            "name": "ASIC short position reports",
            "type": "fundamental",
            "configured": True,
            "note": "Weekly aggregated short positions from ASIC; weekly frequency, public data.",
        },
        {
            "name": "SPI 200 futures",
            "type": "price",
            "configured": yfinance_live,
            "note": yf_note("SPI 200 futures (^APAI=F) for overnight ASX direction"),
        },
        {
            "name": "MSCI Australia",
            "type": "macro",
            "configured": yfinance_live,
            "note": yf_note("Proxy via EWA or AUS iShares ETF"),
        },
        {
            "name": "MSCI World ex-Australia",
            "type": "macro",
            "configured": yfinance_live,
            "note": yf_note("Proxy via VGS, VEU, IWLD, VGAD"),
        },
        {
            "name": "Westpac consumer sentiment",
            "type": "macro",
            "configured": True,
            "note": "Monthly Westpac-Melbourne Institute consumer sentiment; scraped from westpac.com.au / public releases.",
        },
        {
            "name": "ANZ job advertisements",
            "type": "macro",
            "configured": True,
            "note": "Monthly ANZ job ads; public release from anz.com.au.",
        },
        {
            "name": "Ai Group / S&P Global Australia PMI",
            "type": "macro",
            "configured": True,
            "note": "Monthly Australian manufacturing and services PMI; public releases.",
        },
        {
            "name": "CFTC AUD COT",
            "type": "macro",
            "configured": True,
            "note": "Weekly Commitment of Traders report for AUD futures positioning.",
        },
        {
            "name": "China macro proxies",
            "type": "macro",
            "configured": yfinance_live,
            "note": "CNY=X, ^SSEC, FXI, MCHI, KWEB for demand/commodity and currency spillover into ASX.",
        },
        {
            "name": "RBNZ / RBA / Fed policy calendars",
            "type": "calendar",
            "configured": True,
            "note": "Scheduled central bank statements and rate decisions relevant to AUD and ASX.",
        },
        {
            "name": "Australian bank earnings calendar",
            "type": "calendar",
            "configured": yfinance_live,
            "note": "CBA, WBC, NAB, ANZ, MQG reporting dates via Yahoo Finance calendars (best effort).",
        },
        {
            "name": "ASX company announcements",
            "type": "news",
            "configured": asx_key or yfinance_live,
            "note": "ASX announcements feed via ASX API or Yahoo Finance news.",
        },
        {
            "name": "Ausbiz / Livewire / MarketIndex",
            "type": "news",
            "configured": newsapi,
            "note": (
                "Set NEWSAPI_API_KEY to pull AU financial news."
                if not newsapi
                else "Australian financial media aggregation."
            ),
        },
        {
            "name": "Benzinga",
            "type": "news",
            "configured": benzinga,
            "note": (
                "Set BENZINGA_API_KEY."
                if not benzinga
                else "Live AU/US cross-listed news and analyst moves."
            ),
        },
        {
            "name": "Social sentiment (StockTwits / X / Reddit)",
            "type": "news",
            "configured": stocktwits or twitter,
            "note": (
                "Set STOCKTWITS_API_KEY / TWITTER_BEARER_TOKEN / X_BEARER_TOKEN."
                if not (stocktwits or twitter)
                else "Social sentiment for ASX blue-chips."
            ),
        },
        {
            "name": "Australian options / warrants flow",
            "type": "MCP",
            "configured": unusualwhales or asx_key,
            "note": (
                "Set UNUSUAL_WHALES_API_KEY or ASX_MCP_API_KEY."
                if not (unusualwhales or asx_key)
                else "ASX and cross-listed options flow (best effort)."
            ),
        },
        {
            "name": "Alpaca / Polygon US-hours order flow",
            "type": "MCP",
            "configured": alpaca or polygon,
            "note": (
                "Set ALPACA_API_KEY / POLYGON_API_KEY."
                if not (alpaca or polygon)
                else "US pre/after-hours flow affecting ASX ADRs."
            ),
        },
        {
            "name": "OpenFIGI",
            "type": "MCP",
            "configured": openfigi,
            "note": (
                "Set OPENFIGI_API_KEY."
                if not openfigi
                else "Symbology mapping for ASX-to-US ADRs and ETFs."
            ),
        },
        {
            "name": "Financial Modeling Prep",
            "type": "MCP",
            "configured": fmp,
            "note": (
                "Set FINANCIAL_MODELING_PREP_API_KEY / FMP_API_KEY."
                if not fmp
                else "ASX fundamentals, insider transactions and calendar."
            ),
        },
    ]

    return sources
