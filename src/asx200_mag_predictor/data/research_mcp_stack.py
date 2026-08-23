"""Research MCP adapters: TE calendar, Finnhub sentiment, CNBS/WB snapshot, EarningsCalls.

This is the runtime load used by the predictor dashboards. Cursor/Claude
mcpServers JSON lives in .cursor/mcp.json — same four vendors.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

UA = {"User-Agent": "PredictorMCP/1.1", "Accept": "application/json"}
SGT = timezone(timedelta(hours=8))

RESEARCH_MCP_ENDPOINTS = (
    ("trading_economics_mcp", "https://mcp.tradingeconomics.com"),
    ("finnhub_hosted_mcp", "https://mcp.finnhub.io/mcp"),
    ("earningscalls_mcp", "https://mcp.earningscalls.dev/mcp"),
    ("alphavantage_mcp", "https://mcp.alphavantage.co/mcp"),
    ("equibles_hosted", "https://mcp.equibles.com/mcp"),
)

DEFAULT_SENTIMENT_TICKERS = ("NVDA", "005930.KS", "0700.HK", "TSM", "AAPL")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env(*names: str) -> str:
    for n in names:
        v = (os.environ.get(n) or "").strip()
        if v:
            return v
    return ""


def probe_research_mcps(timeout: float = 3.0) -> list[dict[str, Any]]:
    out = []
    for name, url in RESEARCH_MCP_ENDPOINTS:
        try:
            resp = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
            out.append({"name": name, "url": url, "http_status": resp.status_code, "reachable": resp.status_code < 500, "note": "probe_only_auth_may_be_401"})
        except Exception as exc:
            out.append({"name": name, "url": url, "reachable": False, "error": str(exc)[:160]})
    return out


def fetch_te_calendar(days: int = 7) -> dict[str, Any]:
    key = _env("TRADING_ECONOMICS_API_KEY", "TE_API_KEY")
    if not key:
        return {"status": "skipped", "vendor": "trading_economics", "reason": "missing_TRADING_ECONOMICS_API_KEY"}
    start = datetime.now(timezone.utc).date()
    end = start + timedelta(days=days)
    countries = "united states,japan,china,south korea,hong kong,singapore,australia,euro area"
    try:
        resp = requests.get(
            "https://api.tradingeconomics.com/calendar",
            params={"c": key, "d1": start.isoformat(), "d2": end.isoformat(), "country": countries, "importance": "2,3"},
            headers=UA,
            timeout=12.0,
        )
        if resp.status_code != 200:
            return {"status": "error", "vendor": "trading_economics", "http_status": resp.status_code, "body": resp.text[:240]}
        rows = resp.json() if resp.text else []
        if not isinstance(rows, list):
            rows = []
        slim = []
        for ev in rows[:80]:
            slim.append({"date": ev.get("Date") or ev.get("DateUtc") or ev.get("date"), "country": ev.get("Country"), "event": ev.get("Event"), "importance": ev.get("Importance") or ev.get("ImportanceValue"), "actual": ev.get("Actual"), "forecast": ev.get("Forecast") or ev.get("Consensus"), "previous": ev.get("Previous"), "ticker": ev.get("Ticker")})
        return {"status": "ok", "vendor": "trading_economics", "as_of": _now(), "window_sgt_note": "Convert timestamps to SGT. CN PMI Friday is often Saturday SGT.", "count": len(slim), "events": slim}
    except Exception as exc:
        return {"status": "error", "vendor": "trading_economics", "error": str(exc)[:200]}


def fetch_finnhub_sentiment(tickers: tuple[str, ...] = DEFAULT_SENTIMENT_TICKERS) -> dict[str, Any]:
    key = _env("FINNHUB_API_KEY", "FINNHUB_TOKEN")
    if not key:
        return {"status": "skipped", "vendor": "finnhub", "reason": "missing_FINNHUB_API_KEY"}
    items = []
    errors = []
    for sym in tickers:
        try:
            resp = requests.get("https://finnhub.io/api/v1/news-sentiment", params={"symbol": sym, "token": key}, headers=UA, timeout=8.0)
            if resp.status_code != 200:
                errors.append(f"{sym}:http_{resp.status_code}")
                continue
            data = resp.json() or {}
            sent = data.get("sentiment") or {}
            items.append({"symbol": sym, "vendor": "finnhub", "buzz": (data.get("buzz") or {}).get("buzz"), "weekly_average": (data.get("buzz") or {}).get("weeklyAverage"), "company_news_score": sent.get("companyNewsScore"), "sector_avg": sent.get("sectorAverageNewsScore"), "bearish_pct": sent.get("bearishPercent"), "bullish_pct": sent.get("bullishPercent"), "note": "Do not compare this score to AV / Buzzberg / MarketPsych."})
        except Exception as exc:
            errors.append(f"{sym}:{exc}"[:120])
    status = "ok" if items else "error"
    if items and errors:
        status = "degraded"
    return {"status": status, "vendor": "finnhub", "as_of": _now(), "items": items, "errors": errors}


def fetch_cnbs_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"vendor": "mcp-cnbs_equivalent", "as_of": _now()}
    errors: list[str] = []
    try:
        wb = requests.get("https://api.worldbank.org/v2/country/CHN/indicator/FP.CPI.TOTL.ZG", params={"format": "json", "per_page": 3, "mrnev": 3}, headers=UA, timeout=10.0)
        payload = wb.json()
        series = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        snapshot["china_cpi_yoy_wb"] = [{"year": r.get("date"), "value": r.get("value")} for r in (series or []) if r.get("value") is not None][:3]
    except Exception as exc:
        errors.append(f"wb:{exc}"[:140])
    try:
        imf = requests.get("https://www.imf.org/external/datamapper/api/v1/PCPIPCH/CHN/USA/JPN/KOR/SGP/AUS", headers=UA, timeout=10.0)
        snapshot["imf_inflation_pcpipch"] = imf.json() if imf.status_code == 200 else {"http_status": imf.status_code}
    except Exception as exc:
        errors.append(f"imf:{exc}"[:140])
    snapshot["dual_pmi_rule"] = "Report NBS official PMI and Caixin/RatingDog separately. Never merge."
    snapshot["fred_rule"] = "FRED is the archive, not the 8:30 print. Not used as a wire."
    snapshot["errors"] = errors
    snapshot["status"] = "ok" if not errors else "degraded"
    return snapshot


def fetch_earnings_week() -> dict[str, Any]:
    ect = _env("EARNINGSCALLS_API_KEY", "EARNINGS_CALLS_API_KEY")
    if ect:
        try:
            resp = requests.get("https://earningscalls.dev/api/v1/calls", params={"limit": 15}, headers={**UA, "X-API-Key": ect}, timeout=12.0)
            if resp.status_code == 200:
                return {"status": "ok", "vendor": "earningscalls", "as_of": _now(), "payload": resp.json()}
            return {"status": "error", "vendor": "earningscalls", "http_status": resp.status_code, "body": resp.text[:240]}
        except Exception as exc:
            return {"status": "error", "vendor": "earningscalls", "error": str(exc)[:200]}
    key = _env("FINNHUB_API_KEY", "FINNHUB_TOKEN")
    if not key:
        return {"status": "skipped", "vendor": "earningscalls", "reason": "missing_EARNINGSCALLS_API_KEY_and_FINNHUB_API_KEY", "fallback": "calendar_plus_eps_surprise_only"}
    start = datetime.now(timezone.utc).date()
    end = start + timedelta(days=7)
    try:
        resp = requests.get("https://finnhub.io/api/v1/calendar/earnings", params={"from": start.isoformat(), "to": end.isoformat(), "token": key}, headers=UA, timeout=10.0)
        data = resp.json() if resp.status_code == 200 else {}
        return {"status": "ok" if resp.status_code == 200 else "error", "vendor": "finnhub_earnings_calendar_fallback", "note": "EarningsCalls unconfigured — no transcripts, calendar + estimates only.", "http_status": resp.status_code, "earningsCalendar": (data.get("earningsCalendar") or [])[:40] if isinstance(data, dict) else []}
    except Exception as exc:
        return {"status": "error", "vendor": "finnhub_earnings_calendar_fallback", "error": str(exc)[:200]}


def fetch_week_ahead_us_apac() -> dict[str, Any]:
    calendar = fetch_te_calendar()
    prints = fetch_cnbs_snapshot()
    sentiment = fetch_finnhub_sentiment()
    transcripts = fetch_earnings_week()
    probes = probe_research_mcps()
    configured = {
        "trading_economics": calendar.get("status") != "skipped",
        "finnhub": sentiment.get("status") != "skipped",
        "cnbs_equivalent": prints.get("status") in {"ok", "degraded"},
        "earningscalls": transcripts.get("vendor") == "earningscalls" and transcripts.get("status") == "ok",
    }
    return {
        "name": "week_ahead_us_apac",
        "as_of": _now(),
        "tz_display": "SGT",
        "order": ["calendar_te", "prints_cnbs", "sentiment_finnhub", "transcripts_earningscalls"],
        "limits": [
            "Free != print wire. Scrapers miss timestamps, revisions, dual PMI.",
            "FRED is the archive, not the 8:30 print.",
            "CN PMI Friday is often Saturday SGT.",
            "Do not compare Finnhub vs AV vs Buzzberg vs MarketPsych scores.",
            "Re-call after any timestamp — MCP clients cache.",
        ],
        "configured": configured,
        "mcp_probes": probes,
        "calendar": calendar,
        "prints": prints,
        "sentiment": sentiment,
        "transcripts": transcripts,
    }
