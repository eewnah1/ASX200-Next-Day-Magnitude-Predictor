"""Free / freemium MCP-equivalent market stack for US + Asia-Pacific."""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote as urlquote

import requests

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_UA = "Mozilla/5.0 (compatible; PredictorMCP/1.0; +https://github.com/eewnah1)"
REMOTE_MCP_ENDPOINTS = (
    ("stockmcp_hosted", "https://stockmcp.leoguerin.fr/mcp"),
    ("equibles_hosted", "https://mcp.equibles.com/mcp"),
    ("alphavantage_mcp", "https://mcp.alphavantage.co/mcp"),
)
DEFAULT_UNIVERSES: dict[str, dict[str, str]] = {
    "asx200": {"XJO": "^AXJO", "SPI": "AP=F", "CBA": "CBA.AX", "BHP": "BHP.AX", "RIO": "RIO.AX", "NAB": "NAB.AX", "WBC": "WBC.AX", "ANZ": "ANZ.AX", "AUDUSD": "AUDUSD=X", "IRON": "TIO=F", "GOLD": "GC=F", "SPX": "^GSPC", "NDX": "^IXIC", "VIX": "^VIX"},
    "ausuper": {"XJO": "^AXJO", "SPX": "^GSPC", "NDX": "^IXIC", "ACWI": "ACWI", "EEM": "EEM", "AGG": "AGG", "BND": "BND", "GLD": "GLD", "AUDUSD": "AUDUSD=X", "VIX": "^VIX", "STW": "STW.AX", "IOZ": "IOZ.AX"},
    "international": {"ACWI": "ACWI", "VXUS": "VXUS", "EFA": "EFA", "EEM": "EEM", "SPX": "^GSPC", "NDX": "^IXIC", "NKY": "^N225", "HSI": "^HSI", "KS11": "^KS11", "STOXX": "^STOXX50E", "DXY": "DX-Y.NYB", "VIX": "^VIX", "USDJPY": "USDJPY=X"},
    "pinebridge": {"EEM": "EEM", "AAXJ": "AAXJ", "FXI": "FXI", "EWY": "EWY", "EWT": "EWT", "INDA": "INDA", "EWS": "EWS", "NKY": "^N225", "HSI": "^HSI", "KS11": "^KS11", "TWII": "^TWII", "SENSEX": "^BSESN", "USDJPY": "USDJPY=X", "USDCNY": "USDCNY=X", "VIX": "^VIX"},
    "ut_switch": {"STI": "^STI", "SPX": "^GSPC", "NDX": "^IXIC", "HSI": "^HSI", "NKY": "^N225", "XJO": "^AXJO", "EEM": "EEM", "GLD": "GLD", "TLT": "TLT", "USDSGD": "USDSGD=X", "VIX": "^VIX", "ES3": "ES3.SI", "G3B": "G3B.SI"},
}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _round(v: float | None, n: int = 4) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), n)

def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff >= 0: gains += diff
        else: losses -= diff
    if losses == 0: return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def _yahoo_chart(ticker: str, range_: str = "3mo", interval: str = "1d", timeout: float = 8.0) -> dict[str, Any]:
    url = YAHOO_CHART.format(ticker=urlquote(ticker, safe=""))
    resp = requests.get(url, params={"range": range_, "interval": interval, "events": "div,splits"}, headers={"User-Agent": YAHOO_UA, "Accept": "application/json"}, timeout=timeout)
    if resp.status_code != 200:
        return {"ticker": ticker, "error": f"http_{resp.status_code}"}
    payload = resp.json()
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result:
        return {"ticker": ticker, "error": "empty_chart"}
    meta = result.get("meta") or {}
    ts = result.get("timestamp") or []
    ohlc = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = [c for c in (ohlc.get("close") or []) if c is not None]
    if len(closes) < 2:
        return {"ticker": ticker, "last": meta.get("regularMarketPrice"), "error": "insufficient_history"}
    last, prev = closes[-1], closes[-2]
    ret_1d = (last / prev - 1.0) * 100.0 if prev else None
    window_5 = closes[-6:] if len(closes) >= 6 else closes
    ret_5d = (window_5[-1] / window_5[0] - 1.0) * 100.0 if len(window_5) >= 2 and window_5[0] else None
    window_21 = closes[-22:] if len(closes) >= 22 else closes
    ret_21d = (window_21[-1] / window_21[0] - 1.0) * 100.0 if len(window_21) >= 2 and window_21[0] else None
    rets = []
    for i in range(1, min(len(closes), 22)):
        if closes[-i - 1]:
            rets.append(closes[-i] / closes[-i - 1] - 1.0)
    vol = None
    if len(rets) >= 5:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
        vol = math.sqrt(var) * math.sqrt(252.0) * 100.0
    return {"ticker": ticker, "last": last, "prev": prev, "currency": meta.get("currency"), "exchange": meta.get("exchangeName"), "ret_1d_pct": _round(ret_1d), "ret_5d_pct": _round(ret_5d), "ret_21d_pct": _round(ret_21d), "ann_vol_21d_pct": _round(vol), "rsi_14": _round(_rsi(closes, 14)), "as_of": datetime.fromtimestamp(ts[-1], tz=timezone.utc).isoformat() if ts else _now_iso(), "source": "yahoo_chart_v8"}

def _probe_remote_mcp(name: str, url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        resp = requests.get(url, headers={"Accept": "application/json, text/event-stream", "User-Agent": YAHOO_UA}, timeout=timeout, allow_redirects=True)
        return {"name": name, "url": url, "http_status": resp.status_code, "reachable": resp.status_code < 500, "note": "probe_only_no_session"}
    except Exception as exc:
        return {"name": name, "url": url, "http_status": None, "reachable": False, "error": str(exc)[:180]}

def _alpha_vantage_quote(symbol: str) -> dict[str, Any] | None:
    key = (os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get("ALPHA_VANTAGE_API_KEY") or os.environ.get("ALPHAVANTAGE_KEY") or "").strip()
    if not key:
        return None
    try:
        resp = requests.get("https://www.alphavantage.co/query", params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key}, timeout=8.0)
        data = resp.json()
        q = data.get("Global Quote") or {}
        if not q:
            return {"symbol": symbol, "error": list(data.keys())[:3]}
        return {"symbol": symbol, "price": float(q.get("05. price") or 0) or None, "change_pct": float(str(q.get("10. change percent") or "0").replace("%", "") or 0), "source": "alpha_vantage_rest"}
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)[:160]}

def _factor_tilt(quotes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def r(name: str) -> float | None:
        return (quotes.get(name) or {}).get("ret_1d_pct")
    risk_on = [r(n) for n in ("SPX", "NDX", "EEM", "XJO", "NKY", "HSI", "KS11", "AAXJ") if r(n) is not None]
    risk_on_avg = sum(risk_on) / len(risk_on) if risk_on else 0.0
    vix = r("VIX") or 0.0
    score = risk_on_avg - 0.25 * vix
    bias = "risk_on" if score >= 0.35 else ("risk_off" if score <= -0.35 else "neutral")
    return {"bias": bias, "score": _round(score), "confidence": _round(min(0.85, 0.35 + abs(score) / 4.0), 3), "risk_on_avg_1d_pct": _round(risk_on_avg), "vix_1d_pct": _round(vix), "names_used": [n for n, q in quotes.items() if q.get("ret_1d_pct") is not None]}

def fetch_free_mcp_stack(universe: str = "asx200", extra_tickers: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    tickers = dict(DEFAULT_UNIVERSES.get(universe, DEFAULT_UNIVERSES["asx200"]))
    if extra_tickers:
        tickers.update(extra_tickers)
    quotes: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, yf_ticker in tickers.items():
        try:
            quotes[name] = _yahoo_chart(yf_ticker)
            if quotes[name].get("error"):
                errors.append(f"{name}:{quotes[name]['error']}")
        except Exception as exc:
            quotes[name] = {"ticker": yf_ticker, "error": str(exc)[:160]}
            errors.append(f"{name}:{exc}")
    ok_quotes = {k: v for k, v in quotes.items() if v.get("ret_1d_pct") is not None}
    tilt = _factor_tilt(quotes)
    remote = [_probe_remote_mcp(n, u) for n, u in REMOTE_MCP_ENDPOINTS]
    av = _alpha_vantage_quote("SPY")
    sources_ok = ["yahoo_chart_v8"]
    if av and av.get("price"):
        sources_ok.append("alpha_vantage_rest")
    sources_ok.extend([p["name"] for p in remote if p.get("reachable")])
    status = "ok" if ok_quotes else "failed"
    if ok_quotes and errors:
        status = "degraded"
    return {"name": "free_mcp_stack", "status": status, "universe": universe, "as_of": _now_iso(), "elapsed_ms": int((time.time() - started) * 1000), "quotes": quotes, "ok_count": len(ok_quotes), "requested_count": len(tickers), "tilt": tilt, "remote_mcp_probes": remote, "alpha_vantage": av, "sources_ok": sources_ok, "errors": errors[:12], "note": "Yahoo chart is the free MCP data plane for US + APAC. Remote MCP URLs are probed only."}

def overlay_prediction(base: dict[str, Any], stack: dict[str, Any], *, key_field: str = "bias") -> dict[str, Any]:
    out = dict(base)
    tilt = stack.get("tilt") or {}
    out["mcp_stack"] = {"status": stack.get("status"), "universe": stack.get("universe"), "tilt": tilt, "sources_ok": stack.get("sources_ok"), "ok_count": stack.get("ok_count"), "requested_count": stack.get("requested_count"), "as_of": stack.get("as_of")}
    used = list(out.get("mcp_sources_used") or [])
    for src in stack.get("sources_ok") or []:
        if src not in used:
            used.append(src)
    out["mcp_sources_used"] = used
    if tilt.get("bias") and key_field not in out:
        out["mcp_regime"] = tilt.get("bias")
    return out
