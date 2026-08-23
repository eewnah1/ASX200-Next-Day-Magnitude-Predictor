"""MCP stack HTTP routes + mini dashboard."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from asx200_mag_predictor.data.free_mcp_stack import fetch_free_mcp_stack
from asx200_mag_predictor.data.research_mcp_stack import fetch_week_ahead_us_apac

router = APIRouter()

ALLOWED = {"asx200", "ausuper", "international", "pinebridge", "ut_switch"}

PAGE = """<!DOCTYPE html><html><head><meta charset=utf-8><title>MCP stack</title>
<style>body{font-family:sans-serif;background:#0b1220;color:#e2e8f0;margin:24px}pre{background:#111827;padding:12px;border-radius:8px;overflow:auto;max-height:62vh}button{margin:4px;padding:8px 12px}</style></head>
<body><h1>Predictor MCP stack</h1>
<p>Yahoo tilt + TE / Finnhub / CNBS / EarningsCalls week-ahead pack.</p>
<div>
<button onclick=\"go('asx200')\">ASX 200</button>
<button onclick=\"go('international')\">International</button>
<button onclick=\"go('pinebridge')\">PineBridge Asia</button>
<button onclick=\"go('ausuper')\">AusSuper</button>
<button onclick=\"go('ut_switch')\">UT-Switch</button>
<button onclick=\"research()\">Week-ahead US+APAC</button>
</div>
<p id=s></p><pre id=p>loading…</pre>
<script>
async function go(u){
 document.getElementById('s').textContent='Loading '+u;
 const r=await fetch('/api/v1/mcp/stack?universe='+u);
 const d=await r.json();
 document.getElementById('s').textContent=(d.status||'')+' · '+(d.ok_count||0)+'/'+(d.requested_count||0)+' · tilt '+((d.tilt&&d.tilt.bias)||'n/a');
 document.getElementById('p').textContent=JSON.stringify({tilt:d.tilt,sources_ok:d.sources_ok,remote_mcp_probes:d.remote_mcp_probes,quotes:d.quotes,errors:d.errors},null,2);
}
async function research(){
 document.getElementById('s').textContent='Loading week-ahead research MCPs';
 const r=await fetch('/api/v1/mcp/research');
 const d=await r.json();
 document.getElementById('s').textContent='configured '+JSON.stringify(d.configured||{});
 document.getElementById('p').textContent=JSON.stringify(d,null,2);
}
go('asx200');
</script></body></html>"""


@router.get("/mcp/stack")
async def mcp_stack(universe: str = "asx200") -> dict[str, Any]:
    if universe not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"universe must be one of {sorted(ALLOWED)}")
    return fetch_free_mcp_stack(universe)


@router.get("/mcp/research")
async def mcp_research() -> dict[str, Any]:
    return fetch_week_ahead_us_apac()


@router.get("/mcp", response_class=HTMLResponse, include_in_schema=False)
async def mcp_page() -> HTMLResponse:
    return HTMLResponse(PAGE)
