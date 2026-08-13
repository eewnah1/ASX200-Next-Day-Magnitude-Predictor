"""TradingView MCP data adapters.

Wraps two TradingView MCP services:
- atilaahmettaner/tradingview-mcp (PyPI: tradingview-mcp-server)
- fiale-plus/tradingview-mcp-server (npm: tradingview-mcp-server)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from asx200_mag_predictor.logging_config import get_logger

logger = get_logger(__name__)

_TA_TIMEOUT = 30.0
_FIALE_TIMEOUT = 90.0


def _tradingview_cli_path() -> str:
    """Locate the fiale-plus tradingview-cli binary."""
    for base in [Path.home() / ".npm-global" / "bin", Path("/usr") / "local" / "bin"]:
        candidate = base / "tradingview-cli"
        if candidate.exists():
            return str(candidate)
    found = shutil.which("tradingview-cli")
    if found:
        return found
    return "tradingview-cli"


def _run_fiale_cli(*args: str) -> dict[str, Any]:
    """Run fiale-plus tradingview-cli and return parsed JSON."""
    cmd = [_tradingview_cli_path(), *args, "-f", "json"]
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".npm-global" / "bin") + ":" + env.get("PATH", "")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_FIALE_TIMEOUT,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            return {"error": f"fiale-plus CLI failed: {err}"}
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        logger.exception("Could not parse fiale-plus output")
        return {"error": f"fiale-plus output parse error: {exc}"}
    except FileNotFoundError as exc:
        return {"error": f"tradingview-cli not found: {exc}"}
    except subprocess.TimeoutExpired:
        return {"error": "fiale-plus CLI timed out"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("fiale-plus CLI error")
        return {"error": str(exc)}


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


def atila_symbol_analysis(symbol: str, exchange: str) -> dict[str, Any]:
    """Multi-agent technical/sentiment/risk analysis for a symbol."""
    try:
        from tradingview_mcp.core.services.multi_agent_service import (
            run_multi_agent_analysis,
        )

        result = run_multi_agent_analysis(symbol, exchange, "1D")
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
    """Run a TradingView screener preset via fiale-plus/tradingview-mcp-server."""
    return _run_fiale_cli("screen", asset_type, "--preset", preset, "--limit", str(limit))


def fiale_lookup(*symbols: str) -> dict[str, Any]:
    """Look up one or more TradingView symbols via fiale-plus."""
    if not symbols:
        return {"error": "No symbols provided"}
    return _run_fiale_cli("lookup", *symbols)


def fiale_presets() -> dict[str, Any]:
    """List available fiale-plus presets."""
    return _run_fiale_cli("presets")


def get_asx200_insights() -> dict[str, Any]:
    """Combined TradingView MCP insights for the ASX 200 predictor."""
    return {
        "market_snapshot": atila_market_snapshot(),
        "asx200_analysis": atila_symbol_analysis("ASX:XJO", "asx"),
        "asx200_price": atila_price("^AXJO"),
        "quality_screener": fiale_screen("stocks", "quality_stocks", 10),
    }
