"""Entrypoint for deployed FastAPI app."""

import os

# Make the bundled TradingView MCP client fail fast in serverless/cloud hosts
# where repeated retries and cooldown sleeps would block the event loop.
os.environ.setdefault("TRADINGVIEW_MCP_FAILURE_COOLDOWN_S", "0")
os.environ.setdefault("TRADINGVIEW_MCP_RETRY_DELAYS", "0.0")
os.environ.setdefault("TRADINGVIEW_MCP_SOCKET_TIMEOUT", "10")
os.environ.setdefault("TRADINGVIEW_MCP_MAX_INFLIGHT", "1")

from asx200_mag_predictor.api.main import app

__all__ = ["app"]
