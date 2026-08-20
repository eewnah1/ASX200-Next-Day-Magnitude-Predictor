"""Yahoo Finance download helpers with chart API fallback.

Used by fetchers when the yfinance library is rate-limited on cloud hosts.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def _period_to_range(period: str) -> str:
    """Map yfinance-style period strings to Yahoo chart API range values."""
    mapping = {
        "5d": "5d",
        "10d": "1mo",
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "10y": "10y",
        "ytd": "ytd",
        "max": "max",
        "60d": "3mo",
        "120d": "6mo",
        "18mo": "2y",
        "24mo": "2y",
    }
    return mapping.get(period, "1y")


def _yahoo_chart_download(ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    """Direct Yahoo Finance chart v8 API fallback when yfinance is rate-limited."""
    range_ = _period_to_range(period)
    encoded = quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?range={range_}&interval={interval}&includePrePost=false"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            logger.debug("Yahoo chart API %s for %s: %s", resp.status_code, ticker, resp.text[:120])
            return pd.DataFrame()
        payload = resp.json()
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame()
        item = result[0]
        timestamps = item.get("timestamp") or []
        quote_data = ((item.get("indicators") or {}).get("quote") or [{}])[0]
        if not timestamps or not quote_data:
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "Open": quote_data.get("open") or [],
                "High": quote_data.get("high") or [],
                "Low": quote_data.get("low") or [],
                "Close": quote_data.get("close") or [],
                "Volume": quote_data.get("volume") or [],
            },
            index=pd.to_datetime(timestamps, unit="s"),
        )
        df = df.dropna(subset=["Close"], how="any")
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.tz_localize(None)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.debug("Yahoo chart fallback failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def yf_download(
    tickers: list[str],
    period: str = "5d",
    interval: str = "1d",
    retries: int = 2,
) -> pd.DataFrame:
    """Download OHLCV via yfinance with Yahoo chart API fallback on rate-limits."""
    for ticker in tickers:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                    threads=False,
                )
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        level = 1 if "Ticker" in (df.columns.names or []) else 0
                        if ticker in df.columns.get_level_values(level):
                            df = df.xs(ticker, level=level, axis=1, drop_level=True)
                        else:
                            valid = list(df.columns.get_level_values(level).unique())
                            if not valid:
                                continue
                            df = df.xs(valid[0], level=level, axis=1, drop_level=True)
                    return df
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc).lower()
                if "rate" in msg or "too many" in msg or "429" in msg:
                    sleep_s = 1.5 * (attempt + 1)
                    logger.debug(
                        "yfinance rate-limited for %s (attempt %s); sleeping %.1fs",
                        ticker,
                        attempt + 1,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                logger.debug("yfinance failed for %s: %s", ticker, exc)
                break

        chart_df = _yahoo_chart_download(ticker, period=period, interval=interval)
        if not chart_df.empty:
            logger.info("Using Yahoo chart API fallback for %s (%s rows)", ticker, len(chart_df))
            return chart_df

        if last_exc:
            logger.debug("All download paths failed for %s: %s", ticker, last_exc)

    return pd.DataFrame()
