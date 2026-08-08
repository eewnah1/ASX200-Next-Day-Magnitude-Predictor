"""Robust data fetchers with primary + fallback chains and local JSON caching.

Every fetcher is defensive: it returns partial data and flags rather than
raising.  The orchestrator (`DataFetcher`) turns raw prices into the
`RawMarketData` object the feature builder consumes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.logging_config import get_logger
from asx200_mag_predictor.scoring.features import RawMarketData
from asx200_mag_predictor.timezone import now_sydney

logger = get_logger(__name__)

ASX_CASH_TICKERS = ["^AXJO"]
SPI_FUTURES_TICKERS = ["AP=F", "^AP", "SPI1.AX", "^AXJO"]  # last is cash proxy
A_VIX_TICKERS = ["^A-VIX", "^VIX"]
IRON_ORE_TICKERS = ["FE=F", "TIO=F", "MT=F"]
GOLD_TICKERS = ["GC=F"]
OIL_TICKERS = ["CL=F"]
COPPER_TICKERS = ["HG=F"]
AUDUSD_TICKERS = ["AUDUSD=X"]
SP500_TICKERS = ["^GSPC", "ES=F"]
NASDAQ_TICKERS = ["^IXIC", "NQ=F"]
DOW_TICKERS = ["^DJI", "YM=F"]
US10Y_TICKERS = ["^TNX", "^FVX"]
VIX_TICKERS = ["^VIX"]


@dataclass
class DataSnapshot:
    """A serialisable raw snapshot for persistence and debugging."""

    timestamp: str
    source: str
    data: dict[str, Any]


def _yf_download(
    tickers: list[str],
    period: str = "5d",
    interval: str = "1d",
    raise_on_empty: bool = False,
) -> pd.DataFrame:
    """Download from yfinance, trying tickers in order until one returns data."""
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                # If multiple tickers, yfinance returns MultiIndex columns; flatten
                if isinstance(df.columns, pd.MultiIndex):
                    valid = [t for t in tickers if t in df.columns.get_level_values(1)]
                    if not valid:
                        continue
                    df = df.xs(valid[0], level="Ticker", axis=1, drop_level=True)
                return df
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance failed for %s: %s", ticker, exc)
    if raise_on_empty:
        raise RuntimeError(f"No yfinance data for {tickers}")
    return pd.DataFrame()


def _extract_series(df: pd.DataFrame) -> dict[str, list[float]] | None:
    """Extract close/high/low/volume lists from a yfinance DataFrame."""
    if df.empty:
        return None
    try:
        close = df["Close"].dropna().tolist()
        high = df["High"].dropna().tolist()
        low = df["Low"].dropna().tolist()
        volume = df["Volume"].dropna().tolist() if "Volume" in df.columns else []
        return {"close": close, "high": high, "low": low, "volume": volume}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Extract series failed: %s", exc)
        return None


def _latest_change_pct(df: pd.DataFrame) -> float | None:
    """Return the latest close vs previous close percent change."""
    if df.empty or "Close" not in df.columns:
        return None
    try:
        close = df["Close"].dropna()
        if len(close) < 2:
            return None
        return (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100.0
    except Exception as exc:  # noqa: BLE001
        logger.debug("latest_change_pct failed: %s", exc)
        return None


class YFinanceClient:
    """Thin wrapper around yfinance with fallback symbols."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def asx_cash(self) -> dict[str, Any]:
        df = _yf_download(ASX_CASH_TICKERS, period="30d", interval="1d")
        series = _extract_series(df)
        if not series:
            return {}
        return {"ticker": ASX_CASH_TICKERS[0], "series": series}

    def spi_futures(self) -> dict[str, Any]:
        df = _yf_download(SPI_FUTURES_TICKERS, period="5d", interval="1d")
        series = _extract_series(df)
        if not series:
            return {}
        return {"ticker": SPI_FUTURES_TICKERS[0], "series": series}

    def a_vix(self) -> dict[str, Any]:
        df = _yf_download(A_VIX_TICKERS, period="10d", interval="1d")
        series = _extract_series(df)
        if not series:
            return {}
        close = series.get("close", [])
        return {"ticker": A_VIX_TICKERS[0], "close": close[-1] if close else None, "series": series}

    def commodities(self) -> dict[str, Any]:
        result: dict[str, Any] = {"sources": {}}

        def _fetch(name: str, tickers: list[str]) -> float | None:
            df = _yf_download(tickers, period="5d", interval="1d")
            chg = _latest_change_pct(df)
            if chg is not None:
                result["sources"][name] = tickers[0]
            return chg

        result["iron_ore_change_pct"] = _fetch("iron_ore", IRON_ORE_TICKERS)
        result["gold_change_pct"] = _fetch("gold", GOLD_TICKERS)
        result["oil_change_pct"] = _fetch("oil", OIL_TICKERS)
        result["copper_change_pct"] = _fetch("copper", COPPER_TICKERS)
        return result

    def fx(self) -> dict[str, Any]:
        df = _yf_download(AUDUSD_TICKERS, period="5d", interval="1d")
        chg = _latest_change_pct(df)
        return {"aud_usd_change_pct": chg, "ticker": AUDUSD_TICKERS[0]}

    def us_assets(self) -> dict[str, Any]:
        result: dict[str, Any] = {"sources": {}}

        def _fetch(name: str, tickers: list[str]) -> float | None:
            df = _yf_download(tickers, period="5d", interval="1d")
            chg = _latest_change_pct(df)
            if chg is not None:
                result["sources"][name] = tickers[0]
            return chg

        result["us_futures_change_pct"] = _fetch("us_futures", SP500_TICKERS)
        result["sp500_change_pct"] = _fetch("sp500", SP500_TICKERS)
        result["nasdaq_change_pct"] = _fetch("nasdaq", NASDAQ_TICKERS)
        result["dow_change_pct"] = _fetch("dow", DOW_TICKERS)

        # 10y yield: convert % yield change to basis points (1 pct = 100 bps)
        df10y = _yf_download(US10Y_TICKERS, period="5d", interval="1d")
        chg10y = _latest_change_pct(df10y)
        if chg10y is not None:
            result["us_10y_change_bps"] = chg10y * 100.0
            result["sources"]["us_10y"] = US10Y_TICKERS[0]
        else:
            result["us_10y_change_bps"] = None

        result["vix_change_pct"] = _fetch("vix", VIX_TICKERS)
        return result

    def intraday_asx(self) -> dict[str, Any]:
        """Try to grab today's 5m bars; fall back to daily open/close."""
        df = _yf_download(["^AXJO"], period="1d", interval="5m")
        if df.empty:
            # Fall back to daily open/close
            df = _yf_download(["^AXJO"], period="5d", interval="1d")
            if df.empty or "Close" not in df.columns:
                return {}
            close = df["Close"].dropna()
            open_price = df["Open"].dropna()
            volume = df["Volume"].dropna()
            if len(close) < 1:
                return {}
            return {
                "asx_open_to_now_return_pct": (close.iloc[-1] - open_price.iloc[0])
                / open_price.iloc[0]
                * 100.0,
                "current_volume_vs_20d_avg": None,
                "current_range_vs_atr": None,
                "today_volume": int(volume.iloc[-1]) if len(volume) else None,
            }

        # Use 5m bars for current session
        today = now_sydney().date()
        df.index = df.index.tz_convert("Australia/Sydney")
        today_bars = df[df.index.date == today]
        if today_bars.empty:
            today_bars = df
        open_price = float(today_bars["Open"].iloc[0])
        last_close = float(today_bars["Close"].iloc[-1])
        high = float(today_bars["High"].max())
        low = float(today_bars["Low"].min())
        volume = int(today_bars["Volume"].sum())

        # 20-day average volume using daily history
        daily = _yf_download(["^AXJO"], period="40d", interval="1d")
        avg_volume = None
        if not daily.empty and "Volume" in daily.columns:
            avg_volume = daily["Volume"].tail(20).mean()
            avg_volume = float(avg_volume) if avg_volume and avg_volume > 0 else None

        # ATR from daily history for range comparison
        atr = None
        if not daily.empty and len(daily) >= 5:
            try:
                atr = (
                    (
                        daily["High"].tail(5).max()
                        - daily["Low"].tail(5).min()
                    )
                    / daily["Close"].iloc[-1]
                    * 100.0
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("ATR calc failed: %s", exc)

        current_range_pct = (high - low) / open_price * 100.0
        range_vs_atr = None
        if atr:
            range_vs_atr = current_range_pct / atr

        volume_ratio = None
        if volume and avg_volume and avg_volume > 0:
            volume_ratio = volume / avg_volume

        return {
            "asx_open_to_now_return_pct": (last_close - open_price) / open_price * 100.0,
            "current_volume_vs_20d_avg": volume_ratio,
            "current_range_vs_atr": range_vs_atr,
            "today_volume": volume,
            "today_high": high,
            "today_low": low,
        }


class NewsAPICalendar:
    """Economic calendar proxy via NewsAPI headlines."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def fetch(self) -> dict[str, Any]:
        key = self.settings.newsapi_api_key
        if not key:
            logger.info("NewsAPI key missing; calendar fetch skipped.")
            return {}

        keywords = [
            "RBA",
            "Federal Reserve",
            "Fed",
            "ECB",
            "BOE",
            "PBOC",
            "CPI",
            "PPI",
            "GDP",
            "NFP",
            "nonfarm payrolls",
            "PMI",
            "unemployment rate",
            "interest rate",
            "tariff",
            "Australia",
            "China",
            "United States",
        ]
        query = " OR ".join(f'"{k}"' for k in keywords)
        today = now_sydney().strftime("%Y-%m-%d")
        # Look for articles published today; NewsAPI free tier is 24h rolling.
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={requests.utils.quote(query)}"
            f"&from={today}&to={today}"
            "&language=en&sortBy=relevancy&pageSize=50"
            f"&apiKey={key}"
        )
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            return self._score(articles)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsAPI calendar failed: %s", exc)
            return {}

    def _score(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        high_impact_keywords = {
            "rate decision",
            "interest rate",
            "RBA",
            "Federal Reserve",
            "Fed",
            "CPI",
            "PPI",
            "GDP",
            "NFP",
            "nonfarm",
            "PMI",
            "tariff",
            "inflation",
            "recession",
        }
        count = 0
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            if any(k.lower() in text for k in high_impact_keywords):
                count += 1
        # Rough mapping: 0-1 articles -> low, 2-4 -> moderate, 5+ -> high
        return {
            "source": "newsapi",
            "article_count": len(articles),
            "high_impact_24h": min(count, 6),
            "high_impact_48h": min(len(articles) // 10, 3),
        }


class MarketAuxCalendar:
    """Lightweight fallback calendar proxy using MarketAux news."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def fetch(self) -> dict[str, Any]:
        key = self.settings.marketaux_api_key
        if not key:
            return {}
        url = f"https://api.marketaux.com/v1/news/all?api_token={key}&language=en&limit=50"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles = data.get("data", [])
            return {"source": "marketaux", "high_impact_24h": min(len(articles) // 5, 5), "high_impact_48h": 0}
        except Exception as exc:  # noqa: BLE001
            logger.debug("MarketAux calendar failed: %s", exc)
            return {}


class DataFetcher:
    """Orchestrate all fetchers and cache raw snapshots to disk."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.yf = YFinanceClient(settings)
        self.news_calendar = NewsAPICalendar(settings)
        self.marketaux_calendar = MarketAuxCalendar(settings)

    def fetch_all(self) -> RawMarketData:
        """Fetch every data source, cache the snapshot, and return raw data."""
        asx_cash = self.yf.asx_cash()
        spi_futures = self.yf.spi_futures()
        a_vix = self.yf.a_vix()
        commodities = self.yf.commodities()
        fx = self.yf.fx()
        us_assets = self.yf.us_assets()
        volume = self.yf.intraday_asx()

        calendar = self.news_calendar.fetch()
        if not calendar:
            calendar = self.marketaux_calendar.fetch()

        raw = RawMarketData(
            asx_cash=asx_cash,
            spi_futures=spi_futures,
            a_vix=a_vix,
            commodities=commodities,
            fx=fx,
            us_assets=us_assets,
            calendar=calendar,
            volume=volume,
        )
        self._cache_snapshot(raw)
        return raw

    def _cache_snapshot(self, raw: RawMarketData) -> None:
        try:
            snapshot = DataSnapshot(
                timestamp=now_sydney().isoformat(),
                source="data_fetcher",
                data={k: asdict(raw)[k] if k else {} for k in asdict(raw).keys()},
            )
            snapshot_dir = self.settings.data_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_dir / f"{now_sydney().strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(snapshot.data, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache snapshot failed: %s", exc)
