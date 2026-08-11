"""Robust data fetchers with primary + fallback chains and local JSON caching.

Every fetcher is defensive: it returns partial data and rich status metadata
rather than raising.  The orchestrator (`DataFetcher`) turns raw prices into
the `RawMarketData` object the feature builder consumes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
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
IRON_ORE_TICKERS = ["FE=F", "TIO=F", "MT=F", "BHP.AX", "VALE"]
GOLD_TICKERS = ["GC=F"]
SILVER_TICKERS = ["SI=F"]
OIL_TICKERS = ["CL=F"]
COPPER_TICKERS = ["HG=F"]
AUDUSD_TICKERS = ["AUDUSD=X"]
SP500_TICKERS = ["^GSPC", "ES=F"]
NASDAQ_TICKERS = ["^IXIC", "NQ=F"]
DOW_TICKERS = ["^DJI", "YM=F"]
US10Y_TICKERS = ["^TNX", "^FVX"]
VIX_TICKERS = ["^VIX"]

FINANCIALS_BANKS_TICKERS = ["CBA.AX", "NAB.AX", "WBC.AX", "ANZ.AX"]
MATERIALS_MINERS_TICKERS = ["BHP.AX", "RIO.AX", "FMG.AX"]
HOUSING_PROXIES_TICKERS = ["REA.AX", "GMG.AX", "SCG.AX", "LLC.AX"]
CHINA_STEEL_PROPERTY_TICKERS = ["BHP.AX", "RIO.AX", "FMG.AX", "TIO=F", "HG=F"]
HEAVYWEIGHT_TICKERS = ["CBA.AX", "BHP.AX"]


@dataclass
class FetchResult:
    """Result of a single external data fetch."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | stale | failed
    last_success_at: str | None = None
    value: str | None = None
    error: str | None = None
    ticker: str | None = None


@dataclass
class DataSnapshot:
    """A serialisable raw snapshot for persistence and debugging."""

    timestamp: str
    source: str
    data: dict[str, Any]


def _aest_iso() -> str:
    return now_sydney().isoformat()


def _data_timestamp(df: pd.DataFrame) -> datetime | None:
    """Return the timestamp of the last row in a DataFrame, if known."""
    if df.empty:
        return None
    try:
        idx = df.index[-1]
        if isinstance(idx, pd.Timestamp):
            return idx.to_pydatetime()
        return None
    except Exception:  # noqa: BLE001
        return None


def _data_timestamp_str(df: pd.DataFrame) -> str:
    ts = _data_timestamp(df)
    return ts.isoformat() if ts else _aest_iso()


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _yf_download(
    tickers: list[str],
    period: str = "5d",
    interval: str = "1d",
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
                if isinstance(df.columns, pd.MultiIndex):
                    valid = [t for t in tickers if t in df.columns.get_level_values(1)]
                    if not valid:
                        continue
                    df = df.xs(valid[0], level="Ticker", axis=1, drop_level=True)
                return df
        except Exception as exc:  # noqa: BLE001
            logger.debug("yfinance failed for %s: %s", ticker, exc)
    return pd.DataFrame()


def _extract_series(df: pd.DataFrame) -> dict[str, list[float]] | None:
    """Extract open/high/low/close/volume lists from a yfinance DataFrame."""
    if df.empty:
        return None
    try:
        open_ = df["Open"].dropna().tolist() if "Open" in df.columns else []
        close = df["Close"].dropna().tolist()
        high = df["High"].dropna().tolist()
        low = df["Low"].dropna().tolist()
        volume = df["Volume"].dropna().tolist() if "Volume" in df.columns else []
        if not close:
            return None
        return {"open": open_, "close": close, "high": high, "low": low, "volume": volume}
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


def _pct_change_n(df: pd.DataFrame, n: int) -> float | None:
    """Return percent change between last close and close `n` bars back."""
    if df.empty or "Close" not in df.columns:
        return None
    try:
        close = df["Close"].dropna()
        if len(close) < n + 1:
            return None
        return (close.iloc[-1] - close.iloc[-(n + 1)]) / close.iloc[-(n + 1)] * 100.0
    except Exception as exc:  # noqa: BLE001
        logger.debug("pct_change_n failed: %s", exc)
        return None


def _basket_avg_change(
    tickers: list[str], days: int, period: str = "20d"
) -> tuple[float | None, dict[str, float | None], pd.DataFrame | None]:
    """Compute average `days`-bar percent change for a basket of tickers.

    Returns (average_change, per_ticker_changes, last_valid_df).
    """
    changes: dict[str, float | None] = {}
    latest_ts: datetime | None = None
    for ticker in tickers:
        df = _yf_download([ticker], period=period, interval="1d")
        if not df.empty:
            if days == 1:
                chg = _latest_change_pct(df)
            else:
                chg = _pct_change_n(df, days)
            changes[ticker] = chg
            ts = _data_timestamp(df)
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts
        else:
            changes[ticker] = None
    valid = [v for v in changes.values() if v is not None]
    if not valid:
        return None, changes, None
    avg = sum(valid) / len(valid)
    # Build a minimal DataFrame just to carry a timestamp for _data_timestamp callers.
    ts_df = pd.DataFrame(index=pd.DatetimeIndex([latest_ts])) if latest_ts else None
    return avg, changes, ts_df


def _diff_or_none(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _heavyweight_news_boost() -> float:
    """Return a 0-0.5 idiosyncratic boost if major CBA/BHP news is detected."""
    # No API keys by default; return 0.0 and rely on price action.
    # Extend here with NewsAPI / MarketAux sentiment if keys are configured.
    return 0.0


def _last_price_and_date(df: pd.DataFrame) -> tuple[float | None, datetime | None]:
    """Return the last close and its index timestamp."""
    if df.empty or "Close" not in df.columns:
        return None, None
    try:
        close = df["Close"].dropna()
        if close.empty:
            return None, None
        ts = close.index[-1]
        if isinstance(ts, pd.Timestamp):
            return float(close.iloc[-1]), ts.to_pydatetime()
        return float(close.iloc[-1]), None
    except Exception as exc:  # noqa: BLE001
        logger.debug("last_price_and_date failed: %s", exc)
        return None, None


class YFinanceClient:
    """Thin wrapper around yfinance with fallback symbols and status metadata."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def asx_cash(self) -> FetchResult:
        # 120d of daily bars gives enough history for 50-day highs and 14-day RSI.
        df = _yf_download(ASX_CASH_TICKERS, period="120d", interval="1d")
        series = _extract_series(df)
        if not series:
            return FetchResult(
                name="asx_cash",
                status="failed",
                error="Could not download ASX 200 cash data",
                last_success_at=_data_timestamp_str(df),
            )
        last, ts = _last_price_and_date(df)
        return FetchResult(
            name="asx_cash",
            data={
                "ticker": ASX_CASH_TICKERS[0],
                "series": series,
                "last_price_date": ts.isoformat() if ts else None,
            },
            status="ok",
            last_success_at=ts.isoformat() if ts else _data_timestamp_str(df),
            value=f"last {last:.2f}" if last else None,
            ticker=ASX_CASH_TICKERS[0],
        )

    def spi_futures(self) -> FetchResult:
        used_ticker = None
        for ticker in SPI_FUTURES_TICKERS:
            df = _yf_download([ticker], period="10d", interval="1d")
            if not df.empty:
                used_ticker = ticker
                break
        series = _extract_series(df) if not df.empty else None
        if not series:
            return FetchResult(
                name="spi_futures",
                status="failed",
                error="Could not download SPI 200 futures",
                last_success_at=_data_timestamp_str(df),
            )
        # Cash/futures proxies are all valid for SPI basis/momentum; avoid stale.
        status = "ok" if used_ticker else "failed"
        last, ts = _last_price_and_date(df)
        return FetchResult(
            name="spi_futures",
            data={
                "ticker": used_ticker,
                "series": series,
                "cash_proxy": used_ticker == "^AXJO",
                "last_price_date": ts.isoformat() if ts else None,
            },
            status=status,
            last_success_at=ts.isoformat() if ts else _data_timestamp_str(df),
            value=f"last {last:.2f}" if last else None,
            ticker=used_ticker,
        )

    def a_vix(self) -> FetchResult:
        used_ticker = None
        for ticker in A_VIX_TICKERS:
            df = _yf_download([ticker], period="10d", interval="1d")
            if not df.empty:
                used_ticker = ticker
                break
        series = _extract_series(df) if not df.empty else None
        if not series:
            return FetchResult(
                name="a_vix",
                status="failed",
                error="Could not download A-VIX / VIX",
                last_success_at=_data_timestamp_str(df),
            )
        # ^VIX is an acceptable global vol proxy when ^A-VIX is unavailable.
        status = "ok" if used_ticker else "failed"
        close = series.get("close", [])
        last = close[-1] if close else None
        ts = _data_timestamp(df)
        return FetchResult(
            name="a_vix",
            data={
                "ticker": used_ticker,
                "close": last,
                "series": series,
                "last_price_date": ts.isoformat() if ts else None,
            },
            status=status,
            last_success_at=ts.isoformat() if ts else _data_timestamp_str(df),
            value=f"{last:.2f}" if last else None,
            ticker=used_ticker,
        )

    def commodities(self) -> FetchResult:
        result: dict[str, Any] = {"sources": {}}
        errors: list[str] = []
        status = "ok"
        latest_ts: datetime | None = None

        def _fetch(
            name: str, tickers: list[str]
        ) -> tuple[float | None, str | None, str, pd.DataFrame | None]:
            for ticker in tickers:
                df = _yf_download([ticker], period="10d", interval="1d")
                chg = _latest_change_pct(df)
                if chg is not None:
                    # Any ticker in the fallback chain is acceptable.
                    return chg, ticker, "ok", df
            errors.append(f"{name}: all tickers failed")
            return None, None, "failed", None

        for name, tickers, key in [
            ("iron_ore", IRON_ORE_TICKERS, "iron_ore_change_pct"),
            ("gold", GOLD_TICKERS, "gold_change_pct"),
            ("silver", SILVER_TICKERS, "silver_change_pct"),
            ("oil", OIL_TICKERS, "oil_change_pct"),
            ("copper", COPPER_TICKERS, "copper_change_pct"),
        ]:
            chg, used, st, df = _fetch(name, tickers)
            result[key] = chg
            if used:
                result["sources"][name] = used
            if df is not None:
                ts = _data_timestamp(df)
                if ts and (latest_ts is None or ts > latest_ts):
                    latest_ts = ts
            if st != "ok":
                status = st if status == "ok" else status

        if result.get("iron_ore_change_pct") is None:
            status = "failed"

        values = [
            f"{k}={_fmt_pct(result.get(k))}"
            for k in [
                "iron_ore_change_pct",
                "gold_change_pct",
                "silver_change_pct",
                "oil_change_pct",
                "copper_change_pct",
            ]
            if result.get(k) is not None
        ]
        return FetchResult(
            name="commodities",
            data=result,
            status=status,
            last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            value=", ".join(values) if values else None,
            error="; ".join(errors) if errors else None,
        )

    def fx(self) -> FetchResult:
        df = _yf_download(AUDUSD_TICKERS, period="10d", interval="1d")
        chg = _latest_change_pct(df)
        ts = _data_timestamp(df)
        if chg is None:
            return FetchResult(
                name="fx",
                status="failed",
                error="Could not download AUD/USD",
                last_success_at=ts.isoformat() if ts else _aest_iso(),
            )
        return FetchResult(
            name="fx",
            data={
                "aud_usd_change_pct": chg,
                "ticker": AUDUSD_TICKERS[0],
                "last_price_date": ts.isoformat() if ts else None,
            },
            status="ok",
            last_success_at=ts.isoformat() if ts else _aest_iso(),
            value=_fmt_pct(chg),
            ticker=AUDUSD_TICKERS[0],
        )

    def us_assets(self) -> FetchResult:
        result: dict[str, Any] = {"sources": {}}
        errors: list[str] = []
        status = "ok"
        latest_ts: datetime | None = None

        def _fetch(
            name: str, tickers: list[str]
        ) -> tuple[float | None, str | None, str, pd.DataFrame | None]:
            for ticker in tickers:
                df = _yf_download([ticker], period="10d", interval="1d")
                chg = _latest_change_pct(df)
                if chg is not None:
                    # Any ticker in the fallback chain is acceptable.
                    return chg, ticker, "ok", df
            errors.append(f"{name}: all tickers failed")
            return None, None, "failed", None

        for name, tickers, key in [
            ("us_futures", SP500_TICKERS, "us_futures_change_pct"),
            ("sp500", SP500_TICKERS, "sp500_change_pct"),
            ("nasdaq", NASDAQ_TICKERS, "nasdaq_change_pct"),
            ("dow", DOW_TICKERS, "dow_change_pct"),
            ("vix", VIX_TICKERS, "vix_change_pct"),
        ]:
            chg, used, st, df = _fetch(name, tickers)
            result[key] = chg
            if used:
                result["sources"][name] = used
            if df is not None:
                ts = _data_timestamp(df)
                if ts and (latest_ts is None or ts > latest_ts):
                    latest_ts = ts
            if st != "ok":
                status = st if status == "ok" else status

        # 10y yield: convert % yield change to basis points (1 pct = 100 bps)
        df10y = _yf_download(US10Y_TICKERS, period="10d", interval="1d")
        chg10y = _latest_change_pct(df10y)
        if chg10y is not None:
            result["us_10y_change_bps"] = chg10y * 100.0
            result["sources"]["us_10y"] = US10Y_TICKERS[0]
            ts10y = _data_timestamp(df10y)
            if ts10y and (latest_ts is None or ts10y > latest_ts):
                latest_ts = ts10y
        else:
            errors.append("us_10y: all tickers failed")

        values = [
            f"{k}={_fmt_pct(result.get(k))}"
            for k in ["sp500_change_pct", "nasdaq_change_pct", "dow_change_pct", "vix_change_pct"]
            if result.get(k) is not None
        ]
        if result.get("us_10y_change_bps") is not None:
            values.append(f"10y={result['us_10y_change_bps']:+.1f}bps")
        return FetchResult(
            name="us_assets",
            data=result,
            status=status,
            last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            value=", ".join(values) if values else None,
            error="; ".join(errors) if errors else None,
        )

    def financials_vs_materials(self) -> FetchResult:
        """Financials vs Materials relative strength (1d / 3d / 5d)."""
        fin_changes: dict[str, float | None] = {}
        mat_changes: dict[str, float | None] = {}
        latest_ts: datetime | None = None
        status = "ok"

        for days in [1, 2, 3, 5]:
            fin_avg, fin_per_ticker, fin_df = _basket_avg_change(FINANCIALS_BANKS_TICKERS, days)
            mat_avg, mat_per_ticker, mat_df = _basket_avg_change(MATERIALS_MINERS_TICKERS, days)
            fin_changes[f"fin_{days}d"] = fin_avg
            mat_changes[f"mat_{days}d"] = mat_avg
            for df in [fin_df, mat_df]:
                if df is not None:
                    ts = _data_timestamp(df)
                    if ts and (latest_ts is None or ts > latest_ts):
                        latest_ts = ts
            if fin_avg is None or mat_avg is None:
                status = "stale"

        diff_1d = _diff_or_none(fin_changes.get("fin_1d"), mat_changes.get("mat_1d"))
        diff_2d = _diff_or_none(fin_changes.get("fin_2d"), mat_changes.get("mat_2d"))
        diff_3d = _diff_or_none(fin_changes.get("fin_3d"), mat_changes.get("mat_3d"))
        diff_5d = _diff_or_none(fin_changes.get("fin_5d"), mat_changes.get("mat_5d"))
        weighted = None
        if diff_1d is not None and diff_3d is not None and diff_5d is not None:
            weighted = 0.5 * diff_1d + 0.3 * diff_3d + 0.2 * diff_5d
        else:
            status = "stale"

        if all(v is None for v in [diff_1d, diff_3d, diff_5d]):
            return FetchResult(
                name="financials_vs_materials",
                status="failed",
                error="Could not fetch Financials or Materials basket data",
                last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            )

        return FetchResult(
            name="financials_vs_materials",
            data={
                "financials_1d": fin_changes.get("fin_1d"),
                "financials_2d": fin_changes.get("fin_2d"),
                "financials_3d": fin_changes.get("fin_3d"),
                "financials_5d": fin_changes.get("fin_5d"),
                "materials_1d": mat_changes.get("mat_1d"),
                "materials_2d": mat_changes.get("mat_2d"),
                "materials_3d": mat_changes.get("mat_3d"),
                "materials_5d": mat_changes.get("mat_5d"),
                "diff_1d_pct": diff_1d,
                "diff_2d_pct": diff_2d,
                "diff_3d_pct": diff_3d,
                "diff_5d_pct": diff_5d,
                "weighted_diff_pct": weighted,
            },
            status=status,
            last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            value=(
                f"fin={_fmt_pct(fin_changes.get('fin_1d'))}, "
                f"mat={_fmt_pct(mat_changes.get('mat_1d'))}, "
                f"diff={_fmt_pct(weighted)}"
            ),
        )

    def housing_credit_pulse(self) -> FetchResult:
        """Housing & credit pulse proxy from ASX real-estate / property names."""
        avg_1d, per_ticker_1d, df_1d = _basket_avg_change(HOUSING_PROXIES_TICKERS, 1)
        avg_5d, per_ticker_5d, df_5d = _basket_avg_change(HOUSING_PROXIES_TICKERS, 5)
        latest_ts = _data_timestamp(df_1d) if df_1d is not None else None
        if df_5d is not None:
            ts5 = _data_timestamp(df_5d)
            if ts5 and (latest_ts is None or ts5 > latest_ts):
                latest_ts = ts5

        if avg_1d is None and avg_5d is None:
            return FetchResult(
                name="housing_credit",
                status="failed",
                error="Could not fetch housing/credit proxy basket",
                last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            )

        # Base score 5; move with 1d and 5d proxy returns. Coefficients keep output 0-10.
        a1 = avg_1d or 0.0
        a5 = avg_5d or 0.0
        pulse = 5.0 + (a1 * 1.5) + (a5 * 0.5)
        pulse = max(0.0, min(10.0, pulse))

        return FetchResult(
            name="housing_credit",
            data={
                "pulse_score": pulse,
                "proxy_1d_pct": avg_1d,
                "proxy_5d_pct": avg_5d,
                "per_ticker_1d": per_ticker_1d,
                "per_ticker_5d": per_ticker_5d,
                "sources": HOUSING_PROXIES_TICKERS,
            },
            status="ok",
            last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            value=f"pulse={pulse:.1f}/10",
        )

    def china_steel_property(self) -> FetchResult:
        """China steel / property pulse from iron ore, copper and major miners."""
        weights = {
            "TIO=F": 0.25,  # iron ore futures
            "HG=F": 0.20,  # copper futures
            "BHP.AX": 0.20,
            "RIO.AX": 0.15,
            "FMG.AX": 0.20,
        }
        changes: dict[str, float | None] = {}
        latest_ts: datetime | None = None
        weighted_sum = 0.0
        weight_used = 0.0
        for ticker, w in weights.items():
            df = _yf_download([ticker], period="10d", interval="1d")
            chg = _latest_change_pct(df)
            changes[ticker] = chg
            if chg is not None:
                weighted_sum += w * chg
                weight_used += w
                ts = _data_timestamp(df)
                if ts and (latest_ts is None or ts > latest_ts):
                    latest_ts = ts

        if weight_used == 0:
            return FetchResult(
                name="china_pulse",
                status="failed",
                error="Could not fetch China steel/property proxies",
                last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            )

        composite = weighted_sum / weight_used
        status = "ok" if weight_used >= 0.8 else "stale"
        return FetchResult(
            name="china_pulse",
            data={
                "composite_return_pct": composite,
                "per_ticker_1d": changes,
                "coverage": weight_used,
            },
            status=status,
            last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            value=f"composite={composite:+.2f}%",
        )

    def heavyweight_idiosyncratic(self) -> FetchResult:
        """CBA + BHP 1-day return, with optional news boost."""
        changes: dict[str, float | None] = {}
        latest_ts: datetime | None = None
        for ticker in HEAVYWEIGHT_TICKERS:
            df = _yf_download([ticker], period="10d", interval="1d")
            chg = _latest_change_pct(df)
            changes[ticker] = chg
            if chg is not None:
                ts = _data_timestamp(df)
                if ts and (latest_ts is None or ts > latest_ts):
                    latest_ts = ts

        if any(v is None for v in changes.values()):
            return FetchResult(
                name="heavyweight_idio",
                status="stale",
                data=changes,
                error="Could not fetch both CBA and BHP",
                last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            )

        cba = changes["CBA.AX"] or 0.0
        bhp = changes["BHP.AX"] or 0.0
        weighted = 0.55 * cba + 0.45 * bhp
        news_boost = _heavyweight_news_boost()
        return FetchResult(
            name="heavyweight_idio",
            data={
                "cba_change_pct": cba,
                "bhp_change_pct": bhp,
                "weighted_change_pct": weighted,
                "news_boost": news_boost,
            },
            status="ok",
            last_success_at=latest_ts.isoformat() if latest_ts else _aest_iso(),
            value=f"CBA={cba:+.2f}%, BHP={bhp:+.2f}% (boost {news_boost:.0%})",
        )

    def intraday_asx(self) -> FetchResult:
        """Try to grab the latest available 5m bars, fall back to daily close."""
        # 5m data is only retained for the last few trading days by Yahoo.
        df5m = _yf_download(["^AXJO"], period="5d", interval="5m")
        today = now_sydney().date()
        if not df5m.empty:
            try:
                df5m.index = df5m.index.tz_convert("Australia/Sydney")
                today_bars = df5m[df5m.index.date == today]
                if not today_bars.empty and len(today_bars) >= 2:
                    open_price = float(today_bars["Open"].iloc[0])
                    last_close = float(today_bars["Close"].iloc[-1])
                    high = float(today_bars["High"].max())
                    low = float(today_bars["Low"].min())
                    volume = int(today_bars["Volume"].sum())

                    # 20-day average volume using daily history
                    daily = _yf_download(["^AXJO"], period="40d", interval="1d")
                    avg_volume = None
                    if not daily.empty and "Volume" in daily.columns:
                        avg = daily["Volume"].tail(20).mean()
                        avg_volume = float(avg) if avg and avg > 0 else None

                    # ATR from daily history for range comparison
                    atr_pct = None
                    if not daily.empty and len(daily) >= 5:
                        try:
                            atr = daily["High"].tail(5).max() - daily["Low"].tail(5).min()
                            atr_pct = (
                                atr / daily["Close"].iloc[-1] * 100.0
                                if daily["Close"].iloc[-1]
                                else None
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("ATR calc failed: %s", exc)

                    current_range_pct = (high - low) / open_price * 100.0
                    range_vs_atr = None
                    if atr_pct:
                        range_vs_atr = current_range_pct / atr_pct

                    volume_ratio = None
                    if volume and avg_volume and avg_volume > 0:
                        volume_ratio = volume / avg_volume

                    session_return_pct = (last_close - open_price) / open_price * 100.0
                    return FetchResult(
                        name="volume",
                        data={
                            "asx_open_to_now_return_pct": session_return_pct,
                            "current_volume_vs_20d_avg": volume_ratio,
                            "current_range_vs_atr": range_vs_atr,
                            "today_volume": volume,
                            "today_high": high,
                            "today_low": low,
                            "session_date": str(today),
                        },
                        status="ok",
                        last_success_at=today_bars.index[-1].isoformat(),
                        value=f"session return {session_return_pct:+.2f}%",
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Intraday 5m processing failed: %s", exc)

        # Fall back to daily open/close of the most recent trading day
        df = _yf_download(["^AXJO"], period="10d", interval="1d")
        if df.empty or "Close" not in df.columns or "Open" not in df.columns:
            return FetchResult(
                name="volume",
                status="failed",
                error="Could not download ASX session data",
                last_success_at=_aest_iso(),
            )
        try:
            close = df["Close"].dropna()
            open_price = df["Open"].dropna()
            if close.empty or open_price.empty:
                raise ValueError("empty close/open")
            last_close = float(close.iloc[-1])
            first_open = float(open_price.iloc[0])
            session_return_pct = (last_close - first_open) / first_open * 100.0
            session_ts = close.index[-1]
            session_date = session_ts
            if isinstance(session_ts, pd.Timestamp):
                session_date = session_ts.to_pydatetime().date()
            return FetchResult(
                name="volume",
                data={
                    "asx_open_to_now_return_pct": session_return_pct,
                    "current_volume_vs_20d_avg": None,
                    "current_range_vs_atr": None,
                    "today_volume": None,
                    "session_date": str(session_date),
                    "fallback": "daily",
                },
                status="stale",
                last_success_at=session_ts.isoformat()
                if isinstance(session_ts, pd.Timestamp)
                else _aest_iso(),
                value=f"last session return {session_return_pct:+.2f}% (daily fallback)",
            )
        except Exception as exc:  # noqa: BLE001
            return FetchResult(
                name="volume",
                status="failed",
                error=f"Daily fallback failed: {exc}",
                last_success_at=_aest_iso(),
            )


class NewsAPICalendar:
    """Economic calendar proxy via NewsAPI headlines."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def fetch(self) -> FetchResult:
        key = self.settings.newsapi_api_key
        if not key:
            return FetchResult(
                name="calendar",
                status="failed",
                error="NEWSAPI_API_KEY not configured",
                last_success_at=_aest_iso(),
            )

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
            scored = self._score(articles)
            return FetchResult(
                name="calendar",
                data=scored,
                status="ok",
                last_success_at=_aest_iso(),
                value=f"{scored.get('high_impact_24h', 0)} high-impact events (24h)",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsAPI calendar failed: %s", exc)
            return FetchResult(
                name="calendar",
                status="failed",
                error=str(exc),
                last_success_at=_aest_iso(),
            )

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

    def fetch(self) -> FetchResult:
        key = self.settings.marketaux_api_key
        if not key:
            return FetchResult(
                name="calendar",
                status="failed",
                error="MARKETAUX_API_KEY not configured",
                last_success_at=_aest_iso(),
            )
        url = f"https://api.marketaux.com/v1/news/all?api_token={key}&language=en&limit=50"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles = data.get("data", [])
            return FetchResult(
                name="calendar",
                data={
                    "source": "marketaux",
                    "high_impact_24h": min(len(articles) // 5, 5),
                    "high_impact_48h": 0,
                },
                status="ok",
                last_success_at=_aest_iso(),
                value=f"{min(len(articles) // 5, 5)} high-impact events (24h)",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("MarketAux calendar failed: %s", exc)
            return FetchResult(
                name="calendar",
                status="failed",
                error=str(exc),
                last_success_at=_aest_iso(),
            )


class ForexFactoryCalendar:
    """Free economic calendar from Forex Factory (fair-economy media)."""

    URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    FOCUS_COUNTRIES = {"AUD", "USD", "CNY", "All"}
    CACHE_TTL_SECONDS = 6 * 3600

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def fetch(self) -> FetchResult:
        cache = self._load_cache()
        if cache and self._cache_age(cache) < self.CACHE_TTL_SECONDS:
            data = cache.get("data", {})
            return FetchResult(
                name="calendar",
                data=data,
                status="ok",
                last_success_at=cache.get("cached_at", _aest_iso()),
                value=f"{data.get('high_impact_24h', 0)} high-impact events (24h) (cached)",
            )
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(self.URL, headers=headers, timeout=15)
            response.raise_for_status()
            events = response.json()
            parsed = self._parse(events)
            self._save_cache(parsed)
            return FetchResult(
                name="calendar",
                data=parsed,
                status="ok",
                last_success_at=_aest_iso(),
                value=f"{parsed.get('high_impact_24h', 0)} high-impact events (24h)",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("ForexFactory calendar failed: %s", exc)
            if cache:
                data = cache.get("data", {})
                return FetchResult(
                    name="calendar",
                    data=data,
                    status="stale",
                    last_success_at=cache.get("cached_at", _aest_iso()),
                    value=(
                        f"{data.get('high_impact_24h', 0)} high-impact events (24h) (stale cache)"
                    ),
                    error=f"Live fetch failed, using cached calendar: {exc}",
                )
            return FetchResult(
                name="calendar",
                status="failed",
                error=str(exc),
                last_success_at=_aest_iso(),
            )

    def _cache_path(self) -> Any:
        return self.settings.data_dir / "ff_calendar_cache.json"

    def _load_cache(self) -> dict[str, Any] | None:
        try:
            path = self._cache_path()
            if not path.exists():
                return None
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return None

    def _cache_age(self, cache: dict[str, Any]) -> float:
        try:
            cached_at = datetime.fromisoformat(cache.get("cached_at", "1970-01-01T00:00:00"))
        except Exception:  # noqa: BLE001
            return float("inf")
        return (now_sydney() - cached_at).total_seconds()

    def _save_cache(self, data: dict[str, Any]) -> None:
        try:
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path().write_text(
                json.dumps({"cached_at": _aest_iso(), "data": data}, indent=2)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Calendar cache write failed: %s", exc)

    def _parse(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        now = now_sydney()
        h24 = now + timedelta(hours=24)
        h48 = now + timedelta(hours=48)
        high_24h: list[dict[str, Any]] = []
        high_48h: list[dict[str, Any]] = []
        for event in events:
            country = event.get("country", "")
            impact = event.get("impact", "")
            if country not in self.FOCUS_COUNTRIES or impact != "High":
                continue
            try:
                dt = datetime.fromisoformat(event.get("date", ""))
            except Exception:  # noqa: BLE001
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=now.tzinfo)
            ev = {
                "title": event.get("title", ""),
                "country": country,
                "time": dt.isoformat(),
                "forecast": event.get("forecast", ""),
                "previous": event.get("previous", ""),
            }
            if dt >= now and dt <= h48:
                high_48h.append(ev)
                if dt <= h24:
                    high_24h.append(ev)
        return {
            "source": "forexfactory",
            "events_next_48h": high_48h[:20],
            "high_impact_24h": len(high_24h),
            "high_impact_48h": len(high_48h),
        }


class DataFetcher:
    """Orchestrate all fetchers and cache raw snapshots to disk."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.yf = YFinanceClient(settings)
        self.ff_calendar = ForexFactoryCalendar()
        self.news_calendar = NewsAPICalendar(settings)
        self.marketaux_calendar = MarketAuxCalendar(settings)

    def fetch_all(self) -> RawMarketData:
        """Fetch every data source, cache the snapshot, and return raw data."""
        results: dict[str, FetchResult] = {}
        errors: list[str] = []

        results["asx_cash"] = self.yf.asx_cash()
        results["spi_futures"] = self.yf.spi_futures()
        results["a_vix"] = self.yf.a_vix()
        results["commodities"] = self.yf.commodities()
        results["fx"] = self.yf.fx()
        results["us_assets"] = self.yf.us_assets()
        results["financials_vs_materials"] = self.yf.financials_vs_materials()
        results["housing_credit"] = self.yf.housing_credit_pulse()
        results["china_pulse"] = self.yf.china_steel_property()
        results["heavyweight_idio"] = self.yf.heavyweight_idiosyncratic()
        results["volume"] = self.yf.intraday_asx()

        cal = self.ff_calendar.fetch()
        if cal.status != "ok":
            errors.append(f"ForexFactory: {cal.error}")
            cal = self.news_calendar.fetch()
            if cal.status != "ok":
                errors.append(f"NewsAPI: {cal.error}")
                cal = self.marketaux_calendar.fetch()
                if cal.status != "ok":
                    errors.append(f"MarketAux: {cal.error}")
        results["calendar"] = cal

        raw = RawMarketData(
            asx_cash=results["asx_cash"].data,
            spi_futures=results["spi_futures"].data,
            a_vix=results["a_vix"].data,
            commodities=results["commodities"].data,
            fx=results["fx"].data,
            us_assets=results["us_assets"].data,
            financials_vs_materials=results["financials_vs_materials"].data,
            housing_credit=results["housing_credit"].data,
            china_pulse=results["china_pulse"].data,
            heavyweight_idio=results["heavyweight_idio"].data,
            calendar=results["calendar"].data,
            volume=results["volume"].data,
            source_status=[
                {
                    "name": r.name,
                    "status": r.status,
                    "last_success_at": r.last_success_at,
                    "value": r.value,
                    "error": r.error,
                }
                for r in results.values()
            ],
            errors=errors,
        )

        # On weekends or market holidays, yfinance may still return the last
        # trading day's daily bar; mark that as a successful "last available" fetch.
        if not self._has_live_data(raw) and all(r.status != "failed" for r in results.values()):
            for r in results.values():
                if r.status == "ok":
                    r.value = (r.value or "") + " (last available)"

        self._cache_snapshot(raw)
        return raw

    def _has_live_data(self, raw: RawMarketData) -> bool:
        """Heuristic: do we have a current ASX session?"""
        try:
            vol = raw.volume or {}
            return (
                vol.get("session_date") == str(now_sydney().date())
                or vol.get("today_volume") is not None
            )
        except Exception:  # noqa: BLE001
            return False

    def _cache_snapshot(self, raw: RawMarketData) -> None:
        try:
            snapshot = DataSnapshot(
                timestamp=_aest_iso(),
                source="data_fetcher",
                data=asdict(raw),
            )
            snapshot_dir = self.settings.data_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_dir / f"{now_sydney().strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(snapshot.data, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache snapshot failed: %s", exc)

    def load_cached_snapshot(self) -> RawMarketData | None:
        """Load the most recent cached snapshot if available."""
        try:
            snapshot_dir = self.settings.data_dir / "snapshots"
            files = sorted(snapshot_dir.glob("*.json"), reverse=True)
            if not files:
                return None
            data = json.loads(files[0].read_text())
            return RawMarketData(**data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Load cached snapshot failed: %s", exc)
            return None

    def calendar(self) -> FetchResult:
        """Return the first successful economic calendar result."""
        cal = self.ff_calendar.fetch()
        if cal.status == "ok":
            return cal
        cal = self.news_calendar.fetch()
        if cal.status == "ok":
            return cal
        return self.marketaux_calendar.fetch()
