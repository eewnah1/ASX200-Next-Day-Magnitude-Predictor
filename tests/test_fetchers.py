"""Unit tests for the SPI 200 futures fetcher hardening."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from asx200_mag_predictor.data.fetchers import YFinanceClient


def _make_df(timestamp: datetime, close: float = 7000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [close * 0.99],
            "High": [close * 1.01],
            "Low": [close * 0.98],
            "Close": [close],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex([timestamp]),
    )


@pytest.fixture
def client() -> YFinanceClient:
    return YFinanceClient()


def test_spi_futures_primary_ok(client: YFinanceClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh primary futures data -> ok, non-cash proxy."""
    now = datetime(2026, 8, 7, 9, 0)  # Friday 09:00
    prev_close = datetime(2026, 8, 6, 16, 0)  # Thursday 16:00
    ts = datetime(2026, 8, 6, 0, 0)  # Thursday bar

    def _fake_download(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
        if tickers[0] in ("AP=F", "^AP"):
            return _make_df(ts, close=7000.0)
        return pd.DataFrame()

    monkeypatch.setattr("asx200_mag_predictor.data.fetchers._yf_download", _fake_download)
    monkeypatch.setattr("asx200_mag_predictor.data.fetchers.now_sydney", lambda: now)
    monkeypatch.setattr(
        "asx200_mag_predictor.data.fetchers.previous_asx_session_close", lambda dt=None: prev_close
    )

    result = client.spi_futures()
    assert result.status == "ok"
    assert result.ticker == "AP=F"
    assert result.data["source_group"] == "primary"
    assert result.data["cash_proxy"] is False
    assert result.data["last_price_date"] is not None
    assert result.data["data_age_hours"] is not None


def test_spi_futures_fallback_cash_proxy_ok(
    client: YFinanceClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary futures fail; fresh cash proxy fallback -> ok and marked cash_proxy."""
    now = datetime(2026, 8, 10, 9, 0)  # Monday 09:00
    prev_close = datetime(2026, 8, 7, 16, 0)  # Friday 16:00
    ts = datetime(2026, 8, 7, 0, 0)  # Friday bar

    def _fake_download(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
        if tickers[0] == "^AXJO":
            return _make_df(ts, close=7000.0)
        return pd.DataFrame()

    monkeypatch.setattr("asx200_mag_predictor.data.fetchers._yf_download", _fake_download)
    monkeypatch.setattr("asx200_mag_predictor.data.fetchers.now_sydney", lambda: now)
    monkeypatch.setattr(
        "asx200_mag_predictor.data.fetchers.previous_asx_session_close", lambda dt=None: prev_close
    )

    result = client.spi_futures()
    assert result.status == "ok"
    assert result.ticker == "^AXJO"
    assert result.data["source_group"] == "fallback"
    assert result.data["cash_proxy"] is True


def test_spi_futures_stale(client: YFinanceClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Usable ticker exists but data is older than the last cash close -> stale."""
    now = datetime(2026, 8, 11, 9, 0)  # Tuesday 09:00
    prev_close = datetime(2026, 8, 10, 16, 0)  # Monday 16:00
    ts = datetime(2026, 8, 7, 0, 0)  # Friday bar (older than Monday close)

    def _fake_download(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
        if tickers[0] == "AP=F":
            return _make_df(ts, close=7000.0)
        return pd.DataFrame()

    monkeypatch.setattr("asx200_mag_predictor.data.fetchers._yf_download", _fake_download)
    monkeypatch.setattr("asx200_mag_predictor.data.fetchers.now_sydney", lambda: now)
    monkeypatch.setattr(
        "asx200_mag_predictor.data.fetchers.previous_asx_session_close", lambda dt=None: prev_close
    )

    result = client.spi_futures()
    assert result.status == "stale"
    assert result.ticker == "AP=F"
    assert result.error is not None
    assert "stale" in result.error.lower()


def test_spi_futures_missing(client: YFinanceClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """All tickers return empty -> failed."""

    def _fake_download(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr("asx200_mag_predictor.data.fetchers._yf_download", _fake_download)

    result = client.spi_futures()
    assert result.status == "failed"
    assert result.ticker is None
    assert result.error is not None
