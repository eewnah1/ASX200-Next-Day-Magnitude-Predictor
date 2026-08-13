"""API routes for predictions, actuals, calibration and status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.data.fetchers import DataFetcher
from asx200_mag_predictor.logging_config import get_logger
from asx200_mag_predictor.scoring.engine import ScoringEngine
from asx200_mag_predictor.scoring.features import RawMarketData, build_features
from asx200_mag_predictor.storage.repository import Repository
from asx200_mag_predictor.timezone import now_sydney

logger = get_logger(__name__)
router = APIRouter()


class PredictRequest(BaseModel):
    in_position: bool = False
    notes: str = ""


def _repo() -> Repository:
    return Repository()


def _engine() -> ScoringEngine:
    return ScoringEngine(get_settings())


def _fetcher() -> DataFetcher:
    return DataFetcher(get_settings())


def _fallback_raw() -> RawMarketData:
    """Return an empty RawMarketData so a degraded prediction can still be made."""
    return RawMarketData(
        source_status=[],
        errors=["All data fetchers failed; using neutral fallback."],
    )


@router.get("/status")
async def status() -> dict[str, Any]:
    return {
        "now_aest": now_sydney().isoformat(),
        "env": get_settings().app_env,
        "database_url": get_settings().database_url,
    }


@router.post("/predict", response_model=dict[str, Any])
async def predict_manual(body: PredictRequest = Body(default=None)) -> dict[str, Any]:
    """Trigger a fresh prediction from live data and store it."""
    body = body or PredictRequest()
    fetcher = _fetcher()
    raw: RawMarketData | None = None
    fetch_errors: list[str] = []
    try:
        raw = fetcher.fetch_all()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Live fetch failed; trying cache")
        fetch_errors.append(f"Live fetch failed: {exc}")
        raw = fetcher.load_cached_snapshot()
        if not raw:
            fetch_errors.append("No cached snapshot available; using fallback")
            raw = _fallback_raw()

    try:
        features, flags = build_features(raw)
        prediction = _engine().predict(features, flags, in_position=body.in_position)
        if body.notes:
            prediction.notes.append(body.notes)
        if fetch_errors:
            prediction.errors.extend(fetch_errors)
            prediction.degraded = True
            prediction.degraded_sources.extend(["fetch"])
        repo = _repo()
        prediction_id = repo.save_prediction(prediction)
        return {"prediction_id": prediction_id, "prediction": prediction.model_dump(mode="json")}
    except Exception as exc:  # noqa: B008, BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/predictions/latest", response_model=dict[str, Any])
async def latest_prediction() -> dict[str, Any]:
    repo = _repo()
    pred = repo.get_latest_prediction()
    if not pred:
        raise HTTPException(status_code=404, detail="No predictions yet")
    return pred.model_dump(mode="json")


@router.get("/predictions")
async def list_predictions(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    repo = _repo()
    return [p.model_dump(mode="json") for p in repo.list_predictions(limit)]


@router.get("/predictions/{prediction_id}", response_model=dict[str, Any])
async def get_prediction(prediction_id: str) -> dict[str, Any]:
    repo = _repo()
    pred = repo.get_prediction(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return pred.model_dump(mode="json")


@router.post("/actuals/{prediction_id}")
async def record_actual(prediction_id: str, actual_return_pct: float) -> dict[str, Any]:
    repo = _repo()
    pred = repo.get_prediction(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    bucket = repo.save_actual(prediction_id, actual_return_pct)
    return {
        "prediction_id": prediction_id,
        "actual_return_pct": actual_return_pct,
        "actual_bucket": bucket,
    }


@router.get("/calibration")
async def calibration() -> dict[str, Any]:
    repo = _repo()
    return repo.calibration_metrics().model_dump()


@router.get("/calendar")
async def calendar() -> dict[str, Any]:
    fetcher = _fetcher()
    result = fetcher.calendar()
    data = result.data if result.status == "ok" else {}
    if not data:
        data = {"message": "No calendar data available"}
    return {
        **data,
        "status": result.status,
        "error": result.error,
        "last_success_at": result.last_success_at,
    }


@router.get("/backtest/summary")
async def backtest_summary() -> dict[str, Any]:
    """Return the daily-rates high-conviction backtest summary."""
    import json
    from pathlib import Path

    summary_path = Path(__file__).parent.parent / "daily_rates_backtest_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {"error": "Backtest summary not available"}


@router.post("/train-ml")
async def train_ml() -> dict[str, Any]:
    """Train the hybrid ML models on historical data."""
    from asx200_mag_predictor.scoring.ml import MLTrainer

    try:
        trainer = MLTrainer(settings=get_settings())
        trainer.run()
        return {"status": "ok", "message": "ML models trained and saved"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("ML training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/run-daily")
async def run_daily() -> dict[str, Any]:
    """Manual trigger of the daily job."""
    return await predict_manual(PredictRequest(notes="daily-run"))


@router.get("/tradingview/insights")
async def tradingview_insights() -> dict[str, Any]:
    """Combined TradingView MCP market snapshot and ASX 200 analysis."""
    import asyncio

    from asx200_mag_predictor.data.tradingview_mcp import get_asx200_insights

    try:
        return await asyncio.to_thread(get_asx200_insights)
    except Exception as exc:  # noqa: BLE001
        logger.exception("TradingView insights failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/tradingview/market-snapshot")
async def tradingview_market_snapshot() -> dict[str, Any]:
    """Global market snapshot from atilaahmettaner/tradingview-mcp."""
    import asyncio

    from asx200_mag_predictor.data.tradingview_mcp import atila_market_snapshot

    return await asyncio.to_thread(atila_market_snapshot)


@router.get("/tradingview/ta/{symbol:path}")
async def tradingview_ta(symbol: str, exchange: str = "asx") -> dict[str, Any]:
    """Multi-timeframe technical analysis for a TradingView symbol."""
    import asyncio

    from asx200_mag_predictor.data.tradingview_mcp import atila_symbol_analysis

    return await asyncio.to_thread(atila_symbol_analysis, symbol, exchange)


@router.get("/tradingview/price/{symbol:path}")
async def tradingview_price(symbol: str) -> dict[str, Any]:
    """Latest price quote for a single symbol."""
    import asyncio

    from asx200_mag_predictor.data.tradingview_mcp import atila_price

    return await asyncio.to_thread(atila_price, symbol)


@router.get("/tradingview/screen")
async def tradingview_screen(
    asset_type: str = "stocks",
    preset: str = "quality_stocks",
    limit: int = 10,
) -> dict[str, Any]:
    """Run a TradingView screener preset via fiale-plus/tradingview-mcp-server."""
    import asyncio

    from asx200_mag_predictor.data.tradingview_mcp import fiale_screen

    return await asyncio.to_thread(fiale_screen, asset_type, preset, limit)


@router.get("/tradingview/lookup")
async def tradingview_lookup(symbols: str) -> dict[str, Any]:
    """Look up one or more TradingView symbols (comma-separated)."""
    import asyncio

    from asx200_mag_predictor.data.tradingview_mcp import fiale_lookup

    parts = [s.strip() for s in symbols.split(",") if s.strip()]
    if not parts:
        raise HTTPException(status_code=400, detail="No symbols provided")
    return await asyncio.to_thread(fiale_lookup, *parts)
