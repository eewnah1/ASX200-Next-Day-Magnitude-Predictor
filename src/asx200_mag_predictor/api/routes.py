"""API routes for predictions, actuals, calibration and status."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, Query, UploadFile
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

# In-memory upload job store. Jobs are lost on container restart; use a DB or disk if needed.
_upload_jobs: dict[str, dict[str, Any]] = {}


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


def _predict_sync(body: PredictRequest) -> dict[str, Any]:
    """Synchronous prediction pipeline, suitable for asyncio.to_thread."""
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
    except Exception:  # noqa: B008, BLE001
        logger.exception("Prediction failed")
        raise


@router.post("/predict", response_model=dict[str, Any])
async def predict_manual(body: PredictRequest = Body(default=None)) -> dict[str, Any]:
    """Trigger a fresh prediction from live data and store it."""
    body = body or PredictRequest()
    try:
        return await asyncio.to_thread(_predict_sync, body)
    except Exception as exc:  # noqa: BLE001
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
    data = result.data if result.status in ("ok", "degraded") else {}
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
        return cast(dict[str, Any], json.loads(summary_path.read_text()))
    return {"error": "Backtest summary not available"}


@router.post("/train-ml")
async def train_ml(months: int = Query(default=12, ge=3, le=60)) -> dict[str, Any]:
    """Train the hybrid ML models on historical data and persist them.

    Models are written to the persistent DATA_DIR so subsequent cold starts
    (and the next deploy) load them without re-training.
    """
    from asx200_mag_predictor.scoring.ml import HybridML, MLTrainer

    def _run() -> dict[str, Any]:
        settings = get_settings()
        trainer = MLTrainer(settings=settings)
        result = trainer.run(period=f"{months}mo")
        hybrid = HybridML(settings=settings)
        if hasattr(hybrid, "reload"):
            hybrid.reload()
        else:
            hybrid._load()
        out = dict(result or {})
        out["ml_available"] = hybrid.available
        out["model_dir"] = str(hybrid.model_dir)
        return out

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ML training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/ml-status")
async def ml_status() -> dict[str, Any]:
    """Report whether hybrid ML models are loaded and their training metadata."""
    from asx200_mag_predictor.scoring.ml import HybridML
    from asx200_mag_predictor.scoring.seed_provision import ensure_seed_ml_models

    settings = get_settings()
    ensure_seed_ml_models(settings=settings)
    hybrid = HybridML(settings=settings)
    meta = hybrid.metadata() or {}
    return {
        "ml_available": hybrid.available,
        "model_dir": str(hybrid.model_dir),
        "has_primary": hybrid.primary is not None,
        "has_secondary": hybrid.secondary is not None,
        "has_mapper": hybrid.mapper is not None,
        "metadata": meta,
    }


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


def _run_backtest_sync(
    job_id: str,
    file_path: Path,
    target_column: str,
    period: str,
    train_models: bool,
) -> None:
    """Synchronous worker: run backtest + optional model retrain from uploaded CSV."""
    from asx200_mag_predictor.scoring.csv_backtest import (
        run_backtest_and_train,
        write_summary,
    )

    _upload_jobs[job_id]["status"] = "running"
    try:
        settings = get_settings()
        model_dir = settings.ml_model_dir if train_models else None
        summary = run_backtest_and_train(
            csv_path=file_path,
            period=period,
            n_splits=5,
            model_dir=model_dir,
            target_column=target_column,
        )
        write_summary(summary)
        _upload_jobs[job_id].update(
            {
                "status": "completed",
                "completed_at": now_sydney().isoformat(),
                "summary": {
                    "n_rows_tested": summary.get("n_rows_tested"),
                    "three_class_accuracy": summary.get("overall", {}).get(
                        "three_class_accuracy"
                    ),
                    "directional_accuracy": summary.get("overall", {}).get(
                        "directional_accuracy"
                    ),
                    "binary_directional_accuracy": summary.get("binary", {}).get(
                        "directional_accuracy"
                    ),
                    "note": summary.get("note"),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload backtest/train failed")
        _upload_jobs[job_id].update({"status": "failed", "error": str(exc)})


async def _process_upload(
    job_id: str,
    file_path: Path,
    target_column: str,
    period: str,
    train_models: bool,
) -> None:
    """Background worker wrapper that runs the sync backtest in a thread."""
    await asyncio.to_thread(
        _run_backtest_sync, job_id, file_path, target_column, period, train_models
    )


@router.post("/backtest/upload")
async def upload_backtest_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("Australian Shares"),
    period: str = Form("max"),
    train_models: bool = Form(True),
) -> dict[str, Any]:
    """Upload a daily-rates CSV and start an async backtest + model retrain job."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    settings = get_settings()
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    file_path = upload_dir / f"{job_id}_{safe_name}"

    try:
        file_path.write_bytes(await file.read())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc

    _upload_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "file": safe_name,
        "target_column": target_column,
        "period": period,
        "train_models": train_models,
        "submitted_at": now_sydney().isoformat(),
    }
    background_tasks.add_task(
        _process_upload, job_id, file_path, target_column, period, train_models
    )
    return _upload_jobs[job_id]


@router.get("/backtest/upload/status/{job_id}")
async def upload_status(job_id: str) -> dict[str, Any]:
    """Poll the status of a CSV upload backtest/train job."""
    job = _upload_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/debug/seed")
async def debug_seed(csv_path: str = "/data/uploads/placeholder.csv") -> dict[str, Any]:
    """Debug helper: report whether a seed cache exists for a given CSV path."""
    from asx200_mag_predictor.scoring.csv_backtest import _seed_cache_path

    seed = _seed_cache_path(csv_path, "5y")
    return {
        "csv_path": csv_path,
        "seed": str(seed),
        "seed_exists": seed.is_file() if seed else False,
    }


@router.get("/debug/parquet")
async def debug_parquet() -> dict[str, Any]:
    """Read the seed parquet and report its shape."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "seed_csv_cache",
        Path("src/asx200_mag_predictor/data/seed_csv_cache"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            for parquet in candidate.glob("*.parquet"):
                try:
                    import pandas as pd

                    df = pd.read_parquet(parquet)
                    return {
                        "seed": str(parquet),
                        "rows": len(df),
                        "columns": len(df.columns),
                    }
                except Exception as exc:  # noqa: BLE001
                    return {"seed": str(parquet), "error": str(exc)}
    return {"seed": None}


@router.get("/debug/seeds")
async def debug_seeds() -> dict[str, Any]:
    """List bundled seed cache and model directories."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "seed_csv_cache",
        Path("src/asx200_mag_predictor/data/seed_csv_cache"),
    ]
    files: list[str] = []
    for c in candidates:
        if c.is_dir():
            files += [str(p.name) for p in c.glob("*.parquet")]
    settings = get_settings()
    return {
        "seed_candidates": [str(c) for c in candidates],
        "seed_parquet_files": files,
        "data_dir": str(settings.data_dir),
        "ml_model_dir": str(settings.ml_model_dir),
        "ml_model_dir_exists": settings.ml_model_dir.is_dir(),
    }
