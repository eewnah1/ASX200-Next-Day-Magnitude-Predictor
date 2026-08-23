"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from asx200_mag_predictor.api.mcp_router import router as mcp_router
from asx200_mag_predictor.api.routes import router
from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.logging_config import setup_logging
from asx200_mag_predictor.scheduler.jobs import start_scheduler
from asx200_mag_predictor.storage.models import init_db

settings = get_settings()
setup_logging(settings)
logger = logging.getLogger("asx200_mag_predictor.api")


def _wire_yahoo_fallback() -> None:
    try:
        import asx200_mag_predictor.data.fetchers as fetchers_mod
        from asx200_mag_predictor.data.yahoo_download import yf_download

        fetchers_mod._yf_download = yf_download  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not wire Yahoo chart fallback: %s", exc)


def _train_ml_sync(months: int = 12) -> dict:
    from asx200_mag_predictor.scoring.ml import HybridML, MLTrainer

    trainer = MLTrainer(settings=settings)
    result = trainer.run(period=f"{months}mo")
    hybrid = HybridML(settings=settings)
    if hasattr(hybrid, "reload"):
        hybrid.reload()
    else:
        hybrid._load()  # type: ignore[attr-defined]
    out = dict(result or {})
    out["ml_available"] = hybrid.available
    out["model_dir"] = str(hybrid.model_dir)
    return out


async def _ensure_ml_models() -> None:
    try:
        from asx200_mag_predictor.scoring.ml import HybridML
        from asx200_mag_predictor.scoring.seed_provision import ensure_seed_ml_models

        seeded = ensure_seed_ml_models(settings=settings)
        hybrid = HybridML(settings=settings)
        logger.info("ML models on startup: available=%s seeded=%s dir=%s", hybrid.available, seeded, hybrid.model_dir)
        if hybrid.available:
            return
        logger.info("ML models missing — starting background train (12mo)")

        def _run() -> None:
            try:
                out = _train_ml_sync(months=12)
                logger.info("Background ML train finished: status=%s ml_available=%s rows=%s", out.get("status"), out.get("ml_available"), out.get("rows"))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Background ML train failed: %s", exc)

        asyncio.get_running_loop().run_in_executor(None, _run)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML seed/load on startup failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(settings)
    _wire_yahoo_fallback()
    await _ensure_ml_models()
    scheduler = start_scheduler(settings)
    yield
    scheduler.shutdown()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")
app.include_router(mcp_router, prefix="/api/v1")


@app.get("/health")
async def health():
    ml_available = False
    ml_dir = None
    try:
        from asx200_mag_predictor.scoring.ml import HybridML
        hybrid = HybridML(settings=settings)
        ml_available = hybrid.available
        ml_dir = str(hybrid.model_dir)
    except Exception:
        pass
    return {"status": "ok", "env": settings.app_env, "ml_available": ml_available, "ml_model_dir": ml_dir, "mcp_stack": True}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = Path(__file__).with_name("dashboard.html")
    return path.read_text(encoding="utf-8")
