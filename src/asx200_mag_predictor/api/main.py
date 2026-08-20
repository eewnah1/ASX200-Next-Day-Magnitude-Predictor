"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from asx200_mag_predictor.api.routes import router
from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.logging_config import setup_logging
from asx200_mag_predictor.scheduler.jobs import start_scheduler
from asx200_mag_predictor.storage.models import init_db

settings = get_settings()
setup_logging(settings)


def _wire_yahoo_fallback() -> None:
    """Replace fragile yfinance-only download with chart-API-aware implementation."""
    try:
        import asx200_mag_predictor.data.fetchers as fetchers_mod
        from asx200_mag_predictor.data.yahoo_download import yf_download

        fetchers_mod._yf_download = yf_download  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("asx200_mag_predictor.api").warning(
            "Could not wire Yahoo chart fallback: %s", exc
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB, seed ML models if missing, and start the daily scheduler."""
    init_db(settings)
    _wire_yahoo_fallback()
    try:
        import logging

        from asx200_mag_predictor.scoring.ml import HybridML
        from asx200_mag_predictor.scoring.seed_provision import ensure_seed_ml_models

        seeded = ensure_seed_ml_models(settings=settings)
        hybrid = HybridML(settings=settings)
        logging.getLogger("asx200_mag_predictor.api").info(
            "ML models on startup: available=%s seeded=%s dir=%s",
            hybrid.available,
            seeded,
            hybrid.model_dir,
        )
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger("asx200_mag_predictor.api").warning(
            "ML seed/load on startup failed: %s", exc
        )
    scheduler = start_scheduler(settings)
    yield
    scheduler.shutdown()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    ml_available = False
    ml_dir = None
    try:
        from asx200_mag_predictor.scoring.ml import HybridML

        hybrid = HybridML(settings=settings)
        ml_available = hybrid.available
        ml_dir = str(hybrid.model_dir)
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok",
        "env": settings.app_env,
        "ml_available": ml_available,
        "ml_model_dir": ml_dir,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the public predictor dashboard."""
    path = Path(__file__).with_name("dashboard.html")
    return path.read_text(encoding="utf-8")
