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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB and start the daily scheduler on startup."""
    init_db(settings)
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
    return {"status": "ok", "env": settings.app_env}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the public predictor dashboard."""
    path = Path(__file__).with_name("dashboard.html")
    return path.read_text(encoding="utf-8")
