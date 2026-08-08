"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from asx200_mag_predictor.api.routes import router
from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.logging_config import setup_logging
from asx200_mag_predictor.storage.models import init_db

settings = get_settings()
setup_logging(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB on startup; scheduler can be started here too."""
    init_db(settings)
    yield


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
