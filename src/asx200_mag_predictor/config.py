"""Application configuration via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All app configuration."""

    model_config = ConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ASX200 Next-Day Magnitude Predictor"
    app_env: str = "development"
    log_level: str = "INFO"
    tz: str = "Australia/Sydney"

    database_url: str = "sqlite:///./asx200_predictor.db"

    newsapi_api_key: str = ""
    marketaux_api_key: str = ""
    fred_api_key: str = ""
    alphavantage_api_key: str = ""

    schedule_daily: bool = True
    daily_run_time: str = "13:35"
    us_close_update: bool = True
    us_close_run_time: str = "06:30"

    volatility_weight: float = 0.35
    catalyst_weight: float = 0.25
    alignment_weight: float = 0.25
    session_weight: float = 0.10
    spi_basis_weight: float = 0.05

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def data_dir(self) -> Path:
        path = Path("data")
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
