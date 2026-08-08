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

    volatility_weight: float = 0.20
    catalyst_weight: float = 0.14
    alignment_weight: float = 0.16
    session_weight: float = 0.07
    spi_basis_weight: float = 0.05
    financials_vs_materials_weight: float = 0.12
    housing_credit_weight: float = 0.10
    china_steel_property_weight: float = 0.08
    heavyweight_idio_weight: float = 0.08

    rsi_weight: float = 0.10
    ath_distance_weight: float = 0.10
    momentum_exhaustion_weight: float = 0.08
    bollinger_weight: float = 0.05

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

    @property
    def ml_model_dir(self) -> Path:
        path = self.data_dir / "ml_models"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
