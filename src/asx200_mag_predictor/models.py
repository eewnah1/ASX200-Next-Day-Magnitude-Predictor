"""Pydantic domain models used across the app."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from asx200_mag_predictor.timezone import now_sydney


class DataQualityFlags(BaseModel):
    """Health flags for each data source."""

    asx_cash: str = "ok"
    spi_futures: str = "ok"
    a_vix: str = "ok"
    commodities: str = "ok"
    fx: str = "ok"
    us_assets: str = "ok"
    calendar: str = "ok"
    volume: str = "ok"


class FeatureVector(BaseModel):
    """Inputs that drive the rule-based scoring engine."""

    # Volatility / regime
    a_vix: float | None = None
    atr_5d_pct: float | None = None
    realized_vol_annual: float | None = None
    vol_regime: int | None = None

    # Catalyst
    catalyst_score: int | None = None
    high_impact_events_next_24h: int = 0
    high_impact_events_next_48h: int = 0

    # Cross-asset
    us_futures_change_pct: float | None = None
    iron_ore_change_pct: float | None = None
    aud_usd_change_pct: float | None = None
    sp500_change_pct: float | None = None
    nasdaq_change_pct: float | None = None
    dow_change_pct: float | None = None
    us_10y_change_bps: float | None = None
    vix_change_pct: float | None = None
    cross_asset_alignment_score: float | None = None
    cross_asset_magnitude: float | None = None

    # Session character
    asx_open_to_now_return_pct: float | None = None
    current_volume_vs_20d_avg: float | None = None
    current_range_vs_atr: float | None = None
    asx_session_character: str | None = None  # e.g. "trend", "range", "mixed"

    # SPI basis / futures momentum
    spi_basis_pct: float | None = None
    spi_momentum_pct: float | None = None

    # Metadata
    fetched_at: datetime = Field(default_factory=now_sydney)
    sources: dict[str, Any] = Field(default_factory=dict)


class BucketProbabilities(BaseModel):
    """Probabilities for each magnitude bucket."""

    low: float = Field(..., ge=0.0, le=1.0, description="P(|return| < 0.3%)")
    mid: float = Field(..., ge=0.0, le=1.0, description="P(0.3% <= |return| <= 0.5%)")
    high: float = Field(..., ge=0.0, le=1.0, description="P(|return| > 0.5%)")


class FactorBreakdown(BaseModel):
    """Per-factor contribution to the probability of the *high* bucket."""

    volatility: float
    catalyst: float
    alignment: float
    session: float
    spi_basis: float


class Prediction(BaseModel):
    """The public prediction object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | None = None
    prediction_for_date: datetime
    generated_at: datetime = Field(default_factory=now_sydney)
    features: FeatureVector
    probabilities: BucketProbabilities
    bucket: str  # human-readable primary bucket
    confidence: float  # 0-1 explicit confidence
    factor_breakdown: FactorBreakdown
    notes: list[str] = Field(default_factory=list)
    data_quality_flags: DataQualityFlags = Field(default_factory=DataQualityFlags)


class Actual(BaseModel):
    """Outcome used for calibration."""

    prediction_id: str
    actual_abs_return_pct: float
    actual_bucket: str
    recorded_at: datetime = Field(default_factory=now_sydney)


class CalibrationMetrics(BaseModel):
    """Simple calibration metrics."""

    total: int
    correct: int
    hit_rate: float
    by_regime: dict[str, dict[str, float]] = Field(default_factory=dict)
