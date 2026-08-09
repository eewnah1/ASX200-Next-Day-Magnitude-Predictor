"""Pydantic domain models used across the app."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from asx200_mag_predictor.timezone import now_sydney


class DataSourceStatus(BaseModel):
    """Health metadata for a single external data source."""

    name: str
    status: str = "ok"  # ok | stale | failed
    last_success_at: str | None = None
    value: str | None = None
    error: str | None = None


class DataQualityFlags(BaseModel):
    """Health flags for each data source (kept for backward compatibility)."""

    asx_cash: str = "ok"
    spi_futures: str = "ok"
    a_vix: str = "ok"
    commodities: str = "ok"
    fx: str = "ok"
    us_assets: str = "ok"
    calendar: str = "ok"
    volume: str = "ok"
    financials_vs_materials: str = "ok"
    housing_credit: str = "ok"
    china_steel_property: str = "ok"
    heavyweight_idio: str = "ok"


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
    gold_change_pct: float | None = None
    silver_change_pct: float | None = None
    oil_change_pct: float | None = None
    copper_change_pct: float | None = None
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
    spi_short_term_momentum_pct: float | None = None

    # Secondary-model short-term inputs
    overnight_gap_pct: float | None = None
    gap_filled_score: float | None = None
    vwap_distance_pct: float | None = None
    market_breadth_score: float | None = None

    # New high-priority factors
    financials_minus_materials_1d_pct: float | None = None
    financials_minus_materials_2d_pct: float | None = None
    financials_minus_materials_3d_pct: float | None = None
    financials_minus_materials_5d_pct: float | None = None
    financials_minus_materials_weighted_pct: float | None = None
    financials_vs_materials_score: float | None = None

    housing_credit_pulse_score: float | None = None  # 0-10
    housing_credit_pulse_sources: list[str] = Field(default_factory=list)

    china_steel_property_score: float | None = None
    china_steel_property_return_pct: float | None = None
    china_steel_property_sources: list[str] = Field(default_factory=list)

    heavyweight_idio_return_pct: float | None = None
    heavyweight_idio_score: float | None = None
    heavyweight_idio_news_boost: float = 0.0

    # Technical indicators
    rsi_14: float | None = None
    rsi_previous_14: float | None = None
    rsi_slope: float | None = None
    rsi_score: float | None = None
    ath_distance_pct: float | None = None
    high_20d_distance_pct: float | None = None
    high_50d_distance_pct: float | None = None
    ath_score: float | None = None
    asx_1d_return_pct: float | None = None
    asx_2d_return_pct: float | None = None
    asx_3d_return_pct: float | None = None
    index_5d_return_pct: float | None = None
    momentum_exhaustion_score: float | None = None
    bollinger_position: float | None = None
    bollinger_score: float | None = None
    profit_taking_combo_score: float | None = None

    # Metadata
    fetched_at: datetime = Field(default_factory=now_sydney)
    data_as_of: datetime | None = None  # timestamp of the latest available market data
    sources: dict[str, Any] = Field(default_factory=dict)
    source_status: list[DataSourceStatus] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BucketProbabilities(BaseModel):
    """Probabilities for each next-day return bucket."""

    negative: float = Field(..., ge=0.0, le=1.0, description="P(return < 0%)")
    low: float = Field(..., ge=0.0, le=1.0, description="P(0% <= return < 0.3%)")
    mid: float = Field(..., ge=0.0, le=1.0, description="P(0.3% <= return <= 0.5%)")
    high: float = Field(..., ge=0.0, le=1.0, description="P(return > 0.5%)")


class FactorContribution(BaseModel):
    """Human-readable contribution of a single input factor."""

    name: str
    raw_value: float | None = None
    raw_unit: str = ""
    direction: str = "neutral"  # bullish / bearish / neutral
    weight: float = 0.0
    score: float = 0.0
    note: str = ""
    group: str = ""  # e.g. "Primary" or "Secondary"


class FactorBreakdown(BaseModel):
    """Per-factor logit-delta contribution to the probability of the *high* bucket."""

    volatility: float
    catalyst: float
    alignment: float
    session: float
    spi_basis: float
    direction: float


class Prediction(BaseModel):
    """The public prediction object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | None = None
    prediction_for_date: datetime
    generated_at: datetime = Field(default_factory=now_sydney)
    data_as_of: datetime | None = None
    features: FeatureVector
    probabilities: BucketProbabilities
    bucket: str  # human-readable primary bucket
    confidence: float  # 0-1 explicit confidence
    factor_breakdown: FactorBreakdown
    factor_contributions: list[FactorContribution] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    data_quality_flags: DataQualityFlags = Field(default_factory=DataQualityFlags)
    source_status: list[DataSourceStatus] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    degraded: bool = False
    degraded_sources: list[str] = Field(default_factory=list)
    model: str = "Primary"  # Primary | Secondary
    primary_bucket: str = "Neutral"
    secondary_bucket: str | None = None
    primary_score: float = 0.0
    secondary_score: float = 0.0
    ml_available: bool = False
    ml_probabilities: dict[str, Any] | None = None
    ml_feature_importance: list[dict[str, Any]] = Field(default_factory=list)
    ml_fallback_reason: str | None = None

    # Long-only directional recommendation
    recommendation: str = "STAY IN CASH"  # GO LONG | STAY IN CASH | HOLD EXISTING
    recommendation_source: str = "Primary"  # Primary | Secondary
    recommendation_confidence: float = 0.0  # 0-1 confidence in the recommendation
    in_position: bool | None = None


class Actual(BaseModel):
    """Outcome used for calibration."""

    prediction_id: str
    actual_return_pct: float
    actual_bucket: str
    recorded_at: datetime = Field(default_factory=now_sydney)


class CalibrationMetrics(BaseModel):
    """Simple calibration metrics."""

    total: int
    correct: int
    hit_rate: float
    by_regime: dict[str, dict[str, float]] = Field(default_factory=dict)
