"""Unit tests for the rule-based scoring engine."""

from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scoring.engine import (
    PRIMARY_WEIGHTS,
    ScoringEngine,
    _detect_regime,
    _regime_aware_weights,
    bucket_from_return,
)


def _probs_sum_to_one(p):
    probs = p.probabilities
    total = probs.negative + probs.low + probs.mid + probs.high
    assert round(total, 3) == 1.0


def test_probabilities_sum_to_one(engine: ScoringEngine):
    p = engine.predict(FeatureVector())
    _probs_sum_to_one(p)


def test_high_vol_increases_high_bucket(engine: ScoringEngine):
    low = engine.predict(FeatureVector(a_vix=10.0, realized_vol_annual=10.0)).probabilities.high
    high = engine.predict(FeatureVector(a_vix=35.0, realized_vol_annual=35.0)).probabilities.high
    assert high > low


def test_catalyst_increases_high_bucket(engine: ScoringEngine):
    low = engine.predict(
        FeatureVector(catalyst_score=0, high_impact_events_next_24h=0)
    ).probabilities.high
    high = engine.predict(
        FeatureVector(catalyst_score=5, high_impact_events_next_24h=3)
    ).probabilities.high
    assert high > low


def test_trend_session_increases_high_bucket(engine: ScoringEngine):
    low = engine.predict(FeatureVector(asx_session_character="range")).probabilities.high
    high = engine.predict(FeatureVector(asx_session_character="trend")).probabilities.high
    assert high > low


def test_alignment_magnitude_increases_high_bucket(engine: ScoringEngine):
    low = engine.predict(FeatureVector(cross_asset_magnitude=0.0)).probabilities.high
    high = engine.predict(FeatureVector(cross_asset_magnitude=3.0)).probabilities.high
    assert high > low


def test_bucket_from_return():
    assert bucket_from_return(-0.75) == "Large Down"
    assert bucket_from_return(-0.30) == "Neutral"
    assert bucket_from_return(0.40) == "Neutral"
    assert bucket_from_return(0.75) == "Large Up"


def test_mock_feature_dict(engine: ScoringEngine):
    p = engine.predict({"a_vix": 20.0, "catalyst_score": 3})
    _probs_sum_to_one(p)
    assert 0.0 <= p.confidence <= 1.0


def test_detect_regime_financials_led():
    # Strong financials, weak materials/iron/China => financials-led
    fv = FeatureVector(
        financials_minus_materials_weighted_pct=2.5,
        rba_rates_score=2.0,
        housing_credit_pulse_score=1.5,
        sp500_change_pct=1.5,
        iron_ore_change_pct=-1.0,
        china_steel_property_score=-1.0,
    )
    regime, _, conf, _, _ = _detect_regime(fv)
    assert regime == "financials_led"
    assert conf > 0.0


def test_detect_regime_materials_led():
    # Strong materials, weak financials => materials-led
    fv = FeatureVector(
        financials_minus_materials_weighted_pct=-2.0,
        rba_rates_score=-1.0,
        iron_ore_change_pct=2.5,
        china_steel_property_score=2.0,
        copper_change_pct=1.0,
    )
    regime, _, conf, _, _ = _detect_regime(fv)
    assert regime == "materials_led"
    assert conf > 0.0


def test_regime_aware_weights_rebalance():
    w = _regime_aware_weights("financials_led")
    total = sum(w.values())
    assert round(total, 4) == 1.0
    # financials-led boosts financials vs materials and rba rates
    assert w["financials_vs_materials"] > PRIMARY_WEIGHTS["financials_vs_materials"]
    assert w["rba_rates"] > PRIMARY_WEIGHTS["rba_rates"]
    assert w["iron_ore"] < PRIMARY_WEIGHTS["iron_ore"]


def test_prediction_includes_regime(engine: ScoringEngine):
    p = engine.predict(FeatureVector())
    assert p.regime in ("financials_led", "materials_led", "dual_engine", "contested")
    assert p.regime_confidence is not None
    assert 0.0 <= p.regime_confidence <= 1.0
