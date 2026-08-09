"""Unit tests for the rule-based scoring engine."""

from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scoring.engine import ScoringEngine, bucket_from_return


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
