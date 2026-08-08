"""Unit tests for feature construction helpers."""

from asx200_mag_predictor.scoring.features import (
    RawMarketData,
    build_features,
    classify_session,
    compute_catalyst_score,
    compute_cross_asset_alignment,
    compute_vol_regime,
)


def test_compute_vol_regime():
    assert compute_vol_regime(10.0, None, None) == 0
    assert compute_vol_regime(14.0, None, None) == 1
    assert compute_vol_regime(25.0, None, None) == 3
    assert compute_vol_regime(None, None, None) == 1


def test_compute_catalyst_score():
    assert compute_catalyst_score(0, 0) == 0
    assert compute_catalyst_score(3, 0) == 4  # min(3*2,4) + 0
    assert compute_catalyst_score(0, 2) == 1


def test_compute_cross_asset_alignment():
    align, mag = compute_cross_asset_alignment(
        us_futures_change_pct=1.0,
        iron_ore_change_pct=1.5,
        aud_usd_change_pct=0.5,
        sp500_change_pct=0.0,
        nasdaq_change_pct=0.0,
        dow_change_pct=0.0,
        us_10y_change_bps=None,
        vix_change_pct=None,
    )
    assert 0.0 < align <= 1.0
    assert mag > 0.0


def test_classify_session():
    assert classify_session(0.05, 0.9, 0.5) == "range"
    assert classify_session(0.50, 1.3, 1.2) == "trend"
    assert classify_session(0.25, 1.0, 0.9) == "mixed"


def test_build_features_defaults():
    fv, flags = build_features(RawMarketData())
    assert fv.vol_regime is not None
    assert 0 <= fv.catalyst_score <= 5
    assert isinstance(flags.model_dump(), dict)
