"""Shared fixtures."""

import pytest

from asx200_mag_predictor.scoring.engine import ScoringEngine


@pytest.fixture
def engine():
    return ScoringEngine()
