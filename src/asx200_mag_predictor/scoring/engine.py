"""Rule-based scoring engine for ASX200 next-day magnitude probabilities.

Design notes
------------
* The engine is intentionally transparent.  Every factor maps to a logit
  adjustment for the [low, mid, high] buckets, and a final softmax turns the
  combined logit vector into a probability distribution.
* Volatility sets the *baseline* distribution.  The other four factors tilt
  the baseline toward or away from a large move.
* Weights from `Settings` scale each factor's contribution; they are normalised
  internally so the engine is stable regardless of absolute weight size.
* All missing values are filled with neutral defaults and flagged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.logging_config import get_logger
from asx200_mag_predictor.models import (
    BucketProbabilities,
    DataQualityFlags,
    FactorBreakdown,
    FeatureVector,
    Prediction,
)
from asx200_mag_predictor.timezone import now_sydney

logger = get_logger(__name__)

BUCKET_LABELS = ["<0.3%", "0.3%-0.5%", ">0.5%"]
BUCKET_KEYS = ["low", "mid", "high"]

# Baseline bucket distributions indexed by volatility regime (0=calm ... 4=extreme).
# Each vector is [low, mid, high] and sums to 1.
VOL_BASELINES: dict[int, list[float]] = {
    0: [0.60, 0.28, 0.12],  # calm
    1: [0.45, 0.30, 0.25],  # normal
    2: [0.30, 0.30, 0.40],  # elevated
    3: [0.20, 0.25, 0.55],  # high
    4: [0.12, 0.20, 0.68],  # extreme
}


def _softmax(logits: np.ndarray, temperature: float = 0.8) -> np.ndarray:
    """Numerically stable softmax with optional temperature."""
    x = np.asarray(logits, dtype=float) / temperature
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _clamp(value: float | None, low: float, high: float, default: float) -> float:
    if value is None or math.isnan(value):
        return default
    return max(low, min(high, value))


class ScoringEngine:
    """Predict P(|ASX200 next-day return| in bucket) from a FeatureVector."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.weights = {
            "volatility": _clamp(self.settings.volatility_weight, 0.0, 1.0, 0.35),
            "catalyst": _clamp(self.settings.catalyst_weight, 0.0, 1.0, 0.25),
            "alignment": _clamp(self.settings.alignment_weight, 0.0, 1.0, 0.25),
            "session": _clamp(self.settings.session_weight, 0.0, 1.0, 0.10),
            "spi_basis": _clamp(self.settings.spi_basis_weight, 0.0, 1.0, 0.05),
        }
        # Normalise weights so they behave like relative allocations.
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.temperature = 0.85

    def predict(
        self,
        features: FeatureVector | dict[str, Any],
        data_quality_flags: DataQualityFlags | None = None,
        prediction_for: Any | None = None,
    ) -> Prediction:
        """Return a calibrated prediction for the supplied feature set."""
        fv = self._coerce(features)
        flags = data_quality_flags or DataQualityFlags()
        fv, flags = self._fill_missing(fv, flags)

        prediction_for_date = self._resolve_prediction_for(prediction_for)

        # Start with baseline log-probabilities set by volatility regime.
        baseline_probs = VOL_BASELINES.get(fv.vol_regime or 1, VOL_BASELINES[1])
        baseline_logits = np.log(np.maximum(baseline_probs, 1e-9))

        # Build per-factor logit deltas.
        vol_delta, vol_high = self._volatility_delta(fv)
        cat_delta, cat_high = self._catalyst_delta(fv)
        align_delta, align_high = self._alignment_delta(fv)
        sess_delta, sess_high = self._session_delta(fv)
        spi_delta, spi_high = self._spi_delta(fv)

        # Weighted combination.
        combined = baseline_logits.copy()
        combined += self.weights["volatility"] * vol_delta
        combined += self.weights["catalyst"] * cat_delta
        combined += self.weights["alignment"] * align_delta
        combined += self.weights["session"] * sess_delta
        combined += self.weights["spi_basis"] * spi_delta

        probs = _softmax(combined, temperature=self.temperature)

        # Small Laplace-style smoothing to avoid over-confident zeros.
        probs = (probs + 1e-4) / (probs + 1e-4).sum()

        bucket_index = int(np.argmax(probs))
        bucket_label = BUCKET_LABELS[bucket_index]

        confidence = self._compute_confidence(probs, flags)
        factor_breakdown = FactorBreakdown(
            volatility=round(float(vol_high) * self.weights["volatility"], 4),
            catalyst=round(float(cat_high) * self.weights["catalyst"], 4),
            alignment=round(float(align_high) * self.weights["alignment"], 4),
            session=round(float(sess_high) * self.weights["session"], 4),
            spi_basis=round(float(spi_high) * self.weights["spi_basis"], 4),
        )

        notes: list[str] = []
        if baseline_probs[-1] > 0.40:
            notes.append(f"Elevated vol baseline (regime {fv.vol_regime}) tilts toward >0.5%.")
        if fv.catalyst_score and fv.catalyst_score >= 3:
            notes.append(f"High catalyst score ({fv.catalyst_score}) from scheduled events.")
        if fv.cross_asset_magnitude and fv.cross_asset_magnitude > 1.0:
            notes.append(f"Cross-asset magnitude {fv.cross_asset_magnitude:.2f}% is meaningful.")

        return Prediction(
            prediction_for_date=prediction_for_date,
            features=fv,
            probabilities=BucketProbabilities(
                low=round(float(probs[0]), 4),
                mid=round(float(probs[1]), 4),
                high=round(float(probs[2]), 4),
            ),
            bucket=bucket_label,
            confidence=round(confidence, 4),
            factor_breakdown=factor_breakdown,
            notes=notes,
            data_quality_flags=flags,
        )

    # ------------------------------------------------------------------ helpers

    def _coerce(self, features: FeatureVector | dict[str, Any]) -> FeatureVector:
        if isinstance(features, FeatureVector):
            return features
        return FeatureVector(**features)

    def _fill_missing(
        self, fv: FeatureVector, flags: DataQualityFlags
    ) -> tuple[FeatureVector, DataQualityFlags]:
        """Apply neutral defaults and record data-quality issues."""
        if fv.a_vix is None and fv.realized_vol_annual is None and fv.atr_5d_pct is None:
            flags.a_vix = "missing or stale"
            # Conservative neutral regime
            fv = fv.model_copy(update={"vol_regime": 1})

        if fv.vol_regime is None:
            from asx200_mag_predictor.scoring.features import compute_vol_regime

            regime = compute_vol_regime(fv.a_vix, fv.atr_5d_pct, fv.realized_vol_annual)
            fv = fv.model_copy(update={"vol_regime": regime})

        if fv.catalyst_score is None:
            from asx200_mag_predictor.scoring.features import compute_catalyst_score

            cat = compute_catalyst_score(
                fv.high_impact_events_next_24h, fv.high_impact_events_next_48h
            )
            fv = fv.model_copy(update={"catalyst_score": cat})

        if fv.cross_asset_alignment_score is None and fv.cross_asset_magnitude is None:
            from asx200_mag_predictor.scoring.features import compute_cross_asset_alignment

            align, mag = compute_cross_asset_alignment(
                us_futures_change_pct=fv.us_futures_change_pct,
                iron_ore_change_pct=fv.iron_ore_change_pct,
                aud_usd_change_pct=fv.aud_usd_change_pct,
                sp500_change_pct=fv.sp500_change_pct,
                nasdaq_change_pct=fv.nasdaq_change_pct,
                dow_change_pct=fv.dow_change_pct,
                us_10y_change_bps=fv.us_10y_change_bps,
                vix_change_pct=fv.vix_change_pct,
            )
            fv = fv.model_copy(
                update={"cross_asset_alignment_score": align, "cross_asset_magnitude": mag}
            )
        elif fv.cross_asset_alignment_score is None:
            fv = fv.model_copy(update={"cross_asset_alignment_score": 0.0})
        elif fv.cross_asset_magnitude is None:
            fv = fv.model_copy(update={"cross_asset_magnitude": 0.0})

        if fv.asx_session_character is None:
            from asx200_mag_predictor.scoring.features import classify_session

            char = classify_session(
                fv.asx_open_to_now_return_pct,
                fv.current_volume_vs_20d_avg,
                fv.current_range_vs_atr,
            )
            fv = fv.model_copy(update={"asx_session_character": char})

        return fv, flags

    def _resolve_prediction_for(self, prediction_for: Any | None) -> Any:
        """Predictions are for the next ASX trading session close."""
        if prediction_for is None:
            from asx200_mag_predictor.timezone import next_asx_session

            return next_asx_session(now_sydney())
        return prediction_for

    def _volatility_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Extra vol push beyond the baseline regime.

        If implied (A-VIX) is meaningfully above realised vol, raise the
        probability of a large move (uncooked risk).  If realised > implied,
        we are already in a trending market -- also raise the large bucket.
        """
        vix = _clamp(fv.a_vix, 0.0, 200.0, 0.0)
        realised = _clamp(fv.realized_vol_annual, 0.0, 200.0, 0.0)

        if vix > 0 and realised > 0:
            gap = (vix - realised) / 30.0  # normalise to ~ +/-1
        elif vix > 0:
            gap = (vix - 16.0) / 30.0
        elif realised > 0:
            gap = (realised - 16.0) / 30.0
        else:
            gap = 0.0

        gap = _clamp(gap, -1.0, 1.0, 0.0)
        # Positive gap -> high bucket, negative -> low bucket
        high = gap * 0.30
        low = -gap * 0.20
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _catalyst_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Scheduled high-impact events increase tail probability.

        Score 0-5 linearly scales a high-bucket uplift; low bucket falls by the
        same amount (mid keeps the remainder).
        """
        score = _clamp(fv.catalyst_score, 0.0, 5.0, 0.0)
        high = (score / 5.0) * 0.45
        low = -high * 0.70
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _alignment_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Strong directional alignment among US futures, commodities and FX.

        We care about the *magnitude* of aligned moves more than the sign,
        because the prediction target is the *absolute* ASX return.
        A large pro-ASX suite raises the high bucket; a large risk-off suite
        still raises volatility and therefore the high bucket.
        """
        magnitude = _clamp(fv.cross_asset_magnitude, 0.0, 5.0, 0.0)
        alignment = _clamp(fv.cross_asset_alignment_score, -1.0, 1.0, 0.0)

        # Magnitude is the dominant driver; alignment nudges the sign of the
        # low/high split (pro-ASX -> high, risk-off -> low).
        mag_push = min(magnitude / 4.0, 0.40)
        sign_push = alignment * 0.15
        high = mag_push + sign_push
        low = -mag_push * 0.60 + sign_push * 0.30
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _session_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Current ASX session character: trend days are more likely to extend."""
        char = fv.asx_session_character or "unknown"
        if char == "trend":
            high = 0.25
        elif char == "range":
            high = -0.20
        else:
            high = 0.0
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _spi_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """SPI futures basis and momentum confirm / deny the cash move.

        Positive basis + positive momentum = futures leading the cash market
        higher -> raises the high bucket.  The opposite lowers it.
        """
        basis = _clamp(fv.spi_basis_pct, -2.0, 2.0, 0.0)
        momentum = _clamp(fv.spi_momentum_pct, -3.0, 3.0, 0.0)
        combined = (basis + momentum) / 2.0
        high = _clamp(combined, -0.30, 0.30, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _compute_confidence(self, probs: np.ndarray, flags: DataQualityFlags) -> float:
        """Confidence = how far the distribution is from uniform, penalised by missing data."""
        max_prob = float(np.max(probs))
        # Distance from uniform (1/3), normalised to [0, 1]
        base_conf = (max_prob - 1.0 / 3.0) / (2.0 / 3.0)
        base_conf = max(0.0, min(1.0, base_conf))

        # Count non-ok flags
        flag_count = sum(1 for v in flags.model_dump().values() if v != "ok")
        penalty = min(flag_count * 0.08, 0.35)
        return base_conf - penalty


def bucket_from_return(abs_return_pct: float) -> str:
    """Map an actual absolute return to its bucket label."""
    if abs_return_pct < 0.3:
        return BUCKET_LABELS[0]
    if abs_return_pct <= 0.5:
        return BUCKET_LABELS[1]
    return BUCKET_LABELS[2]
