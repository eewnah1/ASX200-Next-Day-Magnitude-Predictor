"""Rule-based scoring engine for ASX200 next-day magnitude and direction probabilities.

Design notes
------------
* The engine is intentionally transparent.  Every factor maps to a logit
  adjustment for the [low, mid, high] magnitude buckets, and a final softmax
  turns the combined logit vector into a conditional positive-move distribution.
* A separate direction model estimates the probability the next-day return is
  negative (< 0%).  The two pieces are combined to produce the final four
  buckets: negative, low (0-0.3%), mid (0.3-0.5%), high (>0.5%).
* Volatility sets the *magnitude baseline* distribution.  The other four
  factors tilt the baseline toward or away from a large move.
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
    DataSourceStatus,
    FactorBreakdown,
    FactorContribution,
    FeatureVector,
    Prediction,
)
from asx200_mag_predictor.timezone import now_sydney

logger = get_logger(__name__)

BUCKET_LABELS = ["<0%", "0%-0.3%", "0.3%-0.5%", ">0.5%"]
BUCKET_KEYS = ["negative", "low", "mid", "high"]

# Baseline magnitude distributions indexed by volatility regime (0=calm ... 4=extreme).
# Each vector is [low, mid, high] for *positive* moves and sums to 1.
# The negative bucket is derived from a separate direction model.
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


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(value: float | None, low: float, high: float, default: float) -> float:
    if value is None or math.isnan(value):
        return default
    return max(low, min(high, value))


def _direction_label(value: float, bullish: str = "bullish", bearish: str = "bearish") -> str:
    if value > 0.1:
        return bullish
    if value < -0.1:
        return bearish
    return "neutral"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


class ScoringEngine:
    """Predict P(|ASX200 next-day return| in bucket) from a FeatureVector."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.weights = {
            "volatility": _clamp(self.settings.volatility_weight, 0.0, 1.0, 0.20),
            "catalyst": _clamp(self.settings.catalyst_weight, 0.0, 1.0, 0.14),
            "alignment": _clamp(self.settings.alignment_weight, 0.0, 1.0, 0.16),
            "session": _clamp(self.settings.session_weight, 0.0, 1.0, 0.07),
            "spi_basis": _clamp(self.settings.spi_basis_weight, 0.0, 1.0, 0.05),
            "financials_vs_materials": _clamp(
                self.settings.financials_vs_materials_weight, 0.0, 1.0, 0.12
            ),
            "housing_credit": _clamp(self.settings.housing_credit_weight, 0.0, 1.0, 0.10),
            "china_steel_property": _clamp(
                self.settings.china_steel_property_weight, 0.0, 1.0, 0.08
            ),
            "heavyweight_idio": _clamp(self.settings.heavyweight_idio_weight, 0.0, 1.0, 0.08),
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

        # Factor-level deltas and raw scores.
        vol_delta, vol_high = self._volatility_delta(fv)
        cat_delta, cat_high = self._catalyst_delta(fv)
        align_delta, align_high = self._alignment_delta(fv)
        sess_delta, sess_high = self._session_delta(fv)
        spi_delta, spi_high = self._spi_delta(fv)
        fvm_delta, fvm_high = self._financials_vs_materials_delta(fv)
        hc_delta, hc_high = self._housing_credit_delta(fv)
        china_delta, china_high = self._china_steel_property_delta(fv)
        hw_delta, hw_high = self._heavyweight_idio_delta(fv)

        baseline_probs = VOL_BASELINES.get(fv.vol_regime or 1, VOL_BASELINES[1])
        baseline_logits = np.log(np.maximum(baseline_probs, 1e-9))

        combined = baseline_logits.copy()
        combined += self.weights["volatility"] * vol_delta
        combined += self.weights["catalyst"] * cat_delta
        combined += self.weights["alignment"] * align_delta
        combined += self.weights["session"] * sess_delta
        combined += self.weights["spi_basis"] * spi_delta
        combined += self.weights["financials_vs_materials"] * fvm_delta
        combined += self.weights["housing_credit"] * hc_delta
        combined += self.weights["china_steel_property"] * china_delta
        combined += self.weights["heavyweight_idio"] * hw_delta

        abs_probs = _softmax(combined, temperature=self.temperature)
        abs_probs = (abs_probs + 1e-4) / (abs_probs + 1e-4).sum()

        direction_score = self._direction_score(fv)
        p_negative = _sigmoid(direction_score * 2.0)

        probs = np.array(
            [
                p_negative,
                (1.0 - p_negative) * float(abs_probs[0]),
                (1.0 - p_negative) * float(abs_probs[1]),
                (1.0 - p_negative) * float(abs_probs[2]),
            ]
        )
        probs = probs / probs.sum()

        bucket_index = int(np.argmax(probs))
        bucket_label = BUCKET_LABELS[bucket_index]

        factor_breakdown = FactorBreakdown(
            volatility=round(float(vol_high) * self.weights["volatility"], 4),
            catalyst=round(float(cat_high) * self.weights["catalyst"], 4),
            alignment=round(float(align_high) * self.weights["alignment"], 4),
            session=round(float(sess_high) * self.weights["session"], 4),
            spi_basis=round(float(spi_high) * self.weights["spi_basis"], 4),
            direction=round(float(direction_score), 4),
        )

        factor_contributions = self._build_factor_contributions(
            fv,
            vol_high,
            cat_high,
            align_high,
            sess_high,
            spi_high,
            fvm_high,
            hc_high,
            china_high,
            hw_high,
        )

        confidence = self._compute_confidence(probs, fv)
        degraded, degraded_sources = self._degraded_status(fv, flags)

        notes: list[str] = []
        if baseline_probs[-1] > 0.40:
            notes.append(f"Elevated vol baseline (regime {fv.vol_regime}) tilts toward >0.5%.")
        if fv.catalyst_score and fv.catalyst_score >= 3:
            notes.append(f"High catalyst score ({fv.catalyst_score}) from scheduled events.")
        if fv.cross_asset_magnitude and fv.cross_asset_magnitude > 1.0:
            notes.append(f"Cross-asset magnitude {fv.cross_asset_magnitude:.2f}% is meaningful.")
        if p_negative > 0.55:
            notes.append(
                f"Bearish direction score {direction_score:.2f} raises probability of a down day."
            )
        elif p_negative < 0.45:
            notes.append(
                f"Bullish direction score {direction_score:.2f} lowers probability of a down day."
            )
        if degraded:
            notes.append(f"Degraded prediction – missing: {', '.join(degraded_sources)}.")

        return Prediction(
            prediction_for_date=prediction_for_date,
            data_as_of=fv.data_as_of,
            features=fv,
            probabilities=BucketProbabilities(
                negative=round(float(probs[0]), 4),
                low=round(float(probs[1]), 4),
                mid=round(float(probs[2]), 4),
                high=round(float(probs[3]), 4),
            ),
            bucket=bucket_label,
            confidence=round(confidence, 4),
            factor_breakdown=factor_breakdown,
            factor_contributions=factor_contributions,
            notes=notes,
            data_quality_flags=flags,
            source_status=fv.source_status,
            errors=fv.errors,
            degraded=degraded,
            degraded_sources=degraded_sources,
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
            flags.a_vix = flags.a_vix if flags.a_vix != "ok" else "missing or stale"
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
        """Extra vol push beyond the baseline regime."""
        vix = _clamp(fv.a_vix, 0.0, 200.0, 0.0)
        realised = _clamp(fv.realized_vol_annual, 0.0, 200.0, 0.0)

        if vix > 0 and realised > 0:
            gap = (vix - realised) / 30.0
        elif vix > 0:
            gap = (vix - 16.0) / 30.0
        elif realised > 0:
            gap = (realised - 16.0) / 30.0
        else:
            gap = 0.0

        gap = _clamp(gap, -1.0, 1.0, 0.0)
        high = gap * 0.30
        low = -gap * 0.20
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _catalyst_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Scheduled high-impact events increase tail probability."""
        score = _clamp(fv.catalyst_score, 0.0, 5.0, 0.0)
        high = (score / 5.0) * 0.45
        low = -high * 0.70
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _alignment_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Strong directional alignment among US futures, commodities and FX."""
        magnitude = _clamp(fv.cross_asset_magnitude, 0.0, 5.0, 0.0)
        alignment = _clamp(fv.cross_asset_alignment_score, -1.0, 1.0, 0.0)

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
        """SPI futures basis and momentum confirm / deny the cash move."""
        basis = _clamp(fv.spi_basis_pct, -2.0, 2.0, 0.0)
        momentum = _clamp(fv.spi_momentum_pct, -3.0, 3.0, 0.0)
        combined = (basis + momentum) / 2.0
        high = _clamp(combined, -0.30, 0.30, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _financials_vs_materials_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Banks vs Miners relative strength tilts the positive-move distribution."""
        score = _clamp(fv.financials_vs_materials_score, -2.0, 2.0, 0.0)
        high = _clamp(score / 2.5, -0.35, 0.35, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _housing_credit_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Housing/credit pulse: strong pulse lifts tail probability."""
        score = _clamp(fv.housing_credit_pulse_score, -2.0, 2.0, 0.0)
        high = _clamp(score / 3.0, -0.30, 0.30, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _china_steel_property_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """China steel/property proxy affects resources-exposed ASX tail."""
        score = _clamp(fv.china_steel_property_score, -2.5, 2.5, 0.0)
        high = _clamp(score / 2.5, -0.40, 0.40, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _heavyweight_idio_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """CBA + BHP idiosyncratic move can drive the index."""
        score = _clamp(fv.heavyweight_idio_score, -2.5, 2.5, 0.0)
        high = _clamp(score / 2.5, -0.40, 0.40, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _direction_score(self, fv: FeatureVector) -> float:
        """Estimate the signed probability of a down day (< 0%)."""
        score = 0.0

        alignment = _clamp(fv.cross_asset_alignment_score, -1.0, 1.0, 0.0)
        score -= 0.40 * alignment

        session_return = _clamp(fv.asx_open_to_now_return_pct, -2.0, 2.0, 0.0)
        score -= 0.20 * (session_return / 0.5)

        spi_combined = 0.0
        if fv.spi_basis_pct is not None:
            spi_combined += _clamp(fv.spi_basis_pct, -2.0, 2.0, 0.0)
        if fv.spi_momentum_pct is not None:
            spi_combined += _clamp(fv.spi_momentum_pct, -3.0, 3.0, 0.0)
        score -= 0.12 * _clamp(spi_combined / 2.0, -1.0, 1.0, 0.0)

        vix_change = _clamp(fv.vix_change_pct, -10.0, 10.0, 0.0)
        score -= 0.04 * _clamp(vix_change / 5.0, -1.0, 1.0, 0.0)

        us10y_change = _clamp(fv.us_10y_change_bps, -20.0, 20.0, 0.0)
        score -= 0.04 * _clamp(us10y_change / 20.0, -1.0, 1.0, 0.0)

        # New high-priority factors
        fvm = _clamp(fv.financials_vs_materials_score, -2.0, 2.0, 0.0)
        score -= 0.14 * fvm

        hc = _clamp(fv.housing_credit_pulse_score, -2.0, 2.0, 0.0)
        score -= 0.12 * hc

        china = _clamp(fv.china_steel_property_score, -2.5, 2.5, 0.0)
        score -= 0.10 * china

        hw = _clamp(fv.heavyweight_idio_score, -2.5, 2.5, 0.0)
        score -= 0.10 * hw

        return _clamp(score, -3.0, 3.0, 0.0)

    def _compute_confidence(self, probs: np.ndarray, fv: FeatureVector) -> float:
        """Confidence = distance from uniform, penalised by missing/stale data."""
        max_prob = float(np.max(probs))
        base_conf = (max_prob - 1.0 / 4.0) / (3.0 / 4.0)
        base_conf = max(0.0, min(1.0, base_conf))

        def _status(s: Any) -> str | None:
            if isinstance(s, DataSourceStatus):
                return s.status
            if isinstance(s, dict):
                return s.get("status")
            return getattr(s, "status", None)

        flag_count = sum(
            1 for s in (fv.source_status or []) if _status(s) not in ("ok", None)
        )
        penalty = min(flag_count * 0.05, 0.25)
        return max(0.2, min(1.0, base_conf - penalty))

    def _degraded_status(
        self, fv: FeatureVector, flags: DataQualityFlags
    ) -> tuple[bool, list[str]]:
        """Return whether the prediction is degraded and the list of problem sources."""
        degraded_sources: list[str] = []
        for s in fv.source_status or []:
            if isinstance(s, DataSourceStatus):
                status = s.status
                name = s.name
            elif isinstance(s, dict):
                status = s.get("status")
                name = s.get("name", "unknown")
            else:
                status = getattr(s, "status", None)
                name = getattr(s, "name", "unknown")
            if status in ("failed", "stale"):
                degraded_sources.append(name or "unknown")
        # Also include flags that are not ok
        for k, v in flags.model_dump().items():
            if v != "ok" and k not in degraded_sources:
                degraded_sources.append(k)
        return bool(degraded_sources), degraded_sources

    def _build_factor_contributions(
        self,
        fv: FeatureVector,
        vol_high: float,
        cat_high: float,
        align_high: float,
        sess_high: float,
        spi_high: float,
        fvm_high: float,
        hc_high: float,
        china_high: float,
        hw_high: float,
    ) -> list[FactorContribution]:
        """Build the human-readable factor contribution list."""
        # 1. US Equity Lead
        us_changes = [
            ("S&P", fv.sp500_change_pct),
            ("Nasdaq", fv.nasdaq_change_pct),
            ("Dow", fv.dow_change_pct),
            ("US futures", fv.us_futures_change_pct),
        ]
        us_values = [v for _, v in us_changes if v is not None]
        us_avg = sum(us_values) / len(us_values) if us_values else None
        us_note = ", ".join(f"{n}={_fmt_pct(v)}" for n, v in us_changes if v is not None)
        us_note = us_note or "No US equity data"
        us_score = _clamp(us_avg, -2.0, 2.0, 0.0) if us_avg is not None else 0.0

        # 2. SPI futures bias
        spi_basis = _clamp(fv.spi_basis_pct, -2.0, 2.0, 0.0)
        spi_momentum = _clamp(fv.spi_momentum_pct, -3.0, 3.0, 0.0)
        spi_combined = (
            (spi_basis + spi_momentum) / 2.0
            if (fv.spi_basis_pct is not None or fv.spi_momentum_pct is not None)
            else None
        )
        spi_note = f"basis {_fmt_pct(fv.spi_basis_pct)}, momentum {_fmt_pct(fv.spi_momentum_pct)}"
        spi_score = _clamp(spi_combined, -1.0, 1.0, 0.0) if spi_combined is not None else 0.0

        # 3. Iron ore
        iron = _clamp(fv.iron_ore_change_pct, -10.0, 10.0, 0.0)

        # 4. Gold & Silver
        if fv.gold_change_pct is not None or fv.silver_change_pct is not None:
            values = [v for v in [fv.gold_change_pct, fv.silver_change_pct] if v is not None]
            pm_avg = sum(values) / len(values) if values else 0.0
            pm_note = (
                f"gold {_fmt_pct(fv.gold_change_pct)}, silver {_fmt_pct(fv.silver_change_pct)}"
            )
            pm_dir = _direction_label(pm_avg)
        else:
            pm_avg = None
            pm_note = "No precious metals data"
            pm_dir = "neutral"

        # 5. AUD/USD
        aud = _clamp(fv.aud_usd_change_pct, -5.0, 5.0, 0.0)

        # 6. Volatility
        realized = f"{fv.realized_vol_annual:.1f}%" if fv.realized_vol_annual is not None else "n/a"
        vol_note = f"A-VIX {fv.a_vix or 'n/a'}, realised {realized}; regime {fv.vol_regime}"
        # High volatility is treated as a tail/negative tail risk for equities.
        vol_dir = "bearish" if (fv.vol_regime or 0) >= 2 else "bullish"

        # 7. Catalyst
        cat = _clamp(fv.catalyst_score, 0.0, 5.0, 0.0)

        # 8. ASX session
        session_ret = _clamp(fv.asx_open_to_now_return_pct, -5.0, 5.0, 0.0)
        session_date = (fv.sources or {}).get("asx_session_date", "latest session")
        session_fallback = (fv.sources or {}).get("asx_session_fallback")
        session_return_fmt = _fmt_pct(fv.asx_open_to_now_return_pct)
        session_note = (
            f"{fv.asx_session_character} session, return {session_return_fmt} ({session_date})"
        )
        if session_fallback:
            session_note += f" [{session_fallback}]"

        # 9. Overall alignment
        alignment = _clamp(fv.cross_asset_alignment_score, -1.0, 1.0, 0.0)
        magnitude = _clamp(fv.cross_asset_magnitude, 0.0, 5.0, 0.0)

        return [
            FactorContribution(
                name="US Equity Lead (S&P / Nasdaq / Dow overnight move)",
                raw_value=us_avg,
                raw_unit="%",
                direction=_direction_label(us_avg if us_avg is not None else 0.0),
                weight=round(self.weights["alignment"], 4),
                score=round(us_score * self.weights["alignment"], 4),
                note=us_note,
            ),
            FactorContribution(
                name="SPI 200 Futures bias",
                raw_value=spi_combined,
                raw_unit="%",
                direction=_direction_label(spi_score),
                weight=round(self.weights["spi_basis"], 4),
                score=round(spi_score * self.weights["spi_basis"], 4),
                note=spi_note,
            ),
            FactorContribution(
                name="Iron Ore change",
                raw_value=fv.iron_ore_change_pct,
                raw_unit="%",
                direction=_direction_label(iron),
                weight=round(self.weights["alignment"] / 3.0, 4),
                score=round(iron / 2.0 * self.weights["alignment"], 4),
                note=(
                    f"Iron ore {_fmt_pct(fv.iron_ore_change_pct)};"
                    " positive supports materials, negative weighs"
                ),
            ),
            FactorContribution(
                name="Gold & Silver change",
                raw_value=pm_avg,
                raw_unit="%",
                direction=pm_dir,
                weight=round(self.weights["alignment"] / 3.0, 4),
                score=round((pm_avg or 0.0) / 2.0 * self.weights["alignment"], 4),
                note=pm_note,
            ),
            FactorContribution(
                name="AUD/USD move",
                raw_value=fv.aud_usd_change_pct,
                raw_unit="%",
                direction=_direction_label(aud),
                weight=round(self.weights["alignment"] / 3.0, 4),
                score=round(aud / 2.0 * self.weights["alignment"], 4),
                note=(
                    f"AUD/USD {_fmt_pct(fv.aud_usd_change_pct)};"
                    " rising AUD is pro-growth / pro-ASX"
                ),
            ),
            FactorContribution(
                name="A-VIX / Volatility Regime",
                raw_value=fv.a_vix,
                raw_unit="index",
                direction=vol_dir,
                weight=round(self.weights["volatility"], 4),
                score=round(vol_high * self.weights["volatility"], 4),
                note=vol_note,
            ),
            FactorContribution(
                name="Catalyst Score (economic calendar)",
                raw_value=cat,
                raw_unit="0-5",
                direction="bearish" if cat >= 3 else "neutral",
                weight=round(self.weights["catalyst"], 4),
                score=round(cat / 5.0 * self.weights["catalyst"], 4),
                note=(
                    f"{int(fv.high_impact_events_next_24h or 0)} high-impact event(s) in next 24h,"
                    f" {int(fv.high_impact_events_next_48h or 0)} in 48h"
                ),
            ),
            FactorContribution(
                name="Current-day / last-session ASX character",
                raw_value=session_ret,
                raw_unit="%",
                direction=_direction_label(session_ret),
                weight=round(self.weights["session"], 4),
                score=round(sess_high * self.weights["session"], 4),
                note=session_note,
            ),
            FactorContribution(
                name="Financials vs Materials Relative Strength",
                raw_value=fv.financials_minus_materials_weighted_pct,
                raw_unit="%",
                direction=_direction_label(fv.financials_vs_materials_score or 0.0),
                weight=round(self.weights["financials_vs_materials"], 4),
                score=round(fvm_high * self.weights["financials_vs_materials"], 4),
                note=(
                    (
                        f"1d {fv.financials_minus_materials_1d_pct:+.2f}%, "
                        f"3d {fv.financials_minus_materials_3d_pct:+.2f}%, "
                        f"5d {fv.financials_minus_materials_5d_pct:+.2f}%"
                    )
                    if fv.financials_minus_materials_1d_pct is not None
                    else "No Financials/Materials data"
                ),
            ),
            FactorContribution(
                name="Housing & Credit Pulse",
                raw_value=fv.housing_credit_pulse_score,
                raw_unit="0-10",
                direction=_direction_label((fv.housing_credit_pulse_score or 5.0) - 5.0),
                weight=round(self.weights["housing_credit"], 4),
                score=round(hc_high * self.weights["housing_credit"], 4),
                note=(
                    f"pulse {fv.housing_credit_pulse_score:.1f}/10 via "
                    f"{', '.join(fv.housing_credit_pulse_sources or ['proxies'])}"
                    if fv.housing_credit_pulse_score is not None
                    else "No housing/credit proxy data"
                ),
            ),
            FactorContribution(
                name="China Steel / Property Pulse",
                raw_value=fv.china_steel_property_return_pct,
                raw_unit="%",
                direction=_direction_label(fv.china_steel_property_score or 0.0),
                weight=round(self.weights["china_steel_property"], 4),
                score=round(china_high * self.weights["china_steel_property"], 4),
                note=(
                    f"composite {fv.china_steel_property_return_pct:+.2f}% "
                    f"({', '.join(fv.china_steel_property_sources or ['proxies'])})"
                    if fv.china_steel_property_return_pct is not None
                    else "No China steel/proxy data"
                ),
            ),
            FactorContribution(
                name="Heavyweight Idiosyncratic Score – CBA + BHP",
                raw_value=fv.heavyweight_idio_return_pct,
                raw_unit="%",
                direction=_direction_label(fv.heavyweight_idio_score or 0.0),
                weight=round(self.weights["heavyweight_idio"], 4),
                score=round(hw_high * self.weights["heavyweight_idio"], 4),
                note=(
                    (
                        f"CBA/BHP weighted {fv.heavyweight_idio_return_pct:+.2f}%"
                        + (
                            f", news boost +{fv.heavyweight_idio_news_boost:.0%}"
                            if fv.heavyweight_idio_news_boost
                            else ", no major idiosyncratic news boost"
                        )
                    )
                    if fv.heavyweight_idio_return_pct is not None
                    else "No CBA/BHP data"
                ),
            ),
            FactorContribution(
                name="Overall Alignment Score",
                raw_value=fv.cross_asset_alignment_score,
                raw_unit="score",
                direction=_direction_label(alignment),
                weight=round(self.weights["alignment"], 4),
                score=round(align_high * self.weights["alignment"], 4),
                note=f"Alignment {alignment:+.2f}, cross-asset magnitude {magnitude:.2f}%",
            ),
        ]


def bucket_from_return(return_pct: float) -> str:
    """Map an actual signed return to its bucket label."""
    if return_pct < 0.0:
        return BUCKET_LABELS[0]
    if return_pct < 0.3:
        return BUCKET_LABELS[1]
    if return_pct <= 0.5:
        return BUCKET_LABELS[2]
    return BUCKET_LABELS[3]
