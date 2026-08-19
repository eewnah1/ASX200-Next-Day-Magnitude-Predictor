"""Two-model rule-based scoring engine for ASX200 next-day moves.

Design notes
------------
* Model 1 (Primary) is a high-conviction large-move classifier with exact
  weights and strict gating rules. It outputs Large Down / Neutral / Large Up.
* Model 2 (Secondary) activates only when Model 1 is Neutral and extracts a
  mild directional bias from short-term inputs. It outputs Mild Bearish Bias /
  Mild Bullish Bias / True Neutral.
* A legacy four-bucket probability distribution (negative, low, mid, high) is
  retained for calibration display compatibility. Missing values are filled
  with neutral defaults and flagged.
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
from asx200_mag_predictor.scoring.daily_rates_overlay import evaluate_high_conviction
from asx200_mag_predictor.scoring.features import (
    score_china_steel_property,
    score_financials_vs_materials,
    score_heavyweight_idio,
    score_housing_credit_pulse,
)
from asx200_mag_predictor.scoring.ml import HybridML
from asx200_mag_predictor.timezone import now_sydney

logger = get_logger(__name__)

BUCKET_LABELS = ["<0%", "0%-0.3%", "0.3%-0.5%", ">0.5%"]
BUCKET_KEYS = ["negative", "low", "mid", "high"]

PRIMARY_BUCKETS = ["Large Down", "Neutral", "Large Up"]
SECONDARY_BUCKETS = ["Mild Bearish Bias", "True Neutral", "Mild Bullish Bias"]

# Exact Model 1 weights (sum to 1.00). Each weight is multiplied by a signed
# score in the [-3.0, +3.0] range so the maximum possible primary score is ±3.0.
# Weights reduced slightly to make room for TradingView MCP-derived factors.
PRIMARY_WEIGHTS: dict[str, float] = {
    "rsi": 0.10,
    "ath_distance": 0.09,
    "financials_vs_materials": 0.10,
    "iron_ore": 0.08,
    "momentum_exhaustion": 0.08,
    "housing_credit": 0.07,
    "heavyweight_idio": 0.07,
    "us_equity_lead": 0.05,
    "china_steel_property": 0.06,
    "gold_silver": 0.02,
    "spi": 0.03,
    "a_vix": 0.01,
    # TradingView MCP enrichment
    "tv_xjo_trend": 0.05,
    "tv_financials_vs_materials": 0.04,
    "tv_heavyweight": 0.03,
    "tv_asian": 0.03,
    "tv_commodity": 0.03,
    "alpha_vantage_global_lead": 0.03,
    "rba_rates": 0.04,
}

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


def _clamp(value: float | None, low: float, high: float, default: float = 0.0) -> float:
    if value is None or math.isnan(value):
        return default
    return max(low, min(high, value))


def _direction_label(value: float, bullish: str = "bullish", bearish: str = "bearish") -> str:
    if value > 0.1:
        return bullish
    if value < -0.1:
        return bearish
    return "neutral"


def _fmt_pct(value: float | None, unit: str = "%") -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}{unit}"


def _china_note(fv: FeatureVector, composite: float | None) -> str:
    if composite is None:
        return "No China steel/proxy data"
    components = fv.tv_china_steel_property_components or {}
    parts = [f"{k}={v:+.2f}%" for k, v in components.items()]
    tv = fv.tv_china_steel_property_return_pct
    base = f"composite {composite:+.2f}%"
    if parts:
        note = f"{base} (TV composite {tv:+.2f}%"
        if components:
            note += "; " + ", ".join(parts)
        note += ")"
        return note
    if tv is not None:
        return f"{base} (TV composite {tv:+.2f}%)"
    return base


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
            "rsi": _clamp(self.settings.rsi_weight, 0.0, 1.0, 0.10),
            "ath_distance": _clamp(self.settings.ath_distance_weight, 0.0, 1.0, 0.10),
            "momentum_exhaustion": _clamp(self.settings.momentum_exhaustion_weight, 0.0, 1.0, 0.08),
            "bollinger": _clamp(self.settings.bollinger_weight, 0.0, 1.0, 0.05),
        }
        # Normalise weights so they behave like relative allocations.
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.temperature = 0.85
        self.hybrid_ml = HybridML(self.settings.ml_model_dir, self.settings)

    def predict(
        self,
        features: FeatureVector | dict[str, Any],
        data_quality_flags: DataQualityFlags | None = None,
        prediction_for: Any | None = None,
        in_position: bool = False,
    ) -> Prediction:
        """Return a two-model prediction for the supplied feature set."""
        fv = self._coerce(features)
        flags = data_quality_flags or DataQualityFlags()
        fv, flags = self._fill_missing(fv, flags)

        prediction_for_date = self._resolve_prediction_for(prediction_for)

        # If the SPI source is degraded, zero its numeric features so it cannot
        # move the legacy probability model or the primary/secondary factors.
        if flags.spi_futures not in ("ok", None):
            fv = fv.model_copy(
                update={
                    "spi_basis_pct": None,
                    "spi_momentum_pct": None,
                    "spi_short_term_momentum_pct": None,
                }
            )

        # TradingView MCP enrichment is optional; zero the features if degraded so
        # stale or partial snapshots cannot bias the legacy/ML models.
        if flags.tradingview not in ("ok", None):
            fv = fv.model_copy(
                update={
                    "tv_xjo_daily_score": None,
                    "tv_xjo_weekly_score": None,
                    "tv_xjo_trend_score": None,
                    "tv_financials_vs_materials_score": None,
                    "tv_financials_minus_materials_pct": None,
                    "tv_heavyweight_avg_score": None,
                    "tv_asian_session_change_pct": None,
                    "tv_commodity_basket_change_pct": None,
                    "tv_commodity_basket_ex_gold_change_pct": None,
                    "tv_commodity_vs_gold_change_pct": None,
                }
            )

        # Alpha Vantage MCP enrichment is optional; zero features if degraded so the
        # legacy/ML models are not biased by stale or partial API data.
        if flags.alpha_vantage not in ("ok", None):
            fv = fv.model_copy(
                update={
                    "av_aud_usd_change_pct": None,
                    "av_spy_change_pct": None,
                    "av_qqq_change_pct": None,
                    "av_gld_change_pct": None,
                    "av_vixy_change_pct": None,
                    "av_us_10y_yield_change_bps": None,
                    "av_us_10y_yield_level": None,
                }
            )

        # RBA / Australian rates expectations are optional; zero if degraded.
        if flags.rba_rates not in ("ok", None):
            fv = fv.model_copy(
                update={
                    "rba_cash_rate_expected_pct": None,
                    "rba_cash_rate_change_bps": None,
                    "au_3y_yield_pct": None,
                    "au_3y_yield_change_bps": None,
                    "au_10y_yield_pct": None,
                    "au_10y_yield_change_bps": None,
                    "rba_rates_score": None,
                }
            )

        # Legacy four-bucket probability distribution is retained for display.
        probs, factor_breakdown = self._legacy_probabilities(fv)

        # Model 1: primary high-conviction large-move classifier (rule + ML hybrid).
        primary_contributions, primary_score, primary_bucket_rule = self._primary_model(fv, flags)
        ml_primary_probs = self.hybrid_ml.primary_probs(fv)
        primary_bucket, primary_fallback = self._hybrid_primary_bucket(
            primary_score, ml_primary_probs, fv, primary_bucket_rule
        )

        # Daily-rates empirical high-conviction overlay.
        hc_signal = evaluate_high_conviction(fv)
        hc_kwargs: dict[str, Any] = {
            "high_conviction": False,
            "high_conviction_bucket": None,
            "high_conviction_historical_accuracy": None,
            "high_conviction_reason": None,
        }
        if hc_signal is not None:
            primary_bucket = hc_signal.bucket
            primary_score = 3.5 if hc_signal.bucket == "Large Up" else -3.5
            ml_primary_probs = {
                "Large Up": 0.95 if hc_signal.bucket == "Large Up" else 0.0,
                "Neutral": 0.05,
                "Large Down": 0.95 if hc_signal.bucket == "Large Down" else 0.0,
            }
            primary_contributions.append(
                FactorContribution(
                    name="Daily Rates Overlay",
                    raw_value=None,
                    raw_unit="",
                    direction="bullish" if hc_signal.bucket == "Large Up" else "bearish",
                    weight=1.0,
                    score=primary_score,
                    note=hc_signal.reason,
                    group="Overlay",
                )
            )
            notes_for_hc = f"High-conviction daily-rates overlay: {hc_signal.reason}"
            hc_kwargs = {
                "high_conviction": True,
                "high_conviction_bucket": hc_signal.bucket,
                "high_conviction_historical_accuracy": hc_signal.historical_accuracy,
                "high_conviction_reason": hc_signal.reason,
            }
        else:
            notes_for_hc = ""

        # Model 2: secondary neutral-zone bias extractor (only if primary is Neutral).
        secondary_contributions: list[FactorContribution] = []
        secondary_score = 0.0
        secondary_bucket: str | None = None
        ml_secondary_probs: dict[str, float] | None = None
        secondary_fallback: str | None = None
        if primary_bucket == "Neutral":
            secondary_contributions, secondary_score, secondary_bucket_rule = self._secondary_model(
                fv, primary_bucket, flags
            )
            ml_secondary_probs = self.hybrid_ml.secondary_probs(fv)
            secondary_bucket, secondary_fallback = self._hybrid_secondary_bucket(
                secondary_score, ml_secondary_probs, secondary_bucket_rule
            )

        factor_contributions = primary_contributions + secondary_contributions

        # The primary bucket is used as the headline bucket.
        # The UI will display the secondary bucket when Model 2 is active.
        model = self._active_model_name(primary_bucket, secondary_bucket)
        recommendation, recommendation_source, confidence = self._build_recommendation(
            primary_bucket=primary_bucket,
            primary_score=primary_score,
            ml_primary_probs=ml_primary_probs,
            secondary_bucket=secondary_bucket,
            secondary_score=secondary_score,
            ml_secondary_probs=ml_secondary_probs,
            fv=fv,
            in_position=in_position,
        )
        degraded, degraded_sources = self._degraded_status(fv, flags)

        ml_available = self.hybrid_ml.available and ml_primary_probs is not None
        fallback_reason = None
        if not ml_available:
            fallback_reason = (
                primary_fallback or secondary_fallback or "ML models not trained or unavailable"
            )

        notes: list[str] = []
        if notes_for_hc:
            notes.append(notes_for_hc)
        notes.append(f"Active model: {model}; primary bucket: {primary_bucket}")
        if ml_available and ml_primary_probs:
            notes.append(
                f"ML primary probabilities: "
                f"Large Down {ml_primary_probs.get('Large Down', 0):.0%}, "
                f"Neutral {ml_primary_probs.get('Neutral', 0):.0%}, "
                f"Large Up {ml_primary_probs.get('Large Up', 0):.0%}"
            )
        if primary_bucket != "Neutral":
            notes.append(
                f"Primary model {primary_bucket} (rule score {primary_score:.2f}); "
                f"gating RSI={_fmt_pct(fv.rsi_14, unit='index')}, "
                f"ATH={_fmt_pct(fv.ath_distance_pct)}"
            )
        if primary_bucket == "Neutral" and secondary_bucket and secondary_bucket != "True Neutral":
            notes.append(
                f"Secondary model selected {secondary_bucket} (rule score {secondary_score:.2f})."
            )
            if ml_available and ml_secondary_probs:
                notes.append(
                    f"ML secondary probabilities: "
                    f"Mild Bearish {ml_secondary_probs.get('Mild Bearish Bias', 0):.0%}, "
                    f"True Neutral {ml_secondary_probs.get('True Neutral', 0):.0%}, "
                    f"Mild Bullish {ml_secondary_probs.get('Mild Bullish Bias', 0):.0%}"
                )
        if fallback_reason:
            notes.append(f"ML fallback: {fallback_reason}; using rule-based output.")
        if degraded:
            notes.append(f"Degraded prediction – missing: {', '.join(degraded_sources)}.")
        notes.append(
            f"Long-only recommendation: {recommendation} "
            f"(source: {recommendation_source}, confidence: {confidence:.1%})"
        )
        if in_position and recommendation == "STAY IN CASH":
            notes.append(
                "Already in position – negative signal converted to HOLD EXISTING (no exit)."
            )

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
            bucket=primary_bucket,
            confidence=round(confidence, 4),
            factor_breakdown=factor_breakdown,
            factor_contributions=factor_contributions,
            notes=notes,
            data_quality_flags=flags,
            source_status=fv.source_status,
            errors=fv.errors,
            degraded=degraded,
            degraded_sources=degraded_sources,
            model=model,
            primary_bucket=primary_bucket,
            secondary_bucket=secondary_bucket,
            primary_score=round(primary_score, 4),
            secondary_score=round(secondary_score, 4),
            ml_available=ml_available,
            ml_probabilities={
                "primary": ml_primary_probs or {},
                "secondary": ml_secondary_probs or {},
            },
            ml_feature_importance=self.hybrid_ml.feature_importance(top=10),
            ml_fallback_reason=fallback_reason,
            recommendation=recommendation,
            recommendation_source=recommendation_source,
            recommendation_confidence=round(confidence, 4),
            in_position=in_position,
            **hc_kwargs,
        )

    # ------------------------------------------------------------------ helpers

    def _legacy_probabilities(self, fv: FeatureVector) -> tuple[np.ndarray, FactorBreakdown]:
        """Compute the legacy four-bucket signed probability distribution."""
        vol_delta, vol_high = self._volatility_delta(fv)
        cat_delta, cat_high = self._catalyst_delta(fv)
        align_delta, align_high = self._alignment_delta(fv)
        sess_delta, sess_high = self._session_delta(fv)
        spi_delta, spi_high = self._spi_delta(fv)
        fvm_delta, fvm_high = self._financials_vs_materials_delta(fv)
        hc_delta, hc_high = self._housing_credit_delta(fv)
        china_delta, china_high = self._china_steel_property_delta(fv)
        hw_delta, hw_high = self._heavyweight_idio_delta(fv)
        rsi_delta, rsi_high = self._rsi_delta(fv)
        ath_delta, ath_high = self._ath_distance_delta(fv)
        mom_delta, mom_high = self._momentum_exhaustion_delta(fv)
        boll_delta, boll_high = self._bollinger_delta(fv)

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
        combined += self.weights["rsi"] * rsi_delta
        combined += self.weights["ath_distance"] * ath_delta
        combined += self.weights["momentum_exhaustion"] * mom_delta
        combined += self.weights["bollinger"] * boll_delta

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

        factor_breakdown = FactorBreakdown(
            volatility=round(float(vol_high) * self.weights["volatility"], 4),
            catalyst=round(float(cat_high) * self.weights["catalyst"], 4),
            alignment=round(float(align_high) * self.weights["alignment"], 4),
            session=round(float(sess_high) * self.weights["session"], 4),
            spi_basis=round(float(spi_high) * self.weights["spi_basis"], 4),
            direction=round(float(direction_score), 4),
        )
        return probs, factor_breakdown

    def _primary_model(
        self, fv: FeatureVector, flags: DataQualityFlags | None = None
    ) -> tuple[list[FactorContribution], float, str]:
        """Model 1: high-conviction large-move classifier with exact weights."""
        flags = flags or DataQualityFlags()
        contributions: list[FactorContribution] = []
        total = 0.0

        def add(
            name: str,
            raw_value: float | None,
            raw_unit: str,
            raw_score: float,
            weight: float,
            note: str,
        ) -> None:
            nonlocal total
            weighted = round(_clamp(raw_score, -3.0, 3.0, 0.0) * weight, 4)
            total += weighted
            contributions.append(
                FactorContribution(
                    name=name,
                    raw_value=raw_value,
                    raw_unit=raw_unit,
                    direction=_direction_label(_clamp(raw_score, -3.0, 3.0, 0.0)),
                    weight=round(weight, 4),
                    score=weighted,
                    note=note,
                    group="Primary",
                )
            )

        # 1. Financials vs Materials Relative Strength (12%)
        fvm_raw = fv.financials_minus_materials_weighted_pct
        fvm_score = _clamp((fv.financials_vs_materials_score or 0.0) * 2.0, -3.0, 3.0)
        fvm_note = (
            f"1d {_fmt_pct(fv.financials_minus_materials_1d_pct)}, "
            f"3d {_fmt_pct(fv.financials_minus_materials_3d_pct)}, "
            f"5d {_fmt_pct(fv.financials_minus_materials_5d_pct)}"
            if fvm_raw is not None
            else "No Financials/Materials data"
        )
        add(
            "Financials vs Materials Relative Strength",
            fvm_raw,
            "%",
            fvm_score,
            PRIMARY_WEIGHTS["financials_vs_materials"],
            fvm_note,
        )

        # 2. Iron Ore change (10%)
        iron = fv.iron_ore_change_pct
        iron_score = _clamp((iron or 0.0) / 1.2, -3.0, 3.0)
        add(
            "Iron Ore change",
            iron,
            "%",
            iron_score,
            PRIMARY_WEIGHTS["iron_ore"],
            f"Iron ore {_fmt_pct(iron)}; positive supports materials, negative weighs",
        )

        # 3. RSI (14) Overbought/Oversold (12%)
        rsi = fv.rsi_14
        rsi_score_val = _clamp((fv.rsi_score or 0.0) * 2.0, -3.0, 3.0)
        add(
            "RSI (14) Overbought / Oversold",
            rsi,
            "index",
            rsi_score_val,
            PRIMARY_WEIGHTS["rsi"],
            f"RSI {rsi:.1f}" if rsi is not None else "No RSI data",
        )

        # 4. Distance from All-Time High (11%)
        ath = fv.ath_distance_pct
        ath_score_val = _clamp(fv.ath_score or 0.0, -1.5, 0.5)
        add(
            "Distance from All-Time High",
            ath,
            "%",
            ath_score_val,
            PRIMARY_WEIGHTS["ath_distance"],
            (
                f"ATH distance {_fmt_pct(ath)}; 20d high {_fmt_pct(fv.high_20d_distance_pct)}, "
                f"50d high {_fmt_pct(fv.high_50d_distance_pct)}"
                if ath is not None
                else "No price history"
            ),
        )

        # 5. Housing & Credit Pulse (8%)
        hc = fv.housing_credit_pulse_score
        hc_score_val = _clamp((score_housing_credit_pulse(hc) or 0.0) * 2.0, -3.0, 3.0)
        add(
            "Housing & Credit Pulse",
            hc,
            "0-10",
            hc_score_val,
            PRIMARY_WEIGHTS["housing_credit"],
            (f"pulse {hc:.1f}/10" if hc is not None else "No housing/credit proxy data"),
        )

        # 6. Heavyweight Idiosyncratic Score – CBA + BHP (8%)
        hw = fv.heavyweight_idio_return_pct
        hw_score_val = _clamp(
            (score_heavyweight_idio(hw, fv.heavyweight_idio_news_boost) or 0.0) * 2.0,
            -3.0,
            3.0,
        )
        hw_note = (
            f"CBA+BHP weighted {hw:+.2f}%"
            + (
                f", news boost +{fv.heavyweight_idio_news_boost:.0%}"
                if fv.heavyweight_idio_news_boost
                else ", no major idiosyncratic news boost"
            )
            if hw is not None
            else "No CBA/BHP data"
        )
        add(
            "Heavyweight Idiosyncratic Score – CBA + BHP",
            hw,
            "%",
            hw_score_val,
            PRIMARY_WEIGHTS["heavyweight_idio"],
            hw_note,
        )

        # 7. US Equity Lead (8%)
        us_changes = [
            ("S&P", fv.sp500_change_pct),
            ("Nasdaq", fv.nasdaq_change_pct),
            ("Dow", fv.dow_change_pct),
            ("US futures", fv.us_futures_change_pct),
        ]
        us_values = [v for _, v in us_changes if v is not None]
        us_avg = sum(us_values) / len(us_values) if us_values else 0.0
        us_note = ", ".join(f"{n}={_fmt_pct(v)}" for n, v in us_changes if v is not None)
        us_score = _clamp(us_avg / 1.0, -3.0, 3.0)
        add(
            "US Equity Lead (S&P / Nasdaq / Dow overnight move)",
            us_avg,
            "%",
            us_score,
            PRIMARY_WEIGHTS["us_equity_lead"],
            us_note or "No US equity data",
        )

        # 8. Short-term Momentum Exhaustion (10%)
        mom = fv.index_5d_return_pct
        mom_score_val = _clamp(
            (fv.momentum_exhaustion_score or 0.0) * 3.0
            + (fv.profit_taking_combo_score or 0.0) * 1.0,
            -3.0,
            3.0,
        )
        add(
            "Short-term Momentum Exhaustion",
            mom,
            "%",
            mom_score_val,
            PRIMARY_WEIGHTS["momentum_exhaustion"],
            (
                f"5d return {_fmt_pct(mom)}"
                + (
                    " – profit-taking combo triggered"
                    if fv.profit_taking_combo_score and fv.profit_taking_combo_score < 0
                    else ""
                )
                if mom is not None
                else "No 5d return data"
            ),
        )

        # 9. China Steel / Property Pulse (7%)
        china = fv.china_steel_property_return_pct
        china_score_val = _clamp((score_china_steel_property(china) or 0.0) * 2.5, -3.0, 3.0)
        add(
            "China Steel / Property Pulse",
            china,
            "%",
            china_score_val,
            PRIMARY_WEIGHTS["china_steel_property"],
            _china_note(fv, china),
        )

        # 10. Gold & Silver change (5%)
        if fv.gold_change_pct is not None or fv.silver_change_pct is not None:
            values = [v for v in [fv.gold_change_pct, fv.silver_change_pct] if v is not None]
            pm_avg = sum(values) / len(values) if values else 0.0
            pm_note = (
                f"gold {_fmt_pct(fv.gold_change_pct)}, silver {_fmt_pct(fv.silver_change_pct)}"
            )
        else:
            pm_avg = 0.0
            pm_note = "No precious metals data"
        pm_score = _clamp(pm_avg / 1.0, -3.0, 3.0)
        add(
            "Gold & Silver change",
            pm_avg,
            "%",
            pm_score,
            PRIMARY_WEIGHTS["gold_silver"],
            pm_note,
        )

        # 11. SPI 200 Futures bias (5%)
        if flags.spi_futures not in ("ok", None):
            spi_combined: float | None = None
            spi_score = 0.0
            spi_note = f"SPI source degraded ({flags.spi_futures}) — factor zeroed"
        else:
            spi_basis = _clamp(fv.spi_basis_pct, -2.0, 2.0, 0.0)
            spi_momentum = _clamp(fv.spi_momentum_pct, -3.0, 3.0, 0.0)
            spi_combined = (
                (spi_basis + spi_momentum) / 2.0
                if (fv.spi_basis_pct is not None or fv.spi_momentum_pct is not None)
                else 0.0
            )
            spi_score = _clamp(spi_combined / 0.5, -3.0, 3.0)
            spi_note = (
                f"basis {_fmt_pct(fv.spi_basis_pct)}, momentum {_fmt_pct(fv.spi_momentum_pct)}"
            )
        add(
            "SPI 200 Futures bias",
            spi_combined,
            "%",
            spi_score,
            PRIMARY_WEIGHTS["spi"],
            spi_note,
        )

        # 12. A-VIX / Volatility Regime (4%)
        a_vix = fv.a_vix
        vol_regime = fv.vol_regime or 1
        vix_score = _clamp(
            -((a_vix or 16.0) - 16.0) / 6.0 - (vol_regime - 1) * 0.4,
            -3.0,
            3.0,
        )
        add(
            "A-VIX / Volatility Regime",
            a_vix,
            "index",
            vix_score,
            PRIMARY_WEIGHTS["a_vix"],
            f"A-VIX {a_vix or 'n/a'}; regime {vol_regime}",
        )

        # 13-17. TradingView MCP enrichment (optional; zeroed if source degraded)
        if flags.tradingview not in ("ok", None):
            tv_note = f"TradingView source degraded ({flags.tradingview}) — factors zeroed"
            add(
                "TV XJO multi-timeframe trend",
                None,
                "score",
                0.0,
                PRIMARY_WEIGHTS["tv_xjo_trend"],
                tv_note,
            )
            add(
                "TV Financials vs Materials",
                None,
                "%",
                0.0,
                PRIMARY_WEIGHTS["tv_financials_vs_materials"],
                tv_note,
            )
            add(
                "TV Heavyweight consensus",
                None,
                "score",
                0.0,
                PRIMARY_WEIGHTS["tv_heavyweight"],
                tv_note,
            )
            add(
                "TV Asian session lead",
                None,
                "%",
                0.0,
                PRIMARY_WEIGHTS["tv_asian"],
                tv_note,
            )
            add(
                "TV Commodity basket vs gold",
                None,
                "%",
                0.0,
                PRIMARY_WEIGHTS["tv_commodity"],
                tv_note,
            )
        else:
            tv_trend = fv.tv_xjo_trend_score
            tv_trend_score = _clamp((tv_trend or 0.0) / 2.0, -3.0, 3.0)
            tv_daily = fv.tv_xjo_daily_score
            tv_weekly = fv.tv_xjo_weekly_score
            add(
                "TV XJO multi-timeframe trend",
                tv_trend,
                "score",
                tv_trend_score,
                PRIMARY_WEIGHTS["tv_xjo_trend"],
                (
                    f"daily {tv_daily}, weekly {tv_weekly}, decision={fv.tv_xjo_decision}"
                    if tv_trend is not None
                    else "No TradingView XJO consensus"
                ),
            )

            tv_fvm = fv.tv_financials_minus_materials_pct
            tv_fvm_score = _clamp(fv.tv_financials_vs_materials_score or 0.0, -3.0, 3.0)
            add(
                "TV Financials vs Materials",
                tv_fvm,
                "%",
                tv_fvm_score,
                PRIMARY_WEIGHTS["tv_financials_vs_materials"],
                f"TV Financials - Materials {_fmt_pct(tv_fvm)}"
                if tv_fvm is not None
                else "No TV sector data",
            )

            tv_hw = fv.tv_heavyweight_avg_score
            tv_hw_score = _clamp((tv_hw or 0.0) / 2.0, -3.0, 3.0)
            add(
                "TV Heavyweight consensus",
                tv_hw,
                "score",
                tv_hw_score,
                PRIMARY_WEIGHTS["tv_heavyweight"],
                f"CBA/BHP/RIO/FMG/WDS avg net score={tv_hw:.2f}"
                if tv_hw is not None
                else "No TV heavyweight data",
            )

            tv_asian = fv.tv_asian_session_change_pct
            tv_asian_score = _clamp((tv_asian or 0.0) / 0.5, -3.0, 3.0)
            add(
                "TV Asian session lead",
                tv_asian,
                "%",
                tv_asian_score,
                PRIMARY_WEIGHTS["tv_asian"],
                f"Nikkei/Hang Seng/STI/KOSPI avg {_fmt_pct(tv_asian)}"
                if tv_asian is not None
                else "No TV Asian data",
            )

            tv_comm = fv.tv_commodity_vs_gold_change_pct
            tv_comm_score = _clamp((tv_comm or 0.0) / 0.5, -3.0, 3.0)
            add(
                "TV Commodity basket vs gold",
                tv_comm,
                "%",
                tv_comm_score,
                PRIMARY_WEIGHTS["tv_commodity"],
                f"industrial basket vs gold {_fmt_pct(tv_comm)}"
                if tv_comm is not None
                else "No TV commodity data",
            )

        # 18. Alpha Vantage global cross-asset lead
        if flags.alpha_vantage not in ("ok", None):
            av_raw = None
            av_score = 0.0
            av_note = f"Alpha Vantage source degraded ({flags.alpha_vantage}) — factor zeroed"
        else:
            av_components: list[tuple[float, float]] = []
            if fv.av_aud_usd_change_pct is not None:
                av_components.append((fv.av_aud_usd_change_pct, 0.25))
            if fv.av_spy_change_pct is not None:
                av_components.append((fv.av_spy_change_pct, 0.30))
            if fv.av_qqq_change_pct is not None:
                av_components.append((fv.av_qqq_change_pct, 0.20))
            if fv.av_gld_change_pct is not None:
                av_components.append((fv.av_gld_change_pct, 0.10))
            if fv.av_vixy_change_pct is not None:
                # VIXY up = risk-off for ASX
                av_components.append((-fv.av_vixy_change_pct, 0.10))
            if fv.av_us_10y_yield_change_bps is not None:
                # rising US 10Y yields are a mild headwind in risk-off terms
                av_components.append((-fv.av_us_10y_yield_change_bps / 10.0, 0.05))
            if av_components:
                av_raw = (
                    sum(v * w for v, w in av_components) / sum(w for _, w in av_components)
                )
            else:
                av_raw = None
            av_score = _clamp((av_raw or 0.0) / 1.0, -3.0, 3.0)
            av_notes: list[str] = []
            if fv.av_aud_usd_change_pct is not None:
                av_notes.append(f"AUD/USD {_fmt_pct(fv.av_aud_usd_change_pct)}")
            if fv.av_spy_change_pct is not None:
                av_notes.append(f"SPY {_fmt_pct(fv.av_spy_change_pct)}")
            if fv.av_qqq_change_pct is not None:
                av_notes.append(f"QQQ {_fmt_pct(fv.av_qqq_change_pct)}")
            if fv.av_gld_change_pct is not None:
                av_notes.append(f"GLD {_fmt_pct(fv.av_gld_change_pct)}")
            if fv.av_vixy_change_pct is not None:
                av_notes.append(f"VIXY (inverted) {_fmt_pct(-fv.av_vixy_change_pct)}")
            if fv.av_us_10y_yield_change_bps is not None:
                av_notes.append(f"US 10Y {_fmt_pct(fv.av_us_10y_yield_change_bps, 'bps')}")
            av_note = "; ".join(av_notes) if av_notes else "No Alpha Vantage data"
        add(
            "Alpha Vantage global cross-asset lead",
            av_raw,
            "%",
            av_score,
            PRIMARY_WEIGHTS["alpha_vantage_global_lead"],
            av_note,
        )

        # 19. RBA / Australian rates expectations (key Financials engine driver)
        if flags.rba_rates not in ("ok", None):
            rba_raw = None
            rba_score = 0.0
            rba_note = f"RBA rates source degraded ({flags.rba_rates}) — factor zeroed"
        else:
            ib1 = fv.rba_cash_rate_change_bps
            yt1 = fv.au_3y_yield_change_bps
            xt1 = fv.au_10y_yield_change_bps
            if ib1 is not None:
                weighted_bps = 0.5 * ib1 + 0.3 * (yt1 or 0.0) + 0.2 * (xt1 or 0.0)
                rba_raw = weighted_bps
                rba_score = _clamp(-weighted_bps / 4.0, -3.0, 3.0)
                rba_note = (
                    f"IB1 cash expectation {fv.rba_cash_rate_expected_pct:.3f}% "
                    f"({ib1:+.2f} bps), 3Y {yt1:+.2f} bps, 10Y {xt1:+.2f} bps"
                )
            else:
                rba_raw = None
                rba_score = 0.0
                rba_note = "No ASX24 rates data"
        add(
            "RBA / Australian rates expectations",
            rba_raw,
            "bps",
            rba_score,
            PRIMARY_WEIGHTS["rba_rates"],
            rba_note,
        )

        # Strict high-conviction gating.
        rsi_for_gate = _clamp(fv.rsi_14, 0.0, 100.0, 50.0)
        ath_for_gate = _clamp(fv.ath_distance_pct, -50.0, 50.0, -5.0)
        iron_for_gate = fv.iron_ore_change_pct
        if total >= 2.3 and rsi_for_gate <= 65 and ath_for_gate <= -1.0:
            primary_bucket = "Large Up"
        elif (
            total <= -2.3
            and rsi_for_gate >= 68
            and (ath_for_gate >= -1.0 or (iron_for_gate is not None and iron_for_gate <= -0.6))
        ):
            primary_bucket = "Large Down"
        else:
            primary_bucket = "Neutral"

        return contributions, round(total, 4), primary_bucket

    def _secondary_model(
        self, fv: FeatureVector, primary_bucket: str, flags: DataQualityFlags | None = None
    ) -> tuple[list[FactorContribution], float, str | None]:
        """Model 2: neutral-zone mild-bias extractor. Activates only when primary is Neutral."""
        if primary_bucket != "Neutral":
            return [], 0.0, None

        flags = flags or DataQualityFlags()
        contributions: list[FactorContribution] = []
        total = 0.0
        bullish = 0
        bearish = 0

        def add(
            name: str,
            raw_value: float | None,
            raw_unit: str,
            raw_score: float,
            weight: float,
            note: str,
        ) -> None:
            nonlocal total, bullish, bearish
            score_clamped = _clamp(raw_score, -3.0, 3.0, 0.0)
            weighted = round(score_clamped * weight, 4)
            total += weighted
            if score_clamped > 0.2:
                bullish += 1
            elif score_clamped < -0.2:
                bearish += 1
            contributions.append(
                FactorContribution(
                    name=name,
                    raw_value=raw_value,
                    raw_unit=raw_unit,
                    direction=_direction_label(score_clamped),
                    weight=round(weight, 4),
                    score=weighted,
                    note=note,
                    group="Secondary",
                )
            )

        # 1. RSI(14) (14%)
        rsi = fv.rsi_14
        rsi_score = _clamp((fv.rsi_score or 0.0) * 2.0, -3.0, 3.0)
        slope = _clamp(fv.rsi_slope, -10.0, 10.0, 0.0)
        slope_score = _clamp(slope * 0.2, -1.0, 1.0)
        rsi_total = _clamp(rsi_score + slope_score, -3.0, 3.0)
        add(
            "RSI(14)",
            rsi,
            "index",
            rsi_total,
            0.14,
            (f"RSI {rsi:.1f}, slope {slope:+.2f}pt" if rsi is not None else "No RSI data"),
        )

        # 2. Short-term Momentum Exhaustion (11%)
        mom = fv.index_5d_return_pct
        mom_score = _clamp(
            ((fv.momentum_exhaustion_score or 0.0) + (fv.profit_taking_combo_score or 0.0)) * 1.5,
            -3.0,
            3.0,
        )
        add(
            "Short-term Momentum Exhaustion",
            mom,
            "%",
            mom_score,
            0.11,
            (
                f"5d return {_fmt_pct(mom)}; profit-taking combo"
                if mom is not None
                else "No 5d return data"
            ),
        )

        # 3. Financials vs Materials (1d & 2d) (13%)
        fvm_1d = fv.financials_minus_materials_1d_pct
        fvm_2d = fv.financials_minus_materials_2d_pct
        fvm_1d_score = _clamp((score_financials_vs_materials(fvm_1d) or 0.0) * 2.0, -3.0, 3.0)
        fvm_2d_score = _clamp((score_financials_vs_materials(fvm_2d) or 0.0) * 2.0, -3.0, 3.0)
        fvm_score = _clamp((fvm_1d_score + fvm_2d_score) / 2.0, -3.0, 3.0)
        fvm_note = (
            f"1d {_fmt_pct(fvm_1d)}, 2d {_fmt_pct(fvm_2d)}"
            if fvm_1d is not None or fvm_2d is not None
            else "No Financials/Materials short-term data"
        )
        add(
            "Financials vs Materials (1d & 2d)",
            fvm_1d,
            "%",
            fvm_score,
            0.13,
            fvm_note,
        )

        # 4. Distance from All-Time High (10%)
        ath = fv.ath_distance_pct
        ath_score_val = _clamp((fv.ath_score or 0.0) * 2.0, -3.0, 3.0)
        add(
            "Distance from All-Time High",
            ath,
            "%",
            ath_score_val,
            0.10,
            (f"ATH distance {_fmt_pct(ath)}" if ath is not None else "No price history"),
        )

        # 5. Iron Ore (9%)
        iron = fv.iron_ore_change_pct
        iron_score = _clamp((iron or 0.0) / 1.2, -3.0, 3.0)
        add(
            "Iron Ore",
            iron,
            "%",
            iron_score,
            0.09,
            f"Iron ore {_fmt_pct(iron)}",
        )

        # 6. Heavyweight (CBA + BHP) (8%)
        hw = fv.heavyweight_idio_return_pct
        hw_score_val = _clamp(
            (score_heavyweight_idio(hw, fv.heavyweight_idio_news_boost) or 0.0) * 2.0,
            -3.0,
            3.0,
        )
        hw_note = (
            f"CBA+BHP weighted {hw:+.2f}%"
            + (
                f", news boost +{fv.heavyweight_idio_news_boost:.0%}"
                if fv.heavyweight_idio_news_boost
                else ", no major idiosyncratic news boost"
            )
            if hw is not None
            else "No CBA/BHP data"
        )
        add(
            "Heavyweight (CBA + BHP)",
            hw,
            "%",
            hw_score_val,
            0.08,
            hw_note,
        )

        # 7. Housing & Credit (7%)
        hc = fv.housing_credit_pulse_score
        hc_score_val = _clamp((score_housing_credit_pulse(hc) or 0.0) * 2.0, -3.0, 3.0)
        add(
            "Housing & Credit",
            hc,
            "0-10",
            hc_score_val,
            0.07,
            (f"pulse {hc:.1f}/10" if hc is not None else "No housing/credit proxy data"),
        )

        # 8. US Equity Lead (7%)
        us_changes = [
            ("S&P", fv.sp500_change_pct),
            ("Nasdaq", fv.nasdaq_change_pct),
            ("Dow", fv.dow_change_pct),
            ("US futures", fv.us_futures_change_pct),
        ]
        us_values = [v for _, v in us_changes if v is not None]
        us_avg = sum(us_values) / len(us_values) if us_values else 0.0
        us_note = ", ".join(f"{n}={_fmt_pct(v)}" for n, v in us_changes if v is not None)
        us_score = _clamp(us_avg / 1.0, -3.0, 3.0)
        add(
            "US Equity Lead",
            us_avg,
            "%",
            us_score,
            0.07,
            us_note or "No US equity data",
        )

        # 9. SPI short-term (6%)
        if flags.spi_futures not in ("ok", None):
            spi_short: float | None = None
            spi_score = 0.0
            spi_note = f"SPI source degraded ({flags.spi_futures}) — factor zeroed"
        else:
            spi_short = fv.spi_short_term_momentum_pct
            if spi_short is None:
                spi_short = fv.spi_momentum_pct
            spi_score = _clamp((spi_short or 0.0) / 0.5, -3.0, 3.0)
            spi_note = "2-4 hour SPI futures bias (daily fallback when intraday unavailable)"
        add(
            "SPI short-term",
            spi_short,
            "%",
            spi_score,
            0.06,
            spi_note,
        )

        # 10. China Pulse (6%)
        china = fv.china_steel_property_return_pct
        china_score_val = _clamp((score_china_steel_property(china) or 0.0) * 2.5, -3.0, 3.0)
        add(
            "China Pulse",
            china,
            "%",
            china_score_val,
            0.06,
            (f"composite {china:+.2f}%" if china is not None else "No China steel/proxy data"),
        )

        # 11. Gold & Silver (5%)
        if fv.gold_change_pct is not None or fv.silver_change_pct is not None:
            values = [v for v in [fv.gold_change_pct, fv.silver_change_pct] if v is not None]
            pm_avg = sum(values) / len(values) if values else 0.0
            pm_note = (
                f"gold {_fmt_pct(fv.gold_change_pct)}, silver {_fmt_pct(fv.silver_change_pct)}"
            )
        else:
            pm_avg = 0.0
            pm_note = "No precious metals data"
        pm_score = _clamp(pm_avg / 1.0, -3.0, 3.0)
        add(
            "Gold & Silver",
            pm_avg,
            "%",
            pm_score,
            0.05,
            pm_note,
        )

        # 12. A-VIX (4%)
        a_vix = fv.a_vix
        vol_regime = fv.vol_regime or 1
        vix_score = _clamp(
            -((a_vix or 16.0) - 16.0) / 6.0 - (vol_regime - 1) * 0.4,
            -3.0,
            3.0,
        )
        add(
            "A-VIX",
            a_vix,
            "index",
            vix_score,
            0.04,
            f"A-VIX {a_vix or 'n/a'}; regime {vol_regime}",
        )

        # Resolve secondary bucket. Require strong alignment across several factors.
        if total >= 0.7 and bullish >= 5 and bearish <= 2:
            secondary_bucket = "Mild Bullish Bias"
        elif total <= -0.7 and bearish >= 5 and bullish <= 2:
            secondary_bucket = "Mild Bearish Bias"
        else:
            secondary_bucket = "True Neutral"

        return contributions, round(total, 4), secondary_bucket

    def _active_model_name(self, primary_bucket: str, secondary_bucket: str | None) -> str:
        if primary_bucket != "Neutral":
            return "Primary"
        if secondary_bucket and secondary_bucket != "True Neutral":
            return "Secondary"
        if secondary_bucket == "True Neutral":
            return "Secondary (True Neutral)"
        return "Primary"

    def _hybrid_primary_bucket(
        self,
        primary_score: float,
        ml_probs: dict[str, float] | None,
        fv: FeatureVector,
        rule_bucket: str,
    ) -> tuple[str, str | None]:
        """Combine rule score and ML probabilities for the primary bucket."""
        if ml_probs is None:
            return rule_bucket, "ML primary model not available"

        p_large_up = ml_probs.get("Large Up", 0.0)
        p_large_down = ml_probs.get("Large Down", 0.0)

        if (primary_score >= 2.0 and p_large_up >= 0.55) or p_large_up >= 0.70:
            return "Large Up", None
        if (primary_score <= -2.0 and p_large_down >= 0.55) or p_large_down >= 0.70:
            return "Large Down", None
        return "Neutral", None

    def _hybrid_secondary_bucket(
        self,
        secondary_score: float,
        ml_probs: dict[str, float] | None,
        rule_bucket: str | None,
    ) -> tuple[str | None, str | None]:
        """Combine secondary rule score and ML probabilities."""
        if ml_probs is None:
            return rule_bucket, "ML secondary model not available"

        p_bull = ml_probs.get("Mild Bullish Bias", 0.0)
        p_bear = ml_probs.get("Mild Bearish Bias", 0.0)

        # Short-term rule support must agree with the ML direction.
        if p_bull >= 0.60 and secondary_score >= 0.4:
            return "Mild Bullish Bias", None
        if p_bear >= 0.60 and secondary_score <= -0.4:
            return "Mild Bearish Bias", None
        return "True Neutral", None

    def _hybrid_confidence(
        self,
        primary_bucket: str,
        primary_score: float,
        ml_primary_probs: dict[str, float] | None,
        secondary_bucket: str | None,
        secondary_score: float,
        ml_secondary_probs: dict[str, float] | None,
        fv: FeatureVector,
    ) -> float:
        """Confidence reflects agreement between rule and ML plus data quality."""
        if primary_bucket != "Neutral":
            ml_prob = 0.0
            if ml_primary_probs:
                ml_prob = ml_primary_probs.get(primary_bucket, 0.0)
            rule_conf = min(1.0, abs(primary_score) / 3.0)
            base_conf = 0.55 + 0.45 * max(ml_prob, rule_conf)
            ml_bucket = (
                max(ml_primary_probs, key=ml_primary_probs.get)
                if ml_primary_probs
                else primary_bucket
            )
            if ml_bucket != primary_bucket:
                base_conf *= 0.85
        else:
            if secondary_bucket and secondary_bucket != "True Neutral":
                ml_prob = 0.0
                if ml_secondary_probs:
                    ml_prob = ml_secondary_probs.get(secondary_bucket, 0.0)
                rule_conf = min(0.85, 0.55 + abs(secondary_score) / 2.0)
                base_conf = 0.5 + 0.4 * max(ml_prob, rule_conf)
                ml_bucket = (
                    max(ml_secondary_probs, key=ml_secondary_probs.get)
                    if ml_secondary_probs
                    else secondary_bucket
                )
                if ml_bucket != secondary_bucket:
                    base_conf *= 0.85
            else:
                base_conf = 0.5

        def _status(s: Any) -> str | None:
            if isinstance(s, DataSourceStatus):
                return s.status
            if isinstance(s, dict):
                return s.get("status")
            return getattr(s, "status", None)

        flag_count = sum(1 for s in (fv.source_status or []) if _status(s) not in ("ok", None))
        penalty = min(flag_count * 0.05, 0.25)
        return max(0.2, min(1.0, base_conf - penalty))

    def _build_recommendation(
        self,
        primary_bucket: str,
        primary_score: float,
        ml_primary_probs: dict[str, float] | None,
        secondary_bucket: str | None,
        secondary_score: float,
        ml_secondary_probs: dict[str, float] | None,
        fv: FeatureVector,
        in_position: bool,
    ) -> tuple[str, str, float]:
        """Produce the final long-only recommendation and its confidence."""
        rsi = _clamp(fv.rsi_14, 0.0, 100.0, 50.0)
        ath = _clamp(fv.ath_distance_pct, -50.0, 50.0, -5.0)
        iron = fv.iron_ore_change_pct
        fvm = _clamp(fv.financials_vs_materials_score, -2.0, 2.0, 0.0)

        # Strong bearish technical signature blocks new longs.
        technicals_bearish = (rsi >= 70) or (rsi >= 65 and ath >= -0.5)

        # Strong downside confirmation (used for STAY IN CASH).
        strong_downside = (
            rsi >= 70 and ath >= -1.0 and ((iron is not None and iron <= -0.6) or fvm <= -0.5)
        )

        p_up = ml_primary_probs.get("Large Up", 0.0) if ml_primary_probs else 0.0
        p_down = ml_primary_probs.get("Large Down", 0.0) if ml_primary_probs else 0.0
        p_mild_bull = (
            ml_secondary_probs.get("Mild Bullish Bias", 0.0) if ml_secondary_probs else 0.0
        )
        p_mild_bear = (
            ml_secondary_probs.get("Mild Bearish Bias", 0.0) if ml_secondary_probs else 0.0
        )

        # Confidence penalty for stale/failed data sources.
        _, degraded_sources = self._degraded_status(fv, DataQualityFlags())
        data_penalty = min(len(degraded_sources) * 0.05, 0.25)

        # 1. Primary GO LONG.
        # Allow daily-rates high-conviction overlay (score >= 3.5) to bypass RSI gating.
        overlay_override = primary_score >= 3.5
        primary_go_long = (
            primary_score >= 1.0 and p_up >= 0.60
            and (not technicals_bearish or overlay_override)
        ) or (
            primary_score >= 2.0 and (not technicals_bearish or overlay_override)
            and ml_primary_probs is None
        )
        if primary_go_long:
            if ml_primary_probs is not None:
                conf = 0.55 + 0.45 * p_up + 0.10 * min(primary_score / 3.0, 1.0)
            else:
                conf = 0.55 + 0.25 * min(primary_score / 3.0, 1.0)
            conf = max(0.35, min(1.0, conf - data_penalty))
            return "GO LONG", "Primary", conf

        # 2. Primary/Secondary STAY IN CASH triggers.
        cash_triggered = (
            primary_score <= -1.0
            or p_down >= 0.55
            or strong_downside
            or (secondary_bucket == "Mild Bearish Bias" and p_mild_bear >= 0.55)
        )

        # 3. Secondary GO LONG (only when primary is unclear / not already cash).
        secondary_go_long = (
            not cash_triggered
            and primary_bucket == "Neutral"
            and secondary_bucket == "Mild Bullish Bias"
            and secondary_score >= 0.4
            and p_mild_bull >= 0.60
            and not technicals_bearish
        )
        if secondary_go_long:
            conf = 0.50 + 0.40 * p_mild_bull + 0.10 * min(secondary_score, 1.0)
            conf = max(0.35, min(1.0, conf - data_penalty))
            return "GO LONG", "Secondary", conf

        # 4. Default: STAY IN CASH (or HOLD EXISTING if already in a position).
        if cash_triggered:
            source = "Secondary" if secondary_bucket == "Mild Bearish Bias" else "Primary"
            conf = (
                0.50 + 0.40 * max(p_down, p_mild_bear) + 0.20 * min(abs(primary_score) / 3.0, 1.0)
            )
        else:
            source = "Primary"
            conf = 0.50 + 0.20 * max(p_up, p_mild_bull)

        conf = max(0.3, min(0.85, conf - data_penalty))
        recommendation = "HOLD EXISTING" if in_position else "STAY IN CASH"
        return recommendation, source, conf

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

    def _rsi_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """RSI overbought/oversold adds mean-reversion pressure."""
        score = _clamp(fv.rsi_score, -2.0, 2.0, 0.0)
        high = _clamp(score / 2.5, -0.45, 0.45, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _ath_distance_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Distance from all-time / trailing highs creates profit-taking risk."""
        score = _clamp(fv.ath_score, -2.0, 2.0, 0.0)
        high = _clamp(score / 2.5, -0.45, 0.45, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _momentum_exhaustion_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Strong run + RSI extreme can trigger profit-taking or snap-back.

        The profit-taking combo (near ATH + overbought RSI + strong run) is blended
        in here to give it extra magnitude/direction impact.
        """
        mom = _clamp(fv.momentum_exhaustion_score, -2.0, 2.0, 0.0)
        combo = _clamp(fv.profit_taking_combo_score or 0.0, -2.0, 0.0, 0.0)
        score = _clamp(mom + combo, -3.5, 3.5, 0.0)
        high = _clamp(score / 3.0, -0.60, 0.60, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _bollinger_delta(self, fv: FeatureVector) -> tuple[np.ndarray, float]:
        """Bollinger Band position flags overextension."""
        score = _clamp(fv.bollinger_score, -1.0, 1.0, 0.0)
        high = _clamp(score / 1.5, -0.35, 0.35, 0.0)
        low = -high * 0.60
        delta = np.array([low, -low - high, high], dtype=float)
        return delta, high

    def _direction_score(self, fv: FeatureVector) -> float:
        """Estimate the signed probability of a down day (< 0%)."""
        score = 0.0

        alignment = _clamp(fv.cross_asset_alignment_score, -1.0, 1.0, 0.0)
        score -= 0.35 * alignment

        session_return = _clamp(fv.asx_open_to_now_return_pct, -2.0, 2.0, 0.0)
        score -= 0.18 * (session_return / 0.5)

        spi_combined = 0.0
        if fv.spi_basis_pct is not None:
            spi_combined += _clamp(fv.spi_basis_pct, -2.0, 2.0, 0.0)
        if fv.spi_momentum_pct is not None:
            spi_combined += _clamp(fv.spi_momentum_pct, -3.0, 3.0, 0.0)
        score -= 0.10 * _clamp(spi_combined / 2.0, -1.0, 1.0, 0.0)

        vix_change = _clamp(fv.vix_change_pct, -10.0, 10.0, 0.0)
        score -= 0.04 * _clamp(vix_change / 5.0, -1.0, 1.0, 0.0)

        us10y_change = _clamp(fv.us_10y_change_bps, -20.0, 20.0, 0.0)
        score -= 0.04 * _clamp(us10y_change / 20.0, -1.0, 1.0, 0.0)

        # Sector / fundamental factors
        fvm = _clamp(fv.financials_vs_materials_score, -2.0, 2.0, 0.0)
        score -= 0.12 * fvm

        hc = _clamp(fv.housing_credit_pulse_score, -2.0, 2.0, 0.0)
        score -= 0.10 * hc

        china = _clamp(fv.china_steel_property_score, -2.5, 2.5, 0.0)
        score -= 0.08 * china

        hw = _clamp(fv.heavyweight_idio_score, -2.5, 2.5, 0.0)
        score -= 0.08 * hw

        # Technical indicators
        tech = (
            _clamp(fv.rsi_score, -2.0, 2.0, 0.0)
            + _clamp(fv.ath_score, -2.0, 2.0, 0.0)
            + _clamp(fv.momentum_exhaustion_score, -2.0, 2.0, 0.0)
            + _clamp(fv.bollinger_score, -1.0, 1.0, 0.0)
            + _clamp(fv.profit_taking_combo_score or 0.0, -2.0, 2.0, 0.0)
        )
        score -= 0.18 * _clamp(tech, -4.0, 4.0, 0.0)

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

        flag_count = sum(1 for s in (fv.source_status or []) if _status(s) not in ("ok", None))
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


def bucket_from_return(return_pct: float) -> str:
    """Map an actual signed return to the primary-model bucket label."""
    if return_pct <= -0.6:
        return PRIMARY_BUCKETS[0]
    if return_pct < 0.6:
        return PRIMARY_BUCKETS[1]
    return PRIMARY_BUCKETS[2]
