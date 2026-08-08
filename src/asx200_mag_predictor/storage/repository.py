"""Repository for predictions, actuals and calibration metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from asx200_mag_predictor.models import (
    BucketProbabilities,
    CalibrationMetrics,
    DataQualityFlags,
    DataSourceStatus,
    FactorBreakdown,
    FactorContribution,
    FeatureVector,
    Prediction,
)
from asx200_mag_predictor.scoring.engine import bucket_from_return
from asx200_mag_predictor.storage.models import (
    ActualRecord,
    PredictionRecord,
    get_engine,
    get_session_maker,
)


class Repository:
    """High-level DB access."""

    def __init__(self, session: Session | None = None, settings=None):
        self.session = session
        if session is None:
            engine = get_engine(settings)
            self.session = get_session_maker(engine)()

    def save_prediction(self, prediction: Prediction) -> str:
        if not prediction.id:
            import uuid

            prediction.id = str(uuid.uuid4())
        record = PredictionRecord(
            id=prediction.id,
            prediction_for_date=prediction.prediction_for_date,
            generated_at=prediction.generated_at,
            data_as_of=prediction.data_as_of,
            features_json=prediction.features.model_dump(mode="json"),
            probabilities_json=prediction.probabilities.model_dump(mode="json"),
            bucket=prediction.bucket,
            confidence=prediction.confidence,
            factor_breakdown_json=prediction.factor_breakdown.model_dump(mode="json"),
            factor_contributions_json=[
                fc.model_dump(mode="json") for fc in prediction.factor_contributions
            ],
            notes_json=prediction.notes,
            data_quality_flags_json=prediction.data_quality_flags.model_dump(mode="json"),
            source_status_json=[s.model_dump(mode="json") for s in prediction.source_status],
            errors_json=prediction.errors,
            degraded=prediction.degraded,
            degraded_sources_json=prediction.degraded_sources,
            model=prediction.model,
            primary_bucket=prediction.primary_bucket,
            secondary_bucket=prediction.secondary_bucket,
            primary_score=prediction.primary_score,
            secondary_score=prediction.secondary_score,
        )
        self.session.add(record)
        self.session.commit()
        return record.id

    def get_prediction(self, prediction_id: str) -> Prediction | None:
        record = self.session.query(PredictionRecord).filter_by(id=prediction_id).first()
        if not record:
            return None
        return self._to_prediction(record)

    def list_predictions(self, limit: int = 100) -> list[Prediction]:
        records = (
            self.session.query(PredictionRecord)
            .order_by(PredictionRecord.generated_at.desc())
            .limit(limit)
            .all()
        )
        return [self._to_prediction(r) for r in records]

    def get_latest_prediction(self) -> Prediction | None:
        record = (
            self.session.query(PredictionRecord)
            .order_by(PredictionRecord.generated_at.desc())
            .first()
        )
        return self._to_prediction(record) if record else None

    def save_actual(self, prediction_id: str, actual_return_pct: float) -> str:
        bucket = bucket_from_return(actual_return_pct)
        record = ActualRecord(
            prediction_id=prediction_id,
            actual_return_pct=actual_return_pct,
            actual_bucket=bucket,
        )
        self.session.add(record)
        self.session.commit()
        return bucket

    def get_actuals(self, limit: int = 1000) -> list[dict[str, Any]]:
        records = (
            self.session.query(ActualRecord)
            .order_by(ActualRecord.recorded_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "prediction_id": r.prediction_id,
                "actual_return_pct": r.actual_return_pct,
                "actual_bucket": r.actual_bucket,
                "recorded_at": r.recorded_at,
            }
            for r in records
        ]

    def calibration_metrics(self) -> CalibrationMetrics:
        """Compute simple hit-rate metrics overall and by vol regime."""
        rows = (
            self.session.query(PredictionRecord, ActualRecord)
            .join(ActualRecord, PredictionRecord.id == ActualRecord.prediction_id)
            .all()
        )
        total = len(rows)
        if not total:
            return CalibrationMetrics(total=0, correct=0, hit_rate=0.0, by_regime={})

        correct = sum(1 for p, a in rows if p.bucket == a.actual_bucket)
        by_regime: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for p, a in rows:
            regime = p.features_json.get("vol_regime", "unknown")
            by_regime[str(regime)]["total"] += 1
            if p.bucket == a.actual_bucket:
                by_regime[str(regime)]["correct"] += 1

        result: dict[str, dict[str, float]] = {}
        for regime, counts in by_regime.items():
            t = counts["total"]
            c = counts["correct"]
            result[regime] = {"total": t, "correct": c, "hit_rate": round(c / t, 4) if t else 0.0}

        return CalibrationMetrics(
            total=total,
            correct=correct,
            hit_rate=round(correct / total, 4),
            by_regime=result,
        )

    def _to_prediction(self, record: PredictionRecord) -> Prediction:
        return Prediction(
            id=record.id,
            prediction_for_date=record.prediction_for_date,
            generated_at=record.generated_at,
            data_as_of=record.data_as_of,
            features=FeatureVector(**record.features_json),
            probabilities=BucketProbabilities(**record.probabilities_json),
            bucket=record.bucket,
            confidence=record.confidence,
            factor_breakdown=FactorBreakdown(**record.factor_breakdown_json),
            factor_contributions=[
                FactorContribution(**fc) for fc in record.factor_contributions_json or []
            ],
            notes=record.notes_json or [],
            data_quality_flags=DataQualityFlags(**record.data_quality_flags_json),
            source_status=[DataSourceStatus(**s) for s in record.source_status_json or []],
            errors=record.errors_json or [],
            degraded=record.degraded if record.degraded is not None else False,
            degraded_sources=record.degraded_sources_json or [],
            model=record.model or "Primary",
            primary_bucket=record.primary_bucket or "Neutral",
            secondary_bucket=record.secondary_bucket,
            primary_score=record.primary_score or 0.0,
            secondary_score=record.secondary_score or 0.0,
        )
