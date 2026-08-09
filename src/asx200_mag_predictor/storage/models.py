"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, Float, ForeignKey, String, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from asx200_mag_predictor.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    """Persisted prediction with full feature vector and metadata."""

    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_for_date: Mapped[datetime] = mapped_column(nullable=False)
    generated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    data_as_of: Mapped[datetime | None] = mapped_column(nullable=True)
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    probabilities_json: Mapped[dict] = mapped_column(JSON, default=dict)
    bucket: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    factor_breakdown_json: Mapped[dict] = mapped_column(JSON, default=dict)
    factor_contributions_json: Mapped[list] = mapped_column(JSON, default=list)
    notes_json: Mapped[list] = mapped_column(JSON, default=list)
    data_quality_flags_json: Mapped[dict] = mapped_column(JSON, default=dict)
    source_status_json: Mapped[list] = mapped_column(JSON, default=list)
    errors_json: Mapped[list] = mapped_column(JSON, default=list)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    degraded_sources_json: Mapped[list] = mapped_column(JSON, default=list)
    model: Mapped[str] = mapped_column(String(20), default="Primary")
    primary_bucket: Mapped[str] = mapped_column(String(20), default="Neutral")
    secondary_bucket: Mapped[str | None] = mapped_column(String(30), nullable=True)
    primary_score: Mapped[float] = mapped_column(Float, default=0.0)
    secondary_score: Mapped[float] = mapped_column(Float, default=0.0)
    ml_available: Mapped[bool] = mapped_column(Boolean, default=False)
    ml_probabilities_json: Mapped[dict] = mapped_column(JSON, default=dict)
    ml_feature_importance_json: Mapped[list] = mapped_column(JSON, default=list)
    ml_fallback_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    recommendation: Mapped[str | None] = mapped_column(String(20), nullable=True)
    recommendation_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    recommendation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    in_position: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    actual: Mapped["ActualRecord"] = relationship(back_populates="prediction", uselist=False)


class ActualRecord(Base):
    """Outcome for a prior prediction, used for calibration."""

    __tablename__ = "actuals"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("predictions.id"), unique=True, nullable=False
    )
    actual_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    actual_bucket: Mapped[str] = mapped_column(String(20), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)

    prediction: Mapped["PredictionRecord"] = relationship(back_populates="actual")


class DataSnapshotRecord(Base):
    """Optional raw snapshot cache in the DB."""

    __tablename__ = "data_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    data_json: Mapped[dict] = mapped_column(JSON, default=dict)


def get_engine(settings: Settings | None = None):
    settings = settings or get_settings()
    url = settings.database_url
    kwargs = {}
    if settings.is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def get_session_maker(engine=None):
    engine = engine or get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _add_missing_columns(engine) -> None:
    """Add columns introduced after the initial schema without losing data."""
    inspector = inspect(engine)
    for table, additions in {
        "predictions": [
            ("data_as_of", "DATETIME"),
            ("factor_contributions_json", "JSON"),
            ("source_status_json", "JSON"),
            ("errors_json", "JSON"),
            ("degraded", "BOOLEAN"),
            ("degraded_sources_json", "JSON"),
            ("model", "VARCHAR"),
            ("primary_bucket", "VARCHAR"),
            ("secondary_bucket", "VARCHAR"),
            ("primary_score", "FLOAT"),
            ("secondary_score", "FLOAT"),
            ("ml_available", "BOOLEAN"),
            ("ml_probabilities_json", "JSON"),
            ("ml_feature_importance_json", "JSON"),
            ("ml_fallback_reason", "VARCHAR"),
            ("recommendation", "VARCHAR"),
            ("recommendation_source", "VARCHAR"),
            ("recommendation_confidence", "FLOAT"),
            ("in_position", "BOOLEAN"),
        ],
        "actuals": [
            ("actual_return_pct", "FLOAT"),
            ("actual_bucket", "VARCHAR"),
            ("recorded_at", "DATETIME"),
        ],
    }.items():
        if table not in inspector.get_table_names():
            continue
        columns = {c["name"] for c in inspector.get_columns(table)}
        with engine.begin() as conn:
            for col, dtype in additions:
                if col not in columns:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}"))


def init_db(settings: Settings | None = None):
    """Create tables if they do not exist and add any missing columns."""
    engine = get_engine(settings)
    Base.metadata.create_all(bind=engine)
    try:
        _add_missing_columns(engine)
    except Exception:  # noqa: BLE001
        pass
    return engine
