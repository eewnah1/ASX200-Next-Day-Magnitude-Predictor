"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, String, create_engine
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
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    probabilities_json: Mapped[dict] = mapped_column(JSON, default=dict)
    bucket: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    factor_breakdown_json: Mapped[dict] = mapped_column(JSON, default=dict)
    notes_json: Mapped[list] = mapped_column(JSON, default=list)
    data_quality_flags_json: Mapped[dict] = mapped_column(JSON, default=dict)

    actual: Mapped["ActualRecord"] = relationship(back_populates="prediction", uselist=False)


class ActualRecord(Base):
    """Outcome for a prior prediction, used for calibration."""

    __tablename__ = "actuals"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("predictions.id"), unique=True, nullable=False
    )
    actual_abs_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
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


def init_db(settings: Settings | None = None):
    """Create tables if they do not exist."""
    engine = get_engine(settings)
    Base.metadata.create_all(bind=engine)
    return engine
