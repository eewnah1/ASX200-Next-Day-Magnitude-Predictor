"""Command-line interface for local operations."""

from __future__ import annotations

import argparse
import json

from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.data.fetchers import DataFetcher
from asx200_mag_predictor.logging_config import setup_logging
from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scheduler.jobs import start_scheduler
from asx200_mag_predictor.scoring.engine import ScoringEngine
from asx200_mag_predictor.scoring.features import build_features
from asx200_mag_predictor.storage.models import init_db
from asx200_mag_predictor.storage.repository import Repository


def cmd_predict(args: argparse.Namespace) -> None:
    settings = get_settings()
    setup_logging(settings)
    init_db(settings)

    if args.mock:
        features = FeatureVector(
            a_vix=args.a_vix,
            realized_vol_annual=args.realized_vol,
            catalyst_score=args.catalyst,
            cross_asset_magnitude=args.cross_mag,
            asx_session_character=args.session,
            spi_basis_pct=args.spi_basis,
            spi_momentum_pct=args.spi_momentum,
        )
        prediction = ScoringEngine(settings).predict(features)
    else:
        raw = DataFetcher(settings).fetch_all()
        features, flags = build_features(raw)
        prediction = ScoringEngine(settings).predict(features, flags)

    print(prediction.model_dump_json(indent=2))


def cmd_calibration(args: argparse.Namespace) -> None:
    init_db(get_settings())
    repo = Repository()
    print(repo.calibration_metrics().model_dump_json(indent=2))


def cmd_record_actual(args: argparse.Namespace) -> None:
    init_db(get_settings())
    repo = Repository()
    repo.save_actual(args.prediction_id, args.actual_return)
    print(f"Recorded actual {args.actual_return}% for prediction {args.prediction_id}")


def cmd_train_ml(args: argparse.Namespace) -> None:
    from asx200_mag_predictor.scoring.ml import MLTrainer

    settings = get_settings()
    setup_logging(settings)
    trainer = MLTrainer(settings=settings)
    period = f"{args.months}mo"
    result = trainer.run(period=period)
    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="ASX200 Next-Day Magnitude Predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Run a prediction")
    p_predict.add_argument("--mock", action="store_true", help="Use mock feature inputs")
    p_predict.add_argument("--a-vix", type=float, default=18.0)
    p_predict.add_argument("--realized-vol", type=float, default=17.0)
    p_predict.add_argument("--catalyst", type=int, default=2)
    p_predict.add_argument("--cross-mag", type=float, default=1.2)
    p_predict.add_argument("--session", default="trend")
    p_predict.add_argument("--spi-basis", type=float, default=0.1)
    p_predict.add_argument("--spi-momentum", type=float, default=0.2)
    p_predict.set_defaults(func=cmd_predict)

    p_cal = sub.add_parser("calibration", help="Show calibration metrics")
    p_cal.set_defaults(func=cmd_calibration)

    p_actual = sub.add_parser("record-actual", help="Record actual outcome for a prediction")
    p_actual.add_argument("prediction_id")
    p_actual.add_argument("--actual-return", type=float, required=True, dest="actual_return")
    p_actual.set_defaults(func=cmd_record_actual)

    p_scheduler = sub.add_parser("run-scheduler", help="Start the daily prediction scheduler")
    p_scheduler.set_defaults(func=cmd_run_scheduler)

    p_train_ml = sub.add_parser("train-ml", help="Train the hybrid ML models on historical data")
    p_train_ml.add_argument("--months", type=int, default=24, help="Months of history to fetch")
    p_train_ml.set_defaults(func=cmd_train_ml)

    args = parser.parse_args()
    args.func(args)


def cmd_run_scheduler(args: argparse.Namespace) -> None:
    settings = get_settings()
    setup_logging(settings)
    scheduler = start_scheduler(settings)
    print("Scheduler started; jobs:", [str(job) for job in scheduler.get_jobs()])
    try:
        while True:
            import time

            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    main()
