"""Centralised logging setup."""

import logging
import sys

from asx200_mag_predictor.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure root logger with a clear, structured-ish format."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)d] | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
