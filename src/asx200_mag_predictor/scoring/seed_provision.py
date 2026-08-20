"""ML model disk provisioning (seed + ensure available)."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from asx200_mag_predictor.config import Settings, get_settings

logger = logging.getLogger(__name__)


def ensure_seed_ml_models(model_dir: Path | None = None, settings: Settings | None = None) -> bool:
    """Copy packaged seed models into the runtime model dir if primary is missing.

    Returns True if models are present after the call (either already there or
    successfully seeded). Defence against empty Render disks / fresh deploys.
    """
    settings = settings or get_settings()
    target = Path(model_dir) if model_dir else settings.ml_model_dir
    target.mkdir(parents=True, exist_ok=True)
    if (target / "primary.pkl").exists() and (target / "mapper.pkl").exists():
        return True

    seed_candidates: list[Path] = []
    try:
        import importlib.resources as resources

        seed_candidates.append(
            Path(str(resources.files("asx200_mag_predictor.data") / "seed_ml_models"))
        )
    except Exception:  # noqa: BLE001
        pass
    seed_candidates.append(Path(__file__).resolve().parent / "seed_ml_models")
    seed_candidates.append(Path("src/asx200_mag_predictor/data/seed_ml_models"))

    for seed in seed_candidates:
        try:
            if not seed.is_dir():
                continue
            src_primary = seed / "primary.pkl"
            if not src_primary.is_file():
                continue
            for name in ("primary.pkl", "secondary.pkl", "mapper.pkl", "metadata.json"):
                src = seed / name
                if src.is_file():
                    shutil.copy2(src, target / name)
            logger.info("Seeded ML models from %s -> %s", seed, target)
            return (target / "primary.pkl").exists()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not seed ML models from %s: %s", seed, exc)
    logger.warning("No seed ML models found; train via CLI or POST /api/v1/train-ml")
    return False
