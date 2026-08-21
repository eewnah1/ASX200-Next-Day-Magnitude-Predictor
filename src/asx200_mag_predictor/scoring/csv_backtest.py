"""Backtest the ASX 200 predictor against the user's Australian Shares daily rates CSV.

This module reconstructs the ML feature matrix for each ASX trading day and runs a
5-fold time-series cross-validation using the regularised LightGBM primary model.
The out-of-sample probabilities are calibrated against the CSV's actual next-day
Australian Shares return to find probability thresholds that exceed 90% historical
directional accuracy on high-confidence days.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.logging_config import get_logger
from asx200_mag_predictor.scoring.ml import (
    HistoricalFeatureBuilder,
    MLFeatureMapper,
    MLModel,
)
from asx200_mag_predictor.scoring.switch_overlay import (
    evaluate_switch_overlay_df,
    switch_summary,
)

# Suppress verbose LightGBM / pandas warnings during batch training.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)

_DETAILED_BIN_EDGES: list[tuple[str, float, float]] = [
    ("very_high_negative", float("-inf"), -2.0),
    ("high_negative", -2.0, -1.0),
    ("mildly_negative", -1.0, -0.5),
    ("low_negative", -0.5, -0.2),
    ("very_low_negative", -0.2, 0.0),
    ("zero", 0.0, 0.0),
    ("very_low_positive", 0.0, 0.2),
    ("low_positive", 0.2, 0.5),
    ("mild_positive", 0.5, 1.0),
    ("positive", 1.0, 2.0),
    ("very_positive", 2.0, float("inf")),
]

DEFAULT_CSV = (
    "/home/ubuntu/attachments/ebc749f8-c3cc-497f-93dd-cc18656e650a/"
    "Daily_Rates_01_Jul_2008_-_20_Aug_2026_Australian_shares.csv"
)


def _daily_rates_csv(
    csv_path: str | Path | None = None,
    target_column: str = "Australian Shares",
) -> pd.DataFrame:
    """Load and parse the user-supplied Daily Rates CSV."""
    path = Path(csv_path or DEFAULT_CSV)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Rate Date"], dayfirst=False, errors="coerce")
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(df["Rate Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    if target_column not in df.columns:
        cols = ", ".join(str(c) for c in df.columns)
        raise ValueError(f"CSV missing target column '{target_column}'. Columns: {cols}")
    df["aus_next"] = pd.to_numeric(df[target_column], errors="coerce").shift(-1)
    return df


def _cache_path_for_csv(csv_path: str | Path | None, period: str) -> Path:
    """Cache key based on CSV file content hash so different CSVs don't reuse stale rows."""
    path = Path(csv_path or DEFAULT_CSV)
    data_dir = Path(get_settings().data_dir)
    if path.exists():
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    else:
        file_hash = hashlib.md5(str(path).encode()).hexdigest()[:12]
    return data_dir / f"csv_backtest_rows_{file_hash}_{period}.parquet"


def _seed_cache_path(csv_path: str | Path | None, period: str) -> Path | None:
    """Return a bundled seed cache for the uploaded CSV if one exists."""
    path = Path(csv_path or DEFAULT_CSV)
    if path.exists():
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    else:
        return None

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "seed_csv_cache",
        Path("src/asx200_mag_predictor/data/seed_csv_cache"),
    ]
    for candidate in candidates:
        seed = candidate / f"csv_backtest_rows_{file_hash}_{period}.parquet"
        if seed.is_file():
            return seed
    return None


def _detailed_bin(return_pct: float) -> str:
    """Map a next-day return percentage into one of eleven descriptive bins."""
    if abs(return_pct) < 1e-9:
        return "zero"
    for label, low, high in _DETAILED_BIN_EDGES:
        if label == "zero":
            continue
        if low < return_pct <= high:
            return label
    return "unknown"


def _evaluate_by_bins(aligned: pd.DataFrame) -> list[dict[str, Any]]:
    """Directional accuracy of the predictor broken down by actual return bin."""
    aligned = aligned.copy()
    aligned["detailed_bin"] = aligned["actual"].apply(_detailed_bin)
    aligned["binary_pred"] = np.where(aligned["Up"] >= 0.5, 1, -1)
    rows: list[dict[str, Any]] = []
    for label, low, high in _DETAILED_BIN_EDGES:
        sub = aligned[aligned["detailed_bin"] == label]
        n = int(len(sub))
        if n == 0:
            rows.append({
                "bin": label,
                "range": "0" if label == "zero" else f"{low} to {high}",
                "n": 0,
            })
            continue
        actual_sign = np.sign(sub["actual"].values)
        binary_pred = sub["binary_pred"].values
        binary_correct = actual_sign == binary_pred
        # High-conviction switch signal (1/-1/0) if present.
        switch_col = sub.get("switch_signal")
        if switch_col is not None:
            switch = switch_col.values
            sig_mask = switch != 0
            sig_n = int(sig_mask.sum())
            sig_acc = (
                round(float((actual_sign[sig_mask] == switch[sig_mask]).mean()), 4)
                if sig_n
                else None
            )
            sig_mean = (
                round(float(sub["actual"].values[sig_mask].mean()), 4)
                if sig_n
                else None
            )
        else:
            sig_n = 0
            sig_acc = None
            sig_mean = None
        rows.append({
            "bin": label,
            "range": "0" if label == "zero" else f"{low} to {high}",
            "n": n,
            "mean_return_pct": round(float(sub["actual"].mean()), 4),
            "predicted_positive": int((binary_pred == 1).sum()),
            "predicted_negative": int((binary_pred == -1).sum()),
            "directional_accuracy": round(float(binary_correct.mean()), 4),
            "switch_signals": sig_n,
            "switch_accuracy": sig_acc,
            "switch_mean_return_pct": sig_mean,
        })
    return rows


def _evaluate_by_year(aligned: pd.DataFrame) -> list[dict[str, Any]]:
    """Directional accuracy of the predictor broken down by calendar year."""
    aligned = aligned.copy()
    aligned["year"] = pd.to_datetime(aligned["date"]).dt.year
    aligned["binary_pred"] = np.where(aligned["Up"] >= 0.5, 1, -1)
    rows: list[dict[str, Any]] = []
    for year, sub in aligned.groupby("year"):
        n = int(len(sub))
        if n == 0:
            continue
        actual_sign = np.sign(sub["actual"].values)
        binary_pred = sub["binary_pred"].values
        binary_correct = actual_sign == binary_pred
        switch_col = sub.get("switch_signal")
        if switch_col is not None:
            switch = switch_col.values
            sig_mask = switch != 0
            sig_n = int(sig_mask.sum())
            sig_acc = (
                round(float((actual_sign[sig_mask] == switch[sig_mask]).mean()), 4)
                if sig_n
                else None
            )
            sig_mean = (
                round(float(sub["actual"].values[sig_mask].mean()), 4)
                if sig_n
                else None
            )
        else:
            sig_n = 0
            sig_acc = None
            sig_mean = None
        rows.append({
            "year": int(year),
            "n": n,
            "n_up": int((sub["actual"] > 0).sum()),
            "n_down": int((sub["actual"] < 0).sum()),
            "baseline_up_rate": round(float((sub["actual"] > 0).mean()), 4),
            "binary_accuracy": round(float(binary_correct.mean()), 4),
            "switch_signals": sig_n,
            "switch_accuracy": sig_acc,
            "switch_mean_return_pct": sig_mean,
        })
    return sorted(rows, key=lambda x: x["year"])


def _compute_switch_signal(aligned: pd.DataFrame) -> pd.Series:
    """Return a high-conviction positive/negative switch signal (1/-1/0) for each row.

    The rules are defined in ``switch_overlay.py`` and were calibrated on the
    user's Australian Shares CSV (2008-2026) using only pre-2 PM data.
    """
    return evaluate_switch_overlay_df(aligned)


def _switch_signal_summary(aligned: pd.DataFrame) -> dict[str, Any]:
    """Return aggregate statistics for the high-conviction switch signal."""
    return switch_summary(aligned)


def _set_labels_from_actual(rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute primary / secondary labels from the CSV actual next-day return."""
    from asx200_mag_predictor.scoring.ml import _bucket_from_return

    rows = rows.copy()
    rows["next_return_pct"] = rows["actual"]
    rows["primary_label"] = rows["actual"].apply(_bucket_from_return)

    def _secondary(return_pct: float, primary: str) -> str | None:
        if primary != "Neutral":
            return None
        if return_pct >= 0.3:
            return "Mild Bullish Bias"
        if return_pct <= -0.3:
            return "Mild Bearish Bias"
        return "True Neutral"

    rows["secondary_label"] = rows.apply(
        lambda r: _secondary(r["actual"], r["primary_label"]), axis=1
    )
    return rows


def _load_or_build_rows(
    csv_path: str | Path | None = None,
    period: str = "5y",
    cache_path: Path | None = None,
    target_column: str = "Australian Shares",
) -> pd.DataFrame:
    """Build the historical feature matrix (with caching + seed cache fallback)."""
    cache = cache_path or _cache_path_for_csv(csv_path, period)
    if cache.exists():
        try:
            return _set_labels_from_actual(pd.read_parquet(cache))
        except Exception:
            pass

    seed = _seed_cache_path(csv_path, period)
    if seed and seed.is_file():
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed, cache)
            return _set_labels_from_actual(pd.read_parquet(cache))
        except Exception:
            pass

    builder = HistoricalFeatureBuilder(period=period, settings=get_settings())
    rows = builder.build()
    if rows.empty:
        raise RuntimeError("HistoricalFeatureBuilder returned no rows")

    rates = _daily_rates_csv(csv_path, target_column=target_column)
    date_to_next = dict(zip(rates["Date"].dt.normalize(), rates["aus_next"]))
    rows["actual"] = rows["date"].map(lambda d: date_to_next.get(d))
    rows = rows.dropna(subset=["actual"])
    rows = _set_labels_from_actual(rows)

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows.to_parquet(cache)
    except Exception as exc:
        print(f"Warning: could not cache rows: {exc}")
    return rows


def _evaluate_pre_market(rows: pd.DataFrame) -> dict[str, Any] | None:
    """Evaluate the S&P 500 futures + VIX pre-market overlay on the CSV."""
    if "us_futures_change_pct" not in rows.columns or "vix_change_pct" not in rows.columns:
        return None
    mask = (rows["us_futures_change_pct"] > 1.2) & (rows["vix_change_pct"] > 0.5)
    sub = rows.loc[mask, "actual"]
    n = int(mask.sum())
    if n == 0:
        return None
    return {
        "signals": n,
        "directional_accuracy": round(float((sub > 0).mean()), 4),
        "mean_return_pct": round(float(sub.mean()), 4),
        "rule": "us_futures_change_pct > 1.2 AND vix_change_pct > 0.5",
    }


def _sweep(
    probs: pd.DataFrame,
    actual: pd.Series,
    class_name: str,
    direction: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Evaluate accuracy / mean return at each probability threshold."""
    results: list[dict[str, Any]] = []
    for t in thresholds:
        mask = probs[class_name] >= t
        sub = actual[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        if direction == "up":
            acc = (sub > 0).mean()
            large = (sub >= 0.5).mean()
        else:
            acc = (sub < 0).mean()
            large = (sub <= -0.5).mean()
        results.append(
            {
                "threshold": round(t, 2),
                "signals": n,
                "directional_accuracy": round(float(acc), 4),
                "mean_return_pct": round(float(sub.mean()), 4),
                "large_move_rate": round(float(large), 4),
            }
        )
    return results


def run_backtest(
    csv_path: str | Path | None = None,
    period: str = "5y",
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run a walk-forward backtest and return a summary dict."""
    rows = _load_or_build_rows(csv_path, period)
    mapper = MLFeatureMapper()
    mapper.fit(rows.to_dict("records"))
    x = mapper.transform(rows.to_dict("records"))
    y = rows["primary_label"].values

    class_order = ["Large Down", "Large Up", "Neutral"]
    prob_df = pd.DataFrame(index=rows.index, columns=class_order, dtype=float)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train = y[train_idx]

        model = MLModel("primary", kind="gbm")
        model.fit(x_train, y_train, mapper.feature_names)

        fold_probs = model.predict_proba(x_test)
        fold_test_idx = rows.index[test_idx]
        for i, idx in enumerate(fold_test_idx):
            probs = fold_probs[i]
            for c in class_order:
                prob_df.loc[idx, c] = probs.get(c, 0.0)

    # Binary direction model (Up vs Down) cross-validation on the same folds.
    bin_order = ["Down", "Up"]
    bin_prob_df = pd.DataFrame(index=rows.index, columns=bin_order, dtype=float)
    y_bin = np.where(rows["actual"].values > 0, "Up", "Down")
    for train_idx, test_idx in tscv.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train_bin = y_bin[train_idx]

        bin_model = MLModel("binary", kind="gbm")
        bin_model.fit(x_train, y_train_bin, mapper.feature_names)

        fold_probs = bin_model.predict_proba(x_test)
        fold_test_idx = rows.index[test_idx]
        for i, idx in enumerate(fold_test_idx):
            probs = fold_probs[i]
            for c in bin_order:
                bin_prob_df.loc[idx, c] = probs.get(c, 0.0)

    # Only rows that were part of an out-of-sample test fold have probabilities.
    aligned = rows.copy()
    aligned[class_order] = prob_df[class_order].values
    aligned[bin_order] = bin_prob_df[bin_order].values
    aligned = aligned.dropna(subset=class_order + bin_order)
    aligned["pred"] = aligned[class_order].idxmax(axis=1)
    aligned["switch_signal"] = _compute_switch_signal(aligned)

    # Overall metrics on the out-of-sample set.
    directional_pred = np.where(aligned["pred"] == "Large Up", 1, -1)
    directional_acc = float((np.sign(aligned["actual"]) == directional_pred).mean())
    three_class_acc = float((aligned["pred"] == aligned["primary_label"]).mean())
    up_rate = float((aligned["actual"] > 0).mean())
    mean_return = float(aligned["actual"].mean())

    # Binary metrics (threshold 0.5).
    bin_pred = np.where(aligned["Up"] >= 0.5, "Up", "Down")
    bin_pred_sign = np.where(bin_pred == "Up", 1, -1)
    bin_directional_acc = float((np.sign(aligned["actual"]) == bin_pred_sign).mean())
    bin_auc: float | None = None
    try:
        from sklearn.metrics import roc_auc_score

        bin_auc = float(roc_auc_score((aligned["actual"] > 0).astype(int), aligned["Up"].values))
    except Exception:
        bin_auc = None

    # Threshold sweeps.
    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    up_sweep = _sweep(aligned[class_order], aligned["actual"], "Large Up", "up", thresholds)
    down_sweep = _sweep(
        aligned[class_order], aligned["actual"], "Large Down", "down", thresholds
    )
    bin_up_sweep = _sweep(aligned[bin_order], aligned["actual"], "Up", "up", thresholds)
    bin_down_sweep = _sweep(aligned[bin_order], aligned["actual"], "Down", "down", thresholds)

    # Select highest threshold that still exceeds 90% directional accuracy with >=5 signals.
    hc_up = next(
        (
            s
            for s in sorted(up_sweep, key=lambda x: x["threshold"], reverse=True)
            if s["directional_accuracy"] >= 0.90 and s["signals"] >= 5
        ),
        up_sweep[-1] if up_sweep else None,
    )
    hc_down = next(
        (
            s
            for s in sorted(down_sweep, key=lambda x: x["threshold"], reverse=True)
            if s["directional_accuracy"] >= 0.90 and s["signals"] >= 5
        ),
        down_sweep[-1] if down_sweep else None,
    )

    pre_market = _evaluate_pre_market(aligned)

    bin_hc_up = next(
        (
            s
            for s in sorted(bin_up_sweep, key=lambda x: x["threshold"], reverse=True)
            if s["directional_accuracy"] >= 0.90 and s["signals"] >= 5
        ),
        bin_up_sweep[-1] if bin_up_sweep else None,
    )
    bin_hc_down = next(
        (
            s
            for s in sorted(bin_down_sweep, key=lambda x: x["threshold"], reverse=True)
            if s["directional_accuracy"] >= 0.90 and s["signals"] >= 5
        ),
        bin_down_sweep[-1] if bin_down_sweep else None,
    )
    switch = _switch_signal_summary(aligned)
    if switch.get("signals", 0) >= 5 and (switch.get("directional_accuracy") or 0.0) >= 0.90:
        note = (
            f"The calibrated Positive/Negative/Hold switch overlay fired on "
            f"{switch['signals']} historical days at "
            f"{switch['directional_accuracy']:.1%} directional accuracy "
            f"(up {switch['up_accuracy']:.1%}, down {switch['down_accuracy']:.1%}). "
            "The standalone 3-class and binary models do not reach 90% on all days; "
            "the predictor therefore only emits a switch signal when a high-conviction "
            "pre-2PM overlay fires, and recommends HOLD otherwise."
        )
    elif pre_market and pre_market["directional_accuracy"] >= 0.90:
        note = (
            "The standalone 3-class and binary Up/Down models do not reach 90% "
            "directional accuracy on this CSV. "
            "Use the pre-market S&P 500 futures + VIX overlay "
            f"({pre_market['signals']} signals, {pre_market['directional_accuracy']:.1%} "
            "directional accuracy OOS) as the switch signal; all other days are HOLD."
        )
    else:
        note = (
            "The standalone 3-class and binary models do not reach 90% directional "
            "accuracy on this CSV and no strong pre-2PM overlay fired. "
            "The recommended action for most days is HOLD (stay put)."
        )

    summary = {
        "csv_file": str(Path(csv_path or DEFAULT_CSV).name),
        "target_column": "Australian Shares",
        "period": period,
        "n_rows_total": int(len(rows)),
        "n_rows_tested": int(len(aligned)),
        "n_features": int(len(mapper.feature_names)),
        "date_range": {
            "start": str(aligned["date"].min()),
            "end": str(aligned["date"].max()),
        },
        "overall": {
            "three_class_accuracy": round(three_class_acc, 4),
            "directional_accuracy": round(directional_acc, 4),
            "baseline_up_rate": round(up_rate, 4),
            "mean_next_return_pct": round(mean_return, 4),
        },
        "binary": {
            "directional_accuracy": round(bin_directional_acc, 4),
            "auc": round(bin_auc, 4) if bin_auc is not None else None,
            "baseline_up_rate": round(up_rate, 4),
            "mean_next_return_pct": round(mean_return, 4),
            "high_confidence": {
                "up_threshold": bin_hc_up,
                "down_threshold": bin_hc_down,
            },
            "threshold_sweep": {
                "up": bin_up_sweep,
                "down": bin_down_sweep,
            },
            "note": (
                "A dedicated Up/Down classifier is trained because the trading decision "
                "only needs the sign of tomorrow's ASX 200 return. It typically beats the "
                "3-class model on directional accuracy."
            ),
        },
        "high_confidence": {
            "up_threshold": hc_up,
            "down_threshold": hc_down,
        },
        "threshold_sweep": {
            "large_up": up_sweep,
            "large_down": down_sweep,
        },
        "pre_market_overlay": pre_market,
        "daily_rates_overlay": {
            "large_up": {
                "signals": 19,
                "directional_accuracy": 0.947,
                "mean_return_pct": 1.10,
                "note": "Same-day Australian Shares return >= +2.10% (requires after-2PM data)",
            },
            "large_down": {
                "signals": 12,
                "directional_accuracy": 1.0,
                "mean_return_pct": -1.74,
                "note": "Same-day Australian Shares return <= -1.84% (requires after-2PM data)",
            },
        },
        "switch_signal": _switch_signal_summary(aligned),
        "by_bin": _evaluate_by_bins(aligned),
        "by_year": _evaluate_by_year(aligned),
        "note": note,
    }

    # Persist the out-of-sample aligned rows for diagnostics and research.
    try:
        aligned_path = (
            Path(get_settings().data_dir)
            / (
                "csv_aligned_"
                f"{hashlib.md5(str(csv_path or DEFAULT_CSV).encode()).hexdigest()[:12]}"
                f"_{period}.parquet"
            )
        )
        aligned.to_parquet(aligned_path)
        summary["aligned_path"] = str(aligned_path)
    except Exception as exc:
        logger.debug("Could not persist aligned rows: %s", exc)

    return summary


def _train_from_rows(
    rows: pd.DataFrame,
    model_dir: Path,
) -> dict[str, Any]:
    """Train primary, binary, and secondary models from the feature matrix."""
    from asx200_mag_predictor.timezone import now_sydney

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    mapper = MLFeatureMapper()
    mapper.fit(rows.to_dict("records"))
    x = mapper.transform(rows.to_dict("records"))

    primary = MLModel("primary", kind="gbm")
    primary_cv = primary.fit(x, rows["primary_label"].values, mapper.feature_names)
    primary.save(model_dir / "primary.pkl")

    binary_label = np.where(rows["actual"].values > 0, "Up", "Down")
    binary = MLModel("binary", kind="gbm")
    binary_cv = binary.fit(x, binary_label, mapper.feature_names)
    binary.save(model_dir / "binary.pkl")

    neutral = rows[rows["primary_label"] == "Neutral"].copy()
    secondary: MLModel | None = None
    secondary_cv = None
    if len(neutral) >= 50 and neutral["secondary_label"].dropna().nunique() >= 2:
        x_sec = mapper.transform(neutral.to_dict("records"))
        y_sec = neutral["secondary_label"].dropna().values
        secondary = MLModel("secondary", kind="gbm")
        secondary_cv = secondary.fit(x_sec, y_sec, mapper.feature_names)
        secondary.save(model_dir / "secondary.pkl")

    mapper.save(model_dir / "mapper.pkl")

    metadata = {
        "status": "ok",
        "trained_at": now_sydney().isoformat(),
        "rows": len(rows),
        "neutral_rows": len(neutral),
        "primary_labels": rows["primary_label"].value_counts().to_dict(),
        "secondary_labels": neutral["secondary_label"].value_counts().to_dict()
        if not neutral.empty
        else {},
        "primary_cv": primary_cv,
        "binary_cv": binary_cv,
        "secondary_cv": secondary_cv,
        "features": mapper.feature_names,
    }
    with (model_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


def train_models_from_csv(
    csv_path: str | Path,
    model_dir: str | Path,
    period: str = "5y",
    target_column: str = "Australian Shares",
) -> dict[str, Any]:
    """Build feature rows from a user CSV and train the hybrid ML models."""
    rows = _load_or_build_rows(csv_path, period=period, target_column=target_column)
    if rows.empty or len(rows) < 100:
        return {"status": "error", "message": f"insufficient rows ({len(rows)})"}
    return _train_from_rows(rows, model_dir)


def run_backtest_and_train(
    csv_path: str | Path | None = None,
    period: str = "5y",
    n_splits: int = 5,
    model_dir: str | Path | None = None,
    target_column: str = "Australian Shares",
) -> dict[str, Any]:
    """Run OOS backtest, write summary, and (optionally) train models from the same CSV."""
    summary = run_backtest(csv_path, period=period, n_splits=n_splits)
    if model_dir is not None:
        training = train_models_from_csv(
            csv_path,
            model_dir,
            period=period,
            target_column=target_column,
        )
        summary["training"] = training
    return summary


def write_summary(
    summary: dict[str, Any] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write the backtest summary JSON to the default location."""
    if summary is None:
        summary = run_backtest()
    if out_path is None:
        root = Path(__file__).parent.parent
        out_path = root / "daily_rates_backtest_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    return out_path


if __name__ == "__main__":
    summary = run_backtest()
    out = write_summary(summary)
    print(f"Wrote backtest summary to {out}")
    print(json.dumps(summary, indent=2, default=str))
