"""Hybrid ML layer for the ASX200 predictor.

Trains two multiclass classifiers (Primary and Secondary) on historical features,
then combines them with the existing rule scores in a transparent hybrid gate.
"""

from __future__ import annotations

import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asx200_mag_predictor.config import Settings, get_settings
from asx200_mag_predictor.logging_config import get_logger
from asx200_mag_predictor.models import FeatureVector
from asx200_mag_predictor.scoring.features import RawMarketData, build_features
from asx200_mag_predictor.timezone import now_sydney


def _fetchers():
    import asx200_mag_predictor.data.fetchers as _fetchers_mod

    return _fetchers_mod


logger = get_logger(__name__)

PRIMARY_LABELS = ["Large Down", "Neutral", "Large Up"]
SECONDARY_LABELS = ["Mild Bearish Bias", "True Neutral", "Mild Bullish Bias"]

ML_BASE_FEATURES = [
    "a_vix",
    "atr_5d_pct",
    "realized_vol_annual",
    "catalyst_score",
    "high_impact_events_next_24h",
    "high_impact_events_next_48h",
    "us_futures_change_pct",
    "iron_ore_change_pct",
    "gold_change_pct",
    "silver_change_pct",
    "oil_change_pct",
    "copper_change_pct",
    "aud_usd_change_pct",
    "sp500_change_pct",
    "nasdaq_change_pct",
    "dow_change_pct",
    "us_10y_change_bps",
    "vix_change_pct",
    "cross_asset_alignment_score",
    "cross_asset_magnitude",
    "asx_open_to_now_return_pct",
    "current_volume_vs_20d_avg",
    "current_range_vs_atr",
    "spi_basis_pct",
    "spi_momentum_pct",
    "spi_short_term_momentum_pct",
    "overnight_gap_pct",
    "gap_filled_score",
    "vwap_distance_pct",
    "market_breadth_score",
    "breadth_pct_above_20d_ma",
    "breadth_pct_above_50d_ma",
    "breadth_pct_above_200d_ma",
    "advance_decline_net",
    "new_20d_highs",
    "new_20d_lows",
    "new_50d_highs",
    "new_50d_lows",
    "breadth_index",
    "breadth_score",
    "asian_session_lead_score",
    "tv_asian_session_change_pct",
    "financials_minus_materials_1d_pct",
    "financials_minus_materials_2d_pct",
    "financials_minus_materials_3d_pct",
    "financials_minus_materials_5d_pct",
    "financials_minus_materials_weighted_pct",
    "financials_vs_materials_score",
    "housing_credit_pulse_score",
    "china_steel_property_score",
    "china_steel_property_return_pct",
    "heavyweight_idio_return_pct",
    "heavyweight_idio_score",
    "heavyweight_idio_news_boost",
    "rsi_14",
    "rsi_previous_14",
    "rsi_slope",
    "rsi_score",
    "ath_distance_pct",
    "high_20d_distance_pct",
    "high_50d_distance_pct",
    "ath_score",
    "asx_1d_return_pct",
    "asx_2d_return_pct",
    "asx_3d_return_pct",
    "index_5d_return_pct",
    "momentum_exhaustion_score",
    "bollinger_position",
    "bollinger_score",
    "profit_taking_combo_score",
    # TradingView MCP enrichment
    "tv_xjo_daily_score",
    "tv_xjo_weekly_score",
    "tv_xjo_trend_score",
    "tv_financials_vs_materials_score",
    "tv_financials_minus_materials_pct",
    "tv_heavyweight_avg_score",
    "tv_commodity_basket_change_pct",
    "tv_commodity_basket_ex_gold_change_pct",
    "tv_commodity_vs_gold_change_pct",
    # Alpha Vantage MCP cross-asset feeds
    "av_aud_usd_change_pct",
    "av_spy_change_pct",
    "av_qqq_change_pct",
    "av_gld_change_pct",
    "av_vixy_change_pct",
    "av_us_10y_yield_change_bps",
    "av_us_10y_yield_level",
    # RBA / Australian rates expectations + TradingView China pulse
    "rba_cash_rate_change_bps",
    "au_3y_yield_change_bps",
    "au_10y_yield_change_bps",
    "rba_rates_score",
    "tv_china_steel_property_return_pct",
    # Regime-aware features (computed by the scoring engine)
    "regime_numeric",
    "regime_confidence",
    # Optional enrichment layers (gracefully degrade to 0 when unavailable)
    "news_sentiment_score",
    "options_positioning_score",
]

ML_INTERACTIONS = [
    ("rsi_14", "ath_distance_pct"),
    ("iron_ore_change_pct", "financials_minus_materials_weighted_pct"),
    ("tv_xjo_trend_score", "tv_asian_session_change_pct"),
    ("tv_heavyweight_avg_score", "tv_commodity_vs_gold_change_pct"),
    ("av_spy_change_pct", "av_aud_usd_change_pct"),
    ("rba_cash_rate_change_bps", "financials_minus_materials_weighted_pct"),
    ("tv_china_steel_property_return_pct", "iron_ore_change_pct"),
    ("regime_numeric", "financials_minus_materials_weighted_pct"),
    ("regime_numeric", "iron_ore_change_pct"),
    ("breadth_score", "financials_minus_materials_weighted_pct"),
    ("breadth_score", "iron_ore_change_pct"),
    ("asian_session_lead_score", "us_equity_lead"),
    ("asian_session_lead_score", "iron_ore_change_pct"),
    ("options_positioning_score", "a_vix"),
    ("news_sentiment_score", "iron_ore_change_pct"),
]


def _clamp(value: float | None, low: float, high: float, default: float = 0.0) -> float:
    if value is None or math.isnan(value):
        return default
    return max(low, min(high, value))


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        f = float(value)
        if math.isnan(f):
            return float("nan")
        return f
    except (TypeError, ValueError):
        return float("nan")


def _bucket_from_return(return_pct: float) -> str:
    """Map an actual signed return to the primary-model bucket label."""
    if return_pct <= -0.6:
        return PRIMARY_LABELS[0]
    if return_pct < 0.6:
        return PRIMARY_LABELS[1]
    return PRIMARY_LABELS[2]


class MLFeatureMapper:
    """Turn a FeatureVector (or dict) into a fixed numeric numpy vector."""

    def __init__(
        self,
        base_features: list[str] | None = None,
        interactions: list[tuple[str, str]] | None = None,
    ):
        self.base_features = list(base_features or ML_BASE_FEATURES)
        self.interactions = list(interactions or ML_INTERACTIONS)
        self.feature_names_ = self.base_features + [f"{a}_x_{b}" for a, b in self.interactions]
        self.fill_: dict[str, float] = {}
        self.fitted = False

    def fit(self, rows: list[dict[str, Any]]) -> None:
        df = pd.DataFrame(rows)
        for col in self.base_features:
            if col in df.columns:
                self.fill_[col] = float(df[col].median())
            else:
                self.fill_[col] = 0.0
        for a, b in self.interactions:
            name = f"{a}_x_{b}"
            if a in df.columns and b in df.columns:
                self.fill_[name] = float((df[a] * df[b]).median())
            else:
                self.fill_[name] = 0.0
        self.fitted = True

    def _base_value(self, row: dict[str, Any], name: str) -> float:
        v = row.get(name)
        f = _safe_float(v)
        if math.isnan(f):
            return self.fill_.get(name, 0.0)
        return f

    def transform_one(self, fv: FeatureVector | dict[str, Any]) -> np.ndarray:
        if isinstance(fv, FeatureVector):
            row = fv.model_dump()
        else:
            row = dict(fv)
        values: list[float] = []
        for name in self.base_features:
            values.append(self._base_value(row, name))
        for a, b in self.interactions:
            av = self._base_value(row, a)
            bv = self._base_value(row, b)
            values.append(av * bv)
        arr = np.array(values, dtype=float)
        for i, name in enumerate(self.feature_names_):
            if math.isnan(arr[i]):
                arr[i] = self.fill_.get(name, 0.0)
        return arr

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.vstack([self.transform_one(r) for r in rows])

    @property
    def feature_names(self) -> list[str]:
        return self.feature_names_

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "MLFeatureMapper | None":
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load ML mapper %s: %s", path, exc)
            return None


class MLModel:
    """Wrap a gradient-boosted / logistic classifier and expose probabilities + importances."""

    def __init__(self, name: str, kind: str = "gbm", classes: list[str] | None = None):
        self.name = name
        self.kind = kind
        self.model: Any = None
        self.classes_: list[str] = list(classes or [])
        self.feature_names_: list[str] = []
        self.importances_: dict[str, float] = {}
        self.label_to_int_: dict[str, int] = {}
        self.int_to_label_: dict[int, str] = {}
        self._train_x: np.ndarray | None = None
        self._train_y: np.ndarray | None = None

    def _pick_classifier(self, n_classes: int):
        """Prefer LightGBM, then XGBoost, then sklearn HistGradient, then LogisticRegression."""
        is_binary = n_classes == 2
        for pkg, cls_name in [("lightgbm", "LGBMClassifier"), ("xgboost", "XGBClassifier")]:
            try:
                module = __import__(pkg, fromlist=[cls_name])
                cls = getattr(module, cls_name)
                if pkg == "lightgbm":
                    return cls(
                        objective="binary" if is_binary else "multiclass",
                        n_estimators=200,
                        learning_rate=0.05,
                        num_leaves=31,
                        reg_alpha=0.01,
                        reg_lambda=0.5,
                        min_child_samples=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        verbosity=-1,
                        n_jobs=1,
                    )
                if pkg == "xgboost":
                    return cls(
                        objective="binary:logistic" if is_binary else "multi:softprob",
                        eval_metric="logloss" if is_binary else "mlogloss",
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        n_jobs=1,
                    )
            except Exception:
                continue
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4)
        except Exception:
            pass
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, multi_class="auto")

    def fit(self, x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
        labels = sorted(set(y))
        self.classes_ = labels
        self.label_to_int_ = {label: i for i, label in enumerate(labels)}
        self.int_to_label_ = {i: label for i, label in enumerate(labels)}
        y_int = np.array([self.label_to_int_[label] for label in y])
        clf = self._pick_classifier(len(labels))
        clf.fit(x, y_int)
        self.model = clf
        self._train_x = x
        self._train_y = y_int
        self.feature_names_ = feature_names
        self.importances_ = self._compute_importances()
        return self._cv_scores(x, y_int)

    def _compute_importances(self) -> dict[str, float]:
        imp: dict[str, float] = {}
        arr: np.ndarray | None = None
        if hasattr(self.model, "feature_importances_"):
            arr = np.asarray(self.model.feature_importances_)
            if not np.any(arr):
                arr = None
        if arr is None and hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_)
            if coef.ndim == 1:
                arr = np.abs(coef)
            else:
                arr = np.mean(np.abs(coef), axis=0)
        if arr is None and self._train_x is not None and self._train_y is not None:
            # Fallback: absolute Pearson correlation with the encoded target.
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.abs(np.corrcoef(self._train_x.T, self._train_y)[:-1, -1])
                corr = np.nan_to_num(corr, nan=0.0)
                arr = corr
            except Exception:
                arr = None
        if arr is None:
            arr = np.zeros(len(self.feature_names_))
        for i, name in enumerate(self.feature_names_):
            imp[name] = float(arr[i]) if i < len(arr) else 0.0
        total = sum(imp.values()) or 1.0
        return {k: round(v / total, 6) for k, v in imp.items()}

    def _cv_scores(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=5)
            accs: list[float] = []
            for train_idx, test_idx in tscv.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 1:
                    continue
                clf = self._pick_classifier(len(self.classes_))
                clf.fit(x_train, y_train)
                preds = clf.predict(x_test)
                accs.append(float(accuracy_score(y_test, preds)))
            return {
                "walk_forward_accuracy": round(float(np.mean(accs)), 4) if accs else None,
                "folds": len(accs),
            }
        except Exception as exc:
            logger.debug("ML CV failed: %s", exc)
            return {"walk_forward_accuracy": None, "folds": 0}

    def predict_proba(self, x: np.ndarray) -> dict[str, float] | list[dict[str, float]] | None:
        if self.model is None:
            return None
        try:
            arr = self.model.predict_proba(x.reshape(1, -1) if x.ndim == 1 else x)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            result = []
            for row in arr:
                result.append(
                    {self.int_to_label_[i]: float(row[i]) for i in range(len(self.classes_))}
                )
            return result[0] if len(result) == 1 else result
        except Exception as exc:
            logger.debug("predict_proba failed: %s", exc)
            return None

    def feature_importance_list(self, top: int = 10) -> list[dict[str, Any]]:
        sorted_imp = sorted(self.importances_.items(), key=lambda x: x[1], reverse=True)
        return [{"feature": k, "importance": round(v, 6)} for k, v in sorted_imp[:top]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "MLModel | None":
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load ML model %s: %s", path, exc)
            return None


class HistoricalFeatureBuilder:
    """Reconstruct historical FeatureVectors for each ASX trading day."""

    def __init__(
        self,
        period: str = "2y",
        min_history: int = 60,
        settings: Settings | None = None,
    ):
        self.period = period
        self.min_history = min_history
        self.settings = settings or get_settings()
        self._cache: dict[str, pd.Series] = {}
        self._asx_df: pd.DataFrame | None = None

    def _close_series(self, ticker: str, period: str | None = None) -> pd.Series:
        fetchers = _fetchers()
        period = period or self.period
        if ticker in self._cache:
            return self._cache[ticker]
        df = fetchers._yf_download([ticker], period=period, interval="1d")
        if df.empty or "Close" not in df.columns:
            self._cache[ticker] = pd.Series(dtype=float)
            return self._cache[ticker]
        s = df["Close"].dropna().copy()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        s.index = idx.normalize()
        self._cache[ticker] = s.sort_index()
        return self._cache[ticker]

    def _asof(self, series: pd.Series, t: datetime) -> float | None:
        try:
            value = series.asof(pd.Timestamp(t))
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return float(value)
        except Exception:
            return None

    def _n_day_return(self, series: pd.Series, t: datetime, n: int) -> float | None:
        if series.empty:
            return None
        cur = self._asof(series, t)
        prev = self._asof(series.shift(n), t)
        if cur is None or prev is None or prev == 0:
            return None
        return (cur / prev - 1.0) * 100.0

    def _first_valid_change(
        self, tickers: list[str], t: datetime, n: int = 1
    ) -> tuple[float | None, str | None]:
        for ticker in tickers:
            s = self._close_series(ticker)
            if s.empty:
                continue
            chg = self._n_day_return(s, t, n)
            if chg is not None:
                return chg, ticker
        return None, None

    def _first_valid_yield_change_bps(
        self, tickers: list[str], t: datetime, n: int = 1
    ) -> tuple[float | None, str | None]:
        """Return the absolute yield change in basis points for the first valid ticker."""
        for ticker in tickers:
            s = self._close_series(ticker)
            if s.empty:
                continue
            cur = self._asof(s, t)
            prev = self._asof(s.shift(n), t)
            if cur is None or prev is None:
                continue
            return (cur - prev) * 100.0, ticker
        return None, None

    def _basket_avg_change(self, tickers: list[str], t: datetime, n: int) -> float | None:
        values = []
        for ticker in tickers:
            chg = self._n_day_return(self._close_series(ticker), t, n)
            if chg is not None:
                values.append(chg)
        return sum(values) / len(values) if values else None

    @staticmethod
    def _diff_or_none(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return a - b

    def _atr_5d(self, df: pd.DataFrame, t: datetime) -> float | None:
        window = df.loc[:t].tail(5)
        if len(window) < 5:
            return None
        try:
            return float(np.mean(window["High"] - window["Low"]))
        except Exception:
            return None

    def _spi_data(self, t: datetime) -> tuple[list[float], str, bool]:
        fetchers = _fetchers()
        for ticker in fetchers.SPI_FUTURES_TICKERS:
            s = self._close_series(ticker)
            if not s.empty and self._asof(s, t) is not None:
                series = s.loc[:t]
                return series.tolist(), ticker, False
        # Fallback to cash proxy.
        s = self._close_series(fetchers.ASX_CASH_TICKERS[0])
        series = s.loc[:t]
        return series.tolist(), fetchers.ASX_CASH_TICKERS[0], True

    def _a_vix_data(self, t: datetime) -> tuple[float | None, list[float], str | None]:
        fetchers = _fetchers()
        for ticker in fetchers.A_VIX_TICKERS:
            s = self._close_series(ticker)
            if not s.empty and self._asof(s, t) is not None:
                series = s.loc[:t]
                return self._asof(s, t), series.tolist(), ticker
        return None, [], None

    def _build_raw_market_data(self, t: datetime) -> RawMarketData:
        fetchers = _fetchers()
        if self._asx_df is None:
            raise RuntimeError("ASX dataframe not loaded")
        df_asx = self._asx_df
        asx_slice = df_asx.loc[:t]

        asx_cash = {
            "ticker": fetchers.ASX_CASH_TICKERS[0],
            "series": {
                "open": asx_slice["Open"].tolist(),
                "high": asx_slice["High"].tolist(),
                "low": asx_slice["Low"].tolist(),
                "close": asx_slice["Close"].tolist(),
                "volume": asx_slice["Volume"].tolist() if "Volume" in asx_slice else [],
            },
            "last_price_date": t.strftime("%Y-%m-%d"),
        }

        spi_series, spi_ticker, spi_proxy = self._spi_data(t)
        spi_futures = {
            "ticker": spi_ticker,
            "series": {"close": spi_series},
            "cash_proxy": spi_proxy,
            "last_price_date": t.strftime("%Y-%m-%d"),
        }

        a_vix_close, a_vix_series, a_vix_ticker = self._a_vix_data(t)
        a_vix: dict[str, Any] = {"ticker": a_vix_ticker}
        if a_vix_close is not None:
            a_vix["close"] = a_vix_close
            a_vix["series"] = {"close": a_vix_series}
            a_vix["last_price_date"] = t.strftime("%Y-%m-%d")

        iron, iron_ticker = self._first_valid_change(fetchers.IRON_ORE_TICKERS, t, 1)
        gold, gold_ticker = self._first_valid_change(fetchers.GOLD_TICKERS, t, 1)
        silver, silver_ticker = self._first_valid_change(fetchers.SILVER_TICKERS, t, 1)
        oil, oil_ticker = self._first_valid_change(fetchers.OIL_TICKERS, t, 1)
        copper, copper_ticker = self._first_valid_change(fetchers.COPPER_TICKERS, t, 1)
        commodities = {
            "iron_ore_change_pct": iron,
            "gold_change_pct": gold,
            "silver_change_pct": silver,
            "oil_change_pct": oil,
            "copper_change_pct": copper,
            "sources": {
                "iron_ore": iron_ticker,
                "gold": gold_ticker,
                "silver": silver_ticker,
                "oil": oil_ticker,
                "copper": copper_ticker,
            },
        }

        aud, _ = self._first_valid_change(fetchers.AUDUSD_TICKERS, t, 1)
        fx = {"aud_usd_change_pct": aud, "ticker": fetchers.AUDUSD_TICKERS[0]}

        sp, sp_t = self._first_valid_change(fetchers.SP500_TICKERS, t, 1)
        nq, nq_t = self._first_valid_change(fetchers.NASDAQ_TICKERS, t, 1)
        dj, dj_t = self._first_valid_change(fetchers.DOW_TICKERS, t, 1)
        vix, vix_t = self._first_valid_change(fetchers.VIX_TICKERS, t, 1)
        us10y, us10y_t = self._first_valid_yield_change_bps(fetchers.US10Y_TICKERS, t, 1)
        us_assets = {
            "us_futures_change_pct": sp,
            "sp500_change_pct": sp,
            "nasdaq_change_pct": nq,
            "dow_change_pct": dj,
            "vix_change_pct": vix,
            "us_10y_change_bps": us10y,
            "sources": {
                "sp500": sp_t,
                "nasdaq": nq_t,
                "dow": dj_t,
                "vix": vix_t,
                "us_10y": us10y_t,
            },
        }

        fin_1d = self._basket_avg_change(fetchers.FINANCIALS_BANKS_TICKERS, t, 1)
        fin_2d = self._basket_avg_change(fetchers.FINANCIALS_BANKS_TICKERS, t, 2)
        fin_3d = self._basket_avg_change(fetchers.FINANCIALS_BANKS_TICKERS, t, 3)
        fin_5d = self._basket_avg_change(fetchers.FINANCIALS_BANKS_TICKERS, t, 5)
        mat_1d = self._basket_avg_change(fetchers.MATERIALS_MINERS_TICKERS, t, 1)
        mat_2d = self._basket_avg_change(fetchers.MATERIALS_MINERS_TICKERS, t, 2)
        mat_3d = self._basket_avg_change(fetchers.MATERIALS_MINERS_TICKERS, t, 3)
        mat_5d = self._basket_avg_change(fetchers.MATERIALS_MINERS_TICKERS, t, 5)
        diffs = {
            "diff_1d_pct": self._diff_or_none(fin_1d, mat_1d),
            "diff_2d_pct": self._diff_or_none(fin_2d, mat_2d),
            "diff_3d_pct": self._diff_or_none(fin_3d, mat_3d),
            "diff_5d_pct": self._diff_or_none(fin_5d, mat_5d),
        }
        weighted = None
        if all(diffs[f"diff_{d}d_pct"] is not None for d in (1, 3, 5)):
            weighted = (
                0.5 * diffs["diff_1d_pct"] + 0.3 * diffs["diff_3d_pct"] + 0.2 * diffs["diff_5d_pct"]
            )
        financials_vs_materials = {**diffs, "weighted_diff_pct": weighted}

        avg_1d = self._basket_avg_change(fetchers.HOUSING_PROXIES_TICKERS, t, 1)
        avg_5d = self._basket_avg_change(fetchers.HOUSING_PROXIES_TICKERS, t, 5)
        pulse = None
        if avg_1d is not None or avg_5d is not None:
            pulse = 5.0 + (avg_1d or 0.0) * 1.5 + (avg_5d or 0.0) * 0.5
            pulse = max(0.0, min(10.0, pulse))
        housing_credit = {"pulse_score": pulse, "sources": fetchers.HOUSING_PROXIES_TICKERS}

        china_weights = {
            "TIO=F": 0.25,
            "HG=F": 0.20,
            "BHP.AX": 0.20,
            "RIO.AX": 0.15,
            "FMG.AX": 0.20,
        }
        weighted_sum = 0.0
        weight_used = 0.0
        per_ticker: dict[str, str] = {}
        for ticker, w in china_weights.items():
            chg = self._n_day_return(self._close_series(ticker), t, 1)
            if chg is not None:
                weighted_sum += w * chg
                weight_used += w
                per_ticker[ticker] = ticker
        composite = weighted_sum / weight_used if weight_used else None
        china_pulse = {
            "composite_return_pct": composite,
            "per_ticker_1d": per_ticker,
            "sources": list(per_ticker.keys()),
        }

        cba = self._n_day_return(self._close_series("CBA.AX"), t, 1)
        bhp = self._n_day_return(self._close_series("BHP.AX"), t, 1)
        weighted_hw = None
        if cba is not None and bhp is not None:
            weighted_hw = 0.55 * cba + 0.45 * bhp
        heavyweight_idio = {
            "weighted_change_pct": weighted_hw,
            "news_boost": 0.0,
            "sources": ["CBA.AX", "BHP.AX"],
        }

        volume: dict[str, Any] = {"fallback": "daily"}
        try:
            row = df_asx.loc[t]
            open_p = float(row["Open"])
            high_p = float(row["High"])
            low_p = float(row["Low"])
            close_p = float(row["Close"])
            vol = float(row["Volume"]) if "Volume" in row else None
            atr_5d = self._atr_5d(df_asx, t)
            session_ret = (close_p - open_p) / open_p * 100.0 if open_p != 0 else None
            vol_ratio = None
            if vol is not None:
                avg_vol_series = df_asx["Volume"].shift(1).rolling(20).mean()
                avg_vol = self._asof(avg_vol_series, t)
                if avg_vol and avg_vol > 0:
                    vol_ratio = vol / avg_vol
            range_ratio = None
            if atr_5d and atr_5d > 0:
                range_ratio = (high_p - low_p) / atr_5d
            volume = {
                "asx_open_to_now_return_pct": session_ret,
                "current_volume_vs_20d_avg": vol_ratio,
                "current_range_vs_atr": range_ratio,
                "session_date": t.strftime("%Y-%m-%d"),
                "fallback": "daily",
            }
        except Exception as exc:
            logger.debug("Volume approx failed for %s: %s", t, exc)

        calendar = {"high_impact_24h": 0, "high_impact_48h": 0}
        breadth: dict[str, Any] = {"breadth_index": None, "breadth_score": None}
        asian_session: dict[str, Any] = {"avg_change_pct": None, "changes_pct": {}}

        source_status = [
            {"name": "asx_cash", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "spi_futures", "status": "ok", "last_success_at": t.isoformat()},
            {
                "name": "a_vix",
                "status": "ok" if a_vix_close is not None else "stale",
                "last_success_at": t.isoformat(),
            },
            {"name": "commodities", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "fx", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "us_assets", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "financials_vs_materials", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "housing_credit", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "china_pulse", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "heavyweight_idio", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "volume", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "breadth", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "asian_session", "status": "ok", "last_success_at": t.isoformat()},
            {"name": "calendar", "status": "ok", "last_success_at": t.isoformat()},
        ]

        return RawMarketData(
            asx_cash=asx_cash,
            spi_futures=spi_futures,
            a_vix=a_vix,
            commodities=commodities,
            fx=fx,
            us_assets=us_assets,
            financials_vs_materials=financials_vs_materials,
            housing_credit=housing_credit,
            china_pulse=china_pulse,
            heavyweight_idio=heavyweight_idio,
            volume=volume,
            calendar=calendar,
            breadth=breadth,
            asian_session=asian_session,
            source_status=source_status,
            errors=[],
        )

    def _fv_to_row(self, fv: FeatureVector, t: datetime, next_return: float) -> dict[str, Any]:
        d = fv.model_dump()

        # Backfill TradingView MCP-derived features with yfinance proxies so the
        # ML model can learn relationships before live MCP data is available.
        if d.get("tv_xjo_trend_score") is None:
            xjo_proxy = _clamp(
                (d.get("rsi_score") or 0.0)
                + (d.get("bollinger_score") or 0.0)
                + (d.get("momentum_exhaustion_score") or 0.0),
                -3.0,
                3.0,
            )
            d["tv_xjo_daily_score"] = xjo_proxy
            d["tv_xjo_weekly_score"] = xjo_proxy
            d["tv_xjo_trend_score"] = xjo_proxy
        if d.get("tv_financials_vs_materials_score") is None:
            d["tv_financials_vs_materials_score"] = d.get("financials_vs_materials_score")
            d["tv_financials_minus_materials_pct"] = d.get(
                "financials_minus_materials_weighted_pct"
            )
        if d.get("tv_heavyweight_avg_score") is None:
            d["tv_heavyweight_avg_score"] = d.get("heavyweight_idio_score")
        if d.get("tv_commodity_basket_change_pct") is None:
            basket = [
                d.get(k) for k in ["iron_ore_change_pct", "copper_change_pct", "oil_change_pct"]
            ]
            basket_values = [v for v in basket if v is not None]
            if basket_values:
                basket_avg = sum(basket_values) / len(basket_values)
                d["tv_commodity_basket_change_pct"] = basket_avg
                d["tv_commodity_basket_ex_gold_change_pct"] = basket_avg
                d["tv_commodity_vs_gold_change_pct"] = basket_avg - (
                    d.get("gold_change_pct") or 0.0
                )

        # Backfill Alpha Vantage MCP-derived features with yfinance proxies for
        # historical training rows (live MCP data only starts after integration).
        if d.get("av_aud_usd_change_pct") is None:
            d["av_aud_usd_change_pct"] = d.get("aud_usd_change_pct")
        if d.get("av_spy_change_pct") is None:
            d["av_spy_change_pct"] = d.get("sp500_change_pct")
        if d.get("av_qqq_change_pct") is None:
            d["av_qqq_change_pct"] = d.get("nasdaq_change_pct")
        if d.get("av_gld_change_pct") is None:
            d["av_gld_change_pct"] = d.get("gold_change_pct")
        if d.get("av_vixy_change_pct") is None:
            d["av_vixy_change_pct"] = d.get("vix_change_pct")
        if d.get("av_us_10y_yield_change_bps") is None:
            d["av_us_10y_yield_change_bps"] = d.get("us_10y_change_bps")
        if d.get("av_us_10y_yield_level") is None:
            d["av_us_10y_yield_level"] = d.get("us_10y_yield_level")

        # Backfill RBA / rates and TradingView China pulse with yfinance proxies.
        if d.get("rba_cash_rate_change_bps") is None:
            # No clean historical AU rates proxy available; leave as neutral.
            d["rba_cash_rate_change_bps"] = 0.0
            d["au_3y_yield_change_bps"] = 0.0
            d["au_10y_yield_change_bps"] = 0.0
            d["rba_rates_score"] = 0.0
        if d.get("tv_china_steel_property_return_pct") is None:
            d["tv_china_steel_property_return_pct"] = d.get("china_steel_property_return_pct")

        # Backfill regime features for historical rows where the engine did not compute them.
        if d.get("regime_numeric") is None:
            d["regime_numeric"] = 0.0
            d["regime_confidence"] = 0.0

        # Backfill breadth and Asian session features for historical rows.
        if d.get("breadth_score") is None:
            d["breadth_score"] = d.get("market_breadth_score") or 0.0
        if d.get("breadth_index") is None:
            d["breadth_index"] = d.get("market_breadth_score") or 0.0
        if d.get("breadth_pct_above_20d_ma") is None:
            d["breadth_pct_above_20d_ma"] = 50.0
            d["breadth_pct_above_50d_ma"] = 50.0
            d["breadth_pct_above_200d_ma"] = 50.0
        if d.get("advance_decline_net") is None:
            d["advance_decline_net"] = 0
            d["new_20d_highs"] = 0
            d["new_20d_lows"] = 0
            d["new_50d_highs"] = 0
            d["new_50d_lows"] = 0
        if d.get("asian_session_lead_score") is None:
            d["asian_session_lead_score"] = d.get("tv_asian_session_change_pct") or 0.0
            if d.get("tv_asian_session_change_pct") is None:
                d["tv_asian_session_change_pct"] = 0.0

        row: dict[str, Any] = {k: d.get(k) for k in ML_BASE_FEATURES}
        row["date"] = t
        row["next_return_pct"] = next_return
        primary_label = _bucket_from_return(next_return)
        row["primary_label"] = primary_label
        if primary_label == "Neutral":
            if next_return >= 0.3:
                row["secondary_label"] = "Mild Bullish Bias"
            elif next_return <= -0.3:
                row["secondary_label"] = "Mild Bearish Bias"
            else:
                row["secondary_label"] = "True Neutral"
        else:
            row["secondary_label"] = None
        return row

    def build(self) -> pd.DataFrame:
        fetchers = _fetchers()
        df_asx = fetchers._yf_download(fetchers.ASX_CASH_TICKERS, period=self.period, interval="1d")
        if df_asx.empty or "Close" not in df_asx.columns:
            logger.error("Could not download ASX cash data for ML training")
            return pd.DataFrame()
        idx = pd.to_datetime(df_asx.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df_asx.index = idx.normalize()
        df_asx = df_asx.sort_index()
        self._asx_df = df_asx

        if len(df_asx) < self.min_history + 2:
            logger.error("Insufficient ASX history for ML training")
            return pd.DataFrame()

        asx_close = df_asx["Close"]
        dates = asx_close.index[self.min_history : -1]
        rows: list[dict[str, Any]] = []

        for t in dates:
            next_val = self._asof(asx_close.shift(-1), t)
            cur = self._asof(asx_close, t)
            if next_val is None or cur is None or cur == 0:
                continue
            next_return = (next_val / cur - 1.0) * 100.0
            try:
                raw = self._build_raw_market_data(t)
                fv, _ = build_features(raw)
                rows.append(self._fv_to_row(fv, t, next_return))
            except Exception as exc:
                logger.debug("Feature build failed for %s: %s", t, exc)
                continue

        return pd.DataFrame(rows)


class MLTrainer:
    """Train, save and reload the hybrid ML models."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.model_dir = Path(model_dir) if model_dir else self.settings.ml_model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, period: str = "2y", min_history: int = 60) -> dict[str, Any]:
        logger.info("Building historical feature dataset for ML training...")
        builder = HistoricalFeatureBuilder(
            period=period, min_history=min_history, settings=self.settings
        )
        df = builder.build()
        if df.empty or len(df) < 100:
            return {
                "status": "error",
                "message": f"insufficient historical data ({len(df)} rows)",
                "rows": len(df),
            }

        logger.info("Training ML models on %s rows", len(df))
        mapper = MLFeatureMapper()
        mapper.fit(df.to_dict("records"))
        x = mapper.transform(df.to_dict("records"))

        primary = MLModel("primary", kind="gbm")
        primary_cv = primary.fit(x, df["primary_label"].values, mapper.feature_names)

        # Binary direction model: the user only needs positive/negative tomorrow,
        # so a dedicated Up/Down classifier is usually more accurate than a
        # 3-class model evaluated on direction.
        binary_label = np.where(df["next_return_pct"].values > 0, "Up", "Down")
        binary = MLModel("binary", kind="gbm")
        binary_cv = binary.fit(x, binary_label, mapper.feature_names)
        binary.save(self.model_dir / "binary.pkl")

        neutral = df[df["primary_label"] == "Neutral"].copy()
        secondary: MLModel | None = None
        secondary_cv = None
        if len(neutral) >= 50 and neutral["secondary_label"].dropna().nunique() >= 2:
            x_sec = mapper.transform(neutral.to_dict("records"))
            y_sec = neutral["secondary_label"].dropna().values
            secondary = MLModel("secondary", kind="gbm")
            secondary_cv = secondary.fit(x_sec, y_sec, mapper.feature_names)
            secondary.save(self.model_dir / "secondary.pkl")
        else:
            logger.warning("Not enough neutral-zone data to train secondary model")

        primary.save(self.model_dir / "primary.pkl")
        mapper.save(self.model_dir / "mapper.pkl")

        metadata = {
            "status": "ok",
            "trained_at": now_sydney().isoformat(),
            "period": period,
            "rows": len(df),
            "neutral_rows": len(neutral),
            "primary_labels": df["primary_label"].value_counts().to_dict(),
            "secondary_labels": neutral["secondary_label"].value_counts().to_dict()
            if not neutral.empty
            else {},
            "primary_cv": primary_cv,
            "binary_cv": binary_cv,
            "secondary_cv": secondary_cv,
            "features": mapper.feature_names,
        }
        with (self.model_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return metadata

    run = train


class HybridML:
    """Load persisted ML models and produce probability forecasts."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.model_dir = Path(model_dir) if model_dir else self.settings.ml_model_dir
        self.mapper: MLFeatureMapper | None = None
        self.primary: MLModel | None = None
        self.secondary: MLModel | None = None
        self.binary: MLModel | None = None
        self.available = False
        self._load()

    def _load(self) -> None:
        mapper_path = self.model_dir / "mapper.pkl"
        if mapper_path.exists():
            self.mapper = MLFeatureMapper.load(mapper_path)
        self.primary = MLModel.load(self.model_dir / "primary.pkl")
        self.secondary = MLModel.load(self.model_dir / "secondary.pkl")
        self.binary = MLModel.load(self.model_dir / "binary.pkl")
        self.available = bool(self.mapper and self.primary)

    def primary_probs(self, fv: FeatureVector) -> dict[str, float] | None:
        if not self.available or self.primary is None or self.mapper is None:
            return None
        x = self.mapper.transform_one(fv).reshape(1, -1)
        return self.primary.predict_proba(x)

    def secondary_probs(self, fv: FeatureVector) -> dict[str, float] | None:
        if not self.available or self.secondary is None or self.mapper is None:
            return None
        x = self.mapper.transform_one(fv).reshape(1, -1)
        return self.secondary.predict_proba(x)

    def binary_probs(self, fv: FeatureVector) -> dict[str, float] | None:
        """Return P(Up) / P(Down) from a dedicated binary direction model."""
        if not self.available or self.binary is None or self.mapper is None:
            return None
        x = self.mapper.transform_one(fv).reshape(1, -1)
        return self.binary.predict_proba(x)

    def feature_importance(self, top: int = 10) -> list[dict[str, Any]]:
        if self.primary:
            return self.primary.feature_importance_list(top)
        return []

    def metadata(self) -> dict[str, Any] | None:
        path = self.model_dir / "metadata.json"
        if not path.exists():
            return None
        try:
            with path.open() as f:
                return json.load(f)
        except Exception as exc:
            logger.debug("Could not read ML metadata: %s", exc)
            return None
