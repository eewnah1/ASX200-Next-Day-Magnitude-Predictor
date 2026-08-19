# ASX200 Predictor — Developer Guide

This guide covers the new regime-aware scoring, optional enrichment layers, and how to add future data sources without breaking the prediction pipeline.

## Architecture

- `data/fetchers.py` — live data orchestration (yfinance, TradingView MCP, Alpha Vantage MCP, calendar, news, options).
- `scoring/features.py` — normalises raw data into a `FeatureVector` and `DataQualityFlags`.
- `scoring/engine.py` — regime detection, two-model scoring, hard/soft gates, sizing/gap notes.
- `scoring/ml.py` — hybrid ML layer with `ML_BASE_FEATURES` and interaction terms.
- `models.py` — Pydantic request/response models and `Prediction`.
- `storage/` — SQLite repository with prediction/actual records and rolling calibration by regime.

## Regime logic

`_detect_regime()` in `scoring/engine.py` classifies each prediction into one of:

- `materials_led`
- `financials_led`
- `dual_engine`
- `contested`
- `range` / fallback

The regime is driven by `financials_minus_materials_weighted_pct`, `china_steel_property_score`, `iron_ore_change_pct`, `us_equity_lead`, and `a_vix`.  It is reported in `Prediction.regime` and persisted to the DB so calibration can be computed by regime.

## Optional enrichment layers

### News / sentiment

- `NewsSentimentFetcher` in `data/news_sentiment_fetcher.py` tries `NewsAPI` then `MarketAux`.
- Config: `news_sentiment_enabled`, `news_headlines_per_entity`, `newsapi_api_key`, `marketaux_api_key`.
- Output is a `SentimentResult` (or dict in `DataFetcher`) with `status`, `score`, `components`.
- `features.py` maps `news_sentiment_score` and `news_sentiment_components` to the `FeatureVector`.
- The ML layer includes `news_sentiment_score`; the primary model shows the factor with weight `0` so it does not disturb calibrated scores.

### Options / positioning

- `OptionsPositioningFetcher` in `data/options_positioning_fetcher.py` tries `AP=F`, `^AXJO`, `SPY`, `QQQ`, `IWM`, `EWA`.
- Config: `options_positioning_enabled`.
- Returns `options_positioning_score`, `options_positioning_note`.
- Wired into `FeatureVector` and `ML_BASE_FEATURES`.

Both layers degrade cleanly: missing keys, timeouts, or empty responses produce `status="failed"` / `score=None` with no hard gate.

## Production hardening

- `DataQualityFlags` tracks every source.
- `_hard_gate_triggered()` blocks a directional recommendation only when a **critical** source fails (`spi_futures`, `commodities`, `financials_vs_materials`, `us_assets`, `asian_session`, `rba_rates`, `asx_cash`).
- `_soft_gate_penalty()` reduces confidence by up to `0.25` for each degraded source.
- Every prediction receives `model_version`, `audit_log_id`, `regime`, `sizing_guidance`, and `gap_risk_note`.
- `Repository.calibration_metrics()` groups hit-rate by regime.

## Adding a future MCP source

1. Add a thin fetcher under `data/` that returns a plain `dict` or `FetchResult`.
2. Call it inside `DataFetcher.fetch_all()` and append its `FetchResult` to `results`.
3. Pass its raw payload into `RawMarketData`.
4. Map the raw payload to typed `FeatureVector` fields in `scoring/features.py`.
5. Add the feature to `ML_BASE_FEATURES` (and optional interactions to `ML_INTERACTIONS`) in `scoring/ml.py`.
6. Update `DataQualityFlags` and `Prediction` in `models.py` if it needs its own flag.
7. Keep it optional: any failure should leave `FeatureVector` values at `None` / `0` and set a degraded flag, never raise.

## SPI 200 freshness

`_is_spi_fresh()` in `data/fetchers.py` accepts a bar whose date is the same as (or later than) the previous ASX close, or whose age in hours is within `spi_freshness_hours` (default `96`).  This prevents spurious degradation when the SPI/cash daily bar is released one session late (e.g. vendor lag over a weekend).

## Testing

- `python3 -m pytest tests -q`
- `python3 -m ruff check src tests notebooks`
- Smoke test: `curl -X POST http://127.0.0.1:8004/api/v1/predict -H 'Content-Type: application/json' -d '{}'`
