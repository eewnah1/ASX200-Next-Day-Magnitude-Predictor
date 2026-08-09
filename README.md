# ASX200 Next-Day Magnitude Predictor

A production-quality Python application that produces calibrated probabilities for the next ASX 200 trading day's absolute percentage move.

## Goal

Before 2pm AEST on ASX trading days, predict whether the next ASX 200 cash session will move:

- `Negative (<0%)`
- `0% – 0.3%`
- `0.3% – 0.5%`
- `> 0.5%`

The app combines a rule-based scoring engine with configurable weights, persistent prediction/actual tracking, and a clean FastAPI + Streamlit interface. It is designed so that an ML calibrator can be dropped in later without changing the data or API contracts.

## Architecture

| Component | Purpose |
|-----------|---------|
| `src/asx200_mag_predictor/data/fetchers.py` | Robust yfinance fetchers with primary + fallback symbols, JSON snapshot caching |
| `src/asx200_mag_predictor/scoring/features.py` | Feature engineering: volatility regime, catalyst score, cross-asset alignment, session character, SPI basis, plus Financials vs Materials, housing/credit pulse, China steel/property and CBA+BHP idiosyncratic factors |
| `src/asx200_mag_predictor/scoring/engine.py` | Rule-based logit-prob engine with configurable weights and confidence |
| `src/asx200_mag_predictor/storage/` | SQLAlchemy models + repository for predictions, actuals, snapshots |
| `src/asx200_mag_predictor/api/` | FastAPI backend (status, predict, history, calibration) |
| `src/asx200_mag_predictor/ui/app.py` | Streamlit dashboard |
| `src/asx200_mag_predictor/scheduler/jobs.py` | APScheduler daily run ~13:30 AEST and optional US-close update |
| `notebooks/backtest.py` | Historical backtest script |

## Tech stack

- Python 3.10+
- FastAPI + Uvicorn
- Streamlit
- SQLAlchemy 2.0 (SQLite default, Postgres via `DATABASE_URL`)
- APScheduler
- yfinance + requests (NewsAPI / MarketAux calendar fallbacks)
- pytest + ruff

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cp .env.example .env
# Edit .env and add NEWSAPI_API_KEY / MARKETAUX_API_KEY if you want the calendar fetcher.
```

## Quick start

Generate a real-data prediction:

```bash
python -m asx200_mag_predictor.cli predict
```

Run a mock prediction (no network):

```bash
python -m asx200_mag_predictor.cli predict --mock
```

Run the FastAPI backend:

```bash
uvicorn asx200_mag_predictor.api.main:app --reload --port 8000
```

Run the Streamlit UI:

```bash
streamlit run src/asx200_mag_predictor/ui/app.py --server.port 8501
```

Run the scheduler in the background:

```bash
python -m asx200_mag_predictor.cli run-scheduler
```

Record the actual next-day move for calibration:

```bash
python -m asx200_mag_predictor.cli record-actual 2026-08-10 0.42
```

## Scoring logic

The scoring engine is intentionally rule-based and well-commented.

1. **Volatility regime** (`0` = very quiet, `4` = very volatile) is derived from A-VIX (or `^VIX` fallback), recent 5-day ATR%, and realized annualized volatility via the rule-of-16.
2. **Catalyst score** (`0–5`) is based on the number of high-impact economic/central-bank headlines in the next 24–48 hours.
3. **Cross-asset alignment** combines US futures, iron ore, gold, oil, copper, AUD/USD, US 10y, and VIX into a signed alignment score and magnitude.
4. **ASX session character** classifies the current session as `trending`, `range-bound`, `mixed`, or `unknown`.
5. **SPI basis / futures momentum** compares SPI futures (or cash proxy) to the ASX 200 cash.

Each regime has a baseline probability distribution. Factor logit-deltas are added and softmaxed to produce the final three probabilities. A confidence score measures how far the predicted distribution is from the baseline.

See:

- `src/asx200_mag_predictor/scoring/engine.py`
- `src/asx200_mag_predictor/scoring/features.py`

## Configuration

All runtime settings are loaded from `.env` via `pydantic-settings`:

- `DATABASE_URL` — default `sqlite:///./asx200_predictor.db`; use Postgres in production.
- `NEWSAPI_API_KEY` / `MARKETAUX_API_KEY` — calendar fallback chain.
- `SCHEDULE_DAILY`, `DAILY_RUN_TIME`, `US_CLOSE_RUN_TIME` — scheduler settings.
- `WEIGHT_*` variables to tweak scoring without touching code.

See `.env.example` for the full list.

## Data sources & fallbacks

| Data | Primary | Fallback |
|------|---------|----------|
| ASX 200 cash | `^AXJO` | — |
| SPI 200 futures | `AP=F` | `^AP`, `SPI1.AX`, `^AXJO` |
| A-VIX | `^A-VIX` | `^VIX` |
| Iron ore | `FE=F` | `TIO=F`, `MT=F` |
| Gold | `GC=F` | — |
| Oil | `CL=F` | — |
| Copper | `HG=F` | — |
| AUD/USD | `AUDUSD=X` | — |
| S&P 500 | `^GSPC` | `ES=F` |
| Nasdaq | `^IXIC` | `NQ=F` |
| Dow | `^DJI` | `YM=F` |
| US 10y | `^TNX` | `^FVX` |
| VIX | `^VIX` | — |
| Calendar | NewsAPI | MarketAux |
| Intraday ASX | 5m yfinance | daily open/close |

### Historical factor coverage & limitations

- `^AXJO` is the anchor and currently provides data back to ~1992. All other
  series are joined to its trading days using `asof` lookups, so a value may be
  the last available close from a non-ASX holiday.
- SPI 200 futures proxies (`AP=F`, `^AP`, `SPI1.AX`) are often unavailable in
  `yfinance` for long history; the builder falls back to `^AXJO` itself for the
  SPI basis/momentum factors. This is documented in the factor table.
- `^A-VIX` has very limited free history, so `^VIX` is used as the volatility
  proxy before `^A-VIX` becomes available.
- Iron ore (`FE=F`, `TIO=F`, `MT=F`) is frequently missing; the fallback chain
  uses the major miners (`BHP.AX`, `RIO.AX`, `FMG.AX`) and commodity proxies.
- Housing & credit and China steel/property pulses are constructed from equity
  proxies (REA/GMG/SCG/LLC, BHP/RIO/FMG/HG) because macro series with daily
  history are not freely available through the chosen sources.
- Older `^AXJO` candles from Yahoo Finance can have `Open=High=Low=Close` and
  zero volume, which makes intraday/session factors default to neutral for those
  dates; the model still uses overnight US, commodity and FX inputs.

See `notebooks/build_historical_factors.py` for the full construction pipeline.

## Tests

```bash
python -m pytest tests -q
ruff check src tests notebooks
```

## Historical factor data & backtest

Build a full daily factor table. The script uses `yfinance` with the same ticker
chains the live fetchers use, falls back transparently when a series is
unavailable, and stores the result as Parquet + CSV in `data/`.

```bash
python notebooks/build_historical_factors.py --period max --output data/historical_factors.parquet --csv data/historical_factors.csv
```

Run the day-by-day backtest. By default it uses a configurable primary rule-score
threshold (score strategy), which produces enough signals for meaningful
statistics. You can also replay the exact `ScoringEngine` recommendation logic
(engine strategy), and enable the trained ML layer with `--ml`.

```bash
# Configurable rule-score strategy (default)
python notebooks/historical_backtest.py --factors data/historical_factors.parquet --start 2010-01-01 --strategy score --primary-threshold 0.6

# Exact engine logic
python notebooks/historical_backtest.py --factors data/historical_factors.parquet --start 2010-01-01 --strategy engine
```

Example output (2010–present, rule-score threshold 0.6, no ML):

| Metric | Value |
|--------|-------|
| Signal days | 368 (8.8% of all days) |
| Hit rate | 64.7% |
| Avg return on signal days | +0.222% |
| Simple total return (signals) | +81.58% |
| Buy & hold annualised | ~4.97% |
| Buy & hold up days | 53.6% |

The original `notebooks/backtest.py` quick script is still available:

```bash
python notebooks/backtest.py --months 12 --mock
python notebooks/backtest.py --months 3
```

## Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501

## Extending

To add an ML calibrator later:

1. Train on `PredictionRecord.feature_vector_json` and the stored `ActualRecord` bucket.
2. Implement a `MLCalibrator.score(fv)` that returns `BucketProbabilities`.
3. In `ScoringEngine.predict()`, call the calibrator after the rule-based distribution or blend the two with a configurable `ml_weight`.

## License

MIT
