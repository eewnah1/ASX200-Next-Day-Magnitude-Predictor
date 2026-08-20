FROM python:3.11-slim-bookworm

WORKDIR /app

# System packages required by lightgbm, xgboost, pandas, scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better layer caching
COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py ./

# Install the package (includes all runtime deps)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Ensure data directory exists (used for SQLite + ML models).
# Also pre-seed ML models into the image so a cold / empty persistent volume
# still has working models on first boot (ensure_seed_ml_models + lifespan
# will copy/seed again at runtime if needed).
RUN mkdir -p /data/ml_models && chmod -R 777 /data && \
    if [ -d src/asx200_mag_predictor/data/seed_ml_models ]; then \
      cp -f src/asx200_mag_predictor/data/seed_ml_models/* /data/ml_models/ || true; \
    fi

ENV APP_ENV=production \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/asx200_predictor.db \
    DATA_DIR=/data \
    TZ=Australia/Sydney

EXPOSE 8000

# Healthcheck used by Fly / Render / Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
