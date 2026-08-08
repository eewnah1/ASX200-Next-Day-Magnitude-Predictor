FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

# Copy application code
COPY main.py ./main.py
COPY src/asx200_mag_predictor/api/dashboard.html ./src/asx200_mag_predictor/api/dashboard.html

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
