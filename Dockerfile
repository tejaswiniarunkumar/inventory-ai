# ── Base image ─────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ────────────────────────────────

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy pipeline code ─────────────────────────────────────────
COPY pipeline/ ./pipeline/
COPY models/   ./models/

# ── Create output directories ──────────────────────────────────
RUN mkdir -p outputs/forecasts \
             outputs/summaries \
             outputs/evaluation \
             data/raw \
             data/processed

# ── Environment variables ──────────────────────────────────────
ENV ANTHROPIC_API_KEY=""
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ─────────────────────────────────────────────────
# Usage:
#   docker run pipeline --mode train
#   docker run pipeline --mode predict
ENTRYPOINT ["python", "pipeline/run_pipeline.py"]
