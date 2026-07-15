# CoinPredictor image: used both for the always-on Streamlit dashboard and
# for the on-demand daily prediction/evaluation jobs (see docker-compose.yml).
FROM python:3.11-slim

# LightGBM needs libgomp at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so code changes don't bust the layer cache.
COPY pyproject.toml requirements.txt ./
COPY src ./src
# CPU-only torch (much smaller than the default CUDA build) is installed
# explicitly before the rest so FinBERT (transformers) can run locally without
# a GPU. Keeps the image lean while still supporting the free Tier-2 sentiment.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

# Pre-download FinBERT into the image so the daily job never fetches it at run
# time. HF_HOME points at a path that is NOT bind-mounted, so the cached weights
# survive in the image layer. First-run download is ~440 MB; the resulting image
# grows by roughly 0.5 GB (weights) + torch. Set to a lighter tier if disk is
# tight by not registering FinBERTSentimentAdapter in registry.MODELS.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert', top_k=None)" \
    || echo "WARN: FinBERT pre-download failed at build time; it will download on first use."

COPY . .

# data/models/logs are bind-mounted at runtime (see docker-compose.yml); these
# just make sure the paths exist if the image is run without mounts.
RUN mkdir -p data/raw data/processed models logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "coinpredictor.predict"]
