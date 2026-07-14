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
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

COPY . .

# data/models/logs are bind-mounted at runtime (see docker-compose.yml); these
# just make sure the paths exist if the image is run without mounts.
RUN mkdir -p data/raw data/processed models logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "coinpredictor.predict"]
