# Match the Python patch version recorded in the Codespaces-built FAISS index.
FROM python:3.13.18-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin banking

COPY requirements-forecast.txt requirements-retrieval.txt requirements-assistant.txt \
     requirements-api.txt requirements-dashboard.txt ./
RUN python -m pip install --no-cache-dir -r requirements-dashboard.txt \
    && python -m pip check

# Only application source is copied. The .dockerignore and bind mounts keep
# original datasets, model artifacts, reports, and credentials outside the image.
COPY banking_observability/ ./banking_observability/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY knowledge/ ./knowledge/
COPY .streamlit/ ./.streamlit/
COPY streamlit_app.py ./streamlit_app.py

RUN mkdir -p /app/data/raw /app/models
USER banking
EXPOSE 8000 8501
