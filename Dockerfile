FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_STORE_DIR=/app/data_store

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY alembic.ini pyproject.toml /app/
COPY crypto_quant_lab /app/crypto_quant_lab

RUN groupadd --gid 10001 cql \
    && useradd --uid 10001 --gid cql --shell /usr/sbin/nologin --create-home cql \
    && mkdir -p /app/data_store \
    && chown -R cql:cql /app

USER cql

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:3000/health || exit 1

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn crypto_quant_lab.app.main:app --host 0.0.0.0 --port 3000 --proxy-headers --forwarded-allow-ips=*"]
