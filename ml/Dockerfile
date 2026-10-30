# ============================================================
# ML Worker Dockerfile
# Multi-stage: builder → runtime
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv para gestión de dependencias rápida
RUN pip install --no-cache-dir uv==0.3.0

COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes instalados desde builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Usuario no-root por seguridad
RUN useradd -m -u 1001 mlworker
USER mlworker

COPY --chown=mlworker:mlworker src/ /app/src/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "src.main"]
