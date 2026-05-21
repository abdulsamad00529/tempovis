# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Installs all Python deps into a virtualenv so the runtime stage stays lean.
FROM python:3.11-slim AS builder

# Build-time system deps (compile extensions, link libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create an isolated virtualenv so we can copy only it to the runtime stage
ENV VIRTUAL_ENV=/app/.venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps only (no compiler toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libfreetype6 \
    libpng16-16 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built virtualenv from builder — keeps this layer small
COPY --from=builder /app/.venv /app/.venv

# Activate the venv for all subsequent RUN / CMD / ENTRYPOINT calls
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy application source (everything except what's in .dockerignore)
COPY app/         ./app/
COPY alembic/     ./alembic/
COPY alembic.ini  ./alembic.ini
COPY pyproject.toml ./pyproject.toml

# Create non-root user before any file writes
RUN groupadd --system tempovis && \
    useradd --system --gid tempovis --no-create-home tempovis && \
    mkdir -p /app/data/plots /app/data && \
    chown -R tempovis:tempovis /app

USER tempovis

# Runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the API.
# Overridden in docker-compose for the worker service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
