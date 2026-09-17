# ── Build stage ───────────────────────────────────────────────────────────────
# Separate build stage keeps the runtime image lean (no compilers, no pip cache)
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    pydantic-settings prometheus-client

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Install the package itself (editable equivalent via PYTHONPATH)
ENV PYTHONPATH=/app

# Persistence volumes — mount these to keep data across container restarts
RUN mkdir -p data mlruns models

# Non-root user — never run ML workloads as root in production
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
