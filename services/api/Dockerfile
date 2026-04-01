FROM python:3.11-slim

WORKDIR /app

# System deps (for psycopg2, etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic pydantic-settings \
    structlog prometheus-client httpx xxhash numpy

# App code
COPY agents/ agents/
COPY services/ services/
COPY models/ models/
COPY configs/ configs/
COPY data/ data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python", "-m", "services.api", "--host", "0.0.0.0", "--port", "8000"]
