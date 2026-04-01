FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    pydantic pydantic-settings structlog prometheus-client \
    httpx xxhash numpy

# App code
COPY agents/ agents/
COPY services/ services/
COPY models/ models/
COPY configs/ configs/
COPY data/ data/

CMD ["python", "-m", "services.pipeline", "--region=amsterdam", "--log-level=INFO"]
