# 🏠 WoningScout

**Autonomous Dutch Housing Market Scout** — a multi-agent ML system that monitors the Dutch property market, predicts fair value, and alerts you when something's underpriced.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What This Does

The Netherlands has a housing shortage of ~390,000 homes. Good listings disappear within days. Buyers refresh Funda 20+ times a day and bid blindly because they have no data on what a property is actually worth.

WoningScout automates the parts of this process that don't require a human:

1. **Monitor** — Polls Funda for new listings every 5 minutes
2. **Analyze** — Engineers 47 features (structural, geospatial, NLP on Dutch descriptions, market context)
3. **Predict** — XGBoost ensemble estimates fair market value with 90% confidence intervals
4. **Compare** — FAISS similarity search finds the most comparable recently-sold properties
5. **Alert** — Notifies you when something is ≥5% undervalued in your target area

The whole thing runs as 5 autonomous agents orchestrated via LangGraph, deployed with Docker Compose, and monitored with Prometheus + Grafana.

## Architecture

```
Funda API ──→ [A1: Ingestion] ──→ [A2: Features] ──→ [A3: Prediction] ──→ [A4: Scoring] ──→ [A5: Alert]
                  │                     │                    │                   │                  │
              deduplicate          47 features          XGBoost + CI        FAISS + livability   email/telegram
              validate             geospatial           shadow model        comparables          rate-limited
              persist              NLP (Dutch)          drift detection     value ratio          nl/en templates
```

**Why not LLM agents?** Because the tasks are deterministic. Ingestion is an API call. Feature engineering is math. Prediction is model inference. LangGraph gives me orchestration primitives (state passing, conditional edges, checkpointing) without the latency and cost of calling an LLM for things a Python function does better.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose v2
- 8GB RAM (FAISS index + spaCy model are memory-hungry)
- Funda API key (optional — leave blank to run in demo mode with fixture data)

### Run with Docker (recommended)

```bash
git clone https://github.com/prakharprakarsh/woningscout.git
cd woningscout

# Configure
cp .env.example .env
# Edit .env if you have a Funda API key, otherwise demo mode works out of the box

# Start everything (API, pipeline, Postgres, Redis, Prometheus, Grafana)
docker compose up -d

# Check it's running
curl http://localhost:8000/health

# Trigger a pipeline run
curl -X POST http://localhost:8000/pipeline/run -H "Content-Type: application/json" -d '{"regions": ["amsterdam"]}'

# Check results
curl http://localhost:8000/listings?undervalued_only=true
```

**Endpoints:**
- API + Swagger docs: http://localhost:8000/docs
- Grafana dashboard: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Run locally (no Docker)

```bash
# Clone and install
git clone https://github.com/prakharprakarsh/woningscout.git
cd woningscout
pip install -e ".[dev]"

# Download Dutch spaCy model (optional, for full NLP features)
python -m spacy download nl_core_news_lg

# Run the pipeline once
python -m services.pipeline --region=amsterdam --once

# Or start the API server
python -m services.api --reload
```

### Run tests

```bash
pytest tests/ -v --cov=agents --cov-report=term-missing
```

## Project Structure

```
woningscout/
├── agents/                  # The 5 autonomous agents
│   ├── base.py              # Abstract agent interface + Prometheus metrics
│   ├── config.py            # Centralized settings (from .env)
│   ├── schemas.py           # Pydantic models for all data types
│   ├── ingestion.py         # A1: Funda API → deduplicate → validate → store
│   ├── features.py          # A2: 47 features (geo, NLP, market, temporal)
│   ├── prediction.py        # A3: XGBoost + bootstrap CI + drift detection
│   ├── scoring.py           # A4: FAISS comparables + livability score
│   └── alerting.py          # A5: User matching + multi-channel dispatch
├── services/
│   ├── api/                 # FastAPI application
│   │   ├── app.py           # Endpoints: health, metrics, listings, pipeline
│   │   └── Dockerfile
│   └── pipeline/            # Pipeline orchestrator + CLI
│       ├── orchestrator.py  # LangGraph DAG + fallback simple pipeline
│       └── Dockerfile
├── models/                  # Model artifacts (gitignored, created by training)
├── configs/
│   └── xgb_v3.yaml          # XGBoost training config (Optuna search space)
├── monitoring/
│   ├── prometheus.yml        # Scrape config
│   ├── dashboards/           # Grafana dashboard JSON
│   └── provisioning/         # Auto-provision datasources
├── scripts/
│   └── init_db.sql           # Database schema (7 tables)
├── data/fixtures/            # Demo data for running without Funda API
├── tests/                    # pytest suite (8 test modules)
├── docker-compose.yml        # Full stack: API, pipeline, Postgres, Redis, Prometheus, Grafana
├── pyproject.toml            # Dependencies + tool config
└── .env.example              # Configuration template
```

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| ML Model | XGBoost (Optuna-tuned) | Best performer on tabular data; 0.891 R² on temporal hold-out |
| Similarity Search | FAISS (IVF2048,Flat) | Sub-millisecond K-NN for comparable properties |
| Agent Orchestration | LangGraph | State management + conditional edges without LLM overhead |
| Feature Store | Redis | 7-day TTL cache for expensive geo features |
| API | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation |
| Database | PostgreSQL 16 | Listings, predictions, user preferences |
| Monitoring | Prometheus + Grafana | Agent run duration, error rates, drift detection |
| NLP | spaCy (nl_core_news_lg) | Dutch description analysis (sentiment, keywords) |
| Deployment | Docker Compose | 6 containers, health checks, auto-restart |

## ML Pipeline Details

### Features (47 across 5 categories)

- **Structural (9):** living area, rooms, bathrooms, build year, energy label, garden, balcony, parking, property type
- **Geospatial (12):** lat/lng, distance to station/centrum/school, supermarket density, green space %, noise, elevation
- **NLP (8):** Dutch sentiment, luxury keywords, renovation detection, description length, English detection
- **Market Context (13):** price/m² for postcode area, days on market, YoY change, bid ratio, income level, mortgage rate
- **Temporal (5):** month, day of week, school holiday, rate change recency, 30-day momentum

### Model Performance

| Model | R² (test) | MAE | MAPE | Notes |
|---|---|---|---|---|
| **XGBoost (Optuna)** | **0.891** | **€16,420** | **5.3%** | Production — 200 trials |
| LightGBM | 0.884 | €17,890 | 5.7% | Shadow model |
| CatBoost | 0.879 | €18,340 | 5.9% | Evaluated |
| Random Forest | 0.852 | €21,100 | 6.8% | Baseline |
| Ridge + poly | 0.791 | €28,400 | 9.2% | Baseline |
| MLP (3 layers) | 0.863 | €19,700 | 6.4% | Abandoned — overfit |

**Evaluation methodology:**
- Temporal split: train (pre-Oct 2023), val (Oct–Dec 2023), test (Jan–Apr 2024)
- No random splitting — avoids leaking future market conditions
- 5-fold stratified CV by province for hyperparameter selection
- SHAP values for interpretability

### Why 0.891 and not 0.94+?

The 0.94+ R² scores you see in Kaggle housing notebooks usually come from random train/test splits (data leakage) or training on a single city. This model is evaluated on a temporal hold-out across all 12 Dutch provinces. 0.891 is an honest number.

## Monitoring

The pipeline exposes Prometheus metrics for:

- **Agent run duration** (histogram, per agent)
- **Agent success/error counts** (counter, per agent)
- **Agent errors by type** (counter)
- **Prediction drift** (PSI, threshold 0.12)
- **API latency** (via FastAPI middleware)

Grafana dashboard auto-provisions on startup at http://localhost:3000.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (Docker/k8s probe) |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/pipeline/run` | Trigger a pipeline scan |
| `GET` | `/pipeline/status` | Last run results |
| `GET` | `/listings` | Scored listings (filterable) |
| `GET` | `/listings/{id}` | Single listing detail + comparables |

Full Swagger docs at `/docs` when the API is running.

## Known Limitations & Tradeoffs

I built this as a portfolio project and learning exercise — it's production-grade in structure but has some honest limitations:

- **Funda API access** requires a partner agreement. Demo mode uses fixture data.
- **Dutch NLP** struggles with compound words (e.g., *driekamerappartement*). Custom splitter covers ~200 terms but doesn't catch everything.
- **Confidence intervals** are bootstrap approximations (n=80). Coverage is 89.2% on a stated 90% interval — drops to ~82% in rural areas with sparse data.
- **Livability weights** are based on an informal survey of ~40 people, biased toward urban young professionals.
- **No image features** (yet). Property photos affect value, but adding a ResNet pipeline was out of scope. It's proxied through NLP keywords for now.

See the [Tradeoffs section in the showcase](dutch-housing-scout.jsx) for more detail on what went wrong and what I'd do differently.

## Roadmap

- [ ] Photo-based condition scoring (ResNet18 on listing images)
- [ ] Fine-tune Dutch BERT on listing descriptions
- [ ] Proper A/B testing framework for model versions
- [ ] GNN on neighborhood relationships
- [ ] User feedback loop — retrain on reported wrong predictions
- [ ] Expand beyond Funda (Pararius, Jaap)

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Prakhar](https://linkedin.com/in/pprakarsh04) — Python, XGBoost, FAISS, LangGraph, FastAPI, Docker*
