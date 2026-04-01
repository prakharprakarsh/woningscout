"""
WoningScout API

FastAPI application providing:
- GET  /health           → health check for Docker/k8s
- GET  /metrics          → Prometheus metrics endpoint
- POST /pipeline/run     → trigger a pipeline run
- GET  /pipeline/status  → last run results
- GET  /listings         → scored listings from last run
- GET  /listings/{id}    → single listing detail with comparables

In production, this would also have:
- WebSocket /ws/alerts   → live alerts stream
- POST /users            → manage user preferences
- GET  /comparables/{id} → FAISS similarity for a specific listing

For now we keep it focused on the core pipeline endpoints.
The API is intentionally thin — most logic lives in the agents.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from agents.config import settings
from agents.schemas import PipelineState, ScoredProperty


logger = structlog.get_logger()


# ── State store (in-memory for demo, Redis/Postgres in production) ────

class AppState:
    """In-memory store for pipeline results.

    In production, this would be Postgres queries. For demo,
    we just keep the last pipeline result in memory.
    """

    def __init__(self):
        self.last_run: Optional[PipelineState] = None
        self.last_run_at: Optional[datetime] = None
        self.total_runs: int = 0
        self.is_running: bool = False
        self.startup_time: datetime = datetime.utcnow()

    @property
    def scored_listings(self) -> list[ScoredProperty]:
        if self.last_run is None:
            return []
        return self.last_run.scored_properties

    def get_listing(self, listing_id: str) -> Optional[ScoredProperty]:
        return next(
            (l for l in self.scored_listings if l.listing_id == listing_id),
            None,
        )


app_state = AppState()


# ── Lifespan (startup/shutdown) ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle hooks."""
    logger.info(
        "api_startup",
        mode="demo" if settings.is_demo_mode else "live",
        regions=settings.regions_list,
    )
    yield
    logger.info("api_shutdown")


# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(
    title="WoningScout API",
    description=(
        "Autonomous Dutch Housing Market Scout — "
        "property valuation, comparable search, and alert service."
    ),
    version="0.9.2",
    lifespan=lifespan,
)

# CORS — permissive for demo, lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    regions: list[str] = Field(
        default_factory=list,
        description="Regions to scan. Empty = use default from config.",
    )


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    ingested: int = 0
    scored: int = 0
    undervalued: int = 0
    alerts_sent: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    mode: str
    uptime_seconds: float
    total_runs: int
    last_run_at: Optional[str] = None
    version: str = "0.9.2"


class ListingSummary(BaseModel):
    listing_id: str
    asking_price: float
    predicted_price: float
    value_ratio: float
    undervalued_pct: float
    livability_score: float
    is_undervalued: bool


class ListingDetail(BaseModel):
    listing_id: str
    asking_price: float
    predicted_price: float
    ci_lower: float
    ci_upper: float
    value_ratio: float
    undervalued_pct: float
    livability: dict
    comparables: list[dict]
    is_undervalued: bool


# ── Health & Metrics ──────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for Docker/k8s probes."""
    uptime = (datetime.utcnow() - app_state.startup_time).total_seconds()
    return HealthResponse(
        status="healthy",
        mode="demo" if settings.is_demo_mode else "live",
        uptime_seconds=round(uptime, 1),
        total_runs=app_state.total_runs,
        last_run_at=app_state.last_run_at.isoformat() if app_state.last_run_at else None,
    )


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus metrics endpoint.

    Exposes agent-level metrics (run duration, error counts)
    plus application-level gauges.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return PlainTextResponse(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )


# ── Pipeline endpoints ────────────────────────────────────────────────

@app.post("/pipeline/run", response_model=PipelineRunResponse, tags=["Pipeline"])
async def trigger_pipeline_run(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger a pipeline run.

    Runs asynchronously in the background. Poll /pipeline/status
    to check results.
    """
    if app_state.is_running:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running. Wait for it to finish.",
        )

    regions = request.regions or settings.regions_list

    async def _run():
        from services.pipeline.orchestrator import run_pipeline

        app_state.is_running = True
        try:
            result = await run_pipeline(regions=regions)
            app_state.last_run = result
            app_state.last_run_at = datetime.utcnow()
            app_state.total_runs += 1
        except Exception as e:
            logger.error("pipeline_run_failed_via_api", error=str(e))
        finally:
            app_state.is_running = False

    background_tasks.add_task(_run)

    return PipelineRunResponse(
        run_id="pending",
        status="started",
        message=f"Pipeline started for regions: {regions}. Poll /pipeline/status for results.",
    )


@app.get("/pipeline/status", response_model=PipelineRunResponse, tags=["Pipeline"])
async def pipeline_status():
    """Get the status and results of the last pipeline run."""
    if app_state.is_running:
        return PipelineRunResponse(
            run_id="in-progress",
            status="running",
            message="Pipeline is currently running...",
        )

    if app_state.last_run is None:
        return PipelineRunResponse(
            run_id="none",
            status="idle",
            message="No pipeline runs yet. POST /pipeline/run to start one.",
        )

    run = app_state.last_run
    return PipelineRunResponse(
        run_id=run.run_id,
        status="completed",
        ingested=len(run.new_listing_ids),
        scored=len(run.scored_properties),
        undervalued=sum(1 for p in run.scored_properties if p.is_undervalued),
        alerts_sent=run.alerts_sent,
        message=run.summary,
    )


# ── Listing endpoints ─────────────────────────────────────────────────

@app.get("/listings", response_model=list[ListingSummary], tags=["Listings"])
async def get_listings(
    undervalued_only: bool = Query(False, description="Only show undervalued properties"),
    min_value_ratio: float = Query(0.0, description="Minimum value ratio filter"),
    sort_by: str = Query("value_ratio", description="Sort field: value_ratio, predicted_price, livability_score"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
):
    """Get scored listings from the last pipeline run.

    Supports filtering by undervalued status and value ratio,
    with sorting options.
    """
    listings = app_state.scored_listings

    if not listings:
        return []

    # Filters
    if undervalued_only:
        listings = [l for l in listings if l.is_undervalued]
    if min_value_ratio > 0:
        listings = [l for l in listings if l.value_ratio >= min_value_ratio]

    # Sort
    sort_key_map = {
        "value_ratio": lambda x: x.value_ratio,
        "predicted_price": lambda x: x.predicted_price,
        "livability_score": lambda x: x.livability.composite,
        "asking_price": lambda x: x.asking_price,
    }
    sort_fn = sort_key_map.get(sort_by, sort_key_map["value_ratio"])
    listings = sorted(listings, key=sort_fn, reverse=True)

    # Limit
    listings = listings[:limit]

    return [
        ListingSummary(
            listing_id=l.listing_id,
            asking_price=l.asking_price,
            predicted_price=l.predicted_price,
            value_ratio=l.value_ratio,
            undervalued_pct=l.undervalued_pct,
            livability_score=l.livability.composite,
            is_undervalued=l.is_undervalued,
        )
        for l in listings
    ]


@app.get("/listings/{listing_id}", response_model=ListingDetail, tags=["Listings"])
async def get_listing_detail(listing_id: str):
    """Get full details for a single listing including comparables."""
    listing = app_state.get_listing(listing_id)
    if listing is None:
        raise HTTPException(
            status_code=404,
            detail=f"Listing {listing_id} not found. Run the pipeline first.",
        )

    return ListingDetail(
        listing_id=listing.listing_id,
        asking_price=listing.asking_price,
        predicted_price=listing.predicted_price,
        ci_lower=listing.ci_lower,
        ci_upper=listing.ci_upper,
        value_ratio=listing.value_ratio,
        undervalued_pct=listing.undervalued_pct,
        livability=listing.livability.model_dump(),
        comparables=[c.model_dump() for c in listing.comparables],
        is_undervalued=listing.is_undervalued,
    )


# ── Run on startup (optional auto-scan) ──────────────────────────────

@app.on_event("startup")
async def run_initial_scan():
    """Optionally run a pipeline scan on API startup.

    In demo mode, this gives us data to serve immediately
    without needing a separate pipeline trigger.
    """
    if settings.is_demo_mode:
        logger.info("auto_running_demo_pipeline_on_startup")

        async def _initial_run():
            # Small delay to let the API finish starting
            await asyncio.sleep(1)
            from services.pipeline.orchestrator import run_pipeline

            app_state.is_running = True
            try:
                result = await run_pipeline()
                app_state.last_run = result
                app_state.last_run_at = datetime.utcnow()
                app_state.total_runs += 1
                logger.info("startup_scan_complete", summary=result.summary)
            except Exception as e:
                logger.error("startup_scan_failed", error=str(e))
            finally:
                app_state.is_running = False

        asyncio.create_task(_initial_run())
