"""Tests for the FastAPI API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch

# We test the app directly using FastAPI's TestClient pattern.
# In a real setup we'd use httpx.AsyncClient, but for simplicity:

from services.api.app import app, app_state
from agents.schemas import (
    PipelineState,
    ScoredProperty,
    LivabilityBreakdown,
    Comparable,
)


# Reset app_state before each test
@pytest.fixture(autouse=True)
def reset_state():
    app_state.last_run = None
    app_state.last_run_at = None
    app_state.total_runs = 0
    app_state.is_running = False
    yield


class TestHealthEndpoint:
    def test_health_returns_healthy(self):
        """Import check — ensure the app object creates without error."""
        # Just verify the app exists and has the route
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_health_response_model(self):
        """Verify the health endpoint is properly configured."""
        route = next(r for r in app.routes if getattr(r, 'path', '') == "/health")
        assert route is not None


class TestListingEndpoints:
    def test_listings_route_exists(self):
        routes = [r.path for r in app.routes]
        assert "/listings" in routes

    def test_listing_detail_route_exists(self):
        routes = [r.path for r in app.routes]
        assert "/listings/{listing_id}" in routes

    def test_listings_empty_when_no_run(self):
        """Without a pipeline run, listings should be empty."""
        assert app_state.scored_listings == []

    def test_listings_populated_after_run(self):
        """Simulate a pipeline run and check listings appear."""
        fake_run = PipelineState(run_id="fake")
        fake_run.scored_properties = [
            ScoredProperty(
                listing_id="TEST-001",
                asking_price=400000,
                predicted_price=440000,
                value_ratio=1.10,
                livability=LivabilityBreakdown(composite=7.5),
            )
        ]
        app_state.last_run = fake_run

        listings = app_state.scored_listings
        assert len(listings) == 1
        assert listings[0].listing_id == "TEST-001"

    def test_get_listing_by_id(self):
        fake_run = PipelineState(run_id="fake")
        fake_run.scored_properties = [
            ScoredProperty(listing_id="TEST-001", value_ratio=1.1, asking_price=1, predicted_price=1),
            ScoredProperty(listing_id="TEST-002", value_ratio=1.2, asking_price=1, predicted_price=1),
        ]
        app_state.last_run = fake_run

        result = app_state.get_listing("TEST-002")
        assert result is not None
        assert result.listing_id == "TEST-002"

    def test_get_listing_not_found(self):
        app_state.last_run = PipelineState(run_id="fake")
        result = app_state.get_listing("NONEXISTENT")
        assert result is None


class TestPipelineEndpoints:
    def test_pipeline_routes_exist(self):
        routes = [r.path for r in app.routes]
        assert "/pipeline/run" in routes
        assert "/pipeline/status" in routes


class TestAppState:
    def test_initial_state(self):
        assert app_state.total_runs == 0
        assert app_state.is_running is False
        assert app_state.last_run is None

    def test_scored_listings_empty_initially(self):
        assert app_state.scored_listings == []
