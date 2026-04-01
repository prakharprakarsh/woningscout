"""Tests for the pipeline orchestrator.

These are integration tests — they run the full pipeline end-to-end
in demo mode. Slower than unit tests but catch wiring issues.
"""

import pytest

from agents.schemas import PipelineState
from services.pipeline.orchestrator import (
    run_pipeline,
    run_simple_pipeline,
    should_alert,
)


class TestConditionalRouting:
    def test_routes_to_alert_when_undervalued(self):
        from agents.schemas import ScoredProperty
        state = PipelineState(run_id="test")
        state.scored_properties = [
            ScoredProperty(
                listing_id="x",
                value_ratio=1.15,
                asking_price=100,
                predicted_price=115,
            )
        ]
        assert should_alert(state) == "alert"

    def test_routes_to_end_when_fair(self):
        from agents.schemas import ScoredProperty
        state = PipelineState(run_id="test")
        state.scored_properties = [
            ScoredProperty(
                listing_id="x",
                value_ratio=1.01,
                asking_price=100,
                predicted_price=101,
            )
        ]
        assert should_alert(state) == "end"

    def test_routes_to_end_when_empty(self):
        state = PipelineState(run_id="test")
        assert should_alert(state) == "end"


class TestSimplePipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_demo_mode(self):
        """Run the full pipeline end-to-end in demo mode.

        This is the most important test — it verifies that all 5 agents
        wire together correctly and produce sensible output.
        """
        state = PipelineState(
            run_id="integration-test",
            target_regions=["amsterdam"],
        )

        result = await run_simple_pipeline(state)

        # Ingestion should have found demo listings
        assert len(result.new_listing_ids) > 0

        # Features should have been computed
        assert len(result.feature_ids) > 0

        # Predictions should exist
        assert len(result.predictions) > 0

        # Scoring should have run
        assert len(result.scored_properties) > 0

        # All scored properties should have valid livability
        for scored in result.scored_properties:
            assert scored.livability.composite > 0
            assert scored.livability.composite <= 10
            assert scored.value_ratio > 0

    @pytest.mark.asyncio
    async def test_pipeline_predictions_sane(self):
        """Predictions should be in a reasonable price range for NL."""
        state = PipelineState(
            run_id="sanity-test",
            target_regions=["amsterdam"],
        )
        result = await run_simple_pipeline(state)

        for pred in result.predictions:
            assert 50_000 < pred.predicted_price < 5_000_000, (
                f"Prediction €{pred.predicted_price:,.0f} outside sane NL range"
            )
            assert pred.ci_lower < pred.predicted_price
            assert pred.ci_upper > pred.predicted_price

    @pytest.mark.asyncio
    async def test_pipeline_stats_populated(self):
        state = PipelineState(
            run_id="stats-test",
            target_regions=["amsterdam"],
        )
        result = await run_simple_pipeline(state)

        # Stats should have been logged by agents
        assert len(result.stats) > 0


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_run_pipeline_convenience(self):
        """Test the public run_pipeline() function."""
        from services.pipeline.orchestrator import _ingestion
        _ingestion._seen_hashes.clear()
        result = await run_pipeline(regions=["amsterdam"])

        assert result.run_id  # should have a run ID
        assert len(result.scored_properties) > 0
        assert result.summary  # should produce a summary string
