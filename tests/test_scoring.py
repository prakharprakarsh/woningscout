"""Tests for the Scoring Agent."""

import pytest

from agents.scoring import (
    ScoringAgent,
    compute_livability,
    _score_transit,
    _score_green,
    _score_amenities,
    _score_schools,
    DEFAULT_LIVABILITY_WEIGHTS,
)
from agents.schemas import (
    PipelineState,
    PricePrediction,
    LivabilityBreakdown,
)


class TestTransitScore:
    def test_very_close(self):
        assert _score_transit(0.1) == 10.0

    def test_walkable(self):
        score = _score_transit(0.7)
        assert 7.0 <= score <= 9.0

    def test_far(self):
        score = _score_transit(3.0)
        assert score < 4.0

    def test_never_below_one(self):
        assert _score_transit(50.0) >= 1.0


class TestGreenScore:
    def test_high_green_area(self):
        assert _score_green(20.0) == 10.0

    def test_low_green_area(self):
        score = _score_green(5.0)
        assert score < 5.0

    def test_capped_at_ten(self):
        assert _score_green(100.0) == 10.0


class TestAmenityScore:
    def test_many_supermarkets(self):
        score = _score_amenities(5, 20)
        assert score >= 8.0

    def test_no_amenities(self):
        assert _score_amenities(0, 0) == 0.0


class TestSchoolScore:
    def test_very_close(self):
        assert _score_schools(0.2) == 10.0

    def test_far(self):
        score = _score_schools(3.0)
        assert score < 5.0


class TestLivabilityComposite:
    def test_all_tens(self):
        """Perfect scores should give ~10."""
        features = {
            "dist_to_nearest_station_km": 0.1,
            "green_space_pct_500m": 25.0,
            "supermarket_count_500m": 5,
            "restaurant_count_1km": 30,
            "postcode_4d": "1072",
            "dist_to_nearest_school_km": 0.2,
        }
        result = compute_livability(features)
        assert result.composite > 8.0

    def test_bad_location(self):
        features = {
            "dist_to_nearest_station_km": 5.0,
            "green_space_pct_500m": 2.0,
            "supermarket_count_500m": 0,
            "restaurant_count_1km": 0,
            "postcode_4d": "9999",
            "dist_to_nearest_school_km": 5.0,
        }
        result = compute_livability(features)
        assert result.composite < 5.0

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_LIVABILITY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_returns_all_components(self):
        features = {
            "dist_to_nearest_station_km": 1.0,
            "green_space_pct_500m": 10.0,
            "supermarket_count_500m": 3,
            "restaurant_count_1km": 10,
            "postcode_4d": "1072",
            "dist_to_nearest_school_km": 0.5,
        }
        result = compute_livability(features)
        assert result.transit > 0
        assert result.safety > 0
        assert result.amenities > 0
        assert result.green > 0
        assert result.schools > 0
        assert result.composite > 0


class TestScoringAgent:
    @pytest.fixture
    def agent(self):
        return ScoringAgent()

    @pytest.fixture
    def state_with_predictions(self, sample_feature_vector):
        state = PipelineState(run_id="test-score")
        state.predictions = [
            PricePrediction(
                listing_id="TEST-001",
                predicted_price=460000,
                ci_lower=435000,
                ci_upper=485000,
            )
        ]
        state._feature_vectors = {
            "TEST-001": sample_feature_vector,
        }
        state._raw_listings = {}
        return state

    @pytest.mark.asyncio
    async def test_scores_predictions(self, agent, state_with_predictions):
        result = await agent.run(state_with_predictions)
        assert len(result.scored_properties) == 1

        scored = result.scored_properties[0]
        assert scored.livability.composite > 0
        assert scored.value_ratio > 0

    @pytest.mark.asyncio
    async def test_undervalued_flagged(self, agent, state_with_predictions):
        """If predicted >> asking, should flag as undervalued."""
        result = await agent.run(state_with_predictions)
        scored = result.scored_properties[0]

        # Asking=425K, predicted=460K → ratio=1.08 → undervalued
        assert scored.value_ratio > 1.0

    @pytest.mark.asyncio
    async def test_comparables_returned(self, agent, state_with_predictions):
        result = await agent.run(state_with_predictions)
        scored = result.scored_properties[0]
        # Demo mode should return some comparables
        assert len(scored.comparables) > 0

    @pytest.mark.asyncio
    async def test_empty_predictions_handled(self, agent):
        state = PipelineState(run_id="test-empty")
        result = await agent.run(state)
        assert result.scored_properties == []
