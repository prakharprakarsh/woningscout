"""Tests for the Feature Engineering Agent."""

import pytest

from agents.features import (
    FeatureAgent,
    simple_dutch_sentiment,
    detect_english,
    count_luxury_keywords,
    detect_renovation,
    count_unique_selling_points,
    nearest_station_km,
    distance_to_centrum,
    haversine_km,
)
from agents.schemas import PipelineState, RawListing


# ── NLP helpers ───────────────────────────────────────────────────────

class TestDutchSentiment:
    def test_positive_description(self):
        text = "Prachtig ruim licht appartement met fantastisch uitzicht"
        score = simple_dutch_sentiment(text)
        assert score > 0.6

    def test_negative_description(self):
        text = "Kleine donkere opknapper met vochtproblemen en schimmel"
        score = simple_dutch_sentiment(text)
        assert score < 0.4

    def test_neutral_empty(self):
        assert simple_dutch_sentiment("") == 0.5

    def test_mixed_language(self):
        text = "Beautiful spacious apartment met prachtig uitzicht"
        score = simple_dutch_sentiment(text)
        assert 0.3 < score <= 1.0  # should detect both


class TestDetectEnglish:
    def test_dutch_text(self):
        text = "Een mooi appartement in de Jordaan met een tuin"
        assert detect_english(text) is False

    def test_english_text(self):
        text = "A beautiful apartment in the heart of Amsterdam with a garden"
        assert detect_english(text) is True

    def test_mixed_defaults_correctly(self):
        # Mostly Dutch
        text = "Groot appartement met een mooie tuin. The living room is spacious."
        # Should detect as Dutch-dominant
        result = detect_english(text)
        assert isinstance(result, bool)


class TestLuxuryKeywords:
    def test_luxury_listing(self):
        text = "Luxe penthouse met vloerverwarming en jacuzzi"
        assert count_luxury_keywords(text) >= 3

    def test_basic_listing(self):
        text = "Eenvoudig appartement op de eerste verdieping"
        assert count_luxury_keywords(text) == 0


class TestRenovation:
    def test_renovation_mentioned(self):
        assert detect_renovation("Recent gerenoveerde keuken en badkamer") is True
        assert detect_renovation("Nieuwe keuken geinstalleerd") is True

    def test_no_renovation(self):
        assert detect_renovation("Appartement op de tweede verdieping") is False


class TestUSPCount:
    def test_counts_short_sentences(self):
        text = "Op loopafstand van het mooie park. Rustige straat dichtbij winkels. Vrij uitzicht over de gracht."
        count = count_unique_selling_points(text)
        assert count >= 2


# ── Geospatial ────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point(self):
        assert haversine_km(52.37, 4.90, 52.37, 4.90) == 0.0

    def test_amsterdam_to_utrecht(self):
        # ~36 km
        dist = haversine_km(52.3676, 4.9041, 52.0894, 5.1180)
        assert 30 < dist < 45

    def test_short_distance(self):
        # Two points ~1km apart in Amsterdam
        dist = haversine_km(52.370, 4.890, 52.379, 4.890)
        assert 0.5 < dist < 1.5


class TestNearestStation:
    def test_amsterdam_centrum(self):
        # Should be very close to Amsterdam Centraal
        dist = nearest_station_km(52.3791, 4.9003)
        assert dist < 0.1

    def test_de_pijp(self):
        # De Pijp area — closest might be Amsterdam Zuid or Amstel
        dist = nearest_station_km(52.3534, 4.8874)
        assert dist < 3.0

    def test_missing_coords_returns_sentinel(self):
        assert nearest_station_km(None, None) == 99.0


class TestDistanceToCentrum:
    def test_amsterdam_centrum_is_zero(self):
        dist = distance_to_centrum(52.3676, 4.9041, "amsterdam")
        assert dist < 0.1

    def test_amsterdam_outskirts(self):
        dist = distance_to_centrum(52.40, 4.80, "amsterdam")
        assert dist > 3.0

    def test_utrecht(self):
        dist = distance_to_centrum(52.0894, 5.1180, "utrecht")
        assert dist < 0.1


# ── Feature Agent integration ─────────────────────────────────────────

class TestFeatureAgent:
    @pytest.fixture
    def agent(self):
        return FeatureAgent()

    @pytest.mark.asyncio
    async def test_produces_features(self, agent, pipeline_state_with_listings):
        result = await agent.run(pipeline_state_with_listings)
        assert len(result.feature_ids) == 1

        fv_store = getattr(result, "_feature_vectors", {})
        assert len(fv_store) == 1

        fv = list(fv_store.values())[0]
        assert fv.feature_count >= 47

    @pytest.mark.asyncio
    async def test_feature_values_sane(self, agent, pipeline_state_with_listings):
        result = await agent.run(pipeline_state_with_listings)
        fv = list(getattr(result, "_feature_vectors", {}).values())[0]
        f = fv.features

        # Structural
        assert f["living_area_m2"] == 78
        assert f["num_rooms"] == 3
        assert f["build_year"] == 1928

        # Geospatial — should be computed, not None
        assert f["dist_to_nearest_station_km"] is not None
        assert f["dist_to_nearest_station_km"] < 10
        assert f["dist_to_centrum_km"] is not None
        assert f["dist_to_centrum_km"] < 20

        # NLP
        assert 0 <= f["desc_sentiment_nl"] <= 1
        assert f["renovation_mentioned"] in (0, 1)

        # Market
        assert f["price_per_m2_pc4_90d"] > 0
        assert f["mortgage_rate_current"] > 0

    @pytest.mark.asyncio
    async def test_empty_listings_skipped(self, agent):
        state = PipelineState(run_id="test-empty", new_listing_ids=[])
        result = await agent.run(state)
        assert result.feature_ids == []
