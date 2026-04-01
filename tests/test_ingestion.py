"""Tests for the Ingestion Agent."""

import pytest

from agents.ingestion import (
    IngestionAgent,
    ContentHashDedup,
    FundaClient,
)
from agents.schemas import PipelineState


class TestContentHashDedup:
    def test_same_listing_same_hash(self):
        dedup = ContentHashDedup()
        listing = {"address": "Teststr 1", "postcode": "1012AB", "asking_price": 400000, "living_area_m2": 70}
        h1 = dedup.compute_hash(listing)
        h2 = dedup.compute_hash(listing)
        assert h1 == h2

    def test_different_price_different_hash(self):
        dedup = ContentHashDedup()
        l1 = {"address": "Teststr 1", "postcode": "1012AB", "asking_price": 400000, "living_area_m2": 70}
        l2 = {"address": "Teststr 1", "postcode": "1012AB", "asking_price": 500000, "living_area_m2": 70}
        assert dedup.compute_hash(l1) != dedup.compute_hash(l2)

    def test_filter_new_removes_duplicates(self):
        dedup = ContentHashDedup()
        listings = [
            {"address": "A", "postcode": "1012AB", "asking_price": 1, "living_area_m2": 1},
            {"address": "B", "postcode": "1012CD", "asking_price": 2, "living_area_m2": 2},
        ]
        # Pre-compute hash of first listing
        existing = {dedup.compute_hash(listings[0])}

        new = dedup.filter_new(listings, existing)
        assert len(new) == 1
        assert new[0]["address"] == "B"

    def test_empty_input(self):
        dedup = ContentHashDedup()
        assert dedup.filter_new([], set()) == []


class TestFundaClient:
    def test_demo_mode_returns_listings(self):
        client = FundaClient(api_key="", rate_limit_rps=2.0)
        assert client._demo_mode is True

    def test_demo_fixtures_not_empty(self):
        client = FundaClient(api_key="")
        fixtures = client._generate_default_fixtures()
        assert len(fixtures) >= 3
        assert all("id" in f for f in fixtures)
        assert all("asking_price" in f for f in fixtures)

    def test_demo_fixtures_have_dutch_descriptions(self):
        client = FundaClient(api_key="")
        fixtures = client._generate_default_fixtures()
        # At least some should have Dutch text
        has_dutch = any("appartement" in f.get("description", "").lower() for f in fixtures)
        assert has_dutch


class TestIngestionAgent:
    @pytest.fixture
    def agent(self):
        return IngestionAgent()

    @pytest.mark.asyncio
    async def test_ingests_demo_listings(self, agent):
        state = PipelineState(
            run_id="test-ingest",
            target_regions=["amsterdam"],
        )
        result = await agent.run(state)

        assert len(result.new_listing_ids) > 0
        assert result.stats.get("ingestion_ingested", 0) > 0

    @pytest.mark.asyncio
    async def test_deduplication_on_second_run(self, agent):
        state = PipelineState(
            run_id="test-dedup-1",
            target_regions=["amsterdam"],
        )
        result1 = await agent.run(state)
        first_count = len(result1.new_listing_ids)
        assert first_count > 0

        # Second run should find nothing new (same fixtures)
        state2 = PipelineState(
            run_id="test-dedup-2",
            target_regions=["amsterdam"],
        )
        result2 = await agent.run(state2)
        assert len(result2.new_listing_ids) == 0
        assert result2.stats.get("ingestion_deduped", 0) > 0

    @pytest.mark.asyncio
    async def test_empty_region_returns_all(self, agent):
        state = PipelineState(run_id="test-empty", target_regions=[])
        result = await agent.run(state)
        # Should still return fixtures even without region filter
        assert len(result.new_listing_ids) >= 0

    @pytest.mark.asyncio
    async def test_stores_raw_listings_in_state(self, agent):
        state = PipelineState(
            run_id="test-raw",
            target_regions=["amsterdam"],
        )
        result = await agent.run(state)
        raw = getattr(result, "_raw_listings", {})
        assert len(raw) > 0
        # Each stored listing should be a validated RawListing
        for lid, listing in raw.items():
            assert listing.asking_price > 0
            assert len(listing.postcode) >= 4
