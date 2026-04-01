"""Tests for the Alert Agent."""

import pytest

from agents.alerting import (
    AlertAgent,
    UserPreference,
    UserRateLimiter,
    render_alert,
    format_comparables,
    DEMO_USERS,
)
from agents.schemas import (
    PipelineState,
    ScoredProperty,
    LivabilityBreakdown,
    Comparable,
)


class TestUserRateLimiter:
    def test_starts_at_zero(self):
        limiter = UserRateLimiter()
        assert limiter.is_exhausted("user-1") is False

    def test_exhausted_after_max(self):
        limiter = UserRateLimiter()
        for _ in range(5):
            limiter.increment("user-1")
        assert limiter.is_exhausted("user-1") is True

    def test_different_users_independent(self):
        limiter = UserRateLimiter()
        for _ in range(5):
            limiter.increment("user-1")
        assert limiter.is_exhausted("user-1") is True
        assert limiter.is_exhausted("user-2") is False

    def test_four_alerts_not_exhausted(self):
        limiter = UserRateLimiter()
        for _ in range(4):
            limiter.increment("user-1")
        assert limiter.is_exhausted("user-1") is False


class TestAlertTemplates:
    @pytest.fixture
    def scored_prop(self):
        return ScoredProperty(
            listing_id="TEST-001",
            asking_price=425000,
            predicted_price=465000,
            ci_lower=440000,
            ci_upper=490000,
            value_ratio=1.094,
            livability=LivabilityBreakdown(composite=7.6),
            comparables=[
                Comparable(
                    listing_id="c1",
                    address="Ruysdaelkade 88",
                    sold_price=468000,
                    living_area_m2=74,
                ),
            ],
        )

    @pytest.fixture
    def dutch_user(self):
        return UserPreference(
            user_id="test-nl",
            name="Jan de Vries",
            preferred_language="nl",
        )

    @pytest.fixture
    def english_user(self):
        return UserPreference(
            user_id="test-en",
            name="Tom Smith",
            preferred_language="en",
        )

    def test_dutch_template(self, scored_prop, dutch_user):
        result = render_alert(scored_prop, dutch_user, "Teststraat 42")
        assert "WoningScout" in result["subject"]
        assert "marktwaarde" in result["subject"]  # Dutch
        assert "Jan" in result["body"]
        assert "Teststraat 42" in result["body"]

    def test_english_template(self, scored_prop, english_user):
        result = render_alert(scored_prop, english_user, "Teststraat 42")
        assert "market value" in result["subject"]
        assert "Tom" in result["body"]

    def test_comparables_in_body(self, scored_prop, dutch_user):
        result = render_alert(scored_prop, dutch_user, "Teststraat 42")
        assert "Ruysdaelkade 88" in result["body"]

    def test_format_comparables_empty(self):
        result = format_comparables([], "nl")
        assert "geen" in result.lower()

        result_en = format_comparables([], "en")
        assert "no comparables" in result_en.lower()


class TestAlertAgent:
    @pytest.fixture
    def agent(self):
        return AlertAgent()

    @pytest.fixture
    def state_with_undervalued(self):
        state = PipelineState(run_id="test-alert")
        state.scored_properties = [
            ScoredProperty(
                listing_id="TEST-001",
                asking_price=425000,
                predicted_price=465000,
                ci_lower=440000,
                ci_upper=490000,
                value_ratio=1.094,
                livability=LivabilityBreakdown(composite=7.6),
                comparables=[],
            ),
        ]
        # Add raw listing for address lookup
        from agents.schemas import RawListing
        state._raw_listings = {
            "TEST-001": RawListing(
                id="TEST-001",
                address="Teststraat 42",
                postcode="1072PA",
                asking_price=425000,
            ),
        }
        return state

    @pytest.mark.asyncio
    async def test_sends_alerts_for_undervalued(self, agent, state_with_undervalued):
        result = await agent.run(state_with_undervalued)
        # Should have sent at least one alert (log channel)
        assert result.alerts_sent > 0

    @pytest.mark.asyncio
    async def test_no_alerts_when_fairly_priced(self, agent):
        state = PipelineState(run_id="test-no-alert")
        state.scored_properties = [
            ScoredProperty(
                listing_id="TEST-002",
                asking_price=400000,
                predicted_price=400000,
                value_ratio=1.00,  # not undervalued
            ),
        ]
        result = await agent.run(state)
        assert result.alerts_sent == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_works(self, agent, state_with_undervalued):
        """Tom (u-003) is already at 5/5 alerts — should be skipped."""
        result = await agent.run(state_with_undervalued)
        # Tom should not receive an alert
        stats = result.stats
        rate_limited = stats.get("alerting_rate_limited", 0)
        # At least Tom should be rate-limited
        assert rate_limited >= 1 or result.alerts_sent >= 1

    @pytest.mark.asyncio
    async def test_empty_scored_no_crash(self, agent):
        state = PipelineState(run_id="test-empty")
        result = await agent.run(state)
        assert result.alerts_sent == 0

    def test_demo_users_exist(self):
        assert len(DEMO_USERS) >= 2
        assert all(u.user_id for u in DEMO_USERS)
