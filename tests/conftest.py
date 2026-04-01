"""
Shared test fixtures.

These fixtures provide realistic-ish test data that all test modules
can use. The listings are loosely based on real Amsterdam properties
but with fictional addresses and prices.
"""

import pytest
from datetime import datetime, timezone

from agents.schemas import (
    PipelineState,
    RawListing,
    FeatureVector,
    PricePrediction,
    ScoredProperty,
    LivabilityBreakdown,
    Comparable,
)


@pytest.fixture
def sample_listing_dict():
    """Raw listing dict as it would come from Funda API."""
    return {
        "id": "TEST-001",
        "address": "Teststraat 42-II",
        "postcode": "1072PA",
        "city": "Amsterdam",
        "region": "amsterdam",
        "asking_price": 425000,
        "property_type": "apartment",
        "living_area_m2": 78,
        "num_rooms": 3,
        "num_bathrooms": 1,
        "build_year": 1928,
        "energy_label": "C",
        "has_garden": False,
        "has_balcony": True,
        "has_parking": False,
        "parking_type": "none",
        "lat": 52.3534,
        "lng": 4.8874,
        "description": (
            "Prachtig 3-kamer appartement op de tweede verdieping "
            "met uitzicht over de gracht. Recent gerenoveerde keuken."
        ),
        "photo_count": 12,
    }


@pytest.fixture
def sample_listing(sample_listing_dict):
    """Validated RawListing from sample dict."""
    return RawListing.model_validate(sample_listing_dict)


@pytest.fixture
def sample_listings_batch():
    """Multiple listings for batch testing."""
    return [
        {
            "id": f"TEST-{i:03d}",
            "address": f"Testlaan {i}",
            "postcode": "1072PA",
            "city": "Amsterdam",
            "region": "amsterdam",
            "asking_price": 350000 + i * 25000,
            "property_type": "apartment",
            "living_area_m2": 55 + i * 5,
            "num_rooms": 2 + (i % 3),
            "build_year": 1920 + i * 10,
            "energy_label": ["D", "C", "B", "A", "C"][i % 5],
            "lat": 52.35 + i * 0.002,
            "lng": 4.88 + i * 0.003,
            "description": f"Test listing {i} met mooie kenmerken.",
            "photo_count": 5 + i,
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_feature_vector():
    """Feature vector for a typical Amsterdam apartment."""
    return FeatureVector(
        listing_id="TEST-001",
        features={
            "living_area_m2": 78.0,
            "num_rooms": 3,
            "num_bathrooms": 1,
            "build_year": 1928,
            "energy_label_ordinal": 5,
            "has_garden": 0,
            "has_balcony": 1,
            "parking_type_enc": 0,
            "property_type_enc": 1,
            "lat": 52.3534,
            "lng": 4.8874,
            "dist_to_nearest_station_km": 0.38,
            "dist_to_centrum_km": 1.2,
            "dist_to_nearest_school_km": 0.45,
            "supermarket_count_500m": 4,
            "restaurant_count_1km": 25,
            "green_space_pct_500m": 14.2,
            "water_body_dist_m": 120,
            "noise_level_estimate": 62.0,
            "elevation_m": -1.2,
            "postal_density_per_km2": 12500,
            "desc_sentiment_nl": 0.68,
            "luxury_keyword_count": 1,
            "renovation_mentioned": 1,
            "desc_word_count": 22,
            "unique_selling_points_count": 2,
            "has_english_text": 0,
            "agent_confidence_tone": 0.35,
            "photo_count": 12,
            "days_on_market_avg_pc4": 18,
            "price_per_m2_pc4_90d": 7140,
            "active_listings_pc4": 23,
            "yoy_price_change_pc4": 0.08,
            "bid_ratio_avg_pc4": 1.04,
            "sold_above_asking_pct_pc4": 62.4,
            "inventory_turnover_pc4": 0.67,
            "new_construction_pc4": 0,
            "avg_income_pc4": 42000,
            "population_growth_pc4": 0.012,
            "mortgage_rate_current": 4.2,
            "consumer_confidence_idx": -22,
            "housing_shortage_idx_province": 3.8,
            "month_listed": 3,
            "day_of_week_listed": 2,
            "is_school_holiday": 0,
            "days_since_last_rate_change": 45,
            "market_momentum_30d_pc4": 0.0067,
            "asking_price": 425000,
        },
    )


@pytest.fixture
def empty_pipeline_state():
    """Fresh pipeline state for testing."""
    return PipelineState(
        run_id="test-run-001",
        target_regions=["amsterdam"],
    )


@pytest.fixture
def pipeline_state_with_listings(empty_pipeline_state, sample_listing):
    """Pipeline state after ingestion."""
    state = empty_pipeline_state
    state.new_listing_ids = [sample_listing.id]
    state._raw_listings = {sample_listing.id: sample_listing}
    return state
