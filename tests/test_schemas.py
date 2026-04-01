"""Tests for data schemas and validation."""

import pytest
from pydantic import ValidationError

from agents.schemas import (
    RawListing,
    FeatureVector,
    PricePrediction,
    ScoredProperty,
    LivabilityBreakdown,
    PipelineState,
    ENERGY_LABEL_ORDINAL,
)


class TestRawListing:
    def test_valid_listing(self, sample_listing_dict):
        listing = RawListing.model_validate(sample_listing_dict)
        assert listing.id == "TEST-001"
        assert listing.asking_price == 425000
        assert listing.postcode == "1072PA"

    def test_postcode_normalization(self):
        """Dutch postcodes with spaces should be normalized."""
        listing = RawListing(
            id="test", postcode="1012 AB", asking_price=300000
        )
        assert listing.postcode == "1012AB"

    def test_postcode_4d(self):
        listing = RawListing(id="t", postcode="1072PA", asking_price=1)
        assert listing.postcode_4d == "1072"

    def test_rejects_missing_postcode(self):
        with pytest.raises(ValidationError):
            RawListing(id="t", postcode="", asking_price=1)

    def test_rejects_negative_price(self):
        with pytest.raises(ValidationError):
            RawListing(id="t", postcode="1012AB", asking_price=-100)

    def test_rejects_zero_price(self):
        with pytest.raises(ValidationError):
            RawListing(id="t", postcode="1012AB", asking_price=0)

    def test_short_postcode_valid(self):
        """PC4-only postcodes (4 chars) should be accepted."""
        listing = RawListing(id="t", postcode="1012", asking_price=1)
        assert listing.postcode == "1012"


class TestFeatureVector:
    def test_feature_count(self, sample_feature_vector):
        # We expect 47 features + asking_price = 48
        assert sample_feature_vector.feature_count >= 47

    def test_empty_features(self):
        fv = FeatureVector(listing_id="t", features={})
        assert fv.feature_count == 0


class TestPricePrediction:
    def test_ci_width(self):
        pred = PricePrediction(
            listing_id="t",
            predicted_price=400000,
            ci_lower=380000,
            ci_upper=420000,
        )
        assert pred.ci_width == 40000
        assert abs(pred.ci_width_pct - 0.1) < 0.001

    def test_rejects_negative_price(self):
        with pytest.raises(ValidationError):
            PricePrediction(
                listing_id="t",
                predicted_price=-1,
                ci_lower=1,
                ci_upper=2,
            )


class TestScoredProperty:
    def test_undervalued_detection(self):
        """value_ratio > 1.05 = undervalued."""
        prop = ScoredProperty(
            listing_id="t",
            asking_price=400000,
            predicted_price=440000,
            value_ratio=1.10,
        )
        assert prop.is_undervalued is True
        assert abs(prop.undervalued_pct - 10.0) < 0.01

    def test_fairly_priced(self):
        prop = ScoredProperty(
            listing_id="t",
            asking_price=400000,
            predicted_price=404000,
            value_ratio=1.01,
        )
        assert prop.is_undervalued is False

    def test_overpriced(self):
        prop = ScoredProperty(
            listing_id="t",
            asking_price=500000,
            predicted_price=420000,
            value_ratio=0.84,
        )
        assert prop.is_undervalued is False
        assert prop.undervalued_pct == 0.0


class TestPipelineState:
    def test_summary(self, empty_pipeline_state):
        s = empty_pipeline_state.summary
        assert "test-run-001" in s

    def test_has_undervalued_false_when_empty(self, empty_pipeline_state):
        assert empty_pipeline_state.has_undervalued is False

    def test_has_undervalued_true(self):
        state = PipelineState(run_id="t")
        state.scored_properties = [
            ScoredProperty(
                listing_id="x",
                value_ratio=1.15,
                asking_price=100,
                predicted_price=115,
            )
        ]
        assert state.has_undervalued is True


class TestEnergyLabelOrdinal:
    def test_ordering(self):
        """A++ should be highest, G lowest."""
        assert ENERGY_LABEL_ORDINAL["A++"] > ENERGY_LABEL_ORDINAL["A"]
        assert ENERGY_LABEL_ORDINAL["A"] > ENERGY_LABEL_ORDINAL["C"]
        assert ENERGY_LABEL_ORDINAL["C"] > ENERGY_LABEL_ORDINAL["G"]
        assert ENERGY_LABEL_ORDINAL["unknown"] == 0
