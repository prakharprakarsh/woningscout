"""Tests for the Prediction Agent.

Focus on behavior, not implementation details:
- Do predictions fall in sane ranges?
- Are CIs actually containing the prediction?
- Does drift detection fire when distributions shift?
- Does rural have wider CIs than urban? (less data = more uncertainty)
"""

import pytest
import numpy as np

from agents.prediction import (
    PredictionAgent,
    DemoModel,
    FEATURE_COLUMNS,
    population_stability_index,
)
from agents.schemas import PipelineState, FeatureVector


class TestDemoModel:
    @pytest.fixture
    def model(self):
        return DemoModel()

    def test_amsterdam_apartment_range(self, model, sample_feature_vector):
        """72m² in De Pijp should predict 350-600K, not 50K or 2M."""
        row = [
            sample_feature_vector.features.get(col, 0) or 0
            for col in FEATURE_COLUMNS
        ]
        X = np.array([row], dtype=np.float64)
        pred = model.predict(X)[0]

        assert 300_000 < pred < 700_000, (
            f"Amsterdam 78m² predicted €{pred:,.0f}, outside sane bounds"
        )

    def test_larger_area_higher_price(self, model):
        """All else equal, bigger apartment should cost more."""
        base = {col: 0 for col in FEATURE_COLUMNS}
        base["living_area_m2"] = 60
        base["price_per_m2_pc4_90d"] = 6000
        base["dist_to_nearest_station_km"] = 1.0
        base["dist_to_centrum_km"] = 2.0
        base["build_year"] = 1960

        small = np.array([[base.get(c, 0) for c in FEATURE_COLUMNS]], dtype=np.float64)

        base["living_area_m2"] = 120
        big = np.array([[base.get(c, 0) for c in FEATURE_COLUMNS]], dtype=np.float64)

        pred_small = model.predict(small)[0]
        pred_big = model.predict(big)[0]

        assert pred_big > pred_small, (
            f"120m² (€{pred_big:,.0f}) should cost more than "
            f"60m² (€{pred_small:,.0f})"
        )

    def test_batch_prediction(self, model):
        """Should handle multiple listings at once."""
        np.random.seed(42)
        X = np.random.rand(10, len(FEATURE_COLUMNS)) * 100
        # Set realistic values for key columns
        X[:, FEATURE_COLUMNS.index("living_area_m2")] = np.random.uniform(40, 150, 10)
        X[:, FEATURE_COLUMNS.index("price_per_m2_pc4_90d")] = np.random.uniform(3000, 8000, 10)

        preds = model.predict(X)
        assert len(preds) == 10
        assert all(p > 0 for p in preds)


class TestPSI:
    def test_identical_distributions(self):
        """Same distribution should have PSI ~0."""
        np.random.seed(42)
        dist = np.random.normal(400000, 50000, 500)
        psi = population_stability_index(dist, dist)
        assert psi < 0.05

    def test_shifted_distribution_detected(self):
        """A 50% price jump should register high PSI."""
        np.random.seed(42)
        reference = np.random.normal(400000, 50000, 500)
        shifted = np.random.normal(600000, 50000, 500)
        psi = population_stability_index(shifted, reference)
        assert psi > 0.12, f"PSI={psi}, expected > 0.12 for shifted dist"

    def test_small_shift_moderate(self):
        """5% price change should be low PSI."""
        np.random.seed(42)
        reference = np.random.normal(400000, 50000, 500)
        slight = np.random.normal(420000, 50000, 500)
        psi = population_stability_index(slight, reference)
        assert psi < 0.15

    def test_small_sample_returns_zero(self):
        """Very small samples shouldn't trigger false drift."""
        psi = population_stability_index(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert psi == 0.0


class TestPredictionAgent:
    @pytest.fixture
    def agent(self):
        return PredictionAgent()

    @pytest.mark.asyncio
    async def test_produces_predictions(self, agent, sample_feature_vector):
        state = PipelineState(run_id="test-pred")
        state.feature_ids = [sample_feature_vector.listing_id]
        state._feature_vectors = {
            sample_feature_vector.listing_id: sample_feature_vector
        }

        result = await agent.run(state)
        assert len(result.predictions) == 1

        pred = result.predictions[0]
        assert pred.predicted_price > 0
        assert pred.ci_lower > 0
        assert pred.ci_upper > pred.ci_lower

    @pytest.mark.asyncio
    async def test_ci_contains_prediction(self, agent, sample_feature_vector):
        state = PipelineState(run_id="test-ci")
        state.feature_ids = [sample_feature_vector.listing_id]
        state._feature_vectors = {
            sample_feature_vector.listing_id: sample_feature_vector
        }

        result = await agent.run(state)
        pred = result.predictions[0]

        assert pred.ci_lower <= pred.predicted_price <= pred.ci_upper

    @pytest.mark.asyncio
    async def test_ci_width_reasonable(self, agent, sample_feature_vector):
        """CI shouldn't be wider than 30% of prediction."""
        state = PipelineState(run_id="test-ci-width")
        state.feature_ids = [sample_feature_vector.listing_id]
        state._feature_vectors = {
            sample_feature_vector.listing_id: sample_feature_vector
        }

        result = await agent.run(state)
        pred = result.predictions[0]

        width_pct = pred.ci_width_pct
        assert width_pct < 0.30, (
            f"CI width is {width_pct:.1%} of prediction, expected < 30%"
        )

    @pytest.mark.asyncio
    async def test_empty_features_skipped(self, agent):
        state = PipelineState(run_id="test-empty")
        state.feature_ids = []
        result = await agent.run(state)
        assert result.predictions == []
