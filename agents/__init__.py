"""
WoningScout Agents

Five autonomous agents that form the property analysis pipeline:

    A1: IngestionAgent   — fetch + deduplicate + validate listings
    A2: FeatureAgent     — compute 47 ML features per listing
    A3: PredictionAgent  — XGBoost price prediction + confidence intervals
    A4: ScoringAgent     — FAISS comparables + livability scoring
    A5: AlertAgent       — match user preferences + send notifications
"""

from agents.ingestion import IngestionAgent
from agents.features import FeatureAgent
from agents.prediction import PredictionAgent
from agents.scoring import ScoringAgent
from agents.alerting import AlertAgent
from agents.base import BaseAgent
from agents.schemas import PipelineState

__all__ = [
    "IngestionAgent",
    "FeatureAgent",
    "PredictionAgent",
    "ScoringAgent",
    "AlertAgent",
    "BaseAgent",
    "PipelineState",
]
