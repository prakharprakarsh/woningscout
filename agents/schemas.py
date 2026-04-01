"""
Data schemas used throughout the pipeline.

These Pydantic models serve double duty:
1. Runtime validation (catching garbage data from Funda early)
2. Documentation (any new contributor can read these to understand the data flow)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────

class PropertyType(str, Enum):
    APARTMENT = "apartment"
    HOUSE = "house"
    VILLA = "villa"
    PENTHOUSE = "penthouse"
    STUDIO = "studio"
    OTHER = "other"


class EnergyLabel(str, Enum):
    A_PLUS_PLUS = "A++"
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    UNKNOWN = "unknown"


ENERGY_LABEL_ORDINAL = {
    "A++": 9, "A+": 8, "A": 7, "B": 6, "C": 5,
    "D": 4, "E": 3, "F": 2, "G": 1, "unknown": 0,
}


# ── Listing (raw from Funda) ─────────────────────────────────────────

class RawListing(BaseModel):
    """Raw listing as received from Funda API.

    We validate the bare minimum here — the feature agent
    handles the heavy transformation work downstream.
    """

    id: str = Field(..., description="Funda listing ID")
    url: str = ""
    address: str = ""
    postcode: str = Field(..., min_length=4, max_length=7)
    city: str = ""
    region: str = ""

    asking_price: float = Field(..., gt=0)
    property_type: PropertyType = PropertyType.OTHER
    living_area_m2: Optional[float] = None
    num_rooms: Optional[int] = None
    num_bathrooms: Optional[int] = None
    build_year: Optional[int] = None
    energy_label: EnergyLabel = EnergyLabel.UNKNOWN

    has_garden: bool = False
    has_balcony: bool = False
    has_parking: bool = False
    parking_type: str = "none"

    lat: Optional[float] = None
    lng: Optional[float] = None

    description: str = ""
    photo_count: int = 0

    listed_at: datetime = Field(default_factory=datetime.utcnow)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("postcode")
    @classmethod
    def normalize_postcode(cls, v: str) -> str:
        """Dutch postcodes: '1012 AB' -> '1012AB', ensure 4+ chars."""
        return v.replace(" ", "").upper()

    @property
    def postcode_4d(self) -> str:
        """First 4 digits of postcode (PC4 level for aggregation)."""
        return self.postcode[:4]


# ── Feature Vector ────────────────────────────────────────────────────

class FeatureVector(BaseModel):
    """Computed features for a single listing.

    47 features across 5 categories. The dict is flat intentionally —
    XGBoost wants a flat feature matrix, and keeping it nested would
    just mean more unpacking code everywhere.
    """

    listing_id: str
    features: dict[str, float | int | bool | None]
    computed_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def feature_count(self) -> int:
        return len(self.features)


# Required features — if any are None after feature engineering,
# we impute with PC4-level medians rather than dropping the row
REQUIRED_FEATURES = [
    "living_area_m2",
    "num_rooms",
    "build_year",
    "asking_price",
    "lat",
    "lng",
    "dist_to_nearest_station_km",
    "price_per_m2_pc4_90d",
]


# ── Prediction ────────────────────────────────────────────────────────

class PricePrediction(BaseModel):
    """Output from the prediction agent."""

    listing_id: str
    predicted_price: float = Field(..., gt=0)
    ci_lower: float = Field(..., gt=0)
    ci_upper: float = Field(..., gt=0)
    model_version: str = ""
    psi_at_inference: float = 0.0  # drift score at time of prediction

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    @property
    def ci_width_pct(self) -> float:
        """CI width as % of predicted price."""
        return self.ci_width / self.predicted_price if self.predicted_price > 0 else 0.0


# ── Comparable (from FAISS) ───────────────────────────────────────────

class Comparable(BaseModel):
    """A recently-sold property found via FAISS similarity search."""

    listing_id: str
    address: str = ""
    sold_price: float = 0.0
    sold_date: Optional[datetime] = None
    living_area_m2: float = 0.0
    property_type: str = ""
    similarity_score: float = 0.0  # lower = more similar (L2 distance)

    @property
    def sold_within_days(self) -> int:
        if not self.sold_date:
            return 999
        return (datetime.now(tz=timezone.utc) - self.sold_date).days


# ── Scored Property ───────────────────────────────────────────────────

class LivabilityBreakdown(BaseModel):
    """Component scores for neighborhood livability."""

    transit: float = Field(0.0, ge=0, le=10)
    safety: float = Field(0.0, ge=0, le=10)
    amenities: float = Field(0.0, ge=0, le=10)
    green: float = Field(0.0, ge=0, le=10)
    schools: float = Field(0.0, ge=0, le=10)
    composite: float = Field(0.0, ge=0, le=10)


class ScoredProperty(BaseModel):
    """Final scored output — prediction + livability + comparables."""

    listing_id: str
    asking_price: float = 0.0
    predicted_price: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    value_ratio: float = 0.0  # predicted / asking (>1.0 = undervalued)
    livability: LivabilityBreakdown = Field(default_factory=LivabilityBreakdown)
    comparables: list[Comparable] = Field(default_factory=list)

    @property
    def is_undervalued(self) -> bool:
        """We consider 5%+ gap as undervalued (value_ratio > 1.05)."""
        return self.value_ratio > 1.05

    @property
    def undervalued_pct(self) -> float:
        return max(0.0, (self.value_ratio - 1.0) * 100)


# ── Pipeline State ────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """Shared state that flows through the agent pipeline.

    Each agent reads what it needs, writes its output, and passes
    the state to the next agent. LangGraph manages this as the
    graph's state object.
    """

    # Control
    run_id: str = ""
    started_at: datetime = Field(default_factory=datetime.utcnow)
    target_regions: list[str] = Field(default_factory=list)
    last_scan_ts: Optional[datetime] = None
    backoff_until: Optional[datetime] = None  # rate limit recovery

    # Ingestion output
    new_listing_ids: list[str] = Field(default_factory=list)

    # Feature output
    feature_ids: list[str] = Field(default_factory=list)

    # Prediction output
    predictions: list[PricePrediction] = Field(default_factory=list)

    # Scoring output
    scored_properties: list[ScoredProperty] = Field(default_factory=list)

    # Alert output
    alerts_sent: int = 0

    # Stats (for monitoring)
    stats: dict[str, int | float] = Field(default_factory=dict)

    @property
    def has_undervalued(self) -> bool:
        return any(p.is_undervalued for p in self.scored_properties)

    @property
    def summary(self) -> str:
        return (
            f"Run {self.run_id}: "
            f"ingested={len(self.new_listing_ids)}, "
            f"scored={len(self.scored_properties)}, "
            f"undervalued={sum(1 for p in self.scored_properties if p.is_undervalued)}, "
            f"alerts={self.alerts_sent}"
        )
