"""
Feature Engineering Agent (A2)

The messiest agent. Computes 47 features per listing across 5 categories:
- Structural (9): from listing metadata directly
- Geospatial (12): distances and POI counts from OpenStreetMap
- NLP (8): sentiment and keyword extraction from Dutch descriptions
- Market context (13): aggregated stats for the postcode area
- Temporal (5): listing timing and market momentum

The geo features are the most expensive — OSM Overpass API has aggressive
rate limiting. We cache heavily in Redis (7d TTL) since POIs don't move.

The Dutch NLP is tricky. spaCy's nl_core_news_lg handles most things but
struggles with compound words ('driekamerappartement') and mixed nl/en
listings. We have a custom compound splitter for ~200 housing terms.
"""

import math
import re
from datetime import datetime
from typing import Optional

from agents.base import BaseAgent
from agents.schemas import (
    ENERGY_LABEL_ORDINAL,
    FeatureVector,
    PipelineState,
    RawListing,
)


# ── Dutch NLP helpers ─────────────────────────────────────────────────

# Common Dutch housing keywords indicating luxury or recent renovation
LUXURY_KEYWORDS_NL = {
    "luxe", "luxueus", "hoogwaardig", "marmeren", "design",
    "state-of-the-art", "exclusief", "penthouse", "suite",
    "jacuzzi", "sauna", "vloerverwarming", "domotica",
}

RENOVATION_KEYWORDS_NL = {
    "gerenoveerd", "renovatie", "vernieuwd", "nieuw", "modern",
    "recent", "opgeknapt", "verbouwd", "gemoderniseerd",
    "nieuwe keuken", "nieuwe badkamer",
}

# Compound word splitter for common housing terms
# Dutch mashes words together: 'driekamerappartement' = drie + kamer + appartement
COMPOUND_SPLITS = {
    "driekamer": "drie kamer",
    "tweekamer": "twee kamer",
    "vierkamer": "vier kamer",
    "vijfkamer": "vijf kamer",
    "bovenwoning": "boven woning",
    "benedenwoning": "beneden woning",
    "tussenwoning": "tussen woning",
    "hoekwoning": "hoek woning",
    "grachtenpand": "grachten pand",
    "dakterras": "dak terras",
    "stadsverwarming": "stads verwarming",
    "vloerverwarming": "vloer verwarming",
    "dubbel glas": "dubbel glas",
    "woonkamer": "woon kamer",
    "slaapkamer": "slaap kamer",
    "badkamer": "bad kamer",
    "balkonkamer": "balkon kamer",
    "dakkapel": "dak kapel",
    "tuinhuis": "tuin huis",
}


def simple_dutch_sentiment(text: str) -> float:
    """Quick-and-dirty sentiment for Dutch listing descriptions.

    Not using spaCy here (too heavy for what we need). Instead, a simple
    positive/negative keyword ratio. Listings are inherently positive
    (sellers are marketing), so the baseline is ~0.6, not 0.5.

    Returns: float in [0.0, 1.0]
    """
    text_lower = text.lower()

    positive = {
        "prachtig", "schitterend", "uniek", "ruim", "licht",
        "karakteristiek", "rustig", "sfeer", "charmant", "fantastisch",
        "geweldig", "perfect", "ideaal", "mooi", "zonnig",
        "recent", "nieuw", "modern", "luxe", "vrij uitzicht",
        "beautiful", "spacious", "bright", "unique", "charming",
        "stunning", "lovely", "excellent", "renovated", "modern",
    }
    negative = {
        "opknapper", "achterstallig", "klein", "donker", "gehorig",
        "slecht", "oud", "verouderd", "reparatie", "lekkage",
        "vocht", "schimmel", "overlast", "herrie", "smal",
    }

    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.5  # neutral

    return round(pos_count / total, 3)


def detect_english(text: str) -> bool:
    """Check if listing description is primarily in English.

    Some luxury listings in Amsterdam are written entirely in English.
    Simple heuristic: if common English articles appear more than Dutch ones.
    """
    text_lower = text.lower()
    english_markers = sum(1 for w in ["the ", " is ", " with ", " and ", " for "] if w in text_lower)
    dutch_markers = sum(1 for w in [" de ", " het ", " een ", " van ", " met ", " en "] if w in text_lower)
    return english_markers > dutch_markers


def count_luxury_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in LUXURY_KEYWORDS_NL if kw in text_lower)


def detect_renovation(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in RENOVATION_KEYWORDS_NL)


def count_unique_selling_points(text: str) -> int:
    """Count sentences that look like USPs (short, punchy, often start with capital)."""
    sentences = re.split(r'[.!]', text)
    return sum(1 for s in sentences if 3 < len(s.split()) < 12 and s.strip())


# ── Geospatial helpers ────────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in kilometers."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Major Dutch train stations (lat, lng) — used for distance calculation
# In production, these come from the NS (Dutch Railways) API
MAJOR_STATIONS = {
    "Amsterdam Centraal": (52.3791, 4.9003),
    "Amsterdam Zuid": (52.3390, 4.8724),
    "Amsterdam Sloterdijk": (52.3890, 4.8375),
    "Amsterdam Amstel": (52.3464, 4.9178),
    "Amsterdam Muiderpoort": (52.3613, 4.9348),
    "Amsterdam Lelylaan": (52.3578, 4.8340),
    "Utrecht Centraal": (52.0894, 5.1101),
    "Rotterdam Centraal": (51.9244, 4.4690),
    "Den Haag Centraal": (52.0802, 4.3250),
    "Leiden Centraal": (52.1663, 4.4815),
}

# City centers for distance calculation
CITY_CENTERS = {
    "amsterdam": (52.3676, 4.9041),
    "utrecht": (52.0894, 5.1180),
    "rotterdam": (51.9244, 4.4777),
    "den haag": (52.0705, 4.3007),
    "leiden": (52.1601, 4.4970),
}


def nearest_station_km(lat: float, lng: float) -> float:
    """Distance to nearest major train station."""
    if lat is None or lng is None:
        return 99.0  # sentinel for missing
    distances = [
        haversine_km(lat, lng, slat, slng)
        for slat, slng in MAJOR_STATIONS.values()
    ]
    return round(min(distances), 3)


def distance_to_centrum(lat: float, lng: float, city: str = "amsterdam") -> float:
    """Distance to city center."""
    if lat is None or lng is None:
        return 99.0
    center = CITY_CENTERS.get(city.lower(), CITY_CENTERS["amsterdam"])
    return round(haversine_km(lat, lng, center[0], center[1]), 3)


# ── Market context (mock for demo, real version queries Postgres) ─────

# Average price per m² by PC4 area (sample data for Amsterdam/Utrecht)
# In production, this is computed from historical sales in the DB
PC4_MARKET_DATA = {
    "1072": {"price_per_m2_90d": 7140, "active_listings": 23, "days_on_market_avg": 18, "yoy_change": 0.08, "bid_ratio": 1.04, "avg_income": 42000},
    "1015": {"price_per_m2_90d": 8200, "active_listings": 15, "days_on_market_avg": 12, "yoy_change": 0.06, "bid_ratio": 1.07, "avg_income": 48000},
    "1018": {"price_per_m2_90d": 6890, "active_listings": 31, "days_on_market_avg": 22, "yoy_change": 0.05, "bid_ratio": 1.02, "avg_income": 38000},
    "1073": {"price_per_m2_90d": 7350, "active_listings": 19, "days_on_market_avg": 16, "yoy_change": 0.07, "bid_ratio": 1.05, "avg_income": 44000},
    "3511": {"price_per_m2_90d": 5100, "active_listings": 28, "days_on_market_avg": 25, "yoy_change": 0.04, "bid_ratio": 1.01, "avg_income": 36000},
}

DEFAULT_MARKET = {
    "price_per_m2_90d": 5500, "active_listings": 20, "days_on_market_avg": 21,
    "yoy_change": 0.05, "bid_ratio": 1.02, "avg_income": 38000,
}


# ── Property type encoding ────────────────────────────────────────────

PROPERTY_TYPE_MAP = {
    "apartment": 1, "house": 2, "villa": 3,
    "penthouse": 4, "studio": 5, "other": 0,
}

PARKING_TYPE_MAP = {
    "none": 0, "street": 1, "permit": 2,
    "garage": 3, "private": 4,
}


# ── The Agent ─────────────────────────────────────────────────────────

class FeatureAgent(BaseAgent):
    """Agent A2: Compute 47 features per listing.

    This agent does NOT call external APIs in demo mode —
    it uses the hardcoded reference data above. In production,
    the geo features would come from OSM Overpass (cached in Redis)
    and market data from a Postgres aggregate query.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "feature_engineering"

    async def _execute(self, state: PipelineState) -> PipelineState:
        if not state.new_listing_ids:
            self.logger.info("no_listings_to_featurize")
            return state

        # Get raw listings from state
        raw_listings: dict = getattr(state, "_raw_listings", {})
        feature_vectors: list[FeatureVector] = []

        for listing_id in state.new_listing_ids:
            listing: Optional[RawListing] = raw_listings.get(listing_id)
            if listing is None:
                self.logger.warning("listing_not_found", listing_id=listing_id)
                continue

            features = self._compute_all_features(listing)

            # Check for missing required features and impute
            missing = [
                k for k in [
                    "living_area_m2", "num_rooms", "lat", "lng",
                    "dist_to_nearest_station_km", "price_per_m2_pc4_90d",
                ]
                if features.get(k) is None
            ]
            if missing:
                self.logger.warning(
                    "imputing_missing_features",
                    listing_id=listing_id,
                    missing=missing,
                )
                features = self._impute_defaults(features, listing)

            feature_vectors.append(FeatureVector(
                listing_id=listing_id,
                features=features,
            ))

        # Store in state for downstream agents
        state.feature_ids = [fv.listing_id for fv in feature_vectors]
        if not hasattr(state, "_feature_vectors"):
            state._feature_vectors = {}  # type: ignore
        for fv in feature_vectors:
            state._feature_vectors[fv.listing_id] = fv  # type: ignore

        self._log_stats(
            state,
            featurized=len(feature_vectors),
            avg_feature_count=round(
                sum(fv.feature_count for fv in feature_vectors) / max(len(feature_vectors), 1),
                1,
            ),
        )

        return state

    def _compute_all_features(self, listing: RawListing) -> dict:
        """Compute all 47 features for a single listing."""
        features = {}

        # ── Structural (9) ────────────────────────────────────
        features["living_area_m2"] = listing.living_area_m2
        features["num_rooms"] = listing.num_rooms
        features["num_bathrooms"] = listing.num_bathrooms
        features["build_year"] = listing.build_year
        features["energy_label_ordinal"] = ENERGY_LABEL_ORDINAL.get(
            listing.energy_label.value, 0
        )
        features["has_garden"] = int(listing.has_garden)
        features["has_balcony"] = int(listing.has_balcony)
        features["parking_type_enc"] = PARKING_TYPE_MAP.get(
            listing.parking_type, 0
        )
        features["property_type_enc"] = PROPERTY_TYPE_MAP.get(
            listing.property_type.value, 0
        )

        # ── Geospatial (12) ───────────────────────────────────
        lat = listing.lat or 52.37  # Amsterdam fallback
        lng = listing.lng or 4.90

        features["lat"] = lat
        features["lng"] = lng
        features["dist_to_nearest_station_km"] = nearest_station_km(lat, lng)
        features["dist_to_centrum_km"] = distance_to_centrum(
            lat, lng, listing.city or "amsterdam"
        )

        # These would come from OSM Overpass in production
        # For demo, we use reasonable estimates based on centrality
        centrality = max(0.1, features["dist_to_centrum_km"])
        features["dist_to_nearest_school_km"] = round(0.3 + centrality * 0.15, 3)
        features["supermarket_count_500m"] = max(1, int(5 - centrality * 0.8))
        features["restaurant_count_1km"] = max(2, int(30 - centrality * 4))
        features["green_space_pct_500m"] = round(8.0 + centrality * 2.5, 1)
        features["water_body_dist_m"] = round(100 + centrality * 80, 0)
        features["noise_level_estimate"] = round(max(40, 70 - centrality * 5), 1)
        features["elevation_m"] = round(-1.5 + centrality * 0.3, 1)  # NL is flat and below sea level
        features["postal_density_per_km2"] = round(max(1000, 15000 - centrality * 2000), 0)

        # ── NLP (8) ───────────────────────────────────────────
        desc = listing.description or ""
        features["desc_sentiment_nl"] = simple_dutch_sentiment(desc)
        features["luxury_keyword_count"] = count_luxury_keywords(desc)
        features["renovation_mentioned"] = int(detect_renovation(desc))
        features["desc_word_count"] = len(desc.split())
        features["unique_selling_points_count"] = count_unique_selling_points(desc)
        features["has_english_text"] = int(detect_english(desc))
        # agent_confidence_tone: how "salesy" the description reads
        # higher word count + more USPs + more luxury keywords = more confident
        features["agent_confidence_tone"] = round(min(1.0, (
            features["luxury_keyword_count"] * 0.15
            + features["unique_selling_points_count"] * 0.1
            + min(features["desc_word_count"], 200) / 400
        )), 3)
        features["photo_count"] = listing.photo_count

        # ── Market Context (13) ───────────────────────────────
        pc4 = listing.postcode_4d
        market = PC4_MARKET_DATA.get(pc4, DEFAULT_MARKET)

        features["days_on_market_avg_pc4"] = market["days_on_market_avg"]
        features["price_per_m2_pc4_90d"] = market["price_per_m2_90d"]
        features["active_listings_pc4"] = market["active_listings"]
        features["yoy_price_change_pc4"] = market["yoy_change"]
        features["bid_ratio_avg_pc4"] = market["bid_ratio"]
        features["sold_above_asking_pct_pc4"] = round(market["bid_ratio"] * 60, 1)  # derived
        features["inventory_turnover_pc4"] = round(12 / max(market["days_on_market_avg"], 1), 2)
        features["new_construction_pc4"] = 0  # would come from CBS
        features["avg_income_pc4"] = market["avg_income"]
        features["population_growth_pc4"] = 0.012  # NL average
        features["mortgage_rate_current"] = 4.2  # as of early 2024
        features["consumer_confidence_idx"] = -22  # CBS, early 2024
        features["housing_shortage_idx_province"] = 3.8  # North Holland

        # ── Temporal (5) ──────────────────────────────────────
        now = listing.listed_at or datetime.utcnow()
        features["month_listed"] = now.month
        features["day_of_week_listed"] = now.weekday()
        features["is_school_holiday"] = int(now.month in (7, 8, 12))  # rough
        features["days_since_last_rate_change"] = 45  # placeholder
        features["market_momentum_30d_pc4"] = round(market["yoy_change"] / 12, 4)

        # ── Derived: asking price (needed for value ratio later) ──
        features["asking_price"] = listing.asking_price

        return features

    def _impute_defaults(self, features: dict, listing: RawListing) -> dict:
        """Fill in missing features with sensible defaults.

        Uses PC4-level medians where available, else NL-wide defaults.
        In production, these medians would come from a pre-computed table.
        """
        defaults = {
            "living_area_m2": 75.0,
            "num_rooms": 3,
            "num_bathrooms": 1,
            "build_year": 1960,
            "lat": 52.37,
            "lng": 4.90,
            "dist_to_nearest_station_km": 1.5,
            "price_per_m2_pc4_90d": 5500,
        }
        for key, default_val in defaults.items():
            if features.get(key) is None:
                features[key] = default_val
        return features
