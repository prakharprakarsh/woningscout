-- WoningScout Database Schema
-- Runs automatically on first docker compose up via init script

-- Listings table (raw from Funda)
CREATE TABLE IF NOT EXISTS listings (
    id VARCHAR(64) PRIMARY KEY,
    content_hash VARCHAR(32) NOT NULL,
    address TEXT NOT NULL DEFAULT '',
    postcode VARCHAR(7) NOT NULL,
    city VARCHAR(100) DEFAULT '',
    region VARCHAR(100) DEFAULT '',
    asking_price NUMERIC(12, 2) NOT NULL,
    property_type VARCHAR(20) DEFAULT 'other',
    living_area_m2 NUMERIC(8, 2),
    num_rooms INTEGER,
    num_bathrooms INTEGER,
    build_year INTEGER,
    energy_label VARCHAR(10) DEFAULT 'unknown',
    has_garden BOOLEAN DEFAULT FALSE,
    has_balcony BOOLEAN DEFAULT FALSE,
    parking_type VARCHAR(20) DEFAULT 'none',
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    description TEXT DEFAULT '',
    photo_count INTEGER DEFAULT 0,
    listed_at TIMESTAMPTZ DEFAULT NOW(),
    scraped_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_listings_postcode ON listings(postcode);
CREATE INDEX IF NOT EXISTS idx_listings_region ON listings(region);
CREATE INDEX IF NOT EXISTS idx_listings_listed_at ON listings(listed_at);
CREATE INDEX IF NOT EXISTS idx_listings_content_hash ON listings(content_hash);

-- Feature vectors (cached, 47 features per listing)
CREATE TABLE IF NOT EXISTS feature_vectors (
    listing_id VARCHAR(64) PRIMARY KEY REFERENCES listings(id),
    features JSONB NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    listing_id VARCHAR(64) NOT NULL REFERENCES listings(id),
    predicted_price NUMERIC(12, 2) NOT NULL,
    ci_lower NUMERIC(12, 2),
    ci_upper NUMERIC(12, 2),
    model_version VARCHAR(50) DEFAULT '',
    psi_at_inference NUMERIC(6, 4) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_listing ON predictions(listing_id);

-- Scored properties (final output)
CREATE TABLE IF NOT EXISTS scored_properties (
    id SERIAL PRIMARY KEY,
    listing_id VARCHAR(64) NOT NULL REFERENCES listings(id),
    predicted_price NUMERIC(12, 2) NOT NULL,
    value_ratio NUMERIC(6, 4) DEFAULT 1.0,
    livability_composite NUMERIC(4, 2) DEFAULT 0,
    livability_transit NUMERIC(4, 2) DEFAULT 0,
    livability_safety NUMERIC(4, 2) DEFAULT 0,
    livability_amenities NUMERIC(4, 2) DEFAULT 0,
    livability_green NUMERIC(4, 2) DEFAULT 0,
    livability_schools NUMERIC(4, 2) DEFAULT 0,
    comparables JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scored_value_ratio ON scored_properties(value_ratio);

-- User preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    email VARCHAR(200) DEFAULT '',
    telegram_id VARCHAR(100) DEFAULT '',
    preferred_language VARCHAR(5) DEFAULT 'nl',
    max_price NUMERIC(12, 2),
    min_rooms INTEGER,
    regions TEXT[] DEFAULT '{}',
    channels TEXT[] DEFAULT '{log}',
    min_value_ratio NUMERIC(4, 2) DEFAULT 1.05,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alert log
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES user_preferences(user_id),
    listing_id VARCHAR(64) NOT NULL REFERENCES listings(id),
    channel VARCHAR(20) NOT NULL,
    priority VARCHAR(10) DEFAULT 'normal',
    sent_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id, sent_at);

-- Pipeline run log
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR(16) PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running',
    ingested INTEGER DEFAULT 0,
    scored INTEGER DEFAULT 0,
    undervalued INTEGER DEFAULT 0,
    alerts_sent INTEGER DEFAULT 0,
    stats JSONB DEFAULT '{}'
);
