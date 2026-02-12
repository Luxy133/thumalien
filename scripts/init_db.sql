-- Initialisation de la base de données Thumalien

CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    uri TEXT UNIQUE NOT NULL,
    cid TEXT,
    author_handle TEXT NOT NULL,
    author_display_name TEXT,
    text_content TEXT NOT NULL,
    clean_text TEXT,
    lang VARCHAR(5),
    created_at TIMESTAMPTZ,
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    like_count INTEGER DEFAULT 0,
    repost_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS analyses (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    credibility_label VARCHAR(20) NOT NULL,
    credibility_score FLOAT NOT NULL,
    scores_detail JSONB,
    dominant_emotion VARCHAR(30),
    emotion_scores JSONB,
    explanation JSONB,
    analyzed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS energy_reports (
    id SERIAL PRIMARY KEY,
    task_name VARCHAR(100),
    duration_seconds FLOAT,
    emissions_kg_co2 FLOAT,
    energy_kwh FLOAT,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_sessions (
    id SERIAL PRIMARY KEY,
    query VARCHAR(200) NOT NULL,
    lang VARCHAR(5),
    num_posts INTEGER,
    num_fiable INTEGER DEFAULT 0,
    num_douteux INTEGER DEFAULT 0,
    num_fake INTEGER DEFAULT 0,
    total_emissions_co2 FLOAT DEFAULT 0.0,
    total_energy_kwh FLOAT DEFAULT 0.0,
    duration_seconds FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_handle);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_label ON analyses(credibility_label);
CREATE INDEX IF NOT EXISTS idx_analyses_post ON analyses(post_id);
