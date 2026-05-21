-- Bubbles schema. Maps the ten-layer cognitive stack (§3.2) to storage.
--   L1 Working Memory ....... transient, held in process (see pipeline.py)
--   L2 Episodic Memory ...... episodes
--   L3 Semantic Memory ...... semantic_beliefs
--   L4 Emotional Memory ..... emotional_timeline
--   L5 Procedural Memory .... procedural
--   L6 Reflective Layer ..... reflections (self-generated abstractions)
--   L10 Identity Graph ...... identity
-- L7 (reconsolidation) and L8 (forgetting) act ON these tables rather than
-- being tables themselves; L9 (predictive prefetch) reads them.

CREATE EXTENSION IF NOT EXISTS vector;

-- L2: event-based autobiographical history. Each row is one stored episode.
CREATE TABLE IF NOT EXISTS episodes (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_access   TIMESTAMPTZ NOT NULL DEFAULT now(),
    role          TEXT NOT NULL,                 -- 'user' | 'assistant'
    content       TEXT NOT NULL,
    embedding     vector(1536),
    salience      REAL NOT NULL,                 -- 1..10 score at write time
    tier_low      INT  NOT NULL,                 -- salience tier band
    tier_high     INT  NOT NULL,
    weekly_decay  REAL NOT NULL,                 -- tier-specific vitality multiplier
    vitality      REAL NOT NULL DEFAULT 1.0,     -- decays weekly; abstract at <0.15
    tone          TEXT,                          -- coarse emotional tone label
    -- snapshot of S at encoding (useful for analysis / reconsolidation)
    s_e REAL, s_k REAL, s_v REAL, s_r REAL,
    reconsolidated_at TIMESTAMPTZ,               -- L7: last time a read mutated it
    abstracted    BOOLEAN NOT NULL DEFAULT FALSE -- L8: rolled up into a belief
);

CREATE INDEX IF NOT EXISTS episodes_embedding_idx
    ON episodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS episodes_vitality_idx ON episodes (vitality);

-- L3: beliefs abstracted from decayed episodes (meaning outlives the event).
CREATE TABLE IF NOT EXISTS semantic_beliefs (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    belief        TEXT NOT NULL,
    embedding     vector(1536),
    tone          TEXT,
    source_count  INT NOT NULL DEFAULT 1,        -- how many episodes rolled in
    confidence    REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS beliefs_embedding_idx
    ON semantic_beliefs USING hnsw (embedding vector_cosine_ops);

-- L4: independent, salience-weighted emotional timeline.
CREATE TABLE IF NOT EXISTS emotional_timeline (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    episode_id    BIGINT REFERENCES episodes(id) ON DELETE SET NULL,
    intensity     REAL NOT NULL,                 -- E at the moment
    tone          TEXT,
    salience      REAL NOT NULL
);

-- L5: learned workflows / interaction habits.
CREATE TABLE IF NOT EXISTS procedural (
    id            BIGSERIAL PRIMARY KEY,
    pattern       TEXT NOT NULL UNIQUE,
    weight        REAL NOT NULL DEFAULT 1.0,
    uses          INT  NOT NULL DEFAULT 0,
    last_used     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- L6: self-generated reflections produced on consolidation.
CREATE TABLE IF NOT EXISTS reflections (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    reflection    TEXT NOT NULL,
    embedding     vector(1536)
);

-- L10: persistent self-model of the user.
CREATE TABLE IF NOT EXISTS identity (
    id            BIGSERIAL PRIMARY KEY,
    attribute     TEXT NOT NULL UNIQUE,
    value         TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.5,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Persisted latent state log so S survives across sessions and feeds the
-- proactive-initiation job.
CREATE TABLE IF NOT EXISTS state_log (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    s_e REAL NOT NULL, s_k REAL NOT NULL, s_v REAL NOT NULL, s_r REAL NOT NULL
);
