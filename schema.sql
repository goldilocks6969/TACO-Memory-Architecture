-- Taco AI OS schema. Maps the ten-layer cognitive stack (§3.2) to storage.
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
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram (BM25-ish) matching for facts

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

-- Structured FACTS — the retrieval target (Phase 1). Episodes stay the raw
-- conversational record (feeding the L4 emotional timeline and L7
-- reconsolidation); facts are extracted, deduplicated, paraphrase-retrievable
-- units. Several columns (emotional_cause, user_belief, retrieval_cues) are
-- TACO-specific and have no Mem0 equivalent.
CREATE TABLE IF NOT EXISTS facts (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Core content
    summary       TEXT NOT NULL,        -- "User's father died of a heart attack last month"
    embedding     vector(1536),         -- text-embedding-3-small for now; Phase 9 → bge-large(1024)

    -- Classification
    event_type    TEXT,                 -- conflict|goal|preference|fear|plan|relationship
                                        -- |achievement|failure|health|money|identity|thread
    fact_type     TEXT,                 -- 'preference'|'event'|'relationship'|'state'|'thread'

    -- Emotional signal (TACO-specific, not in Mem0)
    emotional_tone   TEXT,              -- 'distress' | 'tender' | 'concerned' | ...
    emotional_cause  TEXT,              -- "fear of failing the interview again"
    user_belief      TEXT,              -- "I mess up important opportunities"
    salience         REAL NOT NULL,

    -- Retrieval expansion (cheap recall booster, used by Phase 2 hybrid retrieval)
    retrieval_cues   TEXT[],            -- alt phrasings: ["interview anxiety","fear of failure"]
    cues_text        TEXT,              -- retrieval_cues joined; kept in sync by the app
                                        -- (array_to_string isn't IMMUTABLE, so we can't
                                        -- index the array directly — denormalize instead)

    -- Entity graph (Phase 2)
    entity_keys      TEXT[],            -- ["person:maya", "role:interview"]

    -- Temporal validity (Phase 5)
    valid_from       TIMESTAMPTZ,       -- when the fact became true
    valid_until      TIMESTAMPTZ,       -- null = currently true

    -- Lineage
    source_episode_ids BIGINT[],        -- episodes this fact was drawn from
    superseded_by      BIGINT REFERENCES facts(id),

    -- Status: 'current' | 'outdated' | 'uncertain' | 'resolved'
    validity      TEXT NOT NULL DEFAULT 'current',

    -- Open-thread tracking (when fact_type = 'thread')
    thread_status TEXT,                 -- 'unresolved' | 'in_progress' | 'resolved'
    due_date      TIMESTAMPTZ,

    -- Decay
    vitality      REAL NOT NULL DEFAULT 1.0,
    confidence    REAL NOT NULL DEFAULT 0.7
);
CREATE INDEX IF NOT EXISTS facts_embedding_idx
    ON facts USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS facts_entity_idx ON facts USING gin (entity_keys);
CREATE INDEX IF NOT EXISTS facts_summary_trgm ON facts USING gin (summary gin_trgm_ops);
CREATE INDEX IF NOT EXISTS facts_cues_trgm ON facts USING gin (cues_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS facts_validity_idx ON facts (validity, valid_until);
CREATE INDEX IF NOT EXISTS facts_threads_idx ON facts (thread_status) WHERE fact_type = 'thread';

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
