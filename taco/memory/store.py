"""CRUD over the memory tables and the pgvector kNN candidate cast.

Every memory row carries a ``user_id`` namespace.  Every read filters by
``user_id``; every write stamps it.  ``user_id`` defaults to ``"default"`` for
the single-user CLI; the eval harness assigns a distinct ``user_id`` per
scenario so a grief persona's memories cannot leak into the next persona.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import psycopg

from .episode import Episode
from .fact import Fact
from ..state import LatentState

DEFAULT_USER_ID = "default"


def _vec(embedding: List[float]) -> np.ndarray:
    """pgvector's psycopg adapter binds numpy arrays to the `vector` type."""
    return np.asarray(embedding, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Episodes (L2)
# --------------------------------------------------------------------------- #
def add_episode(conn: psycopg.Connection, ep: Episode, embedding: List[float],
                state: Optional[LatentState] = None,
                user_id: str = DEFAULT_USER_ID) -> int:
    s = state or LatentState()
    ep.user_id = user_id
    row = conn.execute(
        """
        INSERT INTO episodes
            (user_id, role, content, embedding, salience, tier_low, tier_high,
             weekly_decay, vitality, tone, s_e, s_k, s_v, s_r)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """,
        (user_id, ep.role, ep.content, _vec(embedding), ep.salience,
         ep.tier_low, ep.tier_high, ep.weekly_decay, ep.vitality, ep.tone,
         s.E, s.K, s.V, s.R),
    ).fetchone()
    ep.id = row[0]
    return ep.id


def knn_candidates(conn: psycopg.Connection, query_embedding: List[float],
                   k: int, user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """Cast the top-k nearest live episodes by cosine distance (Figure 4 step 1).

    pgvector's `<=>` is cosine distance; similarity = 1 - distance.
    Only this user's non-abstracted episodes with surviving vitality are eligible.
    """
    rows = conn.execute(
        """
        SELECT id, user_id, role, content, salience, tier_low, tier_high,
               weekly_decay, vitality, tone, created_at, last_access,
               s_e, s_v, reconsolidated_at,
               1 - (embedding <=> %s) AS similarity
        FROM episodes
        WHERE user_id = %s AND NOT abstracted AND vitality >= 0.15
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (_vec(query_embedding), user_id, _vec(query_embedding), k),
    ).fetchall()
    out: List[Episode] = []
    for r in rows:
        out.append(Episode(
            id=r[0], user_id=r[1], role=r[2], content=r[3], salience=r[4],
            tier_low=r[5], tier_high=r[6], weekly_decay=r[7], vitality=r[8],
            tone=r[9], created_at=r[10], last_access=r[11], s_e=r[12], s_v=r[13],
            reconsolidated_at=r[14], similarity=max(0.0, float(r[15])),
        ))
    return out


def touch_access(conn: psycopg.Connection, ids: List[int],
                 user_id: str = DEFAULT_USER_ID) -> None:
    """Mark episodes as just retrieved (feeds L7 reconsolidation)."""
    if not ids:
        return
    conn.execute(
        "UPDATE episodes SET last_access = now() "
        "WHERE id = ANY(%s) AND user_id = %s",
        (ids, user_id),
    )


def recent_episodes(conn: psycopg.Connection, limit: int = 6,
                    user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """Most recent stored turns — backs the L1 working-memory window."""
    rows = conn.execute(
        """
        SELECT id, user_id, role, content, salience, tone, created_at
        FROM episodes WHERE user_id = %s
        ORDER BY created_at DESC LIMIT %s
        """,
        (user_id, limit),
    ).fetchall()
    return [Episode(id=r[0], user_id=r[1], role=r[2], content=r[3],
                    salience=r[4], tone=r[5], created_at=r[6])
            for r in reversed(rows)]


# --------------------------------------------------------------------------- #
# Facts (Phase 1) — the structured retrieval target
# --------------------------------------------------------------------------- #
def _cues_text(cues) -> Optional[str]:
    return " ".join(cues) if cues else None


def add_fact(conn: psycopg.Connection, fact: Fact, embedding: List[float],
             source_episode_id: Optional[int] = None,
             state: Optional[LatentState] = None,
             user_id: str = DEFAULT_USER_ID) -> int:
    """Insert a new fact and return its id. `cues_text` is kept in sync with
    `retrieval_cues` (the array can't be trigram-indexed directly)."""
    src = [source_episode_id] if source_episode_id else None
    fact.user_id = user_id
    row = conn.execute(
        """
        INSERT INTO facts
            (user_id, summary, embedding, event_type, fact_type, emotional_tone,
             emotional_cause, user_belief, salience, retrieval_cues, cues_text,
             entity_keys, validity, thread_status, due_date, confidence,
             source_episode_ids)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """,
        (user_id, fact.summary, _vec(embedding), fact.event_type, fact.fact_type,
         fact.emotional_tone, fact.emotional_cause, fact.user_belief,
         fact.salience, fact.retrieval_cues or None, _cues_text(fact.retrieval_cues),
         fact.entity_keys or None, fact.validity, fact.thread_status,
         fact.due_date, fact.confidence, src),
    ).fetchone()
    fact.id = row[0]
    return fact.id


def fact_neighbors(conn: psycopg.Connection, embedding: List[float],
                   k: int = 3, min_sim: float = 0.7,
                   user_id: str = DEFAULT_USER_ID) -> List[Fact]:
    """Nearest CURRENT facts above a similarity floor — the dedup candidates."""
    emb = _vec(embedding)
    rows = conn.execute(
        """
        SELECT id, user_id, summary, salience, emotional_tone, entity_keys,
               1 - (embedding <=> %s) AS sim
        FROM facts
        WHERE user_id = %s AND validity = 'current' AND embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (emb, user_id, emb, k),
    ).fetchall()
    out: List[Fact] = []
    for r in rows:
        if float(r[6]) < min_sim:
            continue
        out.append(Fact(id=r[0], user_id=r[1], summary=r[2], salience=r[3],
                        emotional_tone=r[4], entity_keys=list(r[5] or []),
                        similarity=max(0.0, float(r[6]))))
    return out


def recent_trace_facts(conn: psycopg.Connection, limit: int = 80,
                       user_id: str = DEFAULT_USER_ID) -> List[Fact]:
    """Recent trace/bridge facts with metadata for write-time consolidation."""
    rows = conn.execute(
        """
        SELECT id, user_id, summary, fact_type, event_type, emotional_tone,
               salience, retrieval_cues, entity_keys, validity, confidence,
               created_at
        FROM facts
        WHERE user_id = %s
              AND validity = 'current'
              AND fact_type IN ('trace', 'bridge')
        ORDER BY created_at DESC, id DESC
        LIMIT %s
        """,
        (user_id, limit),
    ).fetchall()
    out: List[Fact] = []
    for r in rows:
        out.append(Fact(
            id=r[0], user_id=r[1], summary=r[2], fact_type=r[3],
            event_type=r[4], emotional_tone=r[5], salience=float(r[6]),
            retrieval_cues=list(r[7] or []), entity_keys=list(r[8] or []),
            validity=r[9], confidence=float(r[10]), created_at=r[11],
        ))
    return list(reversed(out))


def _fact_row_to_episode(r) -> Episode:
    """Adapt a fact row into an Episode so the existing reranker/briefing (which
    are episode-shaped) can consume facts unchanged in Phase 1.

    Row shape: (id, user_id, summary, salience, vitality, tone, created_at,
                similarity[, fact_type, event_type, thread_status,
                retrieval_cues, entity_keys]).
    """
    ep = Episode(
        id=r[0], user_id=r[1], role="memory", content=r[2],
        salience=float(r[3]), vitality=float(r[4]), tone=r[5],
        created_at=r[6], similarity=max(0.0, float(r[7])),
    )
    if len(r) > 8:
        ep.fact_type = r[8]
        ep.event_type = r[9]
        ep.thread_status = r[10]
        ep.retrieval_cues = list(r[11] or [])
        ep.entity_keys = list(r[12] or []) if len(r) > 12 else []
    return ep


def fact_knn_candidates(conn: psycopg.Connection, query_embedding: List[float],
                        k: int, only_current: bool = True,
                        user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """Cast the nearest facts (as Episode adapters) for the briefing path."""
    where = "user_id = %s AND embedding IS NOT NULL"
    if only_current:
        where += " AND validity = 'current'"
    emb = _vec(query_embedding)
    rows = conn.execute(
        f"""
        SELECT id, user_id, summary, salience, vitality, emotional_tone, created_at,
               1 - (embedding <=> %s) AS similarity,
               fact_type, event_type, thread_status, retrieval_cues, entity_keys
        FROM facts
        WHERE {where}
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (emb, user_id, emb, k),
    ).fetchall()
    return [_fact_row_to_episode(r) for r in rows]


def merge_fact(conn: psycopg.Connection, target_id: int, content: str,
               embedding: List[float], source_episode_id: Optional[int],
               cues=None, user_id: str = DEFAULT_USER_ID) -> None:
    """MERGE: refresh an existing fact in place and extend its lineage.

    The ``WHERE id = %s AND user_id = %s`` guard makes a cross-namespace merge
    a silent no-op rather than a data corruption.
    """
    conn.execute(
        """
        UPDATE facts
        SET summary = %s,
            embedding = %s,
            retrieval_cues = COALESCE(retrieval_cues, '{}') || %s,
            cues_text = trim(both ' ' from COALESCE(cues_text,'') || ' ' || %s),
            source_episode_ids = CASE WHEN %s IS NULL THEN source_episode_ids
                ELSE COALESCE(source_episode_ids, '{}') || %s END,
            confidence = least(1.0, confidence + 0.1),
            updated_at = now()
        WHERE id = %s AND user_id = %s
        """,
        (content, _vec(embedding), (cues or []), _cues_text(cues) or "",
         source_episode_id, [source_episode_id] if source_episode_id else None,
         target_id, user_id),
    )


def supersede_fact(conn: psycopg.Connection, old_id: int, new_id: int,
                   user_id: str = DEFAULT_USER_ID) -> None:
    """UPDATE: retire the old fact, pointing it at the new current one."""
    conn.execute(
        """UPDATE facts
           SET validity = 'outdated', valid_until = now(),
               superseded_by = %s, updated_at = now()
           WHERE id = %s AND user_id = %s""",
        (new_id, old_id, user_id),
    )


def mark_fact_outdated(conn: psycopg.Connection, fact_id: int,
                       user_id: str = DEFAULT_USER_ID) -> None:
    """DELETE (soft): the user retracted this fact; keep it but mark it outdated."""
    conn.execute(
        """UPDATE facts SET validity = 'outdated', valid_until = now(),
               updated_at = now() WHERE id = %s AND user_id = %s""",
        (fact_id, user_id),
    )


# --------------------------------------------------------------------------- #
# Phase 2 retrievers — text, cue, and entity-overlap candidate searches that
# complement the semantic kNN.  Each one catches the "extension/function
# not available" failure mode and returns ``[]`` so a deployment without
# pg_trgm doesn't crash the whole hybrid pipeline; the orchestrator logs the
# fallback.
# --------------------------------------------------------------------------- #
def _safe_fact_text_query(conn: psycopg.Connection, sql: str, params: tuple,
                          label: str) -> List[Episode]:
    """Run a pg_trgm-backed query and degrade to ``[]`` on missing extension
    / index — callers (hybrid orchestrator) check the resulting count and
    log a warning when the empty list is the *result of a fallback*."""
    try:
        rows = conn.execute(sql, params).fetchall()
    except psycopg.Error as e:
        # Don't let pg_trgm being unavailable blow up the run; the hybrid
        # orchestrator continues with the remaining retrievers.
        import logging
        logging.getLogger("taco.store").warning(
            "%s failed (%s); falling back to no candidates", label, e
        )
        try:
            conn.rollback()
        except Exception:
            pass
        return []
    return [_fact_row_to_episode(r) for r in rows]


def fact_text_search(conn: psycopg.Connection, query_text: str, k: int = 20,
                     min_similarity: float = 0.05,
                     user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """pg_trgm similarity over ``facts.summary`` — surfaces facts whose
    summary text overlaps the query (paraphrase / surface form match).

    The result list is ordered by trigram similarity descending and capped
    at *k*; rows below ``min_similarity`` are dropped to keep RRF clean.
    The Episode adapter's ``similarity`` field carries the trigram score so
    callers can inspect it.
    """
    if not (query_text or "").strip():
        return []
    sql = """
        SELECT id, user_id, summary, salience, vitality, emotional_tone,
               created_at, similarity(summary, %s) AS sim,
               fact_type, event_type, thread_status, retrieval_cues, entity_keys
        FROM facts
        WHERE user_id = %s AND validity = 'current'
              AND summary %% %s
              AND similarity(summary, %s) >= %s
        ORDER BY similarity(summary, %s) DESC
        LIMIT %s
    """
    return _safe_fact_text_query(
        conn, sql,
        (query_text, user_id, query_text, query_text, min_similarity,
         query_text, k),
        label="store.fact_text_search",
    )


def fact_cue_search(conn: psycopg.Connection, query_text: str, k: int = 20,
                    min_similarity: float = 0.05,
                    user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """pg_trgm similarity over the denormalized ``facts.cues_text``.

    Each fact's ``retrieval_cues`` are joined into ``cues_text`` at write
    time (we can't index an array directly).  This retriever lets a probe
    like "what's my dog's name?" surface a fact whose cues include
    "dog name" / "pet name" even when the summary text doesn't say "dog".
    """
    if not (query_text or "").strip():
        return []
    sql = """
        SELECT id, user_id, summary, salience, vitality, emotional_tone,
               created_at, similarity(COALESCE(cues_text, ''), %s) AS sim,
               fact_type, event_type, thread_status, retrieval_cues, entity_keys
        FROM facts
        WHERE user_id = %s AND validity = 'current'
              AND cues_text IS NOT NULL
              AND cues_text %% %s
              AND similarity(cues_text, %s) >= %s
        ORDER BY similarity(cues_text, %s) DESC
        LIMIT %s
    """
    return _safe_fact_text_query(
        conn, sql,
        (query_text, user_id, query_text, query_text, min_similarity,
         query_text, k),
        label="store.fact_cue_search",
    )


def fact_entity_overlap(conn: psycopg.Connection, entity_keys: List[str],
                        k: int = 20,
                        user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """GIN array-overlap search over ``facts.entity_keys``.

    Returns facts that share at least one ``type:value`` key with the query.
    Ordered by overlap count desc, then salience.  Empty input → empty
    result (no entities means no entity-overlap retrieval to do).
    """
    keys = [k for k in (entity_keys or []) if k]
    if not keys:
        return []
    sql = """
        SELECT id, user_id, summary, salience, vitality, emotional_tone,
               created_at,
               cardinality(ARRAY(SELECT unnest(entity_keys)
                                 INTERSECT SELECT unnest(%s::text[]))) AS overlap,
               fact_type, event_type, thread_status, retrieval_cues, entity_keys
        FROM facts
        WHERE user_id = %s AND validity = 'current'
              AND entity_keys && %s::text[]
        ORDER BY overlap DESC, salience DESC
        LIMIT %s
    """
    return _safe_fact_text_query(
        conn, sql, (keys, user_id, keys, k),
        label="store.fact_entity_overlap",
    )


def fact_arc_origin_search(conn: psycopg.Connection, entity_keys: List[str],
                           k: int = 20,
                           user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """Retrieve early lifecycle traces for origin/trajectory questions.

    This uses only existing trace metadata: ``phase:*``, ``role:*``, ``arc:*``
    and ``time:YYYY-MM-DD`` entity keys. It is the temporal-boundary complement
    to entity-overlap retrieval: when the query asks how something started, the
    earliest matching boundary should get a chance before semantic recency wins.
    """
    keys = [key for key in (entity_keys or []) if key]
    if not keys:
        return []
    sql = """
        WITH candidate AS (
            SELECT id, user_id, summary, salience, vitality, emotional_tone,
                   created_at,
                   cardinality(ARRAY(SELECT unnest(entity_keys)
                                     INTERSECT SELECT unnest(%s::text[]))) AS overlap,
                   fact_type, event_type, thread_status, retrieval_cues, entity_keys,
                   EXISTS(SELECT 1 FROM unnest(entity_keys) k
                          WHERE k IN ('phase:origin', 'role:avoidance', 'role:onset')) AS is_origin,
                   (
                       SELECT MIN(substring(k from 6)::date)
                       FROM unnest(entity_keys) k
                       WHERE k ~ '^time:[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                   ) AS event_date
            FROM facts
            WHERE user_id = %s
                  AND validity = 'current'
                  AND fact_type IN ('trace', 'bridge')
                  AND entity_keys && %s::text[]
        )
        SELECT id, user_id, summary, salience, vitality, emotional_tone,
               created_at, overlap::float / GREATEST(cardinality(%s::text[]), 1),
               fact_type, event_type, thread_status, retrieval_cues, entity_keys
        FROM candidate
        ORDER BY is_origin DESC, event_date ASC NULLS LAST, overlap DESC, salience DESC
        LIMIT %s
    """
    return _safe_fact_text_query(
        conn, sql, (keys, user_id, keys, keys, k),
        label="store.fact_arc_origin_search",
    )


def fact_trajectory_search(conn: psycopg.Connection, entity_keys: List[str],
                           k: int = 20,
                           user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """Retrieve a compact ordered trajectory over bridge/trace facts.

    Bridges are the preferred synthesized answer unit. Supporting trace facts
    still follow as origin, causal antecedent, coping/action, and latest state.
    This uses write-time causal/phase keys instead of asking the LLM to
    rediscover structure from raw top-k fragments.
    """
    keys = [key for key in (entity_keys or []) if key]
    if not keys:
        return []
    sql = """
        WITH candidate AS (
            SELECT id, user_id, summary, salience, vitality, emotional_tone,
                   created_at,
                   cardinality(ARRAY(SELECT unnest(entity_keys)
                                     INTERSECT SELECT unnest(%s::text[]))) AS overlap,
                   fact_type, event_type, thread_status, retrieval_cues, entity_keys,
                   fact_type = 'bridge' AS is_bridge,
                   EXISTS(SELECT 1 FROM unnest(entity_keys) k
                          WHERE k = 'phase:origin') AS is_origin,
                   EXISTS(SELECT 1 FROM unnest(entity_keys) k
                          WHERE k LIKE 'cause:%%' OR k = 'rel:caused_by') AS is_cause,
                   EXISTS(SELECT 1 FROM unnest(entity_keys) k
                          WHERE k LIKE 'coping:%%'
                                OR k IN ('rel:triggered_coping', 'rel:coping_response')) AS is_coping,
                   EXISTS(SELECT 1 FROM unnest(entity_keys) k
                          WHERE k IN ('phase:resolution', 'role:outcome',
                                      'role:resolution', 'rel:supersedes')) AS is_resolution,
                   (
                       SELECT MIN(substring(k from 6)::date)
                       FROM unnest(entity_keys) k
                       WHERE k ~ '^time:[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                   ) AS event_date
            FROM facts
            WHERE user_id = %s
                  AND validity = 'current'
                  AND fact_type IN ('bridge', 'trace')
                  AND entity_keys && %s::text[]
        ),
        ranked AS (
            SELECT *,
                   CASE
                       WHEN is_bridge THEN 0
                       WHEN is_origin THEN 1
                       WHEN is_cause THEN 2
                       WHEN is_coping THEN 3
                       WHEN is_resolution THEN 4
                       ELSE 5
                   END AS chain_rank
            FROM candidate
        )
        SELECT id, user_id, summary, salience, vitality, emotional_tone,
               created_at,
               (1.0 - chain_rank * 0.12)
                   + overlap::float / GREATEST(cardinality(%s::text[]), 1),
               fact_type, event_type, thread_status, retrieval_cues, entity_keys
        FROM ranked
        ORDER BY chain_rank ASC, event_date ASC NULLS LAST, overlap DESC, salience DESC
        LIMIT %s
    """
    return _safe_fact_text_query(
        conn, sql, (keys, user_id, keys, keys, k),
        label="store.fact_trajectory_search",
    )


def due_threads(conn: psycopg.Connection, within_days: int = 7,
                user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """Open threads (L-thread) whose due_date lands within the window — always
    surfaced in the briefing regardless of retrieval score (Phase 1.6)."""
    rows = conn.execute(
        """
        SELECT id, user_id, summary, salience, vitality, emotional_tone,
               created_at, 1.0,
               fact_type, event_type, thread_status, retrieval_cues
        FROM facts
        WHERE user_id = %s AND fact_type = 'thread'
              AND thread_status = 'unresolved'
              AND validity = 'current' AND due_date IS NOT NULL
              AND due_date <= now() + (%s || ' days')::interval
              AND due_date >= now() - interval '1 day'
        ORDER BY due_date ASC
        """,
        (user_id, within_days),
    ).fetchall()
    return [_fact_row_to_episode(r) for r in rows]


# --------------------------------------------------------------------------- #
# Emotional timeline (L4)
# --------------------------------------------------------------------------- #
def add_emotional(conn: psycopg.Connection, episode_id: int, intensity: float,
                  tone: Optional[str], salience: float,
                  user_id: str = DEFAULT_USER_ID) -> None:
    conn.execute(
        """INSERT INTO emotional_timeline (user_id, episode_id, intensity, tone, salience)
           VALUES (%s,%s,%s,%s,%s)""",
        (user_id, episode_id, intensity, tone, salience),
    )


def record_reconsolidation(conn: psycopg.Connection, episode_id: int,
                           new_tone: Optional[str], new_intensity: float,
                           salience: float,
                           user_id: str = DEFAULT_USER_ID) -> None:
    """L7: a retrieval re-experienced this episode from a changed state. Log the
    attenuated emotional charge on the timeline and update the episode's tone +
    reconsolidation timestamp (the trace is mutated by the act of remembering)."""
    conn.execute(
        """INSERT INTO emotional_timeline (user_id, episode_id, intensity, tone, salience)
           VALUES (%s,%s,%s,%s,%s)""",
        (user_id, episode_id, new_intensity, new_tone, salience),
    )
    conn.execute(
        """UPDATE episodes SET tone = %s, reconsolidated_at = now()
           WHERE id = %s AND user_id = %s""",
        (new_tone, episode_id, user_id),
    )


# --------------------------------------------------------------------------- #
# Semantic beliefs (L3)
# --------------------------------------------------------------------------- #
def add_belief(conn: psycopg.Connection, belief: str, embedding: List[float],
               tone: Optional[str], source_count: int = 1,
               confidence: float = 0.5,
               user_id: str = DEFAULT_USER_ID) -> int:
    row = conn.execute(
        """INSERT INTO semantic_beliefs
              (user_id, belief, embedding, tone, source_count, confidence)
           VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
        (user_id, belief, _vec(embedding), tone, source_count, confidence),
    ).fetchone()
    return row[0]


def belief_candidates(conn: psycopg.Connection, query_embedding: List[float],
                      k: int = 2,
                      user_id: str = DEFAULT_USER_ID) -> List[str]:
    rows = conn.execute(
        """SELECT belief FROM semantic_beliefs
           WHERE user_id = %s
           ORDER BY embedding <=> %s LIMIT %s""",
        (user_id, _vec(query_embedding), k),
    ).fetchall()
    return [r[0] for r in rows]


def upsert_belief_evolution(conn: psycopg.Connection, embedding: List[float],
                            belief: str, tone: Optional[str], confidence: float,
                            merge_sim: float = 0.90,
                            user_id: str = DEFAULT_USER_ID) -> int:
    """L7 → L3: rewrite the nearest existing belief if it's close enough, else
    insert. This is how reconsolidation *evolves* an abstraction rather than
    piling up near-duplicates ("I failed" → "that failure made me stronger")."""
    emb = _vec(embedding)
    near = conn.execute(
        """SELECT id, 1 - (embedding <=> %s) AS sim, source_count, confidence
           FROM semantic_beliefs
           WHERE user_id = %s
           ORDER BY embedding <=> %s LIMIT 1""",
        (emb, user_id, emb),
    ).fetchone()
    if near and float(near[1]) >= merge_sim:
        conn.execute(
            """UPDATE semantic_beliefs
               SET belief = %s, embedding = %s, tone = %s,
                   source_count = source_count + 1,
                   confidence = greatest(confidence, %s)
               WHERE id = %s AND user_id = %s""",
            (belief, emb, tone, confidence, near[0], user_id),
        )
        return near[0]
    row = conn.execute(
        """INSERT INTO semantic_beliefs
              (user_id, belief, embedding, tone, source_count, confidence)
           VALUES (%s,%s,%s,%s,2,%s) RETURNING id""",
        (user_id, belief, emb, tone, confidence),
    ).fetchone()
    return row[0]


# --------------------------------------------------------------------------- #
# Identity graph (L10) — UNIQUE (user_id, attribute), so each persona has its
# own self-model.
# --------------------------------------------------------------------------- #
def upsert_identity(conn: psycopg.Connection, attribute: str, value: str,
                    confidence: float = 0.5,
                    user_id: str = DEFAULT_USER_ID) -> None:
    conn.execute(
        """INSERT INTO identity (user_id, attribute, value, confidence, updated_at)
           VALUES (%s,%s,%s,%s, now())
           ON CONFLICT (user_id, attribute)
           DO UPDATE SET value = EXCLUDED.value,
                         confidence = EXCLUDED.confidence,
                         updated_at = now()""",
        (user_id, attribute, value, confidence),
    )


def get_identity(conn: psycopg.Connection, attribute: str,
                 user_id: str = DEFAULT_USER_ID):
    """Current (value, confidence) for an identity attribute, or None."""
    row = conn.execute(
        "SELECT value, confidence FROM identity "
        "WHERE user_id = %s AND attribute = %s",
        (user_id, attribute),
    ).fetchone()
    return (row[0], float(row[1])) if row else None


def identity_snapshot(conn: psycopg.Connection, limit: int = 8,
                      min_confidence: float = 0.0,
                      user_id: str = DEFAULT_USER_ID):
    """The persistent self-model, most-confident attributes first (L10)."""
    rows = conn.execute(
        """SELECT attribute, value, confidence FROM identity
           WHERE user_id = %s AND confidence >= %s
           ORDER BY confidence DESC, updated_at DESC LIMIT %s""",
        (user_id, min_confidence, limit),
    ).fetchall()
    return [(r[0], r[1], float(r[2])) for r in rows]


# --------------------------------------------------------------------------- #
# Latent state log
# --------------------------------------------------------------------------- #
def log_state(conn: psycopg.Connection, s: LatentState,
              user_id: str = DEFAULT_USER_ID) -> None:
    conn.execute(
        "INSERT INTO state_log (user_id, s_e,s_k,s_v,s_r) "
        "VALUES (%s,%s,%s,%s,%s)",
        (user_id, s.E, s.K, s.V, s.R),
    )


def last_state(conn: psycopg.Connection,
               user_id: str = DEFAULT_USER_ID) -> Optional[LatentState]:
    row = conn.execute(
        "SELECT s_e,s_k,s_v,s_r FROM state_log "
        "WHERE user_id = %s ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    if not row:
        return None
    return LatentState(E=row[0], K=row[1], V=row[2], R=row[3])


def recent_states(conn: psycopg.Connection, limit: int = 8,
                  user_id: str = DEFAULT_USER_ID) -> List[LatentState]:
    """Last N persisted states, OLDEST→NEWEST — the trajectory L9 extrapolates."""
    rows = conn.execute(
        "SELECT s_e,s_k,s_v,s_r FROM state_log "
        "WHERE user_id = %s ORDER BY id DESC LIMIT %s",
        (user_id, limit),
    ).fetchall()
    return [LatentState(E=r[0], K=r[1], V=r[2], R=r[3]) for r in reversed(rows)]


def recent_salient_episodes(conn: psycopg.Connection, limit: int = 12,
                            min_salience: float = 5.0,
                            user_id: str = DEFAULT_USER_ID) -> List[Episode]:
    """Recent above-threshold episodes — evidence for identity + theme prediction."""
    rows = conn.execute(
        """SELECT id, user_id, role, content, salience, tone, created_at, s_e, s_v
           FROM episodes WHERE user_id = %s AND salience >= %s
           ORDER BY created_at DESC LIMIT %s""",
        (user_id, min_salience, limit),
    ).fetchall()
    return [Episode(id=r[0], user_id=r[1], role=r[2], content=r[3],
                    salience=r[4], tone=r[5], created_at=r[6], s_e=r[7], s_v=r[8])
            for r in reversed(rows)]


def last_contact_time(conn: psycopg.Connection,
                      user_id: str = DEFAULT_USER_ID) -> Optional[datetime]:
    row = conn.execute(
        "SELECT created_at FROM episodes "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    return row[0] if row else None
