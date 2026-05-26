"""CRUD over the memory tables and the pgvector kNN candidate cast."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import psycopg

from .episode import Episode
from .fact import Fact
from ..state import LatentState


def _vec(embedding: List[float]) -> np.ndarray:
    """pgvector's psycopg adapter binds numpy arrays to the `vector` type."""
    return np.asarray(embedding, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Episodes (L2)
# --------------------------------------------------------------------------- #
def add_episode(conn: psycopg.Connection, ep: Episode, embedding: List[float],
                state: Optional[LatentState] = None) -> int:
    s = state or LatentState()
    row = conn.execute(
        """
        INSERT INTO episodes
            (role, content, embedding, salience, tier_low, tier_high,
             weekly_decay, vitality, tone, s_e, s_k, s_v, s_r)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """,
        (ep.role, ep.content, _vec(embedding), ep.salience, ep.tier_low,
         ep.tier_high, ep.weekly_decay, ep.vitality, ep.tone,
         s.E, s.K, s.V, s.R),
    ).fetchone()
    ep.id = row[0]
    return ep.id


def knn_candidates(conn: psycopg.Connection, query_embedding: List[float],
                   k: int) -> List[Episode]:
    """Cast the top-k nearest live episodes by cosine distance (Figure 4 step 1).

    pgvector's `<=>` is cosine distance; similarity = 1 - distance.
    Only non-abstracted episodes with surviving vitality are eligible.
    """
    rows = conn.execute(
        """
        SELECT id, role, content, salience, tier_low, tier_high, weekly_decay,
               vitality, tone, created_at, last_access,
               s_e, s_v, reconsolidated_at,
               1 - (embedding <=> %s) AS similarity
        FROM episodes
        WHERE NOT abstracted AND vitality >= 0.15
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (_vec(query_embedding), _vec(query_embedding), k),
    ).fetchall()
    out: List[Episode] = []
    for r in rows:
        out.append(Episode(
            id=r[0], role=r[1], content=r[2], salience=r[3], tier_low=r[4],
            tier_high=r[5], weekly_decay=r[6], vitality=r[7], tone=r[8],
            created_at=r[9], last_access=r[10], s_e=r[11], s_v=r[12],
            reconsolidated_at=r[13], similarity=max(0.0, float(r[14])),
        ))
    return out


def touch_access(conn: psycopg.Connection, ids: List[int]) -> None:
    """Mark episodes as just retrieved (feeds L7 reconsolidation)."""
    if not ids:
        return
    conn.execute(
        "UPDATE episodes SET last_access = now() WHERE id = ANY(%s)", (ids,)
    )


def recent_episodes(conn: psycopg.Connection, limit: int = 6) -> List[Episode]:
    """Most recent stored turns — backs the L1 working-memory window."""
    rows = conn.execute(
        """
        SELECT id, role, content, salience, tone, created_at
        FROM episodes ORDER BY created_at DESC LIMIT %s
        """,
        (limit,),
    ).fetchall()
    return [Episode(id=r[0], role=r[1], content=r[2], salience=r[3],
                    tone=r[4], created_at=r[5]) for r in reversed(rows)]


# --------------------------------------------------------------------------- #
# Facts (Phase 1) — the structured retrieval target
# --------------------------------------------------------------------------- #
def _cues_text(cues) -> Optional[str]:
    return " ".join(cues) if cues else None


def add_fact(conn: psycopg.Connection, fact: Fact, embedding: List[float],
             source_episode_id: Optional[int] = None,
             state: Optional[LatentState] = None) -> int:
    """Insert a new fact and return its id. `cues_text` is kept in sync with
    `retrieval_cues` (the array can't be trigram-indexed directly)."""
    src = [source_episode_id] if source_episode_id else None
    row = conn.execute(
        """
        INSERT INTO facts
            (summary, embedding, event_type, fact_type, emotional_tone,
             emotional_cause, user_belief, salience, retrieval_cues, cues_text,
             entity_keys, validity, thread_status, due_date, confidence,
             source_episode_ids)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """,
        (fact.summary, _vec(embedding), fact.event_type, fact.fact_type,
         fact.emotional_tone, fact.emotional_cause, fact.user_belief,
         fact.salience, fact.retrieval_cues or None, _cues_text(fact.retrieval_cues),
         fact.entity_keys or None, fact.validity, fact.thread_status,
         fact.due_date, fact.confidence, src),
    ).fetchone()
    fact.id = row[0]
    return fact.id


def fact_neighbors(conn: psycopg.Connection, embedding: List[float],
                   k: int = 3, min_sim: float = 0.7) -> List[Fact]:
    """Nearest CURRENT facts above a similarity floor — the dedup candidates."""
    emb = _vec(embedding)
    rows = conn.execute(
        """
        SELECT id, summary, salience, emotional_tone, entity_keys,
               1 - (embedding <=> %s) AS sim
        FROM facts
        WHERE validity = 'current' AND embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (emb, emb, k),
    ).fetchall()
    out: List[Fact] = []
    for r in rows:
        if float(r[5]) < min_sim:
            continue
        out.append(Fact(id=r[0], summary=r[1], salience=r[2],
                        emotional_tone=r[3], entity_keys=list(r[4] or []),
                        similarity=max(0.0, float(r[5]))))
    return out


def _fact_row_to_episode(r) -> Episode:
    """Adapt a fact row into an Episode so the existing reranker/briefing (which
    are episode-shaped) can consume facts unchanged in Phase 1."""
    return Episode(
        id=r[0], role="memory", content=r[1], salience=float(r[2]),
        vitality=float(r[3]), tone=r[4], created_at=r[5],
        similarity=max(0.0, float(r[6])),
    )


def fact_knn_candidates(conn: psycopg.Connection, query_embedding: List[float],
                        k: int, only_current: bool = True) -> List[Episode]:
    """Cast the nearest facts (as Episode adapters) for the briefing path."""
    where = "embedding IS NOT NULL" + (" AND validity = 'current'" if only_current else "")
    emb = _vec(query_embedding)
    rows = conn.execute(
        f"""
        SELECT id, summary, salience, vitality, emotional_tone, created_at,
               1 - (embedding <=> %s) AS similarity
        FROM facts
        WHERE {where}
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (emb, emb, k),
    ).fetchall()
    return [_fact_row_to_episode(r) for r in rows]


def merge_fact(conn: psycopg.Connection, target_id: int, content: str,
               embedding: List[float], source_episode_id: Optional[int],
               cues=None) -> None:
    """MERGE: refresh an existing fact in place and extend its lineage."""
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
        WHERE id = %s
        """,
        (content, _vec(embedding), (cues or []), _cues_text(cues) or "",
         source_episode_id, [source_episode_id] if source_episode_id else None,
         target_id),
    )


def supersede_fact(conn: psycopg.Connection, old_id: int, new_id: int) -> None:
    """UPDATE: retire the old fact, pointing it at the new current one."""
    conn.execute(
        """UPDATE facts
           SET validity = 'outdated', valid_until = now(),
               superseded_by = %s, updated_at = now()
           WHERE id = %s""",
        (new_id, old_id),
    )


def mark_fact_outdated(conn: psycopg.Connection, fact_id: int) -> None:
    """DELETE (soft): the user retracted this fact; keep it but mark it outdated."""
    conn.execute(
        """UPDATE facts SET validity = 'outdated', valid_until = now(),
               updated_at = now() WHERE id = %s""",
        (fact_id,),
    )


def due_threads(conn: psycopg.Connection, within_days: int = 7) -> List[Episode]:
    """Open threads (L-thread) whose due_date lands within the window — always
    surfaced in the briefing regardless of retrieval score (Phase 1.6)."""
    rows = conn.execute(
        """
        SELECT id, summary, salience, vitality, emotional_tone, created_at, 1.0
        FROM facts
        WHERE fact_type = 'thread' AND thread_status = 'unresolved'
              AND validity = 'current' AND due_date IS NOT NULL
              AND due_date <= now() + (%s || ' days')::interval
              AND due_date >= now() - interval '1 day'
        ORDER BY due_date ASC
        """,
        (within_days,),
    ).fetchall()
    return [_fact_row_to_episode(r) for r in rows]


# --------------------------------------------------------------------------- #
# Emotional timeline (L4)
# --------------------------------------------------------------------------- #
def add_emotional(conn: psycopg.Connection, episode_id: int, intensity: float,
                  tone: Optional[str], salience: float) -> None:
    conn.execute(
        """INSERT INTO emotional_timeline (episode_id, intensity, tone, salience)
           VALUES (%s,%s,%s,%s)""",
        (episode_id, intensity, tone, salience),
    )


def record_reconsolidation(conn: psycopg.Connection, episode_id: int,
                           new_tone: Optional[str], new_intensity: float,
                           salience: float) -> None:
    """L7: a retrieval re-experienced this episode from a changed state. Log the
    attenuated emotional charge on the timeline and update the episode's tone +
    reconsolidation timestamp (the trace is mutated by the act of remembering)."""
    conn.execute(
        """INSERT INTO emotional_timeline (episode_id, intensity, tone, salience)
           VALUES (%s,%s,%s,%s)""",
        (episode_id, new_intensity, new_tone, salience),
    )
    conn.execute(
        """UPDATE episodes SET tone = %s, reconsolidated_at = now()
           WHERE id = %s""",
        (new_tone, episode_id),
    )


# --------------------------------------------------------------------------- #
# Semantic beliefs (L3)
# --------------------------------------------------------------------------- #
def add_belief(conn: psycopg.Connection, belief: str, embedding: List[float],
               tone: Optional[str], source_count: int = 1,
               confidence: float = 0.5) -> int:
    row = conn.execute(
        """INSERT INTO semantic_beliefs (belief, embedding, tone, source_count, confidence)
           VALUES (%s,%s,%s,%s,%s) RETURNING id""",
        (belief, _vec(embedding), tone, source_count, confidence),
    ).fetchone()
    return row[0]


def belief_candidates(conn: psycopg.Connection, query_embedding: List[float],
                      k: int = 2) -> List[str]:
    rows = conn.execute(
        """SELECT belief FROM semantic_beliefs
           ORDER BY embedding <=> %s LIMIT %s""",
        (_vec(query_embedding), k),
    ).fetchall()
    return [r[0] for r in rows]


def upsert_belief_evolution(conn: psycopg.Connection, embedding: List[float],
                            belief: str, tone: Optional[str], confidence: float,
                            merge_sim: float = 0.90) -> int:
    """L7 → L3: rewrite the nearest existing belief if it's close enough, else
    insert. This is how reconsolidation *evolves* an abstraction rather than
    piling up near-duplicates ("I failed" → "that failure made me stronger")."""
    emb = _vec(embedding)
    near = conn.execute(
        """SELECT id, 1 - (embedding <=> %s) AS sim, source_count, confidence
           FROM semantic_beliefs ORDER BY embedding <=> %s LIMIT 1""",
        (emb, emb),
    ).fetchone()
    if near and float(near[1]) >= merge_sim:
        conn.execute(
            """UPDATE semantic_beliefs
               SET belief = %s, embedding = %s, tone = %s,
                   source_count = source_count + 1,
                   confidence = greatest(confidence, %s)
               WHERE id = %s""",
            (belief, emb, tone, confidence, near[0]),
        )
        return near[0]
    row = conn.execute(
        """INSERT INTO semantic_beliefs (belief, embedding, tone, source_count, confidence)
           VALUES (%s,%s,%s,2,%s) RETURNING id""",
        (belief, emb, tone, confidence),
    ).fetchone()
    return row[0]


# --------------------------------------------------------------------------- #
# Identity graph (L10)
# --------------------------------------------------------------------------- #
def upsert_identity(conn: psycopg.Connection, attribute: str, value: str,
                    confidence: float = 0.5) -> None:
    conn.execute(
        """INSERT INTO identity (attribute, value, confidence, updated_at)
           VALUES (%s,%s,%s, now())
           ON CONFLICT (attribute)
           DO UPDATE SET value = EXCLUDED.value,
                         confidence = EXCLUDED.confidence,
                         updated_at = now()""",
        (attribute, value, confidence),
    )


def get_identity(conn: psycopg.Connection, attribute: str):
    """Current (value, confidence) for an identity attribute, or None."""
    row = conn.execute(
        "SELECT value, confidence FROM identity WHERE attribute = %s",
        (attribute,),
    ).fetchone()
    return (row[0], float(row[1])) if row else None


def identity_snapshot(conn: psycopg.Connection, limit: int = 8,
                      min_confidence: float = 0.0):
    """The persistent self-model, most-confident attributes first (L10)."""
    rows = conn.execute(
        """SELECT attribute, value, confidence FROM identity
           WHERE confidence >= %s
           ORDER BY confidence DESC, updated_at DESC LIMIT %s""",
        (min_confidence, limit),
    ).fetchall()
    return [(r[0], r[1], float(r[2])) for r in rows]


# --------------------------------------------------------------------------- #
# Latent state log
# --------------------------------------------------------------------------- #
def log_state(conn: psycopg.Connection, s: LatentState) -> None:
    conn.execute(
        "INSERT INTO state_log (s_e,s_k,s_v,s_r) VALUES (%s,%s,%s,%s)",
        (s.E, s.K, s.V, s.R),
    )


def last_state(conn: psycopg.Connection) -> Optional[LatentState]:
    row = conn.execute(
        "SELECT s_e,s_k,s_v,s_r FROM state_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return LatentState(E=row[0], K=row[1], V=row[2], R=row[3])


def recent_states(conn: psycopg.Connection, limit: int = 8) -> List[LatentState]:
    """Last N persisted states, OLDEST→NEWEST — the trajectory L9 extrapolates."""
    rows = conn.execute(
        "SELECT s_e,s_k,s_v,s_r FROM state_log ORDER BY id DESC LIMIT %s",
        (limit,),
    ).fetchall()
    return [LatentState(E=r[0], K=r[1], V=r[2], R=r[3]) for r in reversed(rows)]


def recent_salient_episodes(conn: psycopg.Connection, limit: int = 12,
                            min_salience: float = 5.0) -> List[Episode]:
    """Recent above-threshold episodes — evidence for identity + theme prediction."""
    rows = conn.execute(
        """SELECT id, role, content, salience, tone, created_at, s_e, s_v
           FROM episodes WHERE salience >= %s
           ORDER BY created_at DESC LIMIT %s""",
        (min_salience, limit),
    ).fetchall()
    return [Episode(id=r[0], role=r[1], content=r[2], salience=r[3], tone=r[4],
                    created_at=r[5], s_e=r[6], s_v=r[7]) for r in reversed(rows)]


def last_contact_time(conn: psycopg.Connection) -> Optional[datetime]:
    row = conn.execute(
        "SELECT created_at FROM episodes ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None
