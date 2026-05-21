"""CRUD over the memory tables and the pgvector kNN candidate cast."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import psycopg

from .episode import Episode
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
            created_at=r[9], last_access=r[10], similarity=max(0.0, float(r[11])),
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
# Emotional timeline (L4)
# --------------------------------------------------------------------------- #
def add_emotional(conn: psycopg.Connection, episode_id: int, intensity: float,
                  tone: Optional[str], salience: float) -> None:
    conn.execute(
        """INSERT INTO emotional_timeline (episode_id, intensity, tone, salience)
           VALUES (%s,%s,%s,%s)""",
        (episode_id, intensity, tone, salience),
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


def last_contact_time(conn: psycopg.Connection) -> Optional[datetime]:
    row = conn.execute(
        "SELECT created_at FROM episodes ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None
