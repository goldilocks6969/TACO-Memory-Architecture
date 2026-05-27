"""Tier-based decay (L8) and abstraction-before-pruning (→ L3) — §4.4, Figure 6.

Each memory's vitality starts at 1.0 and is multiplied weekly by its
tier-specific rate, so high-salience memories (×0.99/week) are effectively
permanent while small talk (×0.60/week) is pruned in days. When vitality falls
below 0.15 the episode is NOT simply deleted — it is first abstracted into a
durable semantic belief, preserving meaning while losing detail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import psycopg

from .. import config
from . import store as _store


@dataclass
class DecayReport:
    decayed: int
    abstracted: int
    beliefs: List[str]


def vitality_after_weeks(weekly_decay: float, weeks: float) -> float:
    """Return the remaining vitality for a tier after ``weeks``."""
    return weekly_decay ** weeks


def crosses_abstraction_threshold(weekly_decay: float, weeks: float,
                                  threshold: float = config.ABSTRACTION_THRESHOLD) -> bool:
    """Whether a memory should be abstracted before pruning at this age."""
    return vitality_after_weeks(weekly_decay, weeks) < threshold


def recompute_vitality(conn: psycopg.Connection, extra_weeks: float = 0.0,
                       user_id: str = _store.DEFAULT_USER_ID) -> int:
    """Recompute vitality = weekly_decay ^ age_weeks for this user's live episodes.

    Idempotent and time-accurate (matches the Figure 6 curves exactly). Pass
    `extra_weeks` to fast-forward the clock for demos.
    """
    age_weeks = (
        "(EXTRACT(EPOCH FROM (now() - created_at)) / 604800.0 + %s)"
    )
    return conn.execute(
        f"""
        UPDATE episodes
        SET vitality = power(weekly_decay, {age_weeks})
        WHERE user_id = %s AND NOT abstracted
        """,
        (extra_weeks, user_id),
    ).rowcount


def abstract_and_prune(
    conn: psycopg.Connection,
    embed_fn: Callable[[str], List[float]],
    abstract_fn: Callable[[List[str]], str],
    threshold: float = config.ABSTRACTION_THRESHOLD,
    limit: int = 50,
    user_id: str = _store.DEFAULT_USER_ID,
) -> List[str]:
    """Abstract sub-threshold episodes into semantic beliefs, then prune them."""
    rows = conn.execute(
        """
        SELECT id, content, tone FROM episodes
        WHERE user_id = %s AND NOT abstracted AND vitality < %s
        ORDER BY vitality ASC LIMIT %s
        """,
        (user_id, threshold, limit),
    ).fetchall()

    beliefs: List[str] = []
    for ep_id, content, tone in rows:
        belief = abstract_fn([content])
        emb = np.asarray(embed_fn(belief), dtype=np.float32)
        conn.execute(
            """INSERT INTO semantic_beliefs
                  (user_id, belief, embedding, tone, source_count, confidence)
               VALUES (%s,%s,%s,%s,1,0.5)""",
            (user_id, belief, emb, tone),
        )
        # prune the original episode: meaning kept in L3, detail forgotten
        conn.execute(
            "DELETE FROM episodes WHERE id = %s AND user_id = %s",
            (ep_id, user_id),
        )
        beliefs.append(belief)

    return beliefs


def run(conn: psycopg.Connection, embed_fn, abstract_fn,
        extra_weeks: float = 0.0,
        user_id: str = _store.DEFAULT_USER_ID) -> DecayReport:
    decayed = recompute_vitality(conn, extra_weeks=extra_weeks, user_id=user_id)
    beliefs = abstract_and_prune(conn, embed_fn, abstract_fn, user_id=user_id)
    return DecayReport(decayed=decayed, abstracted=len(beliefs), beliefs=beliefs)
