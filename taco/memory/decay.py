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


@dataclass
class DecayReport:
    decayed: int
    abstracted: int
    beliefs: List[str]


def recompute_vitality(conn: psycopg.Connection, extra_weeks: float = 0.0) -> int:
    """Recompute vitality = weekly_decay ^ age_weeks for every live episode.

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
        WHERE NOT abstracted
        """,
        (extra_weeks,),
    ).rowcount


def abstract_and_prune(
    conn: psycopg.Connection,
    embed_fn: Callable[[str], List[float]],
    abstract_fn: Callable[[List[str]], str],
    threshold: float = config.ABSTRACTION_THRESHOLD,
    limit: int = 50,
) -> List[str]:
    """Abstract sub-threshold episodes into semantic beliefs, then prune them."""
    rows = conn.execute(
        """
        SELECT id, content, tone FROM episodes
        WHERE NOT abstracted AND vitality < %s
        ORDER BY vitality ASC LIMIT %s
        """,
        (threshold, limit),
    ).fetchall()

    beliefs: List[str] = []
    for ep_id, content, tone in rows:
        belief = abstract_fn([content])
        emb = np.asarray(embed_fn(belief), dtype=np.float32)
        conn.execute(
            """INSERT INTO semantic_beliefs (belief, embedding, tone, source_count, confidence)
               VALUES (%s,%s,%s,1,0.5)""",
            (belief, emb, tone),
        )
        # prune the original episode: meaning kept in L3, detail forgotten
        conn.execute("DELETE FROM episodes WHERE id = %s", (ep_id,))
        beliefs.append(belief)

    return beliefs


def run(conn: psycopg.Connection, embed_fn, abstract_fn,
        extra_weeks: float = 0.0) -> DecayReport:
    decayed = recompute_vitality(conn, extra_weeks=extra_weeks)
    beliefs = abstract_and_prune(conn, embed_fn, abstract_fn)
    return DecayReport(decayed=decayed, abstracted=len(beliefs), beliefs=beliefs)
