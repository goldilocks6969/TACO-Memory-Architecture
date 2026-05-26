"""L10 — the identity abstraction layer: a persistent self-model of the user.

Episodes are isolated events; semantic beliefs are roll-ups of *fading* episodes.
Neither answers "who is this person, durably?". The identity layer does: a small
set of high-confidence attributes (relationships, values, roles, ongoing
struggles) that persist across sessions and ground every reasoning turn.

It is *abstraction*, not storage: identity is consolidated only from
high-salience moments, and repeated, consistent evidence raises confidence while
contradicting evidence revises the belief. The whole self-model is injected into
the briefing so the engine reasons from a stable picture of the person, not just
the last few things they typed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import psycopg

from .. import config
from . import store
from ..state import LatentState


@dataclass
class IdentityUpdate:
    attribute: str
    value: str
    confidence: float
    changed: bool   # True if this revised an existing value (belief revision)


def reconcile(existing: Optional[Tuple[str, float]], new_value: str,
              new_conf: float,
              reinforce: float = config.IDENTITY_CONF_REINFORCE
              ) -> Tuple[str, float, bool]:
    """Fold new evidence into an existing attribute. Pure, so it is unit-tested.

    - No prior belief → adopt the new value at its stated confidence.
    - Consistent evidence → keep the value, reinforce confidence asymptotically
      toward 1.0 (the more often it recurs, the more certain we are).
    - Contradicting evidence → revise to the new value at its own confidence
      (belief revision; we do not average a person into a contradiction).
    """
    if existing is None:
        return new_value, max(0.0, min(1.0, new_conf)), False

    old_value, old_conf = existing
    consistent = (new_value.strip().lower() == old_value.strip().lower()
                  or new_value.strip().lower() in old_value.strip().lower()
                  or old_value.strip().lower() in new_value.strip().lower())
    if consistent:
        conf = old_conf + (1.0 - old_conf) * reinforce
        # keep the longer (more specific) phrasing of the same fact
        value = new_value if len(new_value) > len(old_value) else old_value
        return value, min(1.0, conf), False
    return new_value, max(0.0, min(1.0, new_conf)), True


def consolidate(conn: psycopg.Connection, content: str, salience: float,
                state: LatentState,
                extract_fn: Callable[[str, List[Tuple]], List[dict]],
                user_id: str = store.DEFAULT_USER_ID,
                ) -> List[IdentityUpdate]:
    """Update the self-model from one (already-stored) high-salience turn.

    Gated by salience so identity is built from moments that matter, not chatter.
    Returns the applied updates (empty when nothing identity-level was found).
    """
    if salience < config.IDENTITY_SALIENCE_MIN:
        return []

    known = store.identity_snapshot(conn, limit=20, user_id=user_id)
    facts = extract_fn(content, known)

    applied: List[IdentityUpdate] = []
    for f in facts:
        attribute = str(f.get("attribute", "")).strip()
        value = str(f.get("value", "")).strip()
        if not attribute or not value:
            continue
        existing = store.get_identity(conn, attribute, user_id=user_id)
        new_value, conf, changed = reconcile(
            existing, value, float(f.get("confidence", 0.5)))
        store.upsert_identity(conn, attribute, new_value, conf, user_id=user_id)
        applied.append(IdentityUpdate(attribute, new_value, conf, changed))
    return applied


def snapshot_lines(conn: psycopg.Connection,
                   user_id: str = store.DEFAULT_USER_ID) -> List[str]:
    """Render the asserted self-model for the briefing (confident attributes only)."""
    rows = store.identity_snapshot(
        conn, limit=config.IDENTITY_TOP_K,
        min_confidence=config.IDENTITY_MIN_CONFIDENCE,
        user_id=user_id)
    return [f"{attr} — {val} (confidence {conf:.0%})" for attr, val, conf in rows]
