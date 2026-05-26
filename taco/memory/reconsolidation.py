"""L7 — reconsolidation: remembering rewrites the memory.

Humans don't replay static memories; recall *re-encodes* them in the present
state. A failure remembered from a steadier place stops being a wound and becomes
"the thing that made me stronger." TACO models this directly: when a memory
encoded in a hot emotional state is retrieved while the user is now calm, three
things happen — its emotional charge attenuates (L4), its semantic abstraction is
rewritten toward an integrated belief (L3), and the trace is timestamped as
reconsolidated so it is not churned every turn.

The event content is never altered; only its *meaning and weight* evolve. This is
deliberately gated and bounded (`RECON_MAX_PER_TURN`) so it stays cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

import psycopg

from .. import config
from . import store
from .episode import Episode
from ..state import LatentState


@dataclass
class ReconsolidationReport:
    rewrites: List[tuple] = field(default_factory=list)  # (episode_id, new_belief)

    @property
    def count(self) -> int:
        return len(self.rewrites)


def is_candidate(encoded_e: Optional[float], current_e: float,
                 reconsolidated_at: Optional[datetime],
                 now: Optional[datetime] = None) -> bool:
    """Eligible iff encoded hot, now substantially calmer, and off cooldown. Pure."""
    if encoded_e is None:
        return False
    if encoded_e < config.RECON_ENCODED_E_MIN:
        return False
    if (encoded_e - current_e) < config.RECON_DROP_MIN:
        return False
    if reconsolidated_at is not None:
        now = now or datetime.now(timezone.utc)
        if reconsolidated_at.tzinfo is None:
            reconsolidated_at = reconsolidated_at.replace(tzinfo=timezone.utc)
        age_days = (now - reconsolidated_at).total_seconds() / 86400.0
        if age_days < config.RECON_COOLDOWN_DAYS:
            return False
    return True


def attenuated_intensity(encoded_e: float) -> float:
    """The emotional charge the memory keeps after being re-held from calm."""
    return max(0.0, encoded_e * config.RECON_ATTENUATION)


def reconsolidate(conn: psycopg.Connection, retrieved: List[Episode],
                  current_state: LatentState, current_tone: str,
                  reconsolidate_fn: Callable[[str, str, str], str],
                  embed_fn: Callable[[str], List[float]],
                  user_id: str = store.DEFAULT_USER_ID,
                  ) -> ReconsolidationReport:
    """Evolve eligible retrieved memories. Bounded to RECON_MAX_PER_TURN rewrites."""
    report = ReconsolidationReport()
    # Most intensely-encoded memories reconsolidate first.
    eligible = [ep for ep in retrieved
                if ep.id and is_candidate(ep.s_e, current_state.E, ep.reconsolidated_at)]
    eligible.sort(key=lambda e: e.s_e or 0.0, reverse=True)

    for ep in eligible[:config.RECON_MAX_PER_TURN]:
        belief = reconsolidate_fn(ep.content, ep.tone or "difficult", current_tone)
        emb = embed_fn(belief)
        # L3: rewrite the abstraction toward the integrated meaning.
        store.upsert_belief_evolution(
            conn, emb, belief, current_tone,
            confidence=min(1.0, 0.6 + 0.1 * (current_state.K / 100.0)),
            user_id=user_id)
        # L4: attenuate the emotional charge; mutate the trace (tone + timestamp).
        store.record_reconsolidation(
            conn, ep.id, new_tone=current_tone,
            new_intensity=attenuated_intensity(ep.s_e or current_state.E),
            salience=ep.salience, user_id=user_id)
        report.rewrites.append((ep.id, belief))
    return report
