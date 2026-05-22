"""The Episode record (L2) and helpers shared across memory subsystems."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Episode:
    """One stored episodic memory plus the metadata the retrieval scorer needs."""

    content: str
    role: str = "user"
    salience: float = 1.0
    tier_low: int = 1
    tier_high: int = 2
    weekly_decay: float = 0.60
    vitality: float = 1.0
    tone: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_access: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: Optional[int] = None

    # snapshot of the state S at encoding + L7 bookkeeping (used by reconsolidation)
    s_e: Optional[float] = None       # emotional intensity when this was encoded
    s_v: Optional[float] = None       # vulnerability when this was encoded
    reconsolidated_at: Optional[datetime] = None

    # transient, populated during retrieval (not persisted)
    similarity: float = 0.0   # cosine sim from the kNN cast, in [0,1]
    score: float = 0.0        # final R(m) after re-ranking

    def age_days(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - self.created_at).total_seconds() / 86400.0)
