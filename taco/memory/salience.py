"""Write-time emotional salience gate (§4.1, §4.3, Figure 5).

Every turn is scored 1–10 and stored only if the score clears the
state-modulated threshold theta(S). Above-threshold episodes are written with
full metadata and a tier-specific decay rate; below-threshold episodes are
dropped (or kept only in the shallow short-term cache, i.e. working memory).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .episode import Episode
from .. import config
from ..state import LatentState


@dataclass
class GateDecision:
    stored: bool
    salience: float
    threshold: float
    tone: str
    tier_label: str


def build_episode(role: str, content: str, analysis: Dict) -> Episode:
    """Construct an Episode with tier + decay rate resolved from its salience."""
    salience = float(analysis["salience"])
    tier = config.tier_for_score(salience)
    return Episode(
        content=content,
        role=role,
        salience=salience,
        tier_low=tier.low,
        tier_high=tier.high,
        weekly_decay=tier.weekly_decay,
        vitality=config.INITIAL_VITALITY,
        tone=analysis.get("tone"),
    )


def gate(salience: float, state: LatentState) -> bool:
    """Store iff salience clears theta(S). The threshold is the first place the
    user's state changes cognition rather than tone (§4.3)."""
    return salience >= state.theta()
