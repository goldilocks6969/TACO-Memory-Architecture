"""Central configuration: every tunable constant from the whitepaper lives here.

Keeping the paper's numbers in one module makes the architecture auditable —
each value below maps to a specific figure or equation in the source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------------------------------- #
# Runtime / providers
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("TACO_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("TACO_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # text-embedding-3-small
PG_DSN = os.getenv("TACO_PG_DSN", "postgresql://localhost:5432/taco")
MOCK = os.getenv("TACO_MOCK", "0") == "1"


# --------------------------------------------------------------------------- #
# Retrieval scoring R(m) — §4.2
#   R(m) = w_sem·sem + w_sal·sal + w_emo·emo + w_rec·rec + w_decay·decay
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: Dict[str, float] = {
    "sem": 0.35,
    "sal": 0.30,
    "emo": 0.20,
    "rec": 0.10,
    "decay": 0.05,
}

# w_emo(V) rises with vulnerability — Figure 7a
W_EMO_MIN = 0.20
W_EMO_MAX = 0.48

EMO_TONE_MATCH_BOOST = 0.30  # +0.3 if memory tone matches current user tone
RECENCY_WINDOW_DAYS = 30     # rec(m) is recency within a 30-day window

CANDIDATE_CAST = 20  # pgvector kNN candidates (Figure 4 step 1)
TOP_K = 4            # memories assembled into the briefing (Figure 4 step 3)


# --------------------------------------------------------------------------- #
# Write-time salience gate — §4.1, §4.3
#   theta(S) ∈ [2.1, 4.3]; default 4.0
# --------------------------------------------------------------------------- #
THETA_DEFAULT = 4.0
THETA_MIN = 2.1   # high E/V → capture more
THETA_MAX = 4.3   # transactional / low engagement → very selective


@dataclass(frozen=True)
class SalienceTier:
    """A salience band: score range, label, and weekly vitality decay rate."""
    low: int
    high: int
    label: str
    weekly_decay: float


# Figure 5 tier table — score range, example category, weekly decay rate.
SALIENCE_TIERS: Tuple[SalienceTier, ...] = (
    SalienceTier(9, 10, "breakups · deaths · identity-level revelations", 0.99),
    SalienceTier(7, 8, "conflict · significant fear · material updates", 0.95),
    SalienceTier(5, 6, "plans · moderate stress · revealed preferences", 0.88),
    SalienceTier(3, 4, "light personal sharing · mild emotional content", 0.75),
    SalienceTier(1, 2, "greetings · small talk · transactional", 0.60),
)

ABSTRACTION_THRESHOLD = 0.15  # §4.4 — below this vitality, abstract then prune
INITIAL_VITALITY = 1.0


def tier_for_score(score: float) -> SalienceTier:
    """Return the salience tier a 1–10 score falls into."""
    s = max(1, min(10, round(score)))
    for tier in SALIENCE_TIERS:
        if tier.low <= s <= tier.high:
            return tier
    return SALIENCE_TIERS[-1]


# --------------------------------------------------------------------------- #
# Latent state bounds — §3.1 (each dimension ∈ [0, 100])
# --------------------------------------------------------------------------- #
STATE_MIN = 0.0
STATE_MAX = 100.0

# Recency half-life (turns of "meaningful contact" decay) — calibrated to
# typical reply cadence. R is inverse time since last meaningful interaction.
RECENCY_HALFLIFE_HOURS = 24.0


@dataclass
class Settings:
    """Mutable runtime view, handy for tests that want to override values."""
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    theta_default: float = THETA_DEFAULT
    candidate_cast: int = CANDIDATE_CAST
    top_k: int = TOP_K
