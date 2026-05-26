"""The four-dimensional latent state S = (E, K, V, R) and its modulation curves.

S is inferred continuously from interaction history (§3.1) and read by every
subsystem before it makes a decision. This module owns two things:

  1. The state itself + how it evolves each turn (`LatentState`, `infer_state`).
  2. The parametric S → parameter curves (§4): theta, w_emo, retrieval weight
     modulation, planning depth, interruption urgency, reinforcement rate.

The curves are deliberately smooth, monotonic, and bounded — the paper argues
these are required properties for a parameter system whose behaviour must be
predictable in production (§4.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict

from . import config


def clamp(x: float, lo: float = config.STATE_MIN, hi: float = config.STATE_MAX) -> float:
    return max(lo, min(hi, x))


@dataclass
class ContentSignal:
    """Per-turn affective read of message content (0–100), the raw input to E and V.

    Produced either by an LLM assessor or rule-based heuristics (§3.1).
    """
    emotional: float   # affective arousal detected in this message
    vulnerability: float  # openness / fragility / disclosure detected


@dataclass
class LatentState:
    """S = (E, K, V, R). Each dimension is bounded to [0, 100] (§3.1)."""

    E: float = 0.0   # Emotional intensity — current weighted affective arousal
    K: float = 0.0   # Engagement streak — depth/consistency of recent interaction
    V: float = 0.0   # Vulnerability — user's current openness / fragility
    R: float = 100.0  # Recency of contact — inverse time since last meaningful turn

    # ---- serialization ---------------------------------------------------- #
    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "LatentState":
        return cls(E=d["E"], K=d["K"], V=d["V"], R=d["R"])

    def __str__(self) -> str:
        return f"S(E={self.E:.0f} K={self.K:.0f} V={self.V:.0f} R={self.R:.0f})"

    # ---- §4.1  write-time salience threshold theta(S) --------------------- #
    def theta(self) -> float:
        """Salience gate threshold, theta(S) ∈ [2.1, 4.3].

        Falls with emotional intensity / vulnerability ("capture more" in
        affectively rich moments, Figure 7b) and rises when engagement is low
        ("be very selective" in transactional mode, §4.1).
        """
        capture_drive = max(self.E, self.V) / 100.0          # 0..1
        base = config.THETA_MAX - (config.THETA_MAX - config.THETA_MIN) * capture_drive
        # Low engagement nudges the threshold back up toward maximum selectivity.
        engagement_penalty = (1.0 - self.K / 100.0)
        theta = base + (config.THETA_MAX - base) * engagement_penalty * 0.5
        return round(max(config.THETA_MIN, min(config.THETA_MAX, theta)), 2)

    def theta_facts(self) -> float:
        """Lower, informational gate for extracted facts (vs. the emotional θ).

        Facts are the retrieval target, so paraphrasable low-arousal content (a
        name, a plan, a preference) should survive even when emotional charge is
        low. Sits a fixed step below θ(S) with a hard floor.
        """
        return max(1.5, self.theta() - 2.0)

    # ---- §4.2  emotional retrieval weight w_emo(V) ------------------------ #
    def w_emo(self) -> float:
        """Emotional retrieval weight, rises linearly with V from 0.20→0.48 (Fig 7a)."""
        return config.W_EMO_MIN + (config.W_EMO_MAX - config.W_EMO_MIN) * (self.V / 100.0)

    def retrieval_weights(self) -> Dict[str, float]:
        """State-modulated R(m) weights, renormalised to sum to 1.

        - w_emo rises with vulnerability (emotional resonance dominates when the
          user is most affected).
        - w_rec is elevated in crisis (high E + low recency) "for safety" (Fig 4).
        - In low-engagement states the defaults dominate (little modulation).
        """
        w = dict(config.DEFAULT_WEIGHTS)
        w["emo"] = self.w_emo()

        crisis = (self.E / 100.0) * (1.0 - self.R / 100.0)  # high arousal + been away
        w["rec"] = w["rec"] + 0.15 * crisis

        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    # ---- §4.6  planning depth -------------------------------------------- #
    def planning_depth(self, max_depth: int = 4) -> int:
        """How many steps the agent may plan ahead (1 = single-turn).

        Scales with engagement streak and trust (proxied by vulnerability):
        a new user gets single-turn responses; a high-streak, high-trust user
        gets multi-step autonomous planning (§4.6).
        """
        trust = 0.6 * self.K + 0.4 * self.V  # 0..100
        return 1 + round((trust / 100.0) * (max_depth - 1))

    # ---- §4.6  interruption timing --------------------------------------- #
    def interruption_urgency(self) -> float:
        """Urgency in [0,1] governing when a held message should surface.

        High emotional intensity + high vulnerability bias toward delivering
        soon; low urgency biases toward never interrupting (§4.6).
        """
        return clamp((self.E + self.V) / 200.0, 0.0, 1.0)

    def interruption_policy(self) -> str:
        u = self.interruption_urgency()
        if u >= 0.6:
            return "within_hour"
        if u >= 0.3:
            return "next_session"
        return "hold"

    # ---- §4.6  reinforcement weighting ----------------------------------- #
    def reinforcement_rate(self, base: float = 1.0) -> float:
        """Effective learning rate — itself a function of state (§4.6).

        Interactions during high-vulnerability, high-streak periods are weighted
        more heavily in shaping future behaviour ("learn faster from the moments
        that matter most").
        """
        return base * (1.0 + (self.V / 100.0) * (self.K / 100.0))


# --------------------------------------------------------------------------- #
# Continuous state inference (§3.1)
# --------------------------------------------------------------------------- #
def recency_score(hours_since_last: float) -> float:
    """R: inverse time since last meaningful contact, on a half-life decay."""
    half = config.RECENCY_HALFLIFE_HOURS
    return clamp(100.0 * (0.5 ** (hours_since_last / half)))


def infer_state(
    prev: LatentState,
    signal: ContentSignal,
    hours_since_last: float,
    *,
    e_alpha: float = 0.5,
    v_alpha: float = 0.5,
) -> LatentState:
    """Evolve S given the previous state, the new message's affective signal, and
    the time gap since the last turn.

    E and V relax toward the current content signal (so neutral content decays
    arousal). K (engagement streak) builds while contact stays frequent and
    erodes across long gaps. R is recomputed from the gap.
    """
    # E, V: exponential relaxation toward this turn's signal.
    E = clamp(prev.E + e_alpha * (signal.emotional - prev.E))
    V = clamp(prev.V + v_alpha * (signal.vulnerability - prev.V))

    # R: recency from the gap before this message arrived.
    R = recency_score(hours_since_last)

    # K: engagement streak. Frequent, deep contact builds it; gaps erode it.
    if hours_since_last <= config.RECENCY_HALFLIFE_HOURS:
        # depth of this turn (use the affective signal as a proxy for depth)
        depth = max(signal.emotional, signal.vulnerability) / 100.0
        K = clamp(prev.K + 8.0 + 12.0 * depth)
    else:
        gaps = hours_since_last / (config.RECENCY_HALFLIFE_HOURS * 7)  # weeks-ish
        K = clamp(prev.K * (0.6 ** gaps))

    return LatentState(E=E, K=K, V=V, R=R)
