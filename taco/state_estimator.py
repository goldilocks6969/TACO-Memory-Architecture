"""Latent state estimation for S = (E, K, V, R).

This module is the isolated "step 1" surface: given a user turn, the previous
state, and the time gap, produce the next bounded latent state. Downstream
memory code should consume the resulting S, not own the estimation policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .memory import extract
from .state import ContentSignal, LatentState, infer_state


@dataclass(frozen=True)
class StateEstimate:
    """A complete per-turn state read.

    ``light`` is kept because the same cheap constrained assessment also feeds
    salience and fact formation. Reusing it avoids a second model call.
    """

    state: LatentState
    signal: ContentSignal
    light: extract.LightExtract
    hours_since_last: float

    def as_payload(self) -> dict:
        """Constrained JSON-friendly state payload for SDK/demo consumers."""
        return {
            "E": round(self.state.E, 2),
            "K": round(self.state.K, 2),
            "V": round(self.state.V, 2),
            "R": round(self.state.R, 2),
        }


def estimate_state(
    message: str,
    previous: Optional[LatentState] = None,
    hours_since_last: float = 0.0,
    *,
    light: Optional[extract.LightExtract] = None,
) -> StateEstimate:
    """Estimate the next latent state from one user turn.

    The assessment path is intentionally narrow: one ``LightExtract`` supplies
    emotional arousal and vulnerability on 0-100 scales, then the smooth state
    transition in ``infer_state`` updates S. Callers that already performed the
    light extraction can pass it in to preserve one assessment call per turn.
    """
    prev = previous or LatentState()
    turn_light = light or extract.light_extract(message)
    signal = ContentSignal(
        emotional=turn_light.emotional,
        vulnerability=turn_light.vulnerability,
    )
    state = infer_state(prev, signal, hours_since_last)
    return StateEstimate(
        state=state,
        signal=signal,
        light=turn_light,
        hours_since_last=hours_since_last,
    )
