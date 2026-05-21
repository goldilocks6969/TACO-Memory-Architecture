"""Proactive initiation (§4.6): the agent acts unprompted when S predicts need.

Two trigger rules from the paper:
  • elevated crisis state + low recency → unprompted check-in
  • high-salience open thread + declining engagement → re-engagement message
"""
from __future__ import annotations

from dataclasses import dataclass

from ..state import LatentState

CRISIS_E = 55.0       # emotional intensity that reads as crisis
LOW_RECENCY = 35.0    # R below this means the user has been away
DECLINING_K = 30.0    # engagement streak this low reads as "declining"


@dataclass
class ProactiveDecision:
    initiate: bool
    kind: str       # 'check_in' | 're_engage' | 'none'
    reason: str


def should_initiate(state: LatentState,
                    has_open_high_salience_thread: bool = False) -> ProactiveDecision:
    """Decide whether to reach out between user-initiated turns."""
    # Rule 1 — crisis + low recency → check in.
    if max(state.E, state.V) >= CRISIS_E and state.R <= LOW_RECENCY:
        return ProactiveDecision(
            True, "check_in",
            f"crisis state (E={state.E:.0f}, V={state.V:.0f}) with low recency "
            f"(R={state.R:.0f})",
        )
    # Rule 2 — open high-salience thread + declining engagement → re-engage.
    if has_open_high_salience_thread and state.K <= DECLINING_K:
        return ProactiveDecision(
            True, "re_engage",
            f"high-salience open thread with declining engagement (K={state.K:.0f})",
        )
    return ProactiveDecision(False, "none", "state does not predict need")
