"""The orchestrator: reads S once and resolves all six subsystem parameters.

This is the "prefrontal cortex" of the architecture (§3). Every subsystem reads
S before it decides; the orchestrator gathers those decisions into a single
CognitivePlan that the turn pipeline executes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from . import proactive
from ..state import LatentState


@dataclass
class CognitivePlan:
    """All six state-dependent decisions for a single turn."""

    # 1. write-time salience gate
    theta: float
    # 2. retrieval weighting
    weights: Dict[str, float]
    # 3. proactive initiation
    proactive: proactive.ProactiveDecision
    # 4. planning depth
    planning_depth: int
    # 5. interruption timing
    interruption: str
    # 6. reinforcement weighting
    reinforcement_rate: float

    def summary(self) -> str:
        w = " ".join(f"{k}={v:.2f}" for k, v in self.weights.items())
        return (
            f"θ={self.theta:.2f} | plan_depth={self.planning_depth} | "
            f"interrupt={self.interruption} | reinforce×{self.reinforcement_rate:.2f}\n"
            f"  weights: {w}\n"
            f"  proactive: {self.proactive.kind} "
            f"({'yes' if self.proactive.initiate else 'no'}) — {self.proactive.reason}"
        )


def plan(state: LatentState,
         has_open_high_salience_thread: bool = False) -> CognitivePlan:
    """Resolve the full cognitive plan from the current state."""
    return CognitivePlan(
        theta=state.theta(),
        weights=state.retrieval_weights(),
        proactive=proactive.should_initiate(state, has_open_high_salience_thread),
        planning_depth=state.planning_depth(),
        interruption=state.interruption_policy(),
        reinforcement_rate=state.reinforcement_rate(),
    )
