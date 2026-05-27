"""Write-time state gate demo.

This small harness isolates step 2 of the TACO thesis: the same borderline turn
can be admitted or dropped depending on the user's latent state at encoding
time. It runs without a database and uses the local/mock light extractor by
default, so it is safe for fast demos and tests.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable, List

from taco import config, state_estimator
from taco.memory import salience
from taco.state import LatentState


@dataclass(frozen=True)
class GateTrace:
    scenario: str
    turn_index: int
    message: str
    E: float
    K: float
    V: float
    R: float
    salience: float
    theta: float
    stored: bool
    tone: str


@dataclass(frozen=True)
class GateScenario:
    name: str
    priming_turns: List[str]
    target_turn: str


TARGET_TURN = "could you help me pick dinner tonight!"

SCENARIOS: List[GateScenario] = [
    GateScenario(
        name="transactional",
        priming_turns=[
            "what time does the pharmacy close",
            "thanks, also remind me to buy batteries",
        ],
        target_turn=TARGET_TURN,
    ),
    GateScenario(
        name="vulnerable",
        priming_turns=[
            "I have never told anyone this but I am scared I cannot handle it",
            "I feel shaky and I need a little help staying grounded",
            "I'm not okay and I can't keep pretending this is fine",
            "honestly I need you to remember this is a hard week",
        ],
        target_turn=TARGET_TURN,
    ),
]


def run_scenario(scenario: GateScenario,
                 *,
                 initial_state: LatentState | None = None,
                 gap_hours: float = 0.5) -> List[GateTrace]:
    """Replay a scenario and record the write gate decision at each turn."""
    state = initial_state or LatentState()
    traces: List[GateTrace] = []
    for idx, message in enumerate([*scenario.priming_turns, scenario.target_turn], 1):
        estimate = state_estimator.estimate_state(
            message, state, gap_hours)
        state = estimate.state
        traces.append(GateTrace(
            scenario=scenario.name,
            turn_index=idx,
            message=message,
            E=round(state.E, 2),
            K=round(state.K, 2),
            V=round(state.V, 2),
            R=round(state.R, 2),
            salience=round(estimate.light.salience, 2),
            theta=state.theta(),
            stored=salience.gate(estimate.light.salience, state),
            tone=estimate.light.tone,
        ))
    return traces


def run_gate_demo(scenarios: Iterable[GateScenario] = SCENARIOS) -> List[GateTrace]:
    """Run all gate scenarios under deterministic local extraction."""
    old_mock = config.MOCK
    old_light_local = config.LIGHT_EXTRACT_LOCAL
    config.MOCK = True
    config.LIGHT_EXTRACT_LOCAL = False
    try:
        traces: List[GateTrace] = []
        for scenario in scenarios:
            traces.extend(run_scenario(scenario))
        return traces
    finally:
        config.MOCK = old_mock
        config.LIGHT_EXTRACT_LOCAL = old_light_local


def summarize_target_decisions(traces: List[GateTrace]) -> dict:
    """Return the target-turn comparison for investor/research demos."""
    target = {
        t.scenario: t for t in traces
        if t.message == TARGET_TURN
    }
    return {
        "target_turn": TARGET_TURN,
        "transactional": asdict(target["transactional"]),
        "vulnerable": asdict(target["vulnerable"]),
        "theta_delta": round(
            target["transactional"].theta - target["vulnerable"].theta, 2),
        "same_content_different_decision": (
            target["transactional"].stored is False
            and target["vulnerable"].stored is True
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TACO write-time state gate demo.")
    parser.add_argument(
        "--summary", action="store_true",
        help="print only the matched target-turn comparison",
    )
    args = parser.parse_args()

    traces = run_gate_demo()
    payload = summarize_target_decisions(traces) if args.summary else [
        asdict(t) for t in traces
    ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
