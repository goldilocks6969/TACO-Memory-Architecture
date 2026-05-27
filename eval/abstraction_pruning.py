"""Abstraction-before-pruning demo.

This isolates step 4 of the TACO thesis: low-vitality episodes are not simply
deleted. The episodic detail fades, but an L3 semantic belief survives.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from taco import config
from taco.memory import decay


EPISODE = (
    "My father criticized my career choice again at dinner, and I felt like "
    "nothing I do is ever enough for him."
)
BELIEF = (
    "This person carries unresolved tension with their father around "
    "expectations, autonomy, and feeling good enough."
)


@dataclass(frozen=True)
class AbstractionTrace:
    episode_detail: str
    tier: str
    weekly_decay: float
    weeks_elapsed: float
    vitality: float
    threshold: float
    action: str
    surviving_belief: str
    detail_retained: bool


def run_demo(weeks_elapsed: float = 7.0) -> AbstractionTrace:
    tier = config.tier_for_score(3.0)
    vitality = decay.vitality_after_weeks(tier.weekly_decay, weeks_elapsed)
    should_abstract = vitality < config.ABSTRACTION_THRESHOLD
    return AbstractionTrace(
        episode_detail=EPISODE,
        tier=f"{tier.low}-{tier.high}",
        weekly_decay=tier.weekly_decay,
        weeks_elapsed=weeks_elapsed,
        vitality=round(vitality, 4),
        threshold=config.ABSTRACTION_THRESHOLD,
        action="abstract_then_prune" if should_abstract else "retain_episode",
        surviving_belief=BELIEF if should_abstract else "",
        detail_retained=not should_abstract,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TACO abstraction-before-pruning demo.")
    parser.add_argument("--weeks", type=float, default=7.0)
    parser.add_argument(
        "--summary", action="store_true",
        help="print the compact abstraction result",
    )
    args = parser.parse_args()

    trace = run_demo(args.weeks)
    payload = asdict(trace)
    if args.summary:
        payload = {
            "vitality": trace.vitality,
            "threshold": trace.threshold,
            "action": trace.action,
            "detail_retained": trace.detail_retained,
            "surviving_belief": trace.surviving_belief,
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
