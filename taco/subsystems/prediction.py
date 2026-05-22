"""L9 — predictive continuity: anticipate state, salience, and need.

The other subsystems are reactive — they read the *current* S. This one reads the
*trajectory* of S and the recent emotional record to anticipate where the user is
heading before they say it: whether affect is escalating or recovering, which
themes are gaining weight (and so likely to become persistent), and what to
prefetch so the right memory is already in hand when the need surfaces.

State projection is deterministic trend extrapolation (no LLM) so it is cheap and
testable; theme detection is a light salience/recency heuristic over recent
significant episodes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config
from ..state import LatentState, clamp
from ..memory.episode import Episode


@dataclass
class PredictiveContinuity:
    projected: LatentState        # the anticipated next state
    trend: str                    # 'escalating' | 'recovering' | 'stable'
    themes: List[str] = field(default_factory=list)   # likely-to-persist themes
    prefetch_query: Optional[str] = None              # anticipatory retrieval cue
    rationale: str = ""

    def render(self) -> str:
        lines = [f"Anticipated trajectory: {self.trend} "
                 f"(projected E={self.projected.E:.0f}, V={self.projected.V:.0f})."]
        if self.themes:
            lines.append("Likely to matter / become persistent: "
                         + "; ".join(self.themes) + ".")
        return "\n".join(f"  {ln}" for ln in lines)


def project_state(history: List[LatentState],
                  damping: float = config.PREDICT_DAMPING) -> LatentState:
    """Extrapolate the next state from the recent trajectory of S. Pure.

    Uses the mean step across the observed window (an EMA-like trend), damped so
    a single spike doesn't overshoot, and clamped to the valid range.
    """
    if not history:
        return LatentState()
    last = history[-1]
    if len(history) < 2:
        return LatentState(E=last.E, K=last.K, V=last.V, R=last.R)

    window = history[-config.PREDICT_HISTORY:]
    n = len(window) - 1

    def mean_step(attr: str) -> float:
        return sum(getattr(window[i + 1], attr) - getattr(window[i], attr)
                   for i in range(n)) / n

    return LatentState(
        E=clamp(last.E + damping * mean_step("E")),
        K=clamp(last.K + damping * mean_step("K")),
        V=clamp(last.V + damping * mean_step("V")),
        R=last.R,
    )


def _classify(current: LatentState, projected: LatentState) -> str:
    d = (projected.E - current.E) + (projected.V - current.V)
    if d >= 12.0:
        return "escalating"
    if d <= -12.0:
        return "recovering"
    return "stable"


def emergent_themes(recent: List[Episode], k: int = 3) -> List[str]:
    """Themes gaining weight: tones recurring across recent significant episodes,
    each anchored to its most salient exemplar. Heuristic, deterministic."""
    if not recent:
        return []
    by_tone: Counter = Counter()
    exemplar = {}
    for ep in recent:
        tone = ep.tone or "neutral"
        if tone in ("neutral", "transactional"):
            continue
        by_tone[tone] += 1
        if tone not in exemplar or ep.salience > exemplar[tone].salience:
            exemplar[tone] = ep
    themes = []
    for tone, count in by_tone.most_common(k):
        if count < 2:  # needs to recur to count as emergent
            continue
        snippet = exemplar[tone].content.strip().rstrip(".")[:60]
        themes.append(f"{tone} thread (\"{snippet}\")")
    return themes


def predict(history: List[LatentState],
            recent: List[Episode]) -> PredictiveContinuity:
    """Resolve the predictive-continuity view from state history + recent episodes."""
    current = history[-1] if history else LatentState()
    projected = project_state(history)
    trend = _classify(current, projected)
    themes = emergent_themes(recent)

    # Prefetch only when the trajectory suggests rising need — anticipatory
    # retrieval is itself state-gated, keeping the common path cheap.
    prefetch_query = None
    if trend == "escalating" and recent:
        hottest = max(recent, key=lambda e: (e.s_e or 0.0, e.salience))
        prefetch_query = hottest.content

    rationale = (f"trend={trend}; "
                 f"ΔE≈{projected.E - current.E:+.0f}, ΔV≈{projected.V - current.V:+.0f}; "
                 f"{len(themes)} emergent theme(s)")
    return PredictiveContinuity(projected=projected, trend=trend, themes=themes,
                                prefetch_query=prefetch_query, rationale=rationale)
