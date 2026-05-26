"""Hierarchical retrieval scoring R(m) and context assembly (§4.2, Figure 4).

    R(m) = w_sem·sem(m) + w_sal·sal(m) + w_emo·emo(m)
           + w_rec·rec(m) + w_decay·decay(m)

The weights are state-modulated (see LatentState.retrieval_weights). The
pipeline casts 20 candidates via pgvector kNN, this module re-ranks them by
R(m), selects the top 4, and assembles them into a single narrative briefing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from .episode import Episode
from .. import config
from ..state import LatentState

if TYPE_CHECKING:  # avoids a runtime memory→subsystems import; duck-typed below
    from ..subsystems.reasoning import ReasoningStance
    from ..subsystems.prediction import PredictiveContinuity


# --------------------------------------------------------------------------- #
# Emotional-tone compatibility (the emo(m) component)
# --------------------------------------------------------------------------- #
_TONE_FAMILY = {
    "distress": "negative", "intense": "negative", "concerned": "negative",
    "tender": "tender", "sad": "negative", "anxious": "negative",
    "transactional": "neutral", "neutral": "neutral",
}


def state_tone(state: LatentState) -> str:
    """Coarse label for the user's *current* emotional tone, from S."""
    if state.E >= 60:
        return "distress" if state.V >= 40 else "intense"
    if state.E >= 30:
        return "tender" if state.V >= 30 else "concerned"
    return "neutral"


def tone_compat(current: str, mem_tone: Optional[str]) -> float:
    """emo(m) ∈ [0,1]: tone compatibility + a +0.3 boost on an exact match (§4.2)."""
    if not mem_tone:
        return 0.3
    base = 0.3
    if _TONE_FAMILY.get(current) == _TONE_FAMILY.get(mem_tone):
        base = 0.6
    if current == mem_tone:
        base += config.EMO_TONE_MATCH_BOOST
    return min(1.0, base)


# --------------------------------------------------------------------------- #
# R(m)
# --------------------------------------------------------------------------- #
def components(ep: Episode, state: LatentState, current_tone: str) -> Dict[str, float]:
    sem = max(0.0, min(1.0, ep.similarity))
    sal = max(0.0, min(1.0, ep.salience / 10.0))
    emo = tone_compat(current_tone, ep.tone)
    rec = max(0.0, 1.0 - ep.age_days() / config.RECENCY_WINDOW_DAYS)
    decay = max(0.0, min(1.0, ep.vitality))
    return {"sem": sem, "sal": sal, "emo": emo, "rec": rec, "decay": decay}


def score(ep: Episode, state: LatentState, weights: Dict[str, float],
          current_tone: str) -> float:
    c = components(ep, state, current_tone)
    ep.score = sum(weights[k] * c[k] for k in c)
    return ep.score


def rerank(candidates: List[Episode], state: LatentState,
           top_k: int = config.TOP_K) -> List[Episode]:
    """Re-rank candidates by state-modulated R(m) and return the top-k."""
    weights = state.retrieval_weights()
    tone = state_tone(state)
    for ep in candidates:
        score(ep, state, weights, tone)
    return sorted(candidates, key=lambda e: e.score, reverse=True)[:top_k]


# --------------------------------------------------------------------------- #
# Context assembly (Figure 4 step 4)
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are the reasoning engine inside a state-dependent cognitive memory "
    "layer. A separate system has inferred the user's current state and "
    "selected the most relevant memories. Respond naturally; let the state, "
    "the memories, and the reasoning directives below shape both what you say "
    "and what you choose to do."
)


def briefing_sections(
    memories: List[Episode], beliefs: List[str], working: List[Episode],
    state: LatentState, stance: Optional["ReasoningStance"] = None,
    identity: Optional[List[str]] = None,
    prediction: Optional["PredictiveContinuity"] = None,
) -> Dict[str, str]:
    """Return the briefing broken down into its logical sections.

    Section keys:
      ``system``     — fixed scaffolding the engine always sees.
      ``state``      — the current S = (E, K, V, R) line and reasoning stance.
      ``identity``   — L10 persistent self-model.
      ``working``    — L1 working-memory transcript.
      ``prediction`` — L9 anticipatory continuity block.
      ``retrieval``  — the retrieved memory / fact payload (the policy-dependent
                       component — what the eval harness charges as
                       retrieval_tokens).

    The eval harness uses these to attribute tokens: ``retrieval`` is the
    retrieval payload, ``system`` is the fixed system prompt, and everything
    else (``state``, ``identity``, ``working``, ``prediction``) is the
    structured "state briefing" overhead TACO adds on top of vanilla RAG.
    """
    state_lines: List[str] = [
        f"User state — emotional intensity {state.E:.0f}/100, engagement "
        f"{state.K:.0f}/100, vulnerability {state.V:.0f}/100, recency "
        f"{state.R:.0f}/100 (current tone: {state_tone(state)})."
    ]
    if stance is not None:
        state_lines.append("")
        state_lines.append(stance.render())

    identity_lines: List[str] = []
    if identity:
        identity_lines.append(
            "Persistent self-model (who this person is, across time):"
        )
        for fact in identity:
            identity_lines.append(f"  • {fact}")

    retrieval_lines: List[str] = []
    if beliefs:
        retrieval_lines.append("Durable, identity-level beliefs about this person:")
        for b in beliefs:
            retrieval_lines.append(f"  • {b}")
    if memories:
        if retrieval_lines:
            retrieval_lines.append("")
        retrieval_lines.append(
            "Psychologically privileged memories (re-ranked by R(m), most "
            "significant first):"
        )
        for m in memories:
            retrieval_lines.append(
                f"  • [{m.tone or 'neutral'}, salience {m.salience:.0f}/10, "
                f"R={m.score:.2f}] {m.content}"
            )

    working_lines: List[str] = []
    if working:
        working_lines.append("Recent conversation (working memory):")
        for w in working:
            working_lines.append(f"  {w.role}: {w.content}")

    prediction_lines: List[str] = []
    if prediction is not None:
        prediction_lines.append("Predictive continuity (anticipatory, not yet stated):")
        prediction_lines.append(prediction.render())

    return {
        "system": _SYSTEM_PROMPT,
        "state": "\n".join(state_lines),
        "identity": "\n".join(identity_lines),
        "retrieval": "\n".join(retrieval_lines),
        "working": "\n".join(working_lines),
        "prediction": "\n".join(prediction_lines),
    }


def _join_sections(sections: Dict[str, str]) -> str:
    """Re-join section blocks with blank-line separators, preserving order."""
    order = ("system", "state", "identity", "retrieval", "working", "prediction")
    blocks = [sections[k] for k in order if sections.get(k)]
    return "\n\n".join(blocks)


def assemble_briefing(memories: List[Episode], beliefs: List[str],
                      working: List[Episode], state: LatentState,
                      stance: Optional["ReasoningStance"] = None,
                      identity: Optional[List[str]] = None,
                      prediction: Optional["PredictiveContinuity"] = None) -> str:
    """Assemble the structured narrative briefing for the single LLM call.

    This is the prompt fragment the reasoning engine sees — not raw top-k
    passages, but a state-aware narrative of who it is talking to, what is most
    relevant right now, and (via the reasoning `stance`) *how to reason about it*.
    The stance is what makes this memory-conditioned rather than memory-augmented:
    it frames the memories as cognitively privileged and shifts the reasoning
    rules with the user's state.
    """
    return _join_sections(briefing_sections(
        memories, beliefs, working, state, stance, identity, prediction))
