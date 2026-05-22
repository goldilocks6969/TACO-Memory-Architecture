"""Memory-conditioned reasoning: S → how the engine reasons over retrieved memory.

This is the layer that turns "smart retrieval" into a *state-conditioned reasoning
architecture*. The other subsystems decide **what** is retrieved (R(m) weights),
**what** is stored (theta), **when** to act (proactive/interruption) and **how
deep** to plan. None of them change how the reasoning engine *treats* the memories
once they are in hand — by default an LLM reads them as ordinary extra context.

The reasoning stance closes that gap. It is the seventh decision the orchestrator
resolves from S: an explicit, state-dependent instruction set that frames the
retrieved memories as *cognitively privileged* — psychologically significant prior
states of this specific person — and tells the engine how to weigh them. The same
retrieved set is reasoned about differently depending on where S sits:

  • in crisis, continuity outranks semantic similarity and identity beliefs are
    connected to the present moment (longitudinal coherence);
  • when the user is open/fragile, emotional resonance leads;
  • in a deep ongoing relationship, the engine reasons *across* history and
    anticipates;
  • in a transactional moment, memories collapse back to plain factual reference
    and continuity is deliberately held light.

Thresholds mirror `state.state_tone` / `proactive` so the modes line up with the
rest of the architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..state import LatentState

# Mode boundaries in S-space (kept consistent with retrieval.state_tone). ------ #
CRISIS_E = 60.0       # high affective arousal → most privileged stance
DISTRESS_V = 40.0     # crisis + this much openness reads as distress
VULNERABLE_V = 50.0   # open / fragile even without full crisis arousal
CONCERN_E = 30.0      # real but moderate affect
ENGAGED_K = 50.0      # established, ongoing relationship (deep streak)
TRANSACTIONAL_E = 15.0
TRANSACTIONAL_V = 15.0
TRANSACTIONAL_K = 25.0


# Always-on framing: the epistemic status of the retrieved block. -------------- #
PRIVILEGE_FRAME = (
    "The memories below are not background context. They are psychologically "
    "privileged information — prior states of this specific person that you are "
    "accountable to. Reason *from* them, not merely *with* them."
)


@dataclass
class ReasoningStance:
    """How the reasoning engine should treat the privileged memories, given S."""

    mode: str                 # crisis | vulnerable | concerned | engaged | transactional | neutral
    memory_status: str        # one-line epistemic framing for this mode
    directives: List[str] = field(default_factory=list)
    # When True, the right memory is the one that preserves the user's story, not
    # the one that merely matches keywords — continuity outranks similarity.
    continuity_over_similarity: bool = False

    def render(self) -> str:
        """The directive block injected into the briefing (Figure 4 step 4)."""
        lines = [
            "How to reason about these memories (state-conditioned):",
            f"  {PRIVILEGE_FRAME}",
            f"  [stance: {self.mode}] {self.memory_status}",
        ]
        lines += [f"  - {d}" for d in self.directives]
        return "\n".join(lines)


def stance(state: LatentState) -> ReasoningStance:
    """Resolve the reasoning stance from the current latent state.

    Precedence runs from the most affectively loaded region of S-space to the
    least, so the most privileged stance always wins when regions overlap.
    """
    E, K, V = state.E, state.K, state.V

    # 1. Crisis — peak emotional *arousal* (E-driven, matching state_tone).
    #    Memories take precedence over generic helpfulness; continuity matters most.
    if E >= CRISIS_E:
        # distress = crisis + high openness; crisis = high arousal, more guarded.
        kind = "distress" if V >= DISTRESS_V else "crisis"
        return ReasoningStance(
            mode=kind,
            memory_status=(
                "These mark this person's most significant prior states; they "
                "take precedence over generic helpfulness."
            ),
            directives=[
                "Prioritize continuity over semantic similarity: the right memory "
                "is the one that keeps this person's story coherent, not the one "
                "that merely matches keywords.",
                "Connect their durable, identity-level beliefs to what they are "
                "feeling right now — name the throughline out loud.",
                "Maintain longitudinal coherence: do not contradict, relitigate, "
                "or appear to have forgotten anything they have disclosed.",
                "Lead with what you remember and the relationship; hold the "
                "emotional weight before offering any solution.",
            ],
            continuity_over_similarity=True,
        )

    # 2. Vulnerable — open and fragile, but not full crisis arousal.
    if V >= VULNERABLE_V:
        return ReasoningStance(
            mode="vulnerable",
            memory_status=(
                "This person is open and fragile right now; the memories are "
                "emotionally charged, not neutral."
            ),
            directives=[
                "Favor emotional resonance and continuity over breadth — reference "
                "specific shared history, gently and precisely.",
                "Connect prior disclosures to the present so they feel known, not "
                "analyzed.",
                "Acknowledge the openness it took to say this.",
            ],
            continuity_over_similarity=True,
        )

    # 3. Concerned — real but moderate affect.
    if E >= CONCERN_E:
        return ReasoningStance(
            mode="concerned",
            memory_status="There is genuine affect here; memories are meaningful.",
            directives=[
                "Weave relevant history into your reasoning; prefer continuity when "
                "memories conflict on relevance.",
                "Stay attentive to escalation — be ready to deepen if affect rises.",
            ],
            continuity_over_similarity=False,
        )

    # 4. Engaged — calm, but a deep established relationship.
    if K >= ENGAGED_K:
        return ReasoningStance(
            mode="engaged",
            memory_status="You share an established, ongoing history with this person.",
            directives=[
                "Reason across the relationship: build on open threads instead of "
                "restating them, and you may anticipate next steps.",
                "Assume shared context; brevity signals familiarity.",
            ],
            continuity_over_similarity=False,
        )

    # 5. Transactional — low stakes on every axis.
    if E < TRANSACTIONAL_E and V < TRANSACTIONAL_V and K < TRANSACTIONAL_K:
        return ReasoningStance(
            mode="transactional",
            memory_status="Low-stakes, task-oriented moment; memories are factual reference only.",
            directives=[
                "Use memories only if directly relevant to the task; do not force "
                "continuity or over-personalize.",
                "Be concise and get to the point.",
            ],
            continuity_over_similarity=False,
        )

    # 6. Neutral — ordinary conversational moment.
    return ReasoningStance(
        mode="neutral",
        memory_status="Ordinary conversational moment.",
        directives=[
            "Use memories where they add genuine relevance; keep continuity light.",
        ],
        continuity_over_similarity=False,
    )
