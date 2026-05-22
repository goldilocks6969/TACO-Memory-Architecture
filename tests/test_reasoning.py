"""Tests for the memory-conditioned reasoning stance (S → reasoning directives)."""
from taco.memory.retrieval import assemble_briefing
from taco.state import LatentState
from taco.subsystems import orchestrator, reasoning


def test_crisis_states_privilege_continuity():
    # High emotional arousal → continuity must outrank semantic similarity.
    distress = reasoning.stance(LatentState(E=85, K=40, V=70, R=20))
    crisis = reasoning.stance(LatentState(E=80, K=40, V=10, R=20))
    assert distress.mode == "distress"
    assert crisis.mode == "crisis"
    for s in (distress, crisis):
        assert s.continuity_over_similarity is True
        blob = " ".join(s.directives).lower()
        assert "continuity over semantic similarity" in blob
        assert "longitudinal coherence" in blob
        assert "identity" in blob  # connect identity beliefs to current state


def test_vulnerable_state_leads_with_resonance():
    s = reasoning.stance(LatentState(E=20, K=30, V=65, R=50))
    assert s.mode == "vulnerable"
    assert s.continuity_over_similarity is True
    assert "resonance" in " ".join(s.directives).lower()


def test_transactional_state_holds_continuity_light():
    s = reasoning.stance(LatentState(E=5, K=10, V=5, R=100))
    assert s.mode == "transactional"
    assert s.continuity_over_similarity is False
    blob = " ".join(s.directives).lower()
    assert "do not force" in blob and "concise" in blob


def test_engaged_state_reasons_across_history():
    s = reasoning.stance(LatentState(E=10, K=80, V=10, R=90))
    assert s.mode == "engaged"
    assert "across the relationship" in " ".join(s.directives).lower()


def test_same_input_different_stance_by_state():
    # The whole point: identical retrieval, different reasoning depending on S.
    calm = reasoning.stance(LatentState(E=2, K=5, V=2, R=100))
    grieving = reasoning.stance(LatentState(E=90, K=60, V=80, R=15))
    assert calm.mode != grieving.mode
    assert calm.continuity_over_similarity is False
    assert grieving.continuity_over_similarity is True


def test_orchestrator_resolves_stance():
    plan = orchestrator.plan(LatentState(E=88, K=50, V=75, R=20))
    assert isinstance(plan.stance, reasoning.ReasoningStance)
    assert plan.stance.mode == "distress"
    assert "distress" in plan.summary()
    assert "continuity > similarity" in plan.summary()


def test_briefing_embeds_privileged_framing_and_directives():
    state = LatentState(E=85, K=50, V=70, R=20)
    s = reasoning.stance(state)
    briefing = assemble_briefing([], [], [], state, s)
    assert "psychologically privileged information" in briefing.lower()
    assert "[stance: distress]" in briefing
    # the directive block actually carries the state-conditioned rule
    assert "continuity over semantic similarity" in briefing.lower()


def test_briefing_without_stance_is_backward_compatible():
    # stance is optional so existing callers / tooling don't break.
    briefing = assemble_briefing([], [], [], LatentState(), None)
    assert "stance:" not in briefing
    assert "reasoning engine" in briefing.lower()
