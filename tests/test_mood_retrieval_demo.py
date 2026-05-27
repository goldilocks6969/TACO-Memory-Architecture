"""Tests for the mood-congruent retrieval demo harness."""
from eval import mood_retrieval
from taco.memory import retrieval
from taco.state import LatentState


def test_no_overlap_probe_naive_loses_taco_wins_in_vulnerable_state():
    trace = mood_retrieval.state_weighted_trace(
        LatentState(E=80, K=60, V=90, R=100))

    assert retrieval.classify_query(trace.query) == "emotional"
    assert trace.mood_congruent_win is True
    assert "park" in trace.naive_top
    assert "father died" in trace.taco_top
    assert trace.weights["emo"] > trace.weights["sem"]
    assert len(trace.taco_sparse) == 2


def test_calm_state_keeps_surface_semantic_match_on_top():
    trace = mood_retrieval.state_weighted_trace(
        LatentState(E=10, K=60, V=0, R=100))

    assert "park" in trace.naive_top
    assert "park" in trace.taco_top
    assert trace.weights["sem"] > trace.weights["emo"]


def test_vulnerability_raises_w_emo_for_same_probe():
    calm = LatentState(E=80, K=60, V=0, R=100)
    vulnerable = LatentState(E=80, K=60, V=100, R=100)

    assert vulnerable.w_emo() > calm.w_emo()
    assert retrieval.state_tone(vulnerable) == "distress"
