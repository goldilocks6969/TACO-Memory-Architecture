"""Tests for the deterministic grief demo trace."""
from eval import grief_demo_trace


def test_grief_demo_trace_proves_core_contrast():
    trace = grief_demo_trace.build_trace()

    assert trace["proof"]["naive_top_is_surface_match"] is True
    assert trace["proof"]["taco_top_is_grief_memory"] is True
    assert "park" in trace["naive"]["top"]
    assert "father died" in trace["taco"]["top"]
    assert trace["query_kind"] == "emotional"


def test_grief_target_has_low_similarity_but_high_taco_score():
    trace = grief_demo_trace.build_trace()
    target = next(c for c in trace["candidates"] if c["kind"] == "target")
    surface = next(c for c in trace["candidates"] if c["kind"] == "surface-match")

    assert target["similarity"] < 0.1
    assert surface["similarity"] > 0.9
    assert target["score"] > surface["score"]
