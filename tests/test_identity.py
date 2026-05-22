"""Tests for the L10 identity abstraction layer (confidence reconciliation)."""
from taco import llm
from taco.memory.identity import reconcile


def test_new_attribute_is_adopted():
    value, conf, changed = reconcile(None, "Maya", 0.6)
    assert value == "Maya" and conf == 0.6 and changed is False


def test_consistent_evidence_reinforces_confidence():
    value, conf, changed = reconcile(("Maya", 0.6), "Maya", 0.6)
    assert value == "Maya"
    assert conf > 0.6          # repeated, consistent evidence raises certainty
    assert conf <= 1.0
    assert changed is False


def test_confidence_is_monotonic_and_bounded_under_repetition():
    conf = 0.5
    prev = conf
    for _ in range(20):
        _, conf, _ = reconcile(("x", conf), "x", 0.5)
        assert conf >= prev      # never decreases when consistent
        assert conf <= 1.0       # asymptotes to, never exceeds, 1.0
        prev = conf


def test_contradiction_triggers_belief_revision():
    value, conf, changed = reconcile(("Maya", 0.9), "Jordan", 0.6)
    assert value == "Jordan"     # revise, do not average a person into a contradiction
    assert conf == 0.6
    assert changed is True


def test_consistent_keeps_more_specific_phrasing():
    value, _, changed = reconcile(("software", 0.5), "software engineer", 0.6)
    assert value == "software engineer"
    assert changed is False


def test_heuristic_identity_extractor_pulls_named_relationships():
    facts = llm.extract_identity("my partner Maya and I just bought a house")
    attrs = {f["attribute"]: f["value"] for f in facts}
    assert attrs.get("relationship:partner") == "Maya"
