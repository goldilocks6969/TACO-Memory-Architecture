"""Tests for the abstraction-before-pruning demo."""
from eval import abstraction_pruning


def test_demo_abstracts_low_vitality_episode_before_pruning():
    trace = abstraction_pruning.run_demo(weeks_elapsed=7)

    assert trace.vitality < trace.threshold
    assert trace.action == "abstract_then_prune"
    assert trace.detail_retained is False
    assert "father" in trace.surviving_belief
    assert "expectations" in trace.surviving_belief


def test_demo_retains_episode_above_threshold():
    trace = abstraction_pruning.run_demo(weeks_elapsed=1)

    assert trace.vitality > trace.threshold
    assert trace.action == "retain_episode"
    assert trace.detail_retained is True
    assert trace.surviving_belief == ""
