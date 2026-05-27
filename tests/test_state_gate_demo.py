"""Tests for the write-time state gate demo harness."""
from eval import state_gate


def test_gate_demo_same_content_different_state_different_decision():
    traces = state_gate.run_gate_demo()
    summary = state_gate.summarize_target_decisions(traces)

    assert summary["same_content_different_decision"] is True
    assert summary["transactional"]["message"] == summary["vulnerable"]["message"]
    assert summary["transactional"]["stored"] is False
    assert summary["vulnerable"]["stored"] is True
    assert summary["vulnerable"]["theta"] < summary["transactional"]["theta"]


def test_gate_demo_keeps_trace_bounded_and_explainable():
    traces = state_gate.run_gate_demo()

    assert traces
    for trace in traces:
        assert 0.0 <= trace.E <= 100.0
        assert 0.0 <= trace.K <= 100.0
        assert 0.0 <= trace.V <= 100.0
        assert 0.0 <= trace.R <= 100.0
        assert 1.0 <= trace.salience <= 10.0
        assert trace.theta >= 2.1
        assert trace.theta <= 4.3
