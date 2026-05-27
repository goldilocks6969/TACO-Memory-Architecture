"""Tests for the isolated latent state estimator."""
from taco import config
from taco.memory.extract import LightExtract
from taco.state import LatentState
from taco import state_estimator


def test_estimator_returns_constrained_payload(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)

    estimate = state_estimator.estimate_state(
        "Honestly I'm scared and I don't know what to do",
        LatentState(),
        hours_since_last=0.5,
    )

    payload = estimate.as_payload()
    assert set(payload) == {"E", "K", "V", "R"}
    assert all(0.0 <= v <= 100.0 for v in payload.values())
    assert estimate.state.E > 0
    assert estimate.state.V > 0


def test_estimator_reuses_precomputed_light_extract(monkeypatch):
    def _boom(_message):
        raise AssertionError("light_extract should not be called")

    monkeypatch.setattr(state_estimator.extract, "light_extract", _boom)
    light = LightExtract(
        emotional=80,
        vulnerability=70,
        salience=8.5,
        tone="distress",
    )

    estimate = state_estimator.estimate_state(
        "precomputed", LatentState(), hours_since_last=1.0, light=light)

    assert estimate.light is light
    assert estimate.signal.emotional == 80
    assert estimate.signal.vulnerability == 70
    assert estimate.state.E == 40
    assert estimate.state.V == 35


def test_vulnerable_turn_opens_write_gate(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)

    calm = state_estimator.estimate_state(
        "what time is the meeting",
        LatentState(),
        hours_since_last=0.5,
    ).state
    vulnerable = state_estimator.estimate_state(
        "I have never told anyone this but I am scared I cannot handle it",
        LatentState(),
        hours_since_last=0.5,
    ).state

    assert vulnerable.theta() < calm.theta()
    assert vulnerable.K >= calm.K
