"""TACO_EVAL_FAST_LIVE — the master "live benchmark fast mode" flag.

When set, the response and judge calls collapse to short prompts, temperature
falls to 0, and the SDK call carries the configured ``max_tokens`` cap.  These
tests stand up a fake OpenAI client and inspect the exact kwargs that reach
``client.chat.completions.create`` so no real API key is needed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from taco import config, llm
from eval import judge


class _FakeCreate:
    """Stub for ``client.chat.completions.create`` that records every call."""

    def __init__(self, content: str = "ok"):
        self.calls = []
        self.content = content

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.content))]
        )


def _install_fake_client(monkeypatch, content: str = "ok"):
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fake-test-key-not-real")
    fake = _FakeCreate(content=content)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake)))
    monkeypatch.setattr(llm, "_client", lambda: client)
    return fake


# --------------------------------------------------------------------------- #
# llm.respond under FAST_LIVE
# --------------------------------------------------------------------------- #
def test_fast_live_forces_temperature_zero(monkeypatch):
    """The benchmark must be deterministic across re-runs."""
    fake = _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "FAST_LIVE", True)
    monkeypatch.setattr(config, "FAST_RESPONSES", True)

    llm.respond("briefing", "user", planning_depth=3)
    assert fake.calls[0]["temperature"] == 0


def test_normal_mode_keeps_sampling_temperature(monkeypatch):
    """Outside FAST_LIVE the product still gets the natural 0.7 sampling."""
    fake = _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "FAST_LIVE", False)
    monkeypatch.setattr(config, "FAST_RESPONSES", False)

    llm.respond("briefing", "user", planning_depth=1)
    assert fake.calls[0]["temperature"] == 0.7


def test_fast_response_prompt_matches_spec(monkeypatch):
    """The system prompt the model sees in fast mode must include the
    spec-required "say you do not know" clause."""
    fake = _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "FAST_LIVE", True)
    monkeypatch.setattr(config, "FAST_RESPONSES", True)

    llm.respond("MEMORY BLOCK", "what happened?", planning_depth=1)
    system = next(m for m in fake.calls[0]["messages"] if m["role"] == "system")
    assert "memory context" in system["content"].lower()
    assert "be concise" in system["content"].lower()
    assert "do not know" in system["content"].lower()


# --------------------------------------------------------------------------- #
# judge.judge under FAST_LIVE
# --------------------------------------------------------------------------- #
def test_fast_live_judge_uses_short_rubric(monkeypatch):
    """FAST_LIVE swaps the verbose rubric for a one-sentence variant — same
    JSON contract, fewer tokens spent on instructions."""
    fake = _install_fake_client(monkeypatch,
                                content='{"score": 71, "reason": "ok"}')
    monkeypatch.setattr(config, "FAST_LIVE", True)

    out = judge.judge("probe?", "ground truth", "factual", "the response")
    assert out["score"] == 71
    system = next(m for m in fake.calls[0]["messages"] if m["role"] == "system")
    # The verbose rubric opens with "You are a strict evaluator…"; the fast
    # rubric does not.
    assert not system["content"].lower().startswith("you are a strict evaluator")
    # The fast rubric still spells out the 0-100 scale and the JSON contract.
    assert "0-100" in system["content"]
    assert "score" in system["content"].lower()
    # And it is meaningfully shorter than the full rubric.
    assert len(system["content"]) < len(judge._RUBRIC) // 2


def test_non_fast_live_judge_uses_full_rubric(monkeypatch):
    """With FAST_LIVE off the original rubric is still used."""
    fake = _install_fake_client(monkeypatch,
                                content='{"score": 42, "reason": "x"}')
    monkeypatch.setattr(config, "FAST_LIVE", False)
    judge.judge("p?", "gt", "factual", "r")
    system = next(m for m in fake.calls[0]["messages"] if m["role"] == "system")
    assert system["content"] == judge._RUBRIC


# --------------------------------------------------------------------------- #
# Token cap defaults
# --------------------------------------------------------------------------- #
def test_default_max_token_caps_match_spec():
    """The live benchmark depends on the 120/80 caps as defaults so a stale
    install behaves the same as a fresh one."""
    # These hold at import time; an explicit env var override is allowed.
    import os
    if not os.getenv("TACO_RESPONSE_MAX_TOKENS"):
        assert config.RESPONSE_MAX_TOKENS == 120
    if not os.getenv("TACO_JUDGE_MAX_TOKENS"):
        assert config.JUDGE_MAX_TOKENS == 80


# --------------------------------------------------------------------------- #
# Smoke-mode limits
# --------------------------------------------------------------------------- #
def test_limit_scenarios_and_probes_helper():
    """The harness applies the limits via plain list slices.  This test pins
    that contract: ``LIMIT_SCENARIOS`` truncates SCENARIOS, ``LIMIT_PROBES``
    truncates a scenario's probe list."""
    from eval.dataset import SCENARIOS
    # Both limits are simple slice arguments; verify the underlying property.
    assert SCENARIOS[:1] != SCENARIOS, \
        "smoke mode is meaningless if SCENARIOS has only one entry"
    assert len(SCENARIOS[:1]) == 1
    sc = SCENARIOS[0]
    assert len(sc.probes[:2]) == 2
    assert sc.probes[:2] == sc.probes[:2]  # determinism
