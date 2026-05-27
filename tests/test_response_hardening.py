"""Response + judge hardening — separate models, max-token caps, SDK timeout,
fast-mode prompt.  These tests stand up a fake OpenAI client and capture the
kwargs of the chat-completion call, so they don't need a real API key.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from taco import config, llm
from eval import judge


# --------------------------------------------------------------------------- #
# Fake OpenAI client
# --------------------------------------------------------------------------- #
class _FakeCreate:
    """Stub for ``client.chat.completions.create``.  Records the kwargs each
    invocation receives and returns a constant response object."""

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
    """Force the live LLM path and intercept the SDK call.  Returns the
    ``_FakeCreate`` so tests can inspect ``.calls``."""
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fake-test-key-not-real")
    fake = _FakeCreate(content=content)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake)))
    monkeypatch.setattr(llm, "_client", lambda: client)
    return fake


# --------------------------------------------------------------------------- #
# llm.respond
# --------------------------------------------------------------------------- #
def test_respond_passes_max_tokens_and_sdk_timeout(monkeypatch):
    """Every SDK call must carry ``max_tokens`` and ``timeout`` so the server
    cannot run a single completion past the configured cap."""
    fake = _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "RESPONSE_MAX_TOKENS", 180)
    monkeypatch.setattr(config, "LLM_REQUEST_TIMEOUT_S", 30.0)

    llm.respond("brief", "user message", planning_depth=1)
    assert fake.calls, "respond did not invoke the SDK"
    kwargs = fake.calls[0]
    assert kwargs["max_tokens"] == 180
    assert kwargs["timeout"] == 30.0


def test_respond_passes_outer_thread_timeout(monkeypatch):
    """The SDK timeout is not enough: the retry wrapper's daemon-thread timeout
    must use the same budget, otherwise wedged sockets linger until the global
    default."""
    _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "LLM_REQUEST_TIMEOUT_S", 17.0)
    seen = {}

    def _spy(fn, **kwargs):
        seen.update(kwargs)
        return fn()

    monkeypatch.setattr(llm, "with_retries", _spy)
    llm.respond("brief", "user message", planning_depth=1)
    assert seen["timeout_per_attempt"] == 17.0


def test_respond_uses_response_model(monkeypatch):
    """``respond`` must hit ``config.RESPONSE_MODEL`` — the dedicated answer
    model — not the extraction model."""
    fake = _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "RESPONSE_MODEL", "model-for-answers")

    llm.respond("brief", "user message", planning_depth=1)
    assert fake.calls[0]["model"] == "model-for-answers"


def test_respond_fast_mode_uses_short_instruction(monkeypatch):
    """In FAST_RESPONSES mode the system message must collapse to the short
    "answer concisely from memory" directive; the briefing moves into the
    user role as raw memory context."""
    fake = _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "FAST_RESPONSES", True)

    llm.respond("STATE+MEMORY BRIEFING TEXT", "what happened?", planning_depth=3)

    messages = fake.calls[0]["messages"]
    system = next(m for m in messages if m["role"] == "system")
    user = next(m for m in messages if m["role"] == "user")
    # Short, fixed system prompt — no "reasoning engine inside a cognitive
    # memory layer" scaffolding.
    assert "concise" in system["content"].lower()
    assert "memory context" in system["content"].lower()
    assert "reasoning engine" not in system["content"].lower()
    assert len(system["content"]) < 200
    # The briefing reached the model in the user message instead.
    assert "STATE+MEMORY BRIEFING TEXT" in user["content"]
    assert "what happened?" in user["content"]


def test_respond_normal_mode_keeps_full_briefing(monkeypatch):
    """With FAST_RESPONSES off, the briefing remains the system content
    exactly as the cognitive layer assembled it — product behaviour intact."""
    fake = _install_fake_client(monkeypatch)
    monkeypatch.setattr(config, "FAST_RESPONSES", False)

    llm.respond("You are the reasoning engine inside ... [full briefing]",
                "user q", planning_depth=1)
    messages = fake.calls[0]["messages"]
    system = next(m for m in messages if m["role"] == "system")
    assert system["content"].startswith("You are the reasoning engine")
    user = next(m for m in messages if m["role"] == "user")
    assert user["content"] == "user q"


# --------------------------------------------------------------------------- #
# eval.judge.judge
# --------------------------------------------------------------------------- #
def test_judge_uses_judge_model_and_max_tokens(monkeypatch):
    """The judge must use ``JUDGE_MODEL`` and ``JUDGE_MAX_TOKENS`` and pass
    the SDK timeout — same hardening as ``respond``."""
    fake = _install_fake_client(monkeypatch, content='{"score": 73, "reason": "ok"}')
    monkeypatch.setattr(config, "JUDGE_MODEL", "model-for-judging")
    monkeypatch.setattr(config, "JUDGE_MAX_TOKENS", 120)
    monkeypatch.setattr(config, "LLM_REQUEST_TIMEOUT_S", 25.0)

    out = judge.judge("probe text", "ground truth", "factual", "the response")
    assert out["score"] == 73
    kwargs = fake.calls[0]
    assert kwargs["model"] == "model-for-judging"
    assert kwargs["max_tokens"] == 120
    assert kwargs["timeout"] == 25.0


def test_judge_passes_outer_thread_timeout(monkeypatch):
    """Judge calls also need the outer timeout budget so benchmark timeout rows
    are produced promptly."""
    _install_fake_client(monkeypatch, content='{"score": 90, "reason": "ok"}')
    monkeypatch.setattr(config, "LLM_REQUEST_TIMEOUT_S", 19.0)
    import taco.retry as retry
    seen = {}
    real = retry.with_retries

    def _spy(fn, **kwargs):
        seen.update(kwargs)
        return real(fn, **kwargs)

    monkeypatch.setattr(retry, "with_retries", _spy)
    out = judge.judge("probe", "ground truth", "factual", "response")
    assert out["score"] == 90
    assert seen["timeout_per_attempt"] == 19.0


# --------------------------------------------------------------------------- #
# Default wiring
# --------------------------------------------------------------------------- #
def test_response_and_judge_models_default_to_llm_model():
    """If TACO_RESPONSE_MODEL / TACO_JUDGE_MODEL are unset, both fall back to
    LLM_MODEL — single-model setups need zero new env vars."""
    assert config.RESPONSE_MODEL == config.LLM_MODEL or \
        config.RESPONSE_MODEL == "model-for-answers"  # in case test reorders
    assert config.JUDGE_MODEL == config.LLM_MODEL or \
        config.JUDGE_MODEL == "model-for-judging"
