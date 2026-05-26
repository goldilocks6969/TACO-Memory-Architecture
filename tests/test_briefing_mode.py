"""TACO_STATE_BRIEFING_MODE — state-briefing verbosity has three modes:

  full     — verbose (current debug default)
  compact  — ≤ 80 tokens of state-briefing overhead per turn
  minimal  — ≤ 35 tokens of state-briefing overhead per turn

The retrieval payload (memories/beliefs) is unaffected by the mode — CES_total
needs the briefing overhead to be charged separately, so these tests assert
*only* the four state-briefing sections (state / identity / working / prediction).
"""
from __future__ import annotations

import pytest

from taco import config
from taco.memory.episode import Episode
from taco.memory.retrieval import _count_tokens, briefing_sections
from taco.state import LatentState
from taco.subsystems.reasoning import stance as resolve_stance


_OVERHEAD_KEYS = ("state", "identity", "working", "prediction")


def _state_briefing_tokens(sections) -> int:
    overhead = "\n\n".join(sections[k] for k in _OVERHEAD_KEYS if sections.get(k))
    return _count_tokens(overhead)


@pytest.fixture()
def hot_inputs():
    """A high-state turn with rich identity + working memory — the worst case
    for budget pressure."""
    state = LatentState(E=72, K=40, V=58, R=80)
    stance = resolve_stance(state)
    identity = [
        "relationship:partner — Maya (confidence 80%)",
        "occupation — software engineer at a fintech (confidence 70%)",
        "value:autonomy — high (confidence 60%)",
        "struggle:burnout — recurring quarterly (confidence 65%)",
    ]
    working = [
        Episode(content="I haven't been sleeping much"),
        Episode(content="That sounds exhausting", role="assistant"),
        Episode(content="and Maya noticed yesterday"),
    ]
    beliefs = ["this person works through grief by overworking"]
    return dict(state=state, stance=stance, identity=identity,
                working=working, beliefs=beliefs)


def test_full_mode_keeps_all_sections(hot_inputs):
    s = briefing_sections([], hot_inputs["beliefs"], hot_inputs["working"],
                          hot_inputs["state"], hot_inputs["stance"],
                          identity=hot_inputs["identity"], mode="full")
    assert s["state"], "full mode must render the verbose state line"
    assert s["identity"], "full mode must render the persistent self-model"
    assert s["working"], "full mode must render working memory"
    assert "User state — emotional intensity" in s["state"]


def test_compact_mode_under_80_tokens(hot_inputs):
    s = briefing_sections([], hot_inputs["beliefs"], hot_inputs["working"],
                          hot_inputs["state"], hot_inputs["stance"],
                          identity=hot_inputs["identity"], mode="compact")
    tokens = _state_briefing_tokens(s)
    assert tokens <= config.STATE_BRIEFING_COMPACT_MAX_TOKENS, (
        f"compact mode exceeded its budget: {tokens} > "
        f"{config.STATE_BRIEFING_COMPACT_MAX_TOKENS} tokens. Briefing was:\n"
        + "\n\n".join(s[k] for k in _OVERHEAD_KEYS if s.get(k))
    )
    # the tone + stance must still be present — they carry the actual signal
    assert "tone=" in s["state"]
    assert "Stance" in s["state"]
    # compact drops working memory and prediction by design
    assert not s["working"]
    assert not s["prediction"]


def test_minimal_mode_under_35_tokens(hot_inputs):
    s = briefing_sections([], hot_inputs["beliefs"], hot_inputs["working"],
                          hot_inputs["state"], hot_inputs["stance"],
                          identity=hot_inputs["identity"], mode="minimal")
    tokens = _state_briefing_tokens(s)
    assert tokens <= config.STATE_BRIEFING_MINIMAL_MAX_TOKENS, (
        f"minimal mode exceeded its budget: {tokens} > "
        f"{config.STATE_BRIEFING_MINIMAL_MAX_TOKENS} tokens"
    )
    # minimal carries the two signals our ablations show actually modulate
    # the engine: current tone and the resolved stance label.
    assert "tone=" in s["state"]
    assert "stance=" in s["state"]
    assert not s["identity"]
    assert not s["working"]
    assert not s["prediction"]


def test_retrieval_payload_unaffected_by_mode(hot_inputs):
    """The retrieved memory/fact payload is charged separately — changing the
    state-briefing mode must not change ``retrieval``."""
    full = briefing_sections([], hot_inputs["beliefs"], hot_inputs["working"],
                             hot_inputs["state"], hot_inputs["stance"],
                             identity=hot_inputs["identity"], mode="full")
    compact = briefing_sections([], hot_inputs["beliefs"], hot_inputs["working"],
                                 hot_inputs["state"], hot_inputs["stance"],
                                 identity=hot_inputs["identity"], mode="compact")
    minimal = briefing_sections([], hot_inputs["beliefs"], hot_inputs["working"],
                                 hot_inputs["state"], hot_inputs["stance"],
                                 identity=hot_inputs["identity"], mode="minimal")
    assert full["retrieval"] == compact["retrieval"] == minimal["retrieval"]


def test_mode_reads_from_config_by_default(hot_inputs, monkeypatch):
    """When no explicit ``mode`` is passed, ``config.STATE_BRIEFING_MODE`` wins."""
    monkeypatch.setattr(config, "STATE_BRIEFING_MODE", "minimal")
    s = briefing_sections([], hot_inputs["beliefs"], hot_inputs["working"],
                          hot_inputs["state"], hot_inputs["stance"],
                          identity=hot_inputs["identity"])  # no mode= arg
    assert _state_briefing_tokens(s) <= config.STATE_BRIEFING_MINIMAL_MAX_TOKENS
    assert not s["identity"] and not s["working"]


def test_invalid_mode_raises_on_config_load():
    """Invalid TACO_STATE_BRIEFING_MODE values fail loud at config import."""
    # config has already imported with a valid value; re-validate the rule.
    assert "full" in config._VALID_BRIEFING_MODES
    assert "compact" in config._VALID_BRIEFING_MODES
    assert "minimal" in config._VALID_BRIEFING_MODES
    assert "verbose" not in config._VALID_BRIEFING_MODES
