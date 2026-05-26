"""Tiered extraction: light on every turn, full only on salient turns (Phase 1).

These exercise the keyless heuristic path (TACO_MOCK), so they are deterministic
and need no API key or database.
"""
import pytest

from taco import config
from taco.memory import extract


@pytest.fixture(autouse=True)
def _mock(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)


def test_light_extract_returns_affect_entities_cues():
    le = extract.light_extract("my partner Maya and I broke up last night")
    assert 0 <= le.emotional <= 100
    assert 0 <= le.vulnerability <= 100
    assert 1 <= le.salience <= 10
    assert isinstance(le.entities, list)
    assert isinstance(le.retrieval_cues, list)
    assert len(le.retrieval_cues) <= 2          # light tier caps cues at 2
    # the relationship / named partner is tagged as an entity
    assert any("maya" in e or "partner" in e for e in le.entities)


def test_light_cues_capped_at_two():
    le = extract.light_extract("my dad died, I got laid off, and I relapsed drinking")
    assert len(le.retrieval_cues) <= 2


def test_full_extract_emits_rich_fact_on_disclosure():
    text = "my dad passed away last night, a heart attack"
    le = extract.light_extract(text)
    fe = extract.full_extract(text, le)
    assert len(fe.facts) >= 1
    f = fe.facts[0]
    assert f.summary
    assert f.salience == le.salience            # salience carried from the light tier
    assert len(f.retrieval_cues) <= 3           # full tier caps cues at 3
    assert f.event_type                          # classified, not blank


def test_full_extract_flags_open_thread():
    text = "I have an interview on friday and I want help preparing"
    le = extract.light_extract(text)
    fe = extract.full_extract(text, le)
    assert any(f.fact_type == "thread" and f.thread_status == "unresolved"
               for f in fe.facts)


def test_transactional_turn_is_low_salience():
    le = extract.light_extract("what time does the post office close on fridays?")
    assert le.salience < 4                        # filler should not clear θ_facts
