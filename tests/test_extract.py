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


def test_light_facts_preserve_quiet_nickname_anchor():
    text = "he always called me 'kiddo'"
    le = extract.light_extract(text)
    facts = extract.light_facts(text, le)
    assert any("kiddo" in f.summary.lower() for f in facts)
    assert extract.has_memory_anchor(text, le)


def test_light_facts_preserve_career_outcome_anchor():
    text = "I GOT THE JOB at Halcyon. better pay even. I could cry"
    le = extract.light_extract(text)
    facts = extract.light_facts(text, le)
    assert any("halcyon" in f.summary.lower() and "hired" in f.summary.lower()
               for f in facts)
    assert any("career outcome" in f.retrieval_cues for f in facts)


def test_light_facts_preserve_health_progress_anchor():
    text = "my A1C dropped a little at the recheck. small win"
    le = extract.light_extract(text)
    facts = extract.light_facts(text, le)
    assert any("a1c dropped" in f.summary.lower() for f in facts)
    assert any("health progress" in f.retrieval_cues for f in facts)


def test_light_facts_preserve_unresolved_relationship_thread():
    text = "Sam reached out. wants to 'talk'. I don't know if I can"
    le = extract.light_extract(text)
    facts = extract.light_facts(text, le)
    assert any(f.fact_type == "thread" and f.thread_status == "unresolved"
               for f in facts), (le, facts)
