"""TACO_EVAL_EXTRACTION_MODE — light / auto / full write-path behaviour.

These tests touch the real schema (the ``facts`` row shape matters), so they
skip when no Postgres test DB is reachable.  The LLM client is never invoked —
``TACO_MOCK=1`` is forced and ``extract.full_extract`` is monkeypatched at
each test's grain to detect call attempts and inject CallTimeout.
"""
from __future__ import annotations

import os
from typing import List

import pytest

from taco import config, db
from taco.memory import extract, store
from taco.memory.fact import Fact
from taco.pipeline import Taco
from taco.retry import CallTimeout

TEST_DSN = os.getenv("TACO_TEST_DSN", "postgresql://localhost:5432/taco_eval")


@pytest.fixture(autouse=True)
def _mock(monkeypatch):
    """All extraction-mode tests run keyless. The light fact builder uses the
    deterministic heuristics, so we get repeatable Fact summaries."""
    monkeypatch.setattr(config, "MOCK", True)


@pytest.fixture()
def conn():
    try:
        db.init_db(TEST_DSN)
        c = db.connect(TEST_DSN)
        c.execute("TRUNCATE facts, episodes, semantic_beliefs, "
                  "emotional_timeline, identity, state_log RESTART IDENTITY")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no Postgres test DB available: {e}")
    yield c
    try:
        c.execute("TRUNCATE facts, episodes, semantic_beliefs, "
                  "emotional_timeline, identity, state_log RESTART IDENTITY")
        c.close()
    except Exception:
        pass


def _user_facts(conn, user_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT id, summary, fact_type, salience, validity, confidence "
        "FROM facts WHERE user_id = %s ORDER BY id",
        (user_id,),
    ).fetchall()
    return [dict(id=r[0], summary=r[1], fact_type=r[2], salience=r[3],
                 validity=r[4], confidence=r[5]) for r in rows]


# --------------------------------------------------------------------------- #
# light_fact builder — pure, no DB
# --------------------------------------------------------------------------- #
def test_light_fact_summary_capped_at_240_chars():
    light = extract.LightExtract(emotional=80, vulnerability=60, salience=9,
                                 tone="distress", entities=[], retrieval_cues=[])
    long_text = ("My father passed away last night from a sudden heart attack. "
                 "I keep replaying his last voicemail in my head. ") * 5
    f = extract.light_fact(long_text, light)
    assert len(f.summary) <= 240, f"summary too long: {len(f.summary)} chars"
    assert f.summary.endswith("…"), "truncated summary should be marked with …"
    # Schema requirements from the spec.
    assert f.salience == 9.0
    assert f.emotional_tone == "distress"
    assert f.validity == "current"
    assert f.confidence == 0.6
    assert len(f.retrieval_cues) <= 3


def test_light_fact_thread_inference():
    """A turn that names a future event becomes a thread, not a generic event."""
    light = extract.LightExtract(emotional=40, vulnerability=20, salience=6,
                                 tone="concerned", entities=[],
                                 retrieval_cues=["interview anxiety"])
    f = extract.light_fact("I have a job interview on friday and I'm freaking out", light)
    assert f.fact_type == "thread"
    assert f.thread_status == "unresolved"


# --------------------------------------------------------------------------- #
# Mode = light
# --------------------------------------------------------------------------- #
def test_light_mode_never_calls_full_extract(conn, monkeypatch):
    """The whole point of LIGHT mode: full_extract MUST NOT fire."""
    monkeypatch.setattr(config, "EXTRACTION_MODE", "light")

    def _boom(*args, **kwargs):
        raise AssertionError("full_extract must not be called in light mode")
    monkeypatch.setattr(extract, "full_extract", _boom)

    taco = Taco(conn, user_id="light_user")
    # phrase chosen so the keyless heuristic scores it above theta_facts
    taco.turn("My father passed away last night from a heart attack",
              hours_since_last=0.5)

    facts = _user_facts(conn, "light_user")
    assert facts, "light mode should still store a fact for a high-salience turn"
    assert all(f["validity"] == "current" for f in facts)
    # confidence on a light fact is the spec-specified 0.6
    assert facts[0]["confidence"] == pytest.approx(0.6, abs=0.001)
    # write-path counters
    assert taco.extraction_stats["light_facts_created"] == 1
    assert taco.extraction_stats["full_extract_attempts"] == 0


def test_light_mode_creates_facts_from_salient_messages(conn, monkeypatch):
    """Multiple high-salience turns each yield exactly one light fact, with
    no LLM calls and no decide_action involvement."""
    monkeypatch.setattr(config, "EXTRACTION_MODE", "light")
    monkeypatch.setattr(extract, "full_extract", lambda *a, **k:
                        pytest.fail("full_extract should not run"))

    taco = Taco(conn, user_id="light_multi")
    msgs = [
        "I got laid off from my tech job today and I'm devastated",
        "My partner Maya and I broke up last night after three years",
        "I have a final cancer biopsy result coming on monday",
    ]
    for m in msgs:
        taco.turn(m, hours_since_last=0.5)

    facts = _user_facts(conn, "light_multi")
    assert len(facts) >= 3, f"expected ≥ 3 light facts, got: {facts}"
    assert taco.extraction_stats["light_facts_created"] >= 3


def test_light_mode_synthesizes_causal_bridge(conn, monkeypatch):
    monkeypatch.setattr(config, "EXTRACTION_MODE", "light")
    monkeypatch.setattr(config, "SKIP_IDENTITY_DURING_INGEST", True)

    taco = Taco(conn, user_id="bridge_ingest")
    taco.turn(
        "[2024-10-15 session p1 turn 5] There's been constant workplace harassment, but I haven’t addressed it.",
        hours_since_last=0.5,
        generate_response=False,
    )
    taco.turn(
        "[2024-11-05 session p1 turn 21] Therapy has been grounding and helped me set boundaries around work stress.",
        hours_since_last=24,
        generate_response=False,
    )

    rows = conn.execute(
        "SELECT summary, entity_keys FROM facts "
        "WHERE user_id = %s AND fact_type = 'bridge'",
        ("bridge_ingest",),
    ).fetchall()
    assert rows
    summary, keys = rows[0]
    assert "workplace harassment" in summary
    assert "therapy" in summary
    assert "rel:triggered_coping" in keys
    assert taco.extraction_stats["bridge_facts_created"] >= 1


def test_light_mode_skips_trivial_low_salience(conn, monkeypatch):
    """The salience gate in Taco.turn protects the fact store from chatter."""
    monkeypatch.setattr(config, "EXTRACTION_MODE", "light")
    monkeypatch.setattr(extract, "full_extract", lambda *a, **k:
                        pytest.fail("full_extract should not run"))

    taco = Taco(conn, user_id="light_trivial")
    # plain greetings / transactional small talk land at salience 1-2,
    # which is well below theta_facts() at baseline state.
    for trivial in ("hi", "hello", "thanks", "ok", "got it"):
        taco.turn(trivial, hours_since_last=0.5)

    facts = _user_facts(conn, "light_trivial")
    assert facts == [], f"trivial turns should not create facts: {facts}"
    assert taco.extraction_stats["light_facts_created"] == 0


# --------------------------------------------------------------------------- #
# Mode = auto
# --------------------------------------------------------------------------- #
def test_auto_mode_falls_back_to_light_fact_on_timeout(conn, monkeypatch):
    """If full_extract times out, the light fact must survive and the counters
    must register the fallback."""
    monkeypatch.setattr(config, "EXTRACTION_MODE", "auto")
    # force the auto thresholds to be easy to satisfy
    monkeypatch.setattr(config, "AUTO_FULL_MIN_SALIENCE", 1.0)
    monkeypatch.setattr(config, "AUTO_FULL_MAX_CHARS", 100_000)

    def _hang(*args, **kwargs):
        raise CallTimeout("LLM call hung (>60s) — label='extract.full_extract'")
    monkeypatch.setattr(extract, "full_extract", _hang)

    taco = Taco(conn, user_id="auto_timeout")
    taco.turn("My father passed away last night from a heart attack",
              hours_since_last=0.5)

    facts = _user_facts(conn, "auto_timeout")
    assert facts, "auto mode must keep the light fact when full_extract hangs"
    assert taco.extraction_stats["light_facts_created"] == 1
    assert taco.extraction_stats["full_extract_attempts"] == 1
    assert taco.extraction_stats["full_extract_successes"] == 0
    assert taco.extraction_stats["full_extract_timeouts"] == 1
    assert taco.extraction_stats["full_extract_fallbacks"] == 1


def test_auto_mode_skips_full_extract_under_threshold(conn, monkeypatch):
    """Below the salience cutoff, AUTO mode acts like LIGHT — no full_extract."""
    monkeypatch.setattr(config, "EXTRACTION_MODE", "auto")
    monkeypatch.setattr(config, "AUTO_FULL_MIN_SALIENCE", 9.0)  # very high bar
    monkeypatch.setattr(extract, "full_extract", lambda *a, **k:
                        pytest.fail("full_extract should be skipped here"))

    taco = Taco(conn, user_id="auto_low")
    # "My partner Maya broke up with me last night" lands ~7–8 in the
    # heuristic analyzer; we set the bar at 9.0 so it's skipped.
    taco.turn("My partner Maya broke up with me last night",
              hours_since_last=0.5)
    assert taco.extraction_stats["full_extract_attempts"] == 0
    assert taco.extraction_stats["light_facts_created"] >= 1


# --------------------------------------------------------------------------- #
# Mode = full
# --------------------------------------------------------------------------- #
def test_full_mode_still_calls_full_extract(conn, monkeypatch):
    """Original code path must be intact."""
    monkeypatch.setattr(config, "EXTRACTION_MODE", "full")

    called = {"n": 0}
    real_full = extract.full_extract

    def _spy(text, light):
        called["n"] += 1
        return real_full(text, light)
    monkeypatch.setattr(extract, "full_extract", _spy)

    taco = Taco(conn, user_id="full_user")
    taco.turn("My father passed away last night from a heart attack",
              hours_since_last=0.5)
    assert called["n"] == 1, "full mode must invoke full_extract for salient turns"
    assert taco.extraction_stats["full_extract_attempts"] == 1
    assert taco.extraction_stats["full_extract_successes"] == 1
    # full mode does not create explicit light facts
    assert taco.extraction_stats["light_facts_created"] == 0
