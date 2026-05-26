"""``Taco.turn(generate_response=False)`` — ingest-only path for the eval
harness.  Skips the ``llm.respond`` call while still running the full
write path (light extraction, state inference, episodes, facts, identity,
state logging).
"""
from __future__ import annotations

import os

import pytest

from taco import config, db, llm
from taco.pipeline import Taco

TEST_DSN = os.getenv("TACO_TEST_DSN", "postgresql://localhost:5432/taco_eval")


@pytest.fixture(autouse=True)
def _mock(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)
    # LIGHT extraction keeps every test under one second.
    monkeypatch.setattr(config, "EXTRACTION_MODE", "light")


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


def _count(conn, table: str, user_id: str) -> int:
    return conn.execute(
        f"SELECT count(*) FROM {table} WHERE user_id = %s", (user_id,)
    ).fetchone()[0]


def test_generate_response_false_does_not_call_respond(conn, monkeypatch):
    """The whole point of the ingest-only path: respond MUST NOT fire."""
    def _boom(*args, **kwargs):
        raise AssertionError("llm.respond must not be called when "
                             "generate_response=False")
    monkeypatch.setattr(llm, "respond", _boom)

    taco = Taco(conn, user_id="ingest_only")
    result = taco.turn("My father passed away last night from a heart attack",
                       hours_since_last=0.5, generate_response=False)

    assert result.response == ""
    # The empty assistant stub must not be appended to working memory,
    # otherwise the next ingest turn's briefing would contain a literal "" turn.
    assert all(w.role != "assistant" or w.content for w in taco.working), \
        "empty assistant stub should not enter working memory"


def test_generate_response_false_still_writes_memory(conn, monkeypatch):
    """State inference, episode + fact writes, and the state log all still
    run on the ingest-only path."""
    monkeypatch.setattr(llm, "respond", lambda *a, **k:
                        pytest.fail("respond should not be called"))

    taco = Taco(conn, user_id="ingest_writes")
    taco.turn("My father passed away last night from a heart attack",
              hours_since_last=0.5, generate_response=False)

    # The high-salience turn should produce an episode and a fact.
    assert _count(conn, "episodes", "ingest_writes") == 1
    assert _count(conn, "facts", "ingest_writes") == 1
    # Emotional timeline + state log fire on every above-threshold turn.
    assert _count(conn, "emotional_timeline", "ingest_writes") == 1
    assert _count(conn, "state_log", "ingest_writes") == 1
    # And the light-fact counter is bumped (proves the write path ran).
    assert taco.extraction_stats["light_facts_created"] == 1


def test_default_turn_still_calls_respond(conn, monkeypatch):
    """Product behaviour unchanged: ``generate_response`` defaults to True."""
    calls = {"n": 0}

    def _spy(briefing, message, planning_depth):
        calls["n"] += 1
        return "ok"
    monkeypatch.setattr(llm, "respond", _spy)

    taco = Taco(conn, user_id="normal_turn")
    result = taco.turn("My father passed away last night from a heart attack",
                       hours_since_last=0.5)
    assert calls["n"] == 1
    assert result.response == "ok"
    # The assistant reply IS appended to working memory in normal mode.
    assert any(w.role == "assistant" and w.content == "ok" for w in taco.working)


def test_skip_flag_default_off_for_non_eval_users():
    """The CLI / single-user path must default to ``respond`` running.
    Only the eval harness flips ``SKIP_RESPOND_DURING_INGEST`` to True."""
    # config was imported with the env var unset; default is False.
    assert config.SKIP_RESPOND_DURING_INGEST is False or \
        os.getenv("TACO_EVAL_SKIP_RESPOND_DURING_INGEST") == "1"
