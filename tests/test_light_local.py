"""``TACO_EVAL_LIGHT_EXTRACT_LOCAL=1`` — force the keyless heuristic path inside
``extract.light_extract`` even when an API key is configured.

The live benchmark depends on this: the LLM-backed light tier can wedge during
long scenario ingests in the same way ``llm.respond`` can.  These tests assert
that with the flag on, **no chat completion call is made anywhere in the
ingest path**, while a usable fact is still created.
"""
from __future__ import annotations

import os

import pytest

from taco import config, db, embeddings, llm
from taco.memory import extract
from taco.pipeline import Taco

TEST_DSN = os.getenv("TACO_TEST_DSN", "postgresql://localhost:5432/taco_eval")


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


def _no_llm_environment(monkeypatch):
    """Configure ``config`` so ``llm.available()`` would return True (an API
    key is set, MOCK is off), but make ``llm._client()`` raise on use — so any
    chat call from anywhere fails the test loudly."""
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fake-test-key-not-real")
    monkeypatch.setattr(config, "EXTRACTION_MODE", "light")
    monkeypatch.setattr(config, "LIGHT_EXTRACT_LOCAL", True)

    def _no_client(*args, **kwargs):
        raise AssertionError(
            "llm._client() must not be invoked when "
            "LIGHT_EXTRACT_LOCAL=1 + EXTRACTION_MODE=light + "
            "generate_response=False"
        )
    monkeypatch.setattr(llm, "_client", _no_client)

    # Embeddings are *allowed* per spec when absolutely required (we need a
    # vector for the fact row).  Stub them so the test doesn't hit the
    # embeddings API either — the test point is the chat client, not the
    # embeddings endpoint.
    mock_vec = tuple(0.0 if i else 1.0 for i in range(config.EMBED_DIM))
    monkeypatch.setattr(embeddings, "embed", lambda text: mock_vec)


def test_light_local_does_not_call_llm_during_ingest(conn, monkeypatch):
    """LIGHT mode + LIGHT_EXTRACT_LOCAL=1 + generate_response=False must run
    a full ingest with zero chat-completion calls."""
    _no_llm_environment(monkeypatch)

    # Reset the one-shot announce flag so the log emission is exercised at
    # least once during the test run.
    monkeypatch.setattr(extract, "_LIGHT_LOCAL_ANNOUNCED", False)

    taco = Taco(conn, user_id="light_local_user")
    # A moderate-salience turn so the salience gate passes for fact storage
    # (theta_facts ≈ 2.3) but stays *below* IDENTITY_SALIENCE_MIN (5.0) so
    # identity.consolidate doesn't attempt llm.extract_identity.
    result = taco.turn("I'm worried about my upcoming interview",
                       hours_since_last=0.5,
                       generate_response=False)

    # No response, no LLM call — but the fact is still there.
    assert result.response == ""
    rows = conn.execute(
        "SELECT summary, salience FROM facts WHERE user_id = %s",
        ("light_local_user",),
    ).fetchall()
    assert rows, "light_local ingest must still create a fact"
    assert "interview" in rows[0][0].lower()
    # The light-fact counter confirms the write path executed.
    assert taco.extraction_stats["light_facts_created"] == 1


def test_light_local_returns_full_lightextract_schema(monkeypatch):
    """The heuristic light_extract carries every field the spec requires
    downstream: salience, tone, vulnerability, retrieval_cues."""
    _no_llm_environment(monkeypatch)
    light = extract.light_extract(
        "I broke up with my partner last night and I'm devastated")
    # Affect dimensions populated (spec req 4: salience, emotional_tone,
    # vulnerability)
    assert light.salience > 0
    assert isinstance(light.tone, str) and light.tone
    assert light.vulnerability >= 0
    assert light.emotional >= 0
    # Retrieval cues pulled from the heuristic cue map (spec req 4)
    assert any("breakup" in c or "broke" in c for c in light.retrieval_cues), \
        f"expected a breakup-related cue, got: {light.retrieval_cues}"
    # The downstream light_fact builder infers fact_type/event_type from the
    # raw text (spec req 4: "basic event_type/fact_type if inferable")
    fact = extract.light_fact(
        "I broke up with my partner last night and I'm devastated", light)
    assert fact.fact_type in ("event", "thread")
    assert fact.event_type  # heuristic_event_type returns something for "broke up"


def test_light_local_default_off_for_non_eval_users():
    """Outside the eval harness the LLM-backed light tier remains the default
    — the keyless heuristic only kicks in when the eval explicitly enables it."""
    # Imported with the env var unset (the pytest run unsets it via clean env);
    # if the developer ran pytest with the var set we treat that as opt-in.
    assert config.LIGHT_EXTRACT_LOCAL is False or \
        os.getenv("TACO_EVAL_LIGHT_EXTRACT_LOCAL") == "1"
