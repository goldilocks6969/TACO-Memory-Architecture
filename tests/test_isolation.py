"""Cross-scenario isolation: ``user_id`` namespacing must prevent grief
memories from leaking into the job_loss scenario's retrieval.

Skipped when no Postgres test DB is reachable. The DB cost is unavoidable —
the bug being regressed against was a real cross-row SQL leak that no in-memory
mock can faithfully reproduce.
"""
from __future__ import annotations

import os

import pytest

from taco import config, db
from taco.memory import store
from taco.memory.fact import Fact

TEST_DSN = os.getenv("TACO_TEST_DSN", "postgresql://localhost:5432/taco_eval")


@pytest.fixture(autouse=True)
def _mock(monkeypatch):
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


def _emb(idx: int):
    """A one-hot embedding so retrieval ordering is deterministic in the test."""
    v = [0.0] * config.EMBED_DIM
    v[idx] = 1.0
    return v


def test_father_death_does_not_leak_into_job_loss(conn):
    """The bug the professor flagged: grief memories surfaced in job_loss probes."""
    # Ingest the grief scenario — father died.
    grief_fact = Fact(summary="user's father passed away from a heart attack",
                      salience=10, emotional_tone="distress")
    grief_id = store.add_fact(conn, grief_fact, _emb(0),
                              user_id="grief")
    assert grief_id

    # Ingest the job_loss scenario — laid off (intentionally given the SAME
    # embedding so vector-similarity alone cannot keep them apart; only the
    # user_id filter can).
    job_fact = Fact(summary="user was laid off from their tech job last week",
                    salience=9, emotional_tone="distress")
    job_id = store.add_fact(conn, job_fact, _emb(0),
                            user_id="job_loss")
    assert job_id

    # Querying the job_loss namespace must NOT see the grief memory.
    job_hits = store.fact_knn_candidates(conn, _emb(0), k=10,
                                         user_id="job_loss")
    contents = [ep.content for ep in job_hits]
    assert any("laid off" in c for c in contents), \
        f"job_loss namespace should surface its own fact, got {contents}"
    for ep in job_hits:
        assert ep.user_id == "job_loss", \
            f"cross-namespace leak: {ep.user_id=} content={ep.content!r}"
    assert not any("father" in c or "heart attack" in c for c in contents), \
        f"father-death memory leaked into job_loss: {contents}"

    # Symmetric direction: querying grief must NOT see the layoff memory.
    grief_hits = store.fact_knn_candidates(conn, _emb(0), k=10,
                                           user_id="grief")
    for ep in grief_hits:
        assert ep.user_id == "grief"
    assert not any("laid off" in ep.content for ep in grief_hits)


def test_fact_neighbors_respects_user_id(conn):
    """``fact_neighbors`` (the dedup query) must also namespace by user_id —
    otherwise a new fact in user A could be merged into user B's record."""
    store.add_fact(conn, Fact(summary="user's father passed away from heart attack",
                              salience=10),
                   _emb(0), user_id="grief")

    # Same embedding, but querying under a different user — must see nothing.
    nb = store.fact_neighbors(conn, _emb(0), k=5, min_sim=0.0,
                              user_id="job_loss")
    assert nb == [], f"fact_neighbors leaked across user_id: {nb}"

    # Same query within the original user — finds it.
    nb_same = store.fact_neighbors(conn, _emb(0), k=5, min_sim=0.0,
                                   user_id="grief")
    assert len(nb_same) == 1
    assert nb_same[0].user_id == "grief"


def test_episodes_and_beliefs_isolated(conn):
    """Episodes and semantic beliefs are also namespaced (the harness assertion
    looks at ``episode.user_id`` via the retrieved candidates)."""
    from taco.memory.episode import Episode

    ep_grief = Episode(content="I lost my dad", salience=10, tone="distress")
    store.add_episode(conn, ep_grief, _emb(0), user_id="grief")
    ep_job = Episode(content="I got laid off", salience=9, tone="distress")
    store.add_episode(conn, ep_job, _emb(0), user_id="job_loss")

    job_eps = store.knn_candidates(conn, _emb(0), k=10, user_id="job_loss")
    assert all(e.user_id == "job_loss" for e in job_eps)
    assert not any("dad" in e.content for e in job_eps)

    # Beliefs
    store.add_belief(conn, "this person is grieving",
                     _emb(0), tone="distress", user_id="grief")
    beliefs_job = store.belief_candidates(conn, _emb(0), k=5,
                                          user_id="job_loss")
    assert beliefs_job == []
    beliefs_grief = store.belief_candidates(conn, _emb(0), k=5,
                                            user_id="grief")
    assert "grieving" in beliefs_grief[0]
