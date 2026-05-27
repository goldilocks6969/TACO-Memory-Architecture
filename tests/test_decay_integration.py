"""Integration coverage for abstraction-before-pruning."""
from __future__ import annotations

import os

import pytest

from taco import db
from taco.memory import decay, store
from taco.memory.episode import Episode
from taco.state import LatentState

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


def _emb(_text: str):
    return [0.0] * 1536


def test_abstract_and_prune_writes_belief_before_deleting_episode(conn):
    user_id = "abstract_before_prune"
    ep = Episode(
        content="Father criticized my career choice again at dinner.",
        salience=3,
        weekly_decay=0.75,
        vitality=0.10,
        tone="concerned",
    )
    store.add_episode(conn, ep, _emb(ep.content), LatentState(), user_id=user_id)

    def abstract_fn(texts):
        assert texts == [ep.content]
        return (
            "This person carries unresolved tension with their father around "
            "expectations and autonomy."
        )

    beliefs = decay.abstract_and_prune(
        conn, _emb, abstract_fn, user_id=user_id)

    assert beliefs == [
        "This person carries unresolved tension with their father around "
        "expectations and autonomy."
    ]
    assert conn.execute(
        "SELECT count(*) FROM episodes WHERE user_id = %s", (user_id,)
    ).fetchone()[0] == 0
    row = conn.execute(
        "SELECT belief, tone, source_count FROM semantic_beliefs WHERE user_id = %s",
        (user_id,),
    ).fetchone()
    assert row[0] == beliefs[0]
    assert row[1] == "concerned"
    assert row[2] == 1
