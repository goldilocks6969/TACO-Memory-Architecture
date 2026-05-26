"""Fact write operations: ADD / UPDATE / MERGE / DELETE / NOOP (Phase 1).

`decide_action`'s heuristic is pure and always runs. The lifecycle test exercises
the real store mutations and is skipped when no Postgres test DB is reachable.
"""
import os

import pytest

from taco import config, db
from taco.memory import operations, store
from taco.memory.fact import Fact
from taco.memory.operations import ADD, DELETE, MERGE, NOOP, UPDATE

TEST_DSN = os.getenv("TACO_TEST_DSN", "postgresql://localhost:5432/taco_eval")


# --------------------------------------------------------------------------- #
# decide_action heuristic (no DB, no API)
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _mock(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)


def _neighbor(fid, sim):
    return Fact(id=fid, summary="existing fact", similarity=sim)


def test_no_neighbors_is_add():
    assert operations.decide_action(Fact(summary="new"), []).kind == ADD


def test_exact_duplicate_is_noop():
    a = operations.decide_action(Fact(summary="dup"), [_neighbor(1, 0.985)])
    assert a.kind == NOOP and a.target_id == 1


def test_close_match_merges():
    a = operations.decide_action(Fact(summary="extra detail"), [_neighbor(2, 0.90)])
    assert a.kind == MERGE and a.target_id == 2


def test_distinct_neighbor_still_adds():
    # a neighbour above the 0.7 floor but below the merge band is still new info
    assert operations.decide_action(Fact(summary="diff"), [_neighbor(3, 0.74)]).kind == ADD


# --------------------------------------------------------------------------- #
# store round-trip (needs a Postgres test DB; skipped otherwise)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def conn():
    try:
        db.init_db(TEST_DSN)
        c = db.connect(TEST_DSN)
        c.execute("TRUNCATE facts RESTART IDENTITY")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no Postgres test DB available: {e}")
    yield c
    try:
        c.execute("TRUNCATE facts RESTART IDENTITY")
        c.close()
    except Exception:
        pass


def _emb(idx: int):
    v = [0.0] * config.EMBED_DIM
    v[idx] = 1.0
    return v


def test_fact_lifecycle(conn):
    # ADD two orthogonal facts
    id1 = operations.apply(conn, operations.Action(ADD),
                           Fact(summary="User wants to become a lawyer", salience=6),
                           _emb(0), None)
    id2 = operations.apply(conn, operations.Action(ADD),
                           Fact(summary="User has a dog named Max", salience=5),
                           _emb(1), None)
    assert id1 and id2

    # neighbours near e0 find the lawyer fact
    nb = store.fact_neighbors(conn, _emb(0), k=3, min_sim=0.7)
    assert any(n.id == id1 for n in nb)

    # UPDATE: the career change supersedes the lawyer fact
    new = Fact(summary="User is building AI memory infra, no longer law", salience=7)
    rid = operations.apply(conn, operations.Action(UPDATE, target_id=id1,
                                                   content=new.summary),
                           new, _emb(0), None)
    assert rid and rid != id1
    validity, superseded_by = conn.execute(
        "SELECT validity, superseded_by FROM facts WHERE id=%s", (id1,)).fetchone()
    assert validity == "outdated" and superseded_by == rid

    # only_current retrieval drops the outdated fact, keeps the new one
    ids = {e.id for e in store.fact_knn_candidates(conn, _emb(0), k=5, only_current=True)}
    assert id1 not in ids and rid in ids

    # MERGE: extra detail folds into the dog fact in place (no new row)
    before = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    operations.apply(conn, operations.Action(MERGE, target_id=id2,
                     content="User has a labradoodle named Max who ate the couch"),
                     Fact(summary="Max ate the couch"), _emb(1), 7)
    after = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    assert after == before
    assert "labradoodle" in conn.execute(
        "SELECT summary FROM facts WHERE id=%s", (id2,)).fetchone()[0]

    # DELETE: soft-retire the dog fact
    operations.apply(conn, operations.Action(DELETE, target_id=id2),
                     Fact(summary="x"), _emb(1), None)
    assert conn.execute(
        "SELECT validity FROM facts WHERE id=%s", (id2,)).fetchone()[0] == "outdated"
