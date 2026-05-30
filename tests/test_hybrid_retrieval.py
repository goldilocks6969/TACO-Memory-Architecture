"""Phase 2 hybrid retrieval — semantic + summary-trigram + cue-trigram +
entity-overlap, RRF-fused, then state-tilted.

These tests need a Postgres DB with pg_trgm enabled (the schema sets that up
automatically); they skip when one isn't reachable.  Embeddings are mocked
via ``TACO_MOCK=1`` so no network access is required.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from taco import config, db, embeddings
from taco.memory import retrieval, store
from taco.memory.fact import Fact
from taco.memory import rerank as taco_rerank
from taco.state import LatentState

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


def _ingest_fact(conn, *, user_id: str, fact: Fact, anchor: str) -> int:
    """Insert *fact* with the embedding derived from *anchor* — the mock
    embedder is deterministic, so two facts that share the same anchor
    cluster together in cosine space."""
    return store.add_fact(conn, fact, embeddings.embed_list(anchor),
                          source_episode_id=None, user_id=user_id)


def _hybrid(conn, query: str, *, user_id: str, top_k: int = 6):
    q_emb = embeddings.embed_list(query)
    state = LatentState()
    return retrieval.hybrid_retrieve(conn, query, q_emb, state,
                                      user_id=user_id, top_k=top_k)


# --------------------------------------------------------------------------- #
# (a) dog-name test — cues_text trigram should surface the right fact
# --------------------------------------------------------------------------- #
def test_dog_name_via_cue_trigram(conn):
    """Cue retrieval must surface a fact whose retrieval_cues contain
    "dog name" / "pet name" / "Max" even when the cosine similarity is
    weak (the user spec's flagship Phase-2 case)."""
    dog_fact = Fact(
        summary="User has a labradoodle named Max who ate the couch.",
        fact_type="event",
        emotional_tone="tender",
        salience=6.0,
        retrieval_cues=["dog name", "pet name", "labradoodle", "Max"],
        entity_keys=["pet:dog", "entity:max"],
    )
    _ingest_fact(conn, user_id="dog_user", fact=dog_fact,
                 anchor="user labradoodle max")
    # A few distractor facts at orthogonal embedding positions
    _ingest_fact(conn, user_id="dog_user",
                 fact=Fact(summary="User likes pizza.", salience=2,
                           retrieval_cues=["food preference"]),
                 anchor="pizza food")
    _ingest_fact(conn, user_id="dog_user",
                 fact=Fact(summary="User watched a movie tuesday.", salience=2,
                           retrieval_cues=["movie night"]),
                 anchor="movie night tuesday")

    top, stats = _hybrid(conn, "what's my dog's name?", user_id="dog_user")
    assert top, "hybrid retrieval returned nothing"
    contents = [ep.content for ep in top]
    assert any("labradoodle named Max" in c for c in contents), \
        f"dog fact not retrieved; got {contents}"
    # The cue retriever should have fired
    assert stats["candidates_cues"] >= 1


# --------------------------------------------------------------------------- #
# (b) grief cue test
# --------------------------------------------------------------------------- #
def test_grief_cue_retrieval(conn):
    """A paraphrased emotional probe ("I saw an old man and broke down")
    must reach a fact whose summary uses different surface words but whose
    cues include "old man reminded me of dad"."""
    grief_fact = Fact(
        summary="User's father died of a heart attack.",
        fact_type="event",
        emotional_tone="distress",
        salience=10.0,
        retrieval_cues=["grief", "bereavement", "father loss",
                        "old man reminded me of dad"],
        entity_keys=["relationship:father"],
    )
    _ingest_fact(conn, user_id="grief_user", fact=grief_fact,
                 anchor="user father died heart attack")
    # filler
    _ingest_fact(conn, user_id="grief_user",
                 fact=Fact(summary="User had toast for breakfast.", salience=1,
                           retrieval_cues=["breakfast"]),
                 anchor="toast breakfast")

    top, stats = _hybrid(conn, "I saw an old man and broke down",
                         user_id="grief_user")
    contents = [ep.content for ep in top]
    assert any("father died" in c.lower() for c in contents), \
        f"grief fact not retrieved; got {contents}"


# --------------------------------------------------------------------------- #
# (c) entity overlap test
# --------------------------------------------------------------------------- #
def test_entity_overlap_retrieval(conn):
    """When the query mentions a company name, the entity-overlap retriever
    must surface a fact whose ``entity_keys`` contain that company."""
    halcyon_fact = Fact(
        summary="User got hired at Halcyon.",
        fact_type="event",
        emotional_tone="hopeful",
        salience=8.0,
        retrieval_cues=["new job", "career win"],
        entity_keys=["company:halcyon", "org:halcyon"],
    )
    _ingest_fact(conn, user_id="job_user", fact=halcyon_fact,
                 anchor="user hired halcyon")
    _ingest_fact(conn, user_id="job_user",
                 fact=Fact(summary="User has a cat named Whiskers.", salience=3,
                           retrieval_cues=["cat name"],
                           entity_keys=["pet:cat", "entity:whiskers"]),
                 anchor="cat whiskers")

    # Either path should work — we'll search by the entity name itself.
    top, stats = _hybrid(conn, "where did I get hired at Halcyon?",
                         user_id="job_user")
    contents = [ep.content for ep in top]
    assert any("Halcyon" in c for c in contents), \
        f"halcyon fact not retrieved; got {contents}"


# --------------------------------------------------------------------------- #
# (d) filler suppression — high-salience must outrank low-salience trivia
# --------------------------------------------------------------------------- #
def test_filler_suppressed_by_state_tilt(conn):
    """A high-salience grief fact must outrank a low-salience filler in the
    final top-k, even when both retrievers see them.  This is the
    write-time-salience invariant the spec calls out (the R(m) tilt is the
    last step in hybrid_retrieve)."""
    _ingest_fact(conn, user_id="filler_user",
                 fact=Fact(summary="User's mother died last spring.",
                           salience=10, emotional_tone="distress",
                           retrieval_cues=["mother loss", "grief"]),
                 anchor="mother died spring")
    _ingest_fact(conn, user_id="filler_user",
                 fact=Fact(summary="User mentioned the post office.",
                           salience=1, retrieval_cues=["post office hours"]),
                 anchor="post office hours")

    top, _ = _hybrid(conn, "tell me about my mother",
                     user_id="filler_user", top_k=2)
    # Mother fact must come first; filler should be far below (or absent).
    assert "mother died" in top[0].content.lower()


def test_origin_retrieval_prefers_earliest_arc_boundary(conn):
    """Origin-aware retrieval should recover the beginning of an arc, not only
    the most recent or highest-salience trace."""
    early = Fact(
        summary="Trace [work_career / avoidance / origin / 2024-10-15]: User ignored workplace harassment at first.",
        fact_type="trace",
        event_type="work_career",
        salience=7,
        retrieval_cues=["work career arc", "origin of work career", "initial response"],
        entity_keys=[
            "arc:work_career", "role:avoidance", "phase:origin",
            "label:conflict_marker", "time:2024-10-15",
        ],
    )
    late = Fact(
        summary="Trace [work_career / escalation / transition / 2025-10-28]: Work tension escalated much later.",
        fact_type="trace",
        event_type="work_career",
        salience=9,
        retrieval_cues=["work career arc", "conflict or contradiction"],
        entity_keys=[
            "arc:work_career", "role:escalation", "phase:transition",
            "label:conflict_marker", "time:2025-10-28",
        ],
    )
    _ingest_fact(conn, user_id="origin_user", fact=late, anchor="late work tension")
    _ingest_fact(conn, user_id="origin_user", fact=early, anchor="early harassment")

    top, stats = _hybrid(
        conn,
        "did I address the harassment issue when it first started?",
        user_id="origin_user",
    )
    assert stats["candidates_origin"] >= 1
    assert "ignored workplace harassment at first" in top[0].content


def test_trajectory_retrieval_returns_causal_chain(conn):
    """Trajectory retrieval should assemble origin, cause, and coping traces
    for sequence/evolution questions."""
    origin = Fact(
        summary="Trace [work_career / avoidance / origin / 2024-10-15]: User avoided addressing workplace harassment.",
        fact_type="trace",
        event_type="work_career",
        salience=7,
        retrieval_cues=["work career arc", "origin of work career"],
        entity_keys=[
            "arc:work_career", "role:avoidance", "phase:origin",
            "cause:workplace_conflict", "rel:caused_by",
            "label:conflict_marker", "time:2024-10-15",
        ],
    )
    stressor = Fact(
        summary="Trace [work_career / escalation / transition / 2024-10-25]: Work harassment made the user feel powerless.",
        fact_type="trace",
        event_type="work_career",
        salience=8,
        retrieval_cues=["workplace conflict", "causal antecedent"],
        entity_keys=[
            "arc:work_career", "role:escalation", "phase:transition",
            "cause:workplace_conflict", "rel:caused_by", "rel:same_arc_as",
            "label:conflict_marker", "time:2024-10-25",
        ],
    )
    coping = Fact(
        summary="Trace [mental_health_coping / attempted_action / transition / 2024-11-05]: Therapy helped the user set boundaries around work stress.",
        fact_type="trace",
        event_type="mental_health_coping",
        salience=8,
        retrieval_cues=["therapy", "triggered coping"],
        entity_keys=[
            "arc:mental_health_coping", "role:attempted_action",
            "phase:transition", "coping:therapy", "coping:boundary_setting",
            "cause:workplace_conflict", "rel:triggered_coping",
            "rel:coping_response", "time:2024-11-05",
        ],
    )
    distractor = Fact(
        summary="Trace [mental_health_coping / onset / origin / 2025-10-28]: The user started a painting class much later.",
        fact_type="trace",
        event_type="mental_health_coping",
        salience=9,
        retrieval_cues=["painting", "coping"],
        entity_keys=[
            "arc:mental_health_coping", "role:onset", "phase:origin",
            "coping:creative_expression", "rel:coping_response",
            "time:2025-10-28",
        ],
    )
    _ingest_fact(conn, user_id="trajectory_user", fact=distractor,
                 anchor="later painting class")
    _ingest_fact(conn, user_id="trajectory_user", fact=origin,
                 anchor="early work harassment")
    _ingest_fact(conn, user_id="trajectory_user", fact=stressor,
                 anchor="work harassment powerless")
    _ingest_fact(conn, user_id="trajectory_user", fact=coping,
                 anchor="therapy boundaries work stress")

    top, stats = _hybrid(
        conn,
        "what was the sequence of events that led me to start therapy?",
        user_id="trajectory_user",
    )
    contents = "\n".join(ep.content for ep in top)
    assert stats["candidates_trajectory"] >= 3
    assert "avoided addressing workplace harassment" in contents
    assert "made the user feel powerless" in contents
    assert "Therapy helped the user set boundaries" in contents


def test_trajectory_retrieval_prefers_bridge_fact(conn):
    bridge = Fact(
        summary="Bridge [mental_health_coping / triggered_coping / 2024-11-05]: The user's therapy emerged as a coping response after workplace conflict and relationship loss.",
        fact_type="bridge",
        event_type="mental_health_coping",
        salience=8,
        retrieval_cues=["causal bridge", "what led to coping", "therapy"],
        entity_keys=[
            "arc:mental_health_coping", "role:attempted_action",
            "phase:transition", "rel:caused_by", "rel:triggered_coping",
            "rel:coping_response", "cause:workplace_conflict",
            "cause:relationship_loss", "coping:therapy", "time:2024-11-05",
        ],
    )
    support = Fact(
        summary="Trace [mental_health_coping / onset / origin / 2025-01-05]: Therapy has been helpful later.",
        fact_type="trace",
        event_type="mental_health_coping",
        salience=9,
        retrieval_cues=["therapy", "coping"],
        entity_keys=[
            "arc:mental_health_coping", "role:onset", "phase:origin",
            "coping:therapy", "rel:coping_response", "time:2025-01-05",
        ],
    )
    _ingest_fact(conn, user_id="bridge_user", fact=support,
                 anchor="therapy helpful later")
    _ingest_fact(conn, user_id="bridge_user", fact=bridge,
                 anchor="therapy because workplace conflict relationship loss")

    top, stats = _hybrid(
        conn,
        "what was the sequence of events that led me to start therapy?",
        user_id="bridge_user",
    )
    assert stats["candidates_trajectory"] >= 1
    assert top[0].fact_type == "bridge"
    assert "workplace conflict and relationship loss" in top[0].content


# --------------------------------------------------------------------------- #
# (e) scenario isolation (regression for the earlier cross-scenario leak)
# --------------------------------------------------------------------------- #
def test_scenario_isolation_under_hybrid(conn):
    """Hybrid retrieval must namespace by ``user_id`` on every candidate
    source, including the new pg_trgm + array-overlap ones.  A grief fact
    must not appear in the job_loss user's retrieval."""
    _ingest_fact(conn, user_id="grief",
                 fact=Fact(summary="User's father died of a heart attack.",
                           salience=10, retrieval_cues=["father loss"],
                           entity_keys=["relationship:father"]),
                 anchor="father died heart attack")
    _ingest_fact(conn, user_id="job_loss",
                 fact=Fact(summary="User was laid off from their tech job.",
                           salience=9, retrieval_cues=["job loss"],
                           entity_keys=["job"]),
                 anchor="laid off tech job")

    top_job, _ = _hybrid(conn, "what happened to my dad?",
                         user_id="job_loss")
    # The job_loss namespace has no grief fact; the dad-mentioning query
    # must NOT pull one from another namespace.
    for ep in top_job:
        assert ep.user_id == "job_loss"
        assert "father" not in ep.content.lower()

    top_grief, _ = _hybrid(conn, "what happened to my dad?",
                           user_id="grief")
    assert any("father" in ep.content.lower() for ep in top_grief)


# --------------------------------------------------------------------------- #
# (f) graceful fallback — pg_trgm / spaCy / cross-encoder missing
# --------------------------------------------------------------------------- #
def test_hybrid_survives_text_retriever_failure(conn, monkeypatch):
    """If ``fact_text_search`` raises (e.g. pg_trgm missing on a deployment),
    hybrid retrieval still returns semantic candidates and does not crash."""
    _ingest_fact(conn, user_id="resilient",
                 fact=Fact(summary="User's father died of a heart attack.",
                           salience=10, retrieval_cues=["father loss"]),
                 anchor="father died")

    # Force both trigram retrievers to act like pg_trgm is missing.
    monkeypatch.setattr(store, "fact_text_search", lambda *a, **k: [])
    monkeypatch.setattr(store, "fact_cue_search", lambda *a, **k: [])

    top, stats = _hybrid(conn, "tell me about my father",
                         user_id="resilient")
    assert top, "hybrid must keep returning semantic candidates on fallback"
    assert any("father died" in ep.content.lower() for ep in top)
    assert stats["candidates_summary"] == 0
    assert stats["candidates_cues"] == 0
    assert stats["candidates_semantic"] >= 1


def test_extract_query_entities_handles_missing_spacy():
    """The regex layer alone must extract enough entity keys to be useful —
    spaCy is optional."""
    keys = taco_rerank.extract_query_entities("My partner Maya and I broke up")
    assert "relationship:partner" in keys
    # "Maya" mid-sentence picked up by the proper-noun regex
    assert any("maya" in k for k in keys)


def test_extract_query_cues_basic_paths():
    """Sanity: each cue family in the map fires on a representative probe."""
    assert "dog name" in taco_rerank.extract_query_cues("what's my dog's name?")
    assert any("grief" in c for c in
               taco_rerank.extract_query_cues("I saw an old man"))
    assert "interview" in taco_rerank.extract_query_cues(
        "any tips for my interview tomorrow?")


def test_rrf_fuse_basic_correctness():
    """A fact ranked first in two retrievers should outrank a fact ranked
    second in one."""
    from taco.memory.episode import Episode
    a = Episode(id=1, content="A")
    b = Episode(id=2, content="B")
    c = Episode(id=3, content="C")
    # ranking A in two lists; B only in one (rank 1); C in one (rank 2)
    fused = taco_rerank.rrf_fuse([[a, c], [a, b]], k=60)
    assert fused[0].id == 1, "A wins on cumulative RRF"
    # B and C both appear once at rank 1 and rank 2 respectively
    ids = [ep.id for ep in fused]
    assert set(ids) == {1, 2, 3}
