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


def test_light_extract_emits_structural_labels_and_temporal_anchor():
    text = "[2025-02-15 session p1_conv_9 turn 2] I started painting to manage my stress"
    le = extract.light_extract(text)
    assert "temporal_event" in le.structural_labels
    assert "state_change" in le.structural_labels
    assert "coping_strategy" in le.structural_labels
    assert le.temporal_anchor == "2025-02-15"


def test_light_fact_materializes_structural_labels_for_retrieval():
    text = "[2024-10-15 session p1_conv_1 turn 7] I ignored the harassment at first hoping it would stop"
    le = extract.light_extract(text)
    fact = extract.light_fact(text, le)
    assert "conflict_marker" in le.structural_labels
    assert "state_change" in le.structural_labels
    assert "label:conflict_marker" in fact.entity_keys
    assert "time:2024-10-15" in fact.entity_keys
    assert any("conflict" in cue for cue in fact.retrieval_cues)


def test_light_facts_emit_production_arc_trace():
    text = (
        "[2024-10-15 session p1 turn 7] I ignored the workplace harassment "
        "at first hoping it would stop"
    )
    le = extract.light_extract(text)
    facts = extract.light_facts(text, le)
    trace = next(f for f in facts if f.fact_type == "trace")
    assert trace.event_type == "work_career"
    assert "arc:work_career" in trace.entity_keys
    assert "role:avoidance" in trace.entity_keys
    assert "phase:origin" in trace.entity_keys
    assert "label:conflict_marker" in trace.entity_keys
    assert trace.salience >= 7.0


def test_arc_trace_labels_avoidance_from_unaddressed_conflict():
    text = (
        "[2024-10-15 session p1 turn 5] There's been constant workplace "
        "harassment, but I haven’t addressed it."
    )
    le = extract.light_extract(text)
    trace = next(f for f in extract.light_facts(text, le) if f.fact_type == "trace")
    assert "role:avoidance" in trace.entity_keys
    assert "phase:origin" in trace.entity_keys


def test_arc_trace_preserves_origin_even_without_explicit_arc():
    text = (
        "[2024-10-15 session p1 turn 7] It's been months, maybe longer. "
        "I think I ignored it at first because I thought it would stop."
    )
    le = extract.light_extract(text)
    traces = [f for f in extract.light_facts(text, le) if f.fact_type == "trace"]
    assert traces
    assert "arc:open_arc" in traces[0].entity_keys
    assert "phase:origin" in traces[0].entity_keys
    assert "role:avoidance" in traces[0].entity_keys


def test_arc_trace_stitches_causal_work_conflict():
    text = (
        "[2024-10-25 session p1 turn 9] I keep thinking about the harassment "
        "at work. Ignoring it for so long just made me feel powerless."
    )
    le = extract.light_extract(text)
    trace = next(f for f in extract.light_facts(text, le) if f.fact_type == "trace")
    assert "cause:workplace_conflict" in trace.entity_keys
    assert "rel:caused_by" in trace.entity_keys
    assert "rel:same_arc_as" in trace.entity_keys


def test_arc_trace_stitches_coping_response():
    text = (
        "[2025-05-20 session p1 turn 11] I started painting like they suggested "
        "and it helps me manage my stress."
    )
    le = extract.light_extract(text)
    trace = next(f for f in extract.light_facts(text, le) if f.fact_type == "trace")
    assert "arc:mental_health_coping" in trace.entity_keys
    assert "coping:creative_expression" in trace.entity_keys
    assert "rel:triggered_coping" in trace.entity_keys
    assert "rel:coping_response" in trace.entity_keys


def test_arc_trace_stitches_therapy_to_coping():
    text = (
        "[2024-11-05 session p1 turn 21] Therapy has been grounding and helped "
        "me set boundaries around work stress."
    )
    le = extract.light_extract(text)
    trace = next(f for f in extract.light_facts(text, le) if f.fact_type == "trace")
    assert "coping:therapy" in trace.entity_keys
    assert "coping:boundary_setting" in trace.entity_keys
    assert "cause:workplace_conflict" in trace.entity_keys
    assert "rel:triggered_coping" in trace.entity_keys


def test_session_metadata_does_not_count_as_therapy():
    text = (
        "[2024-09-20 session esc786 turn 4] No, the company hasn't set up "
        "a system for passing along info that way."
    )
    le = extract.light_extract(text)
    facts = extract.light_facts(text, le)
    assert all("coping:therapy" not in (fact.entity_keys or []) for fact in facts)


def test_bridge_fact_synthesizes_causal_coping_summary():
    prior = extract.Fact(
        id=11,
        summary="Trace [work_career / avoidance / origin / 2024-10-15]: User avoided addressing workplace harassment.",
        fact_type="trace",
        event_type="work_career",
        salience=7,
        entity_keys=[
            "arc:work_career", "cause:workplace_conflict",
            "rel:caused_by", "time:2024-10-15",
        ],
    )
    current = extract.Fact(
        summary="Trace [mental_health_coping / attempted_action / transition / 2024-11-05]: Therapy helped with work stress.",
        fact_type="trace",
        event_type="mental_health_coping",
        salience=8,
        emotional_tone="concerned",
        entity_keys=[
            "arc:mental_health_coping", "coping:therapy",
            "rel:triggered_coping", "rel:coping_response",
            "cause:workplace_conflict", "time:2024-11-05",
        ],
    )
    bridge = extract.bridge_fact(current, [prior])
    assert bridge is not None
    assert bridge.fact_type == "bridge"
    assert "workplace harassment" in bridge.summary
    assert "therapy" in bridge.summary
    assert "rel:triggered_coping" in bridge.entity_keys
    assert "cause:workplace_conflict" in bridge.entity_keys
    assert "coping:therapy" in bridge.entity_keys
    assert "support_fact:11" in bridge.entity_keys


def test_bridge_fact_rejects_generic_nearby_dialogue():
    prior = extract.Fact(
        id=12,
        summary=(
            "Trace [work_career / escalation / transition / 2024-09-20]: "
            "No, the company hasn't set up a system for passing along info that way."
        ),
        fact_type="trace",
        event_type="work_career",
        salience=7,
        entity_keys=[
            "arc:work_career", "cause:workplace_conflict",
            "rel:caused_by", "time:2024-09-20",
        ],
    )
    current = extract.Fact(
        summary="Trace [mental_health_coping / attempted_action / transition / 2024-09-20]: Therapy felt like a step toward coping.",
        fact_type="trace",
        event_type="mental_health_coping",
        salience=8,
        entity_keys=[
            "arc:mental_health_coping", "coping:therapy",
            "rel:triggered_coping", "rel:coping_response",
            "cause:workplace_conflict", "time:2024-09-20",
        ],
    )
    assert extract.bridge_fact(current, [prior]) is None


def test_bridge_fact_uses_specific_event_snippets():
    prior_work = extract.Fact(
        id=21,
        summary="Trace [work_career / escalation / transition / 2024-10-25]: The user had a breakdown at work after workplace harassment.",
        fact_type="trace",
        event_type="work_career",
        salience=8,
        entity_keys=[
            "arc:work_career", "cause:workplace_conflict",
            "rel:caused_by", "time:2024-10-25",
        ],
    )
    prior_rel = extract.Fact(
        id=20,
        summary="Trace [relationships / outcome / transition / 2024-10-14]: The user went through a breakup.",
        fact_type="trace",
        event_type="relationships",
        salience=8,
        entity_keys=[
            "arc:relationships", "cause:relationship_loss",
            "rel:caused_by", "time:2024-10-14",
        ],
    )
    current = extract.Fact(
        summary="Trace [mental_health_coping / attempted_action / transition / 2024-10-25]: Therapy felt like a step toward coping.",
        fact_type="trace",
        event_type="mental_health_coping",
        salience=8,
        entity_keys=[
            "arc:mental_health_coping", "coping:therapy",
            "rel:triggered_coping", "rel:coping_response",
            "cause:workplace_conflict", "cause:relationship_loss",
            "time:2024-10-25",
        ],
    )
    bridge = extract.bridge_fact(current, [prior_rel, prior_work])
    assert bridge is not None
    assert "workplace harassment" in bridge.summary or "a breakdown" in bridge.summary
    assert "relationship loss" in bridge.summary
    assert "support_fact:20" in bridge.entity_keys
    assert "support_fact:21" in bridge.entity_keys


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
