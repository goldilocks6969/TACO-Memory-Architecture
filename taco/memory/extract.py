"""Tiered fact extraction (Phase 1).

Two tiers keep cost bounded — every BYO-key user pays for each LLM call:

  • `light_extract` runs on EVERY turn. One cheap JSON call reads affect
    (emotional / vulnerability / salience / tone) plus a couple of entities and
    retrieval cues. It replaces the old standalone `llm.analyze` in the pipeline.

  • `full_extract` runs only when a turn is salient enough to be worth storing as
    structured facts. It produces the rich schema — `emotional_cause`,
    `user_belief`, `event_type`, capped `retrieval_cues`, open-thread fields —
    the signals that make a fact retrievable across paraphrased future queries.

Both tiers fall back to deterministic heuristics under TACO_MOCK=1 so the whole
pipeline runs keyless; with a key but TACO_MOCK unset they fail loud (see
`llm.available`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config, llm
from ..retry import with_retries
from .fact import Fact


@dataclass
class LightExtract:
    """Cheap per-turn read: affect + a few entities + ≤2 retrieval cues."""
    emotional: float
    vulnerability: float
    salience: float
    tone: str
    entities: List[str] = field(default_factory=list)
    retrieval_cues: List[str] = field(default_factory=list)


@dataclass
class FullExtract:
    """The structured facts pulled from a salient turn."""
    facts: List[Fact] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Heuristic fallbacks (TACO_MOCK=1) — deterministic, keyless
# --------------------------------------------------------------------------- #
_REL_ROLES = (
    "partner", "wife", "husband", "girlfriend", "boyfriend", "fiancé", "fiancee",
    "mother", "mom", "father", "dad", "son", "daughter", "sister", "brother",
    "friend", "boss", "manager", "therapist", "doctor", "roommate", "cofounder",
)

# keyword → a reformulated cue someone might later use to ask about this
_CUE_MAP = (
    (("died", "passed away", "funeral", "heart attack"), "grief over losing someone"),
    (("breakup", "broke up", "broke it off"), "the breakup"),
    (("divorce",), "the divorce"),
    (("laid off", "fired", "let go", "lost my job"), "losing the job"),
    (("interview",), "interview anxiety"),
    (("got the job", "hired", "better pay", "raise"), "career outcome"),
    (("diagnos", "cancer", "metformin", "diabetes", "a1c"), "the health diagnosis"),
    (("dropped", "progress", "small win"), "progress update"),
    (("relapse", "sober", "quit drinking", "recovery"), "the recovery struggle"),
    (("fraud", "imposter", "not good enough", "failing"), "feeling not good enough"),
    (("promotion", "got the job", "raise"), "the career win"),
    (("called me", "nickname"), "personal nickname"),
    (("reached out", "wants to talk"), "unresolved relationship thread"),
)

_EVENT_KEYWORDS = (
    (("died", "passed away", "funeral", "heart attack", "diagnos"), "health"),
    (("breakup", "broke up", "divorce", "partner", "marriage"), "relationship"),
    (("laid off", "fired", "promotion", "job", "interview", "work"), "goal"),
    (("scared", "terrified", "anxious", "afraid", "worried"), "fear"),
    (("fight", "argument", "conflict", "criticized"), "conflict"),
    (("plan", "going to", "next week", "tomorrow", "friday"), "plan"),
)

# A thread = a future time cue AND a commitment/event noun (so "interview on
# friday" is a thread but "post office hours on fridays" is not).
_THREAD_TIME = ("tomorrow", "tonight", "this weekend", "next week", "monday",
                "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_THREAD_NOUN = ("interview", "meeting", "appointment", "exam", "midterm", "deadline",
                "surgery", "date", "trip", "wedding", "presentation", "review",
                "funeral", "hearing", "court", "ceremony", "call")


def _is_thread(text: str) -> bool:
    t = text.lower()
    return any(c in t for c in _THREAD_TIME) and any(n in t for n in _THREAD_NOUN)


def _heuristic_entities(text: str) -> List[str]:
    out: List[str] = []
    roles = "|".join(_REL_ROLES)
    for m in re.finditer(rf"\bmy ({roles})\b(?:[, ]+(?:named|called)\s+)?\s*([A-Z][a-z]+)?",
                         text):
        out.append(f"relationship:{m.group(1).lower()}")
        if m.group(2):
            out.append(f"person:{m.group(2).lower()}")
    # standalone proper nouns (not sentence-initial) as coarse entities
    for m in re.finditer(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-z]{2,})\b", text):
        tok = m.group(1).lower()
        if tok not in ("i", "i'm") and f"person:{tok}" not in out:
            out.append(f"entity:{tok}")
    # de-dup, cap
    seen, uniq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e); uniq.append(e)
    return uniq[:5]


def _heuristic_cues(text: str, limit: int) -> List[str]:
    t = text.lower()
    cues: List[str] = []
    for keys, cue in _CUE_MAP:
        if any(k in t for k in keys) and cue not in cues:
            cues.append(cue)
        if len(cues) >= limit:
            break
    return cues[:limit]


def _dedupe(items: List[str], limit: int) -> List[str]:
    seen, out = set(), []
    for item in items:
        if not item:
            continue
        key = item.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
        if len(out) >= limit:
            break
    return out


def _heuristic_event_type(text: str) -> str:
    t = text.lower()
    for keys, et in _EVENT_KEYWORDS:
        if any(k in t for k in keys):
            return et
    return "event"


def _heuristic_light(text: str) -> LightExtract:
    a = llm._heuristic_analyze(text)
    return LightExtract(
        emotional=a["emotional"], vulnerability=a["vulnerability"],
        salience=a["salience"], tone=a["tone"],
        entities=_heuristic_entities(text),
        retrieval_cues=_heuristic_cues(text, limit=2),
    )


def _heuristic_full(text: str, light: LightExtract) -> FullExtract:
    is_thread = _is_thread(text)
    fact = Fact(
        summary=text.strip(),
        fact_type="thread" if is_thread else "event",
        event_type=_heuristic_event_type(text),
        emotional_tone=light.tone,
        salience=light.salience,
        retrieval_cues=_heuristic_cues(text, limit=3),
        entity_keys=light.entities,
        thread_status="unresolved" if is_thread else None,
        confidence=0.6,
    )
    return FullExtract(facts=[fact])


# --------------------------------------------------------------------------- #
# Light-fact synthesis — the cheap, hang-free fact builder used by the LIGHT
# extraction mode (and by AUTO as a baseline before the LLM upgrade attempt).
# No LLM call: only the LightExtract output plus the deterministic
# event/thread heuristics already used by the keyless fallback.
# --------------------------------------------------------------------------- #
_LIGHT_SUMMARY_MAX_CHARS = 240


def _condense(text: str, max_chars: int = _LIGHT_SUMMARY_MAX_CHARS) -> str:
    """Trim ``text`` to ``max_chars`` graphemes, on a word boundary, with an
    ellipsis when truncated."""
    s = " ".join(text.strip().split())
    if len(s) <= max_chars:
        return s
    cut = s[: max_chars - 1].rstrip()
    sp = cut.rfind(" ")
    if sp > max_chars - 40:  # don't lose too much to alignment
        cut = cut[:sp].rstrip()
    return cut + "…"


def light_fact(text: str, light: LightExtract,
               max_chars: int = _LIGHT_SUMMARY_MAX_CHARS) -> Fact:
    """Build one usable structured Fact directly from ``light`` and the raw
    message, with no LLM round-trip.

    Used by the LIGHT extraction mode as the canonical fact, and by AUTO as
    the always-stored baseline before any (optional) full_extract upgrade
    attempt.  Carries the schema fields the eval harness needs (salience,
    tone, retrieval_cues, entity_keys, thread_status) populated from the
    cheap signals the light tier already produced.
    """
    is_thread = _is_thread(text)
    # Prefer the LLM-suggested cues; fall back to the heuristic cue map so a
    # message that didn't surface any cues still gets a useful one.
    cues = list(dict.fromkeys(
        list(light.retrieval_cues or []) + _heuristic_cues(text, limit=3)
    ))[:3]
    return Fact(
        summary=_condense(text, max_chars=max_chars),
        fact_type="thread" if is_thread else "event",
        event_type="thread" if is_thread else _heuristic_event_type(text),
        emotional_tone=light.tone,
        emotional_cause=None,
        user_belief=None,
        salience=light.salience,
        retrieval_cues=cues,
        entity_keys=list(light.entities)[:6],
        validity="current",
        thread_status="unresolved" if is_thread else None,
        due_date=None,
        confidence=0.6,
    )


def _anchor_fact(summary: str, light: LightExtract, *,
                 fact_type: str = "event", event_type: str = "event",
                 cues: Optional[List[str]] = None,
                 entities: Optional[List[str]] = None,
                 salience_floor: float = 4.0,
                 thread_status: Optional[str] = None) -> Fact:
    """Build a durable anchor fact from a deterministic pattern.

    Anchor facts are not a replacement for salience; they are the informational
    half of write-time salience. A nickname, outcome, medication, date, or
    progress marker may be emotionally quiet in isolation, but it is exactly the
    kind of continuity anchor a state-aware memory layer should preserve when it
    appears inside an ongoing high-salience thread.
    """
    merged_cues = _dedupe(list(cues or []) + list(light.retrieval_cues or []), 3)
    merged_entities = _dedupe(list(entities or []) + list(light.entities or []), 6)
    return Fact(
        summary=_condense(summary),
        fact_type=fact_type,
        event_type=event_type,
        emotional_tone=light.tone,
        salience=max(light.salience, salience_floor),
        retrieval_cues=merged_cues,
        entity_keys=merged_entities,
        validity="current",
        thread_status=thread_status,
        confidence=0.7,
    )


def anchor_facts(text: str, light: LightExtract) -> List[Fact]:
    """Extract cheap durable anchors from one turn.

    This is the production-oriented middle tier between raw light facts and an
    LLM-backed full extraction pass. It keeps the write path hang-free while
    preserving the concrete details that long-horizon continuity depends on.
    """
    s = " ".join(text.strip().split())
    t = s.lower()
    facts: List[Fact] = []

    # Personal names / nicknames: "he always called me 'kiddo'".
    for m in re.finditer(r"\b(?:called me|calls me|used to call me)\s+['\"]?([^'\".,!?]+)['\"]?", s, re.I):
        nick = m.group(1).strip()
        if nick:
            facts.append(_anchor_fact(
                f"The user was called '{nick}' by someone important to them.",
                light,
                fact_type="relationship",
                event_type="identity",
                cues=["nickname", "what they were called", nick],
                salience_floor=6.5,
            ))

    # Relationship updates: "Sam reached out. wants to 'talk'."
    if "reached out" in t and ("talk" in t or "call" in t or "meet" in t):
        facts.append(_anchor_fact(
            s,
            light,
            fact_type="thread",
            event_type="relationship",
            cues=["unresolved relationship thread", "reached out", "wants to talk"],
            salience_floor=6.5,
            thread_status="unresolved",
        ))

    # Career outcomes supersede prior interview/job-loss uncertainty.
    if re.search(r"\b(got the job|hired|accepted an offer)\b", t):
        org = None
        m = re.search(r"\b(?:at|with)\s+([A-Z][A-Za-z0-9&.-]+)", s)
        if m:
            org = m.group(1)
        summary = f"The user got hired{f' at {org}' if org else ''}."
        if "better pay" in t or "raise" in t:
            summary += " The role improves their pay."
        entities = [f"org:{org.lower()}"] if org else []
        facts.append(_anchor_fact(
            summary,
            light,
            fact_type="event",
            event_type="achievement",
            cues=["career outcome", "new job", "hired", org or ""],
            entities=entities,
            salience_floor=7.5,
        ))

    # Health progress markers: "my A1C dropped..." should answer progress probes.
    if "a1c" in t and any(k in t for k in ("dropped", "down", "lower", "improved", "better")):
        facts.append(_anchor_fact(
            "The user's A1C dropped at a recheck, indicating health progress.",
            light,
            fact_type="event",
            event_type="health",
            cues=["health progress", "A1C improved", "diabetes progress"],
            salience_floor=7.0,
        ))

    # Baby/name/gender anchors.
    m = re.search(r"\b(?:naming|name)\s+(?:her|him|the baby)\s+([A-Z][a-z]+)", s)
    if not m:
        m = re.search(r"\bthinking of naming (?:her|him|the baby)\s+([A-Z][a-z]+)", s)
    if m:
        name = m.group(1)
        facts.append(_anchor_fact(
            f"The user is considering the baby name {name}.",
            light,
            fact_type="state",
            event_type="identity",
            cues=["baby name", "name considered", name],
            entities=[f"person:{name.lower()}"],
            salience_floor=7.0,
        ))
    if "it's a girl" in t or "it is a girl" in t:
        facts.append(_anchor_fact(
            "The user is expecting a girl.",
            light,
            fact_type="state",
            event_type="identity",
            cues=["baby gender", "girl", "expecting"],
            salience_floor=6.5,
        ))

    # Relocation / housing settledness.
    if "found an apartment" in t:
        place = None
        m = re.search(r"\bin\s+([A-Z][A-Za-z]+)", s)
        if m:
            place = m.group(1)
        facts.append(_anchor_fact(
            f"The user found an apartment{f' in {place}' if place else ''}.",
            light,
            fact_type="event",
            event_type="plan",
            cues=["housing settled", "found apartment", place or ""],
            entities=[f"place:{place.lower()}"] if place else [],
            salience_floor=6.5,
        ))

    # Academic recovery / grade outcomes.
    m = re.search(r"\bgot (?:a |an )?([A-F][+-]?)\b.*\b(midterm|exam|test|class)\b", s, re.I)
    if not m:
        m = re.search(r"\b([A-F][+-]?)\b.*\b(midterm|exam|test|class)\b", s, re.I)
    if m and any(k in t for k in ("got", "understand", "midterm", "exam")):
        grade = m.group(1).upper()
        facts.append(_anchor_fact(
            f"The user got a {grade} on the recent academic assessment.",
            light,
            fact_type="event",
            event_type="achievement",
            cues=["grade update", "academic progress", grade],
            salience_floor=6.5,
        ))

    return facts


def light_facts(text: str, light: LightExtract) -> List[Fact]:
    """Return deterministic facts for LIGHT mode.

    The first fact is the broad event summary. Anchor facts add structured
    retrieval handles for continuity-critical details. De-dup by summary so a
    simple turn still stores exactly one fact.
    """
    facts = [light_fact(text, light)]
    facts.extend(anchor_facts(text, light))
    out: List[Fact] = []
    by_summary: Dict[str, Fact] = {}
    for f in facts:
        key = f.summary.lower()
        existing = by_summary.get(key)
        if existing is not None:
            existing.salience = max(existing.salience, f.salience)
            existing.retrieval_cues = _dedupe(
                existing.retrieval_cues + f.retrieval_cues, 3)
            existing.entity_keys = _dedupe(
                existing.entity_keys + f.entity_keys, 6)
            if f.fact_type and existing.fact_type == "event":
                existing.fact_type = f.fact_type
            if f.event_type and existing.event_type == "event":
                existing.event_type = f.event_type
            if f.thread_status:
                existing.thread_status = f.thread_status
            continue
        by_summary[key] = f
        out.append(f)
    return out


def has_memory_anchor(text: str, light: LightExtract) -> bool:
    """Whether a low-arousal turn still carries a durable continuity anchor."""
    return bool(anchor_facts(text, light))


# --------------------------------------------------------------------------- #
# LLM tiers
# --------------------------------------------------------------------------- #
_LIGHT_SYS = (
    "You are the interoceptive sensor and entity tagger of a cognitive memory "
    "layer. Read the user's message and respond with strict JSON only:\n"
    '{"emotional": <0-100>, "vulnerability": <0-100>, "salience": <1-10>, '
    '"tone": "<one lowercase word>", "entities": ["type:value", ...], '
    '"retrieval_cues": ["...", "..."]}\n'
    "emotional = affective arousal; vulnerability = openness/fragility/disclosure; "
    "salience = worth remembering long-term (10=breakup/death/identity, 1=small talk). "
    "entities are typed keys like person:maya, role:interview, org:nimbus, place:lisbon. "
    "retrieval_cues are AT MOST 2 alternative ways someone might later ask about this "
    "— reformulations of the underlying meaning, not synonyms of the words used."
)

_FULL_SYS = (
    "You extract durable, structured FACTS from a user's message for a long-term "
    "memory system. Return strict JSON: {\"facts\": [ {...}, ... ]}. Produce 1-3 "
    "facts; fewer is better. Each fact:\n"
    '{"summary": "<third-person, durable statement>", '
    '"fact_type": "preference|event|relationship|state|thread", '
    '"event_type": "conflict|goal|preference|fear|plan|relationship|achievement|failure|health|money|identity|thread", '
    '"emotional_tone": "<word>", '
    '"emotional_cause": "<why it carries emotional charge, or empty>", '
    '"user_belief": "<the self-belief it implies, e.g. \'I mess up opportunities\', or empty>", '
    '"retrieval_cues": ["<alt phrasing>", ...], '
    '"entity_keys": ["type:value", ...], '
    '"validity": "current|outdated|uncertain", '
    '"thread_status": "unresolved|in_progress|resolved or null", '
    '"due_date": "<ISO date if there is a future commitment/deadline, else null>"}\n'
    "retrieval_cues: AT MOST 3 reformulations of the meaning someone might ask about "
    "later (not word synonyms). Use fact_type='thread' with thread_status='unresolved' "
    "for an open loop with a future action or time (e.g. an upcoming interview)."
)


# One-shot announce flag so the eval log doesn't print the "local light_extract"
# notice on every turn — once per process is enough to confirm the path.
_LIGHT_LOCAL_ANNOUNCED = False


def _maybe_announce_local() -> None:
    global _LIGHT_LOCAL_ANNOUNCED
    if not _LIGHT_LOCAL_ANNOUNCED:
        print("[extract] local light_extract used "
              "(TACO_EVAL_LIGHT_EXTRACT_LOCAL=1)", flush=True)
        _LIGHT_LOCAL_ANNOUNCED = True


def light_extract(text: str) -> LightExtract:
    """Cheap per-turn extraction.

    Three paths, in priority order:

    * ``config.LIGHT_EXTRACT_LOCAL`` (eval default) — always use the keyless
      heuristic, even when an API key is configured.  The live benchmark
      depends on this: the LLM-backed light tier can wedge during long
      scenario ingests, and the heuristic carries every field the downstream
      ``light_fact`` builder needs (salience, tone, vulnerability,
      retrieval_cues, entities).
    * ``llm.available()`` false (TACO_MOCK=1 / no key) — heuristic, as before.
    * Otherwise — one cheap LLM JSON call, bounded by ``CallTimeout``.
    """
    if config.LIGHT_EXTRACT_LOCAL:
        _maybe_announce_local()
        return _heuristic_light(text)
    if not llm.available():
        return _heuristic_light(text)
    resp = with_retries(lambda: llm._client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": _LIGHT_SYS},
                  {"role": "user", "content": text}],
        temperature=0,
        response_format={"type": "json_object"},
    ), label="extract.light_extract")
    try:
        d = json.loads(resp.choices[0].message.content)
    except Exception:
        return _heuristic_light(text)
    return LightExtract(
        emotional=float(d.get("emotional", 0)),
        vulnerability=float(d.get("vulnerability", 0)),
        salience=max(1.0, min(10.0, float(d.get("salience", 1)))),
        tone=str(d.get("tone", "neutral")),
        entities=[str(e) for e in (d.get("entities") or [])][:5],
        retrieval_cues=[str(c) for c in (d.get("retrieval_cues") or [])][:2],
    )


def full_extract(text: str, light: LightExtract) -> FullExtract:
    """Rich extraction for salient turns. Heuristic under TACO_MOCK=1, else one call."""
    if not llm.available():
        return _heuristic_full(text, light)
    resp = with_retries(lambda: llm._client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": _FULL_SYS},
                  {"role": "user", "content": text}],
        temperature=0.2,
        response_format={"type": "json_object"},
    ), label="extract.full_extract")
    try:
        raw = json.loads(resp.choices[0].message.content).get("facts", [])
    except Exception:
        return _heuristic_full(text, light)
    facts: List[Fact] = []
    for f in raw:
        summary = str(f.get("summary", "")).strip()
        if not summary:
            continue
        facts.append(_fact_from_json(f, light))
    return FullExtract(facts=facts or _heuristic_full(text, light).facts)


def _fact_from_json(f: Dict, light: LightExtract) -> Fact:
    due = f.get("due_date")
    return Fact(
        summary=str(f["summary"]).strip(),
        fact_type=_clean(f.get("fact_type")) or "event",
        event_type=_clean(f.get("event_type")),
        emotional_tone=_clean(f.get("emotional_tone")) or light.tone,
        emotional_cause=_clean(f.get("emotional_cause")),
        user_belief=_clean(f.get("user_belief")),
        salience=light.salience,
        retrieval_cues=[str(c) for c in (f.get("retrieval_cues") or [])][:3],
        entity_keys=[str(e) for e in (f.get("entity_keys") or [])][:6],
        validity=_clean(f.get("validity")) or "current",
        thread_status=_clean(f.get("thread_status")),
        due_date=_parse_iso(due) if due else None,
        confidence=0.7,
    )


def _clean(v) -> Optional[str]:
    """Normalise model output: empty/null-ish strings → None."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "n/a", ""):
        return None
    return s


def _parse_iso(v):
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
