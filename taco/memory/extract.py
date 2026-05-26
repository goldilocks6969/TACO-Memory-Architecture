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
    (("diagnos", "cancer", "metformin", "diabetes"), "the health diagnosis"),
    (("relapse", "sober", "quit drinking", "recovery"), "the recovery struggle"),
    (("fraud", "imposter", "not good enough", "failing"), "feeling not good enough"),
    (("promotion", "got the job", "raise"), "the career win"),
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


def light_extract(text: str) -> LightExtract:
    """Cheap per-turn extraction. Heuristic under TACO_MOCK=1, else one LLM call."""
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
