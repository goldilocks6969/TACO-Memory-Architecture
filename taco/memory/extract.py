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
    """Cheap per-turn read: affect + entities + structural memory signals."""
    emotional: float
    vulnerability: float
    salience: float
    tone: str
    entities: List[str] = field(default_factory=list)
    retrieval_cues: List[str] = field(default_factory=list)
    structural_labels: List[str] = field(default_factory=list)
    temporal_anchor: Optional[str] = None


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

_STRUCTURAL_LABELS = (
    ("temporal_event", (
        "yesterday", "today", "tomorrow", "last night", "last week", "next week",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    )),
    ("state_change", (
        "started", "began", "stopped", "quit", "decided", "realized",
        "learned", "became", "got better", "improved", "changed", "resigned",
        "broke up", "moved", "moving", "diagnosed", "got the job", "hired",
        "at first", "initially", "eventually", "later",
    )),
    ("conflict_marker", (
        "conflict", "fight", "argument", "harassment", "ignored", "avoid",
        "didn't address", "did not address", "hoping it would stop", "tension",
        "boundary", "complaint", "criticized", "silent treatment",
    )),
    ("relationship_update", (
        "partner", "dating", "relationship", "breakup", "broke up", "reconciled",
        "reached out", "colleague", "friend", "mother", "father", "boss",
    )),
    ("coping_strategy", (
        "therapy", "therapist", "painting", "writing", "journal", "vacation",
        "sabbatical", "coping", "manage my stress", "manage stress", "self-care",
        "mental health", "breathing", "meditation",
    )),
    ("medical_or_safety", (
        "diagnosis", "diagnosed", "doctor", "medication", "metformin", "a1c",
        "panic attack", "chest pain", "hospital", "surgery", "biopsy",
    )),
    ("identity_preference", (
        "i like", "i love", "i hate", "i prefer", "i'm the kind of",
        "i am the kind of", "i want to be", "my values", "important to me",
    )),
    ("uncertainty_or_unknown", (
        "i don't know", "i do not know", "not sure", "unknown", "can't tell",
        "cannot tell", "maybe", "uncertain",
    )),
    ("resolution_or_outcome", (
        "resolved", "worked out", "cleared the air", "got the job", "resigned",
        "reconciled", "apologized", "improved", "small win", "outcome",
    )),
)

_STRUCTURAL_CUE_MAP = {
    "temporal_event": "timeline event",
    "state_change": "state change",
    "conflict_marker": "conflict or contradiction",
    "relationship_update": "relationship update",
    "coping_strategy": "coping strategy",
    "medical_or_safety": "health or safety",
    "identity_preference": "identity or preference",
    "uncertainty_or_unknown": "unknown or unspecified",
    "resolution_or_outcome": "resolution or outcome",
}
_STRUCTURAL_CUE_PRIORITY = (
    "conflict_marker",
    "coping_strategy",
    "resolution_or_outcome",
    "relationship_update",
    "temporal_event",
    "state_change",
    "medical_or_safety",
    "identity_preference",
    "uncertainty_or_unknown",
)

_ARC_TYPES = (
    "work_career",
    "relationships",
    "family",
    "health_safety",
    "mental_health_coping",
    "identity_values",
    "life_transition",
    "education_growth",
    "finance_security",
    "legal_admin",
)

_ARC_KEYWORDS = (
    ("work_career", (
        "work", "job", "boss", "manager", "coworker", "colleague", "career",
        "interview", "hired", "laid off", "fired", "promotion", "harassment",
        "workplace",
    )),
    ("relationships", (
        "partner", "wife", "husband", "girlfriend", "boyfriend", "dating",
        "breakup", "broke up", "relationship", "friend", "roommate",
    )),
    ("family", (
        "mother", "mom", "father", "dad", "parent", "sister", "brother",
        "son", "daughter", "child", "baby", "family",
    )),
    ("health_safety", (
        "doctor", "diagnosis", "diagnosed", "medication", "hospital",
        "surgery", "a1c", "diabetes", "cancer", "panic attack", "safety",
    )),
    ("mental_health_coping", (
        "therapy", "therapist", "stress", "anxiety", "depression", "coping",
        "journal", "painting", "meditation", "breathing", "mental health",
    )),
    ("identity_values", (
        "values", "identity", "important to me", "i want to be", "i prefer",
        "i hate", "i love", "the kind of person",
    )),
    ("life_transition", (
        "moving", "moved", "relocated", "apartment", "new city", "pregnant",
        "pregnancy", "married", "divorce", "graduated",
    )),
    ("education_growth", (
        "school", "class", "exam", "midterm", "grade", "teacher", "college",
        "university", "study", "homework", "academic",
    )),
    ("finance_security", (
        "money", "rent", "debt", "bills", "salary", "pay", "raise",
        "savings", "financial",
    )),
    ("legal_admin", (
        "court", "lawyer", "attorney", "legal", "hearing", "complaint",
        "hr", "paperwork", "insurance",
    )),
)

_ARC_ROLE_KEYWORDS = (
    ("avoidance", (
        "ignored", "avoid", "didn't address", "did not address",
        "haven't addressed", "haven’t addressed", "not addressed", "ignoring",
        "hoping it would stop", "kept quiet", "said nothing",
    )),
    ("onset", (
        "started", "began", "first", "at first", "initially", "diagnosed",
        "found out", "happened",
    )),
    ("escalation", (
        "worse", "escalated", "again", "kept happening", "more intense",
        "spiraled", "overwhelmed",
    )),
    ("attempted_action", (
        "tried", "started therapy", "went to therapy", "talked to",
        "reported", "asked for help", "called", "planned", "decided to",
    )),
    ("state_change", (
        "realized", "changed", "became", "stopped", "quit", "got better",
        "improved", "learned", "started",
    )),
    ("resolution", (
        "resolved", "settled", "cleared the air", "apologized",
        "reconciled", "fixed",
    )),
    ("outcome", (
        "got the job", "hired", "accepted", "rejected", "resigned",
        "broke up", "moved", "small win", "outcome",
    )),
    ("maintenance", (
        "still", "continue", "continuing", "ongoing", "keeping up",
        "sticking with",
    )),
    ("uncertainty", (
        "not sure", "i don't know", "i do not know", "maybe", "uncertain",
        "can't tell", "cannot tell",
    )),
)

_TRACE_LABELS = {
    "state_change",
    "conflict_marker",
    "relationship_update",
    "coping_strategy",
    "identity_preference",
    "uncertainty_or_unknown",
    "resolution_or_outcome",
}

_CAUSE_KEYWORDS = (
    ("cause:workplace_conflict", (
        "harassment", "workplace", "work stress", "boss", "manager",
        "critical", "argument about work", "work boundaries",
    )),
    ("cause:relationship_loss", (
        "breakup", "broke up", "cheating", "partner cheating", "divorce",
        "split up",
    )),
    ("cause:relationship_conflict", (
        "argument", "fight", "tension", "space", "boundaries", "old patterns",
        "reconciled", "cleared the air",
    )),
    ("cause:health_event", (
        "diagnosed", "diagnosis", "hospitalized", "doctor", "panic attack",
        "health scare", "medication",
    )),
    ("cause:family_stress", (
        "mother", "father", "mom", "dad", "child", "family", "parent",
        "brother", "sister",
    )),
    ("cause:identity_pressure", (
        "not good enough", "overreacting", "fraud", "imposter", "values",
        "who i am", "self-worth",
    )),
)

_COPING_KEYWORDS = (
    # Deliberately exclude the bare word "session": eval/product transcripts
    # often include session metadata, which is not evidence of therapy.
    ("coping:therapy", ("therapy", "therapist", "counseling", "counsellor")),
    ("coping:creative_expression", (
        "painting", "writing", "journal", "art class", "blog", "self-expression",
    )),
    ("coping:boundary_setting", (
        "boundary", "boundaries", "cut back", "reduced my hours", "leave work",
    )),
    ("coping:mindfulness", ("mindfulness", "meditation", "breathing")),
    ("coping:social_support", (
        "talk to", "opened up", "trusted colleague", "support", "confide",
    )),
    ("coping:formal_action", ("hr", "reported", "complaint", "lawyer", "doctor")),
)

_CAUSE_READABLE = {
    "cause:workplace_conflict": "workplace conflict",
    "cause:relationship_loss": "relationship loss",
    "cause:relationship_conflict": "relationship conflict",
    "cause:health_event": "health stress",
    "cause:family_stress": "family stress",
    "cause:identity_pressure": "identity pressure",
}

_COPING_READABLE = {
    "coping:therapy": "therapy",
    "coping:creative_expression": "creative expression",
    "coping:boundary_setting": "boundary setting",
    "coping:mindfulness": "mindfulness",
    "coping:social_support": "social support",
    "coping:formal_action": "formal action",
}

_CAUSAL_RELATION_CUES = (
    ("rel:led_to", (
        "led to", "made me", "caused", "because of", "prompted", "pushed me",
        "why i", "it's why", "that has really helped",
    )),
    ("rel:triggered_coping", (
        "helped", "helps", "coping", "manage", "grounding", "escape",
        "self-care", "therapy", "therapist",
    )),
    ("rel:same_arc_as", (
        "still", "again", "back to", "reminded me", "kept thinking",
        "processing everything that happened before", "old patterns",
    )),
    ("rel:supersedes", (
        "instead", "now", "no longer", "changed", "got better", "improved",
    )),
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


def _temporal_anchor(text: str) -> Optional[str]:
    t = text.strip()
    m = re.search(r"\[(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        return m.group(1)
    lower = t.lower()
    for cue in ("today", "yesterday", "tomorrow", "last night", "last week",
                "next week", "this weekend"):
        if cue in lower:
            return cue
    return None


def _heuristic_structural_labels(text: str) -> List[str]:
    t = text.lower()
    labels: List[str] = []
    for label, keys in _STRUCTURAL_LABELS:
        if any(k in t for k in keys):
            labels.append(label)
    if _temporal_anchor(text) and "temporal_event" not in labels:
        labels.append("temporal_event")
    return _dedupe(labels, 5)


def _structural_cues(light: LightExtract) -> List[str]:
    labels = list(light.structural_labels or [])
    labels.sort(key=lambda label: (
        _STRUCTURAL_CUE_PRIORITY.index(label)
        if label in _STRUCTURAL_CUE_PRIORITY else len(_STRUCTURAL_CUE_PRIORITY)
    ))
    return [
        _STRUCTURAL_CUE_MAP[label]
        for label in labels
        if label in _STRUCTURAL_CUE_MAP
    ]


def _structural_entity_keys(light: LightExtract) -> List[str]:
    keys = [f"label:{label}" for label in (light.structural_labels or [])]
    if light.temporal_anchor:
        keys.append(f"time:{light.temporal_anchor.lower()}")
    return keys


def _infer_arc_types(text: str, light: LightExtract) -> List[str]:
    """Broad product domains this turn updates.

    Arc types are intentionally coarse so customer applications can rely on
    them across many verticals without inheriting benchmark-specific behavior.
    """
    t = text.lower()
    arcs: List[str] = []
    for arc, keys in _ARC_KEYWORDS:
        if any(k in t for k in keys):
            arcs.append(arc)
    labels = set(light.structural_labels or [])
    if "coping_strategy" in labels:
        arcs.append("mental_health_coping")
    if "medical_or_safety" in labels:
        arcs.append("health_safety")
    if "relationship_update" in labels:
        arcs.append("relationships")
    if "identity_preference" in labels:
        arcs.append("identity_values")
    return _dedupe(arcs, 3)


def _infer_arc_role(text: str, light: LightExtract) -> str:
    t = text.lower()
    for role, keys in _ARC_ROLE_KEYWORDS:
        if any(k in t for k in keys):
            return role
    labels = set(light.structural_labels or [])
    if "conflict_marker" in labels:
        return "escalation"
    if "resolution_or_outcome" in labels:
        return "outcome"
    if "uncertainty_or_unknown" in labels:
        return "uncertainty"
    if "state_change" in labels:
        return "state_change"
    if "temporal_event" in labels:
        return "onset"
    return "maintenance"


def _infer_arc_phase(text: str, role: str, light: LightExtract) -> str:
    """Lifecycle phase used for origin/change retrieval.

    This is a write-time salience signal: the first avoidant response to a
    problem can matter months later even when the affect score is quiet.
    """
    t = text.lower()
    labels = set(light.structural_labels or [])
    if role in {"avoidance", "onset"} or any(
        cue in t for cue in ("at first", "initially", "first started", "began", "started")
    ):
        return "origin"
    if role in {"resolution", "outcome"}:
        return "resolution"
    if role in {"escalation", "attempted_action", "state_change"}:
        return "transition"
    if "uncertainty_or_unknown" in labels:
        return "uncertainty"
    return "maintenance"


def _trace_salience_floor(light: LightExtract, role: str) -> float:
    labels = set(light.structural_labels or [])
    floor = 6.0
    if labels & {"conflict_marker", "medical_or_safety"}:
        floor = max(floor, 7.0)
    if labels & {"coping_strategy", "relationship_update", "resolution_or_outcome"}:
        floor = max(floor, 6.5)
    if role in {"avoidance", "resolution", "outcome"}:
        floor = max(floor, 7.0)
    return floor


def _trace_cues(arc: str, role: str, phase: str, light: LightExtract) -> List[str]:
    label_cues = _structural_cues(light)
    readable_arc = arc.replace("_", " ")
    readable_role = role.replace("_", " ")
    readable_phase = phase.replace("_", " ")
    return _dedupe(
        [
            f"{readable_arc} arc",
            f"{readable_role} in {readable_arc}",
            f"{readable_phase} of {readable_arc}" if phase == "origin" else "",
            *label_cues,
            "timeline change" if "temporal_event" in (light.structural_labels or []) else "",
            "initial response" if role in {"onset", "avoidance"} else "",
        ],
        3,
    )


def _keyword_keys(text: str, mapping) -> List[str]:
    t = text.lower()
    keys: List[str] = []
    for key, needles in mapping:
        if any(needle in t for needle in needles):
            keys.append(key)
    return _dedupe(keys, 6)


def _causal_entity_keys(text: str, arc: str, role: str,
                        light: LightExtract) -> List[str]:
    labels = set(light.structural_labels or [])
    causes = _keyword_keys(text, _CAUSE_KEYWORDS)
    coping = _keyword_keys(text, _COPING_KEYWORDS)
    relations = _keyword_keys(text, _CAUSAL_RELATION_CUES)

    if "conflict_marker" in labels and not causes:
        if arc == "work_career":
            causes.append("cause:workplace_conflict")
        elif arc == "relationships":
            causes.append("cause:relationship_conflict")
    if "coping_strategy" in labels and coping and not relations:
        relations.append("rel:triggered_coping")
    if role in {"state_change", "attempted_action"} and coping:
        relations.append("rel:triggered_coping")
    if role in {"resolution", "outcome"}:
        relations.append("rel:supersedes")
    if role in {"avoidance", "escalation"} and causes:
        relations.append("rel:same_arc_as")

    # Domain-level stressor keys are intentionally coarse. They let future
    # trajectory retrieval assemble chains without depending on scenario nouns.
    if causes:
        relations.append("rel:caused_by")
    if coping:
        relations.append("rel:coping_response")

    return _dedupe([*causes, *coping, *relations], 10)


def _causal_cues(causal_keys: List[str]) -> List[str]:
    out: List[str] = []
    for key in causal_keys:
        if key.startswith("cause:"):
            out.append(key.replace("cause:", "").replace("_", " "))
        elif key.startswith("coping:"):
            out.append(key.replace("coping:", "").replace("_", " "))
        elif key == "rel:triggered_coping":
            out.append("triggered coping")
        elif key == "rel:caused_by":
            out.append("causal antecedent")
    return _dedupe(out, 3)


def _keys_with_prefix(fact: Fact, prefix: str) -> List[str]:
    return [
        key for key in (fact.entity_keys or [])
        if str(key).startswith(prefix)
    ]


def _first_time_key(fact: Fact) -> Optional[str]:
    for key in fact.entity_keys or []:
        if str(key).startswith("time:"):
            return str(key).split(":", 1)[1]
    return None


def _readable_keys(keys: List[str], labels: Dict[str, str]) -> List[str]:
    return _dedupe([labels.get(key, key.split(":", 1)[-1].replace("_", " "))
                    for key in keys], 4)


_BRIDGE_SNIPPET_PATTERNS = (
    (re.compile(r"\b(workplace\s+)?harassment\b", re.I), "workplace harassment"),
    (re.compile(r"\bbreak(?:ing|up|s)?\s+up\b|\bbreakup\b|\brelationship loss\b", re.I), "relationship loss"),
    (re.compile(r"\bbreakdown at work\b|\bbroke down at work\b|\bbreakdown\b", re.I), "a breakdown"),
    (re.compile(r"\bpanic attack\b", re.I), "a panic attack"),
    (re.compile(r"\bboss resigned\b|\bmanager resigned\b", re.I), "a boss resignation"),
    (re.compile(r"\bargument\b|\bfight\b|\btension\b", re.I), "relationship conflict"),
    (re.compile(r"\bhospitalized\b|\bhospital\b", re.I), "a hospitalization"),
    (re.compile(r"\bdiagnos(?:is|ed)\b", re.I), "a diagnosis"),
)


def _strip_trace_prefix(summary: str) -> str:
    text = re.sub(r"^(?:Trace|Bridge)\s+\[[^\]]+\]:\s*", "", summary or "").strip()
    text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
    return text


def _event_snippet(fact: Fact) -> Optional[str]:
    text = _strip_trace_prefix(fact.summary)
    if not text:
        return None
    for pattern, label in _BRIDGE_SNIPPET_PATTERNS:
        if pattern.search(text):
            return label
    cleaned = text
    cleaned = re.sub(r"\b[Tt]he user(?:'s)?\b", "the user", cleaned)
    cleaned = re.sub(r"\b[Ii]\s+(?:had|have|was|am|feel|felt|keep|kept|started|began|found|got|agreed)\b", "", cleaned)
    cleaned = re.sub(r"\b(my|me|I'm|I’m|I've|I’ve|I)\b", "the user", cleaned, flags=re.I)
    cleaned = " ".join(cleaned.replace("…", "").strip(" .").split())
    if not cleaned:
        return None
    if len(cleaned) > 72:
        cleaned = cleaned[:72].rsplit(" ", 1)[0].strip()
    return cleaned[:1].lower() + cleaned[1:]


def _concrete_event_snippet(fact: Fact) -> Optional[str]:
    """A stricter snippet for bridge evidence.

    Bridges are durable synthesized memories, so their antecedents should come
    from concrete event language rather than arbitrary nearby dialogue.
    """
    text = _strip_trace_prefix(fact.summary)
    if not text:
        return None
    for pattern, label in _BRIDGE_SNIPPET_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _has_concrete_coping_evidence(fact: Fact, coping_keys: List[str]) -> bool:
    text = _strip_trace_prefix(fact.summary).lower()
    evidence = {
        "coping:therapy": (
            "therapy", "therapist", "counseling", "counselling",
            "started attending therapy", "went to therapy",
        ),
        "coping:creative_expression": (
            "painting", "writing", "journal", "art class", "blog",
            "self-expression",
        ),
        "coping:boundary_setting": (
            "boundary", "boundaries", "cut back", "reduced my hours",
            "leave work",
        ),
        "coping:mindfulness": ("mindfulness", "meditation", "breathing"),
        "coping:social_support": (
            "talk to", "opened up", "trusted colleague", "support", "confide",
        ),
        "coping:formal_action": (
            "hr", "reported", "complaint", "lawyer", "doctor",
        ),
    }
    return any(
        any(needle in text for needle in evidence.get(key, ()))
        for key in coping_keys
    )


def _bridge_antecedents(supporting_facts: List[Fact],
                        cause_labels: List[str]) -> List[str]:
    snippets = _dedupe([
        snippet for fact in supporting_facts
        for snippet in [_concrete_event_snippet(fact) or _event_snippet(fact)]
        if snippet
    ], 3)
    if snippets:
        return snippets
    return cause_labels


def _join_human(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def bridge_fact(current: Fact, recent: List[Fact]) -> Optional[Fact]:
    """Synthesize a compact causal bridge when coping/action follows causes.

    This is intentionally conservative: a bridge is written only when the
    current trace contains a concrete coping/action key and prior trace facts
    contain at least one causal antecedent. It gives retrieval a durable
    "why this coping began" unit without embedding whole session blobs.
    """
    current_keys = set(current.entity_keys or [])
    coping_keys = _keys_with_prefix(current, "coping:")
    if not coping_keys:
        return None
    if not _first_time_key(current):
        return None
    if not _has_concrete_coping_evidence(current, coping_keys):
        return None
    if not (
        "rel:triggered_coping" in current_keys
        or "rel:coping_response" in current_keys
        or current.fact_type == "bridge"
    ):
        return None

    cause_keys: List[str] = []
    supporting_arcs: List[str] = []
    supporting_facts: List[Fact] = []
    for fact in recent:
        if (fact.id is not None and fact.id == current.id) or fact.fact_type == "bridge":
            continue
        keys = set(fact.entity_keys or [])
        fact_causes = _keys_with_prefix(fact, "cause:")
        if not fact_causes:
            continue
        if not _first_time_key(fact):
            continue
        if not _concrete_event_snippet(fact):
            continue
        if keys & current_keys or "rel:same_arc_as" in keys or "rel:caused_by" in keys:
            cause_keys.extend(fact_causes)
            supporting_arcs.extend(_keys_with_prefix(fact, "arc:"))
            supporting_facts.append(fact)
        if len(_dedupe(cause_keys, 4)) >= 3:
            break

    cause_keys = _dedupe(cause_keys + _keys_with_prefix(current, "cause:"), 4)
    if not cause_keys or not supporting_facts:
        return None

    coping_labels = _readable_keys(coping_keys, _COPING_READABLE)
    cause_labels = _readable_keys(cause_keys, _CAUSE_READABLE)
    antecedents = _bridge_antecedents(supporting_facts, cause_labels)
    time = _first_time_key(current)
    prefix = "Bridge [mental_health_coping / triggered_coping"
    if time:
        prefix += f" / {time}"
    prefix += "]"
    summary = (
        f"{prefix}: The user's {_join_human(coping_labels)} emerged as a coping "
        f"response after {_join_human(antecedents)}."
    )
    entity_keys = _dedupe(
        [
            "arc:mental_health_coping",
            "role:attempted_action",
            "phase:transition",
            "rel:caused_by",
            "rel:triggered_coping",
            "rel:coping_response",
            *cause_keys,
            *coping_keys,
            *_keys_with_prefix(current, "time:"),
            *supporting_arcs,
            *[f"support_fact:{fact.id}" for fact in supporting_facts
              if fact.id is not None],
        ],
        18,
    )
    cues = _dedupe(
        [
            "causal bridge",
            "what led to coping",
            *cause_labels,
            *coping_labels,
        ],
        3,
    )
    return Fact(
        summary=_condense(summary, max_chars=220),
        fact_type="bridge",
        event_type="mental_health_coping",
        emotional_tone=current.emotional_tone,
        salience=max(current.salience, 7.5),
        retrieval_cues=cues,
        entity_keys=entity_keys,
        validity="current",
        confidence=0.72,
    )


def arc_trace_facts(text: str, light: LightExtract) -> List[Fact]:
    """Create compact lifecycle traces for turns that alter a durable arc."""
    labels = set(light.structural_labels or [])
    # A trace is a lifecycle update, not just another copy of every salient
    # fact. Relationship/temporal labels are supporting metadata unless paired
    # with an actual change, conflict, action, outcome, or uncertainty signal.
    lifecycle_labels = labels.intersection(_TRACE_LABELS) - {"relationship_update"}
    if not lifecycle_labels:
        return []
    arcs = _infer_arc_types(text, light)
    if not arcs:
        arcs = ["open_arc"]
    role = _infer_arc_role(text, light)
    phase = _infer_arc_phase(text, role, light)
    time = light.temporal_anchor
    facts: List[Fact] = []
    for arc in arcs:
        prefix = f"Trace [{arc} / {role} / {phase}"
        if time:
            prefix += f" / {time}"
        prefix += "]"
        causal_keys = _causal_entity_keys(text, arc, role, light)
        entity_keys = _dedupe(
            [
                f"arc:{arc}",
                f"role:{role}",
                f"phase:{phase}",
                *[f"label:{label}" for label in sorted(labels)],
                *( [f"time:{time.lower()}"] if time else [] ),
                *causal_keys,
                *list(light.entities or []),
            ],
            18,
        )
        cues = _dedupe(
            _trace_cues(arc, role, phase, light) + _causal_cues(causal_keys),
            3,
        )
        facts.append(Fact(
            summary=_condense(f"{prefix}: {text.strip()}", max_chars=220),
            fact_type="trace",
            event_type=arc,
            emotional_tone=light.tone,
            salience=max(light.salience, _trace_salience_floor(light, role)),
            retrieval_cues=cues,
            entity_keys=entity_keys,
            validity="current",
            confidence=0.65,
        ))
    return facts


def _heuristic_event_type(text: str) -> str:
    t = text.lower()
    for keys, et in _EVENT_KEYWORDS:
        if any(k in t for k in keys):
            return et
    return "event"


def _heuristic_light(text: str) -> LightExtract:
    a = llm._heuristic_analyze(text)
    labels = _heuristic_structural_labels(text)
    return LightExtract(
        emotional=a["emotional"], vulnerability=a["vulnerability"],
        salience=a["salience"], tone=a["tone"],
        entities=_heuristic_entities(text),
        retrieval_cues=_heuristic_cues(text, limit=2),
        structural_labels=labels,
        temporal_anchor=_temporal_anchor(text),
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
    cues = _dedupe(
        list(light.retrieval_cues or [])
        + _heuristic_cues(text, limit=3)
        + _structural_cues(light),
        3,
    )
    entity_keys = _dedupe(
        list(light.entities or []) + _structural_entity_keys(light),
        8,
    )
    return Fact(
        summary=_condense(text, max_chars=max_chars),
        fact_type="thread" if is_thread else "event",
        event_type="thread" if is_thread else _heuristic_event_type(text),
        emotional_tone=light.tone,
        emotional_cause=None,
        user_belief=None,
        salience=light.salience,
        retrieval_cues=cues,
        entity_keys=entity_keys,
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
    merged_cues = _dedupe(
        list(cues or []) + list(light.retrieval_cues or []) + _structural_cues(light),
        3,
    )
    merged_entities = _dedupe(
        list(entities or []) + list(light.entities or []) + _structural_entity_keys(light),
        8,
    )
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
    facts.extend(arc_trace_facts(text, light))
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
    return bool(anchor_facts(text, light) or arc_trace_facts(text, light))


# --------------------------------------------------------------------------- #
# LLM tiers
# --------------------------------------------------------------------------- #
_LIGHT_SYS = (
    "You are the interoceptive sensor and entity tagger of a cognitive memory "
    "layer. Read the user's message and respond with strict JSON only:\n"
    '{"emotional": <0-100>, "vulnerability": <0-100>, "salience": <1-10>, '
    '"tone": "<one lowercase word>", "entities": ["type:value", ...], '
    '"retrieval_cues": ["...", "..."], '
    '"structural_labels": ["..."], "temporal_anchor": "<date/cue or null>"}\n'
    "emotional = affective arousal; vulnerability = openness/fragility/disclosure; "
    "salience = worth remembering long-term (10=breakup/death/identity, 1=small talk). "
    "entities are typed keys like person:maya, role:interview, org:nimbus, place:lisbon. "
    "retrieval_cues are AT MOST 2 alternative ways someone might later ask about this "
    "— reformulations of the underlying meaning, not synonyms of the words used. "
    "structural_labels are AT MOST 5 from: temporal_event, state_change, "
    "conflict_marker, relationship_update, coping_strategy, medical_or_safety, "
    "identity_preference, uncertainty_or_unknown, resolution_or_outcome. "
    "temporal_anchor is an explicit date like 2025-02-15 or a cue like 'last night'."
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
        structural_labels=_dedupe(
            [
                str(label)
                for label in (d.get("structural_labels") or [])
                if str(label) in _STRUCTURAL_CUE_MAP
            ]
            + _heuristic_structural_labels(text),
            5,
        ),
        temporal_anchor=(
            str(d.get("temporal_anchor")).strip()
            if d.get("temporal_anchor") not in (None, "", "null", "None")
            else _temporal_anchor(text)
        ),
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
        retrieval_cues=_dedupe(
            [str(c) for c in (f.get("retrieval_cues") or [])]
            + _structural_cues(light),
            3,
        ),
        entity_keys=_dedupe(
            [str(e) for e in (f.get("entity_keys") or [])]
            + _structural_entity_keys(light),
            8,
        ),
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
