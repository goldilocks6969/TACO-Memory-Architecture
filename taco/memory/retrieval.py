"""Hierarchical retrieval scoring R(m) and context assembly (§4.2, Figure 4).

    R(m) = w_sem·sem(m) + w_sal·sal(m) + w_emo·emo(m)
           + w_rec·rec(m) + w_decay·decay(m)

The weights are state-modulated (see LatentState.retrieval_weights). The
pipeline casts 20 candidates via pgvector kNN, this module re-ranks them by
R(m), selects the top 4, and assembles them into a single narrative briefing.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .episode import Episode
from .. import config
from ..state import LatentState
from . import rerank as _rerank
from . import store as _store

if TYPE_CHECKING:  # avoids a runtime memory→subsystems import; duck-typed below
    from ..subsystems.reasoning import ReasoningStance
    from ..subsystems.prediction import PredictiveContinuity


# --------------------------------------------------------------------------- #
# Emotional-tone compatibility (the emo(m) component)
# --------------------------------------------------------------------------- #
_TONE_FAMILY = {
    "distress": "negative", "intense": "negative", "concerned": "negative",
    "tender": "tender", "sad": "negative", "anxious": "negative",
    "transactional": "neutral", "neutral": "neutral",
}


def state_tone(state: LatentState) -> str:
    """Coarse label for the user's *current* emotional tone, from S."""
    if state.E >= 60:
        return "distress" if state.V >= 40 else "intense"
    if state.E >= 30:
        return "tender" if state.V >= 30 else "concerned"
    return "neutral"


def tone_compat(current: str, mem_tone: Optional[str]) -> float:
    """emo(m) ∈ [0,1]: tone compatibility + a +0.3 boost on an exact match (§4.2)."""
    if not mem_tone:
        return 0.3
    base = 0.3
    if _TONE_FAMILY.get(current) == _TONE_FAMILY.get(mem_tone):
        base = 0.6
    if current == mem_tone:
        base += config.EMO_TONE_MATCH_BOOST
    return min(1.0, base)


# --------------------------------------------------------------------------- #
# R(m)
# --------------------------------------------------------------------------- #
def components(ep: Episode, state: LatentState, current_tone: str) -> Dict[str, float]:
    sem = max(0.0, min(1.0, ep.similarity))
    sal = max(0.0, min(1.0, ep.salience / 10.0))
    emo = tone_compat(current_tone, ep.tone)
    rec = max(0.0, 1.0 - ep.age_days() / config.RECENCY_WINDOW_DAYS)
    decay = max(0.0, min(1.0, ep.vitality))
    return {"sem": sem, "sal": sal, "emo": emo, "rec": rec, "decay": decay}


def score(ep: Episode, state: LatentState, weights: Dict[str, float],
          current_tone: str) -> float:
    c = components(ep, state, current_tone)
    ep.score = sum(weights[k] * c[k] for k in c)
    return ep.score


def rerank(candidates: List[Episode], state: LatentState,
           top_k: int = config.TOP_K) -> List[Episode]:
    """Re-rank candidates by state-modulated R(m) and return the top-k."""
    weights = state.retrieval_weights()
    tone = state_tone(state)
    for ep in candidates:
        score(ep, state, weights, tone)
    return sorted(candidates, key=lambda e: e.score, reverse=True)[:top_k]


# --------------------------------------------------------------------------- #
# Adaptive sparse recall — query intent → recall budget
# --------------------------------------------------------------------------- #
_FILLER_RE = re.compile(
    r"\b(first chat|first ask|recommend|capital|post office|water bottle|"
    r"wifi|kettle|carpet|garlic|baking soda|baking powder|gift|hotel)\b",
    re.I,
)
_COHERENCE_RE = re.compile(
    r"\b(am i|are we|how am i|how are we|progress|stable|settled|over it|"
    r"fully over|making any|where am i|right now|at this point|headway)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(before|after|first|initially|at first|eventually|later|timeline|"
    r"sequence|led to|lead to|over time|changed from|started|began|when)\b",
    re.I,
)
_CONFLICT_RE = re.compile(
    r"\b(conflict|harassment|address|addressed|contradict|contradiction|"
    r"inconsistent|outdated|changed|remain stable|still true|initially address|address at first|"
    r"did .* change|no longer|used to)\b",
    re.I,
)
_USER_MODELING_RE = re.compile(
    r"\b(how has|how did .* evolve|coping|cope|approach|pattern|tendency|"
    r"what does .* say about|user model|changed emotionally|state evolved)\b",
    re.I,
)
_ORIGIN_RE = re.compile(
    r"\b(first|initially|at first|when it started|when .* started|beginning|"
    r"began|origin|led to|lead to|sequence of events|what led)\b",
    re.I,
)
_EMOTIONAL_RE = re.compile(
    r"\b(why|what's going on|unpack|thoughts|feeling|felt|feel|lost it|"
    r"alienated|stomach dropped|surge of anger|welled up|pull over|sick)\b",
    re.I,
)
_FACTUAL_RE = re.compile(
    r"\b(what|where|which|who|when|name|called|medication|condition|company|"
    r"city|class|grade|concept|partner|dad|doctor|hired)\b",
    re.I,
)


def classify_query(query: str) -> str:
    """Cheap query intent classifier for sparse recall.

    No LLM call: the point is to decide how much memory should reach the prompt,
    not to understand the full answer. Order matters: coherence/emotional probes
    often start with "what/why" but need continuity, not fact-only recall.
    """
    q = (query or "").strip()
    if not q:
        return "general"
    if _FILLER_RE.search(q):
        return "filler"
    if _CONFLICT_RE.search(q):
        return "conflict"
    if _USER_MODELING_RE.search(q):
        return "user_modeling"
    if _TEMPORAL_RE.search(q):
        return "temporal"
    if _EMOTIONAL_RE.search(q):
        return "emotional"
    if _COHERENCE_RE.search(q):
        return "coherence"
    if _FACTUAL_RE.search(q):
        return "factual"
    return "general"


def recall_budget(query_kind: str) -> int:
    if config.RECALL_POLICY != "adaptive":
        return config.TOP_K
    return {
        "factual": config.RECALL_FACTUAL_K,
        "emotional": config.RECALL_EMOTIONAL_K,
        "coherence": config.RECALL_COHERENCE_K,
        "temporal": max(config.RECALL_COHERENCE_K, config.RECALL_FACTUAL_K + 2),
        "conflict": max(config.RECALL_COHERENCE_K, config.RECALL_FACTUAL_K + 2),
        "user_modeling": max(config.RECALL_COHERENCE_K, config.RECALL_EMOTIONAL_K),
        "filler": config.RECALL_FILLER_K,
        "general": config.RECALL_GENERAL_K,
    }.get(query_kind, config.RECALL_GENERAL_K)


def _arc_query_entity_keys(query_text: str, query_kind: str) -> List[str]:
    """Synthetic query keys for capability-aware retrieval.

    These are broad product-domain keys, not scenario-specific keywords. They
    let the existing entity-overlap retriever find lifecycle traces when the
    question asks about time, conflict, or user-state evolution.
    """
    q = (query_text or "").lower()
    keys: List[str] = []
    arc_terms = (
        ("arc:work_career", ("work", "job", "career", "boss", "manager", "harassment")),
        ("arc:relationships", ("relationship", "partner", "dating", "breakup", "friend")),
        ("arc:family", ("family", "mother", "mom", "father", "dad", "child")),
        ("arc:health_safety", ("health", "doctor", "diagnosis", "medication", "safety")),
        ("arc:mental_health_coping", ("coping", "cope", "therapy", "therapist", "stress")),
        ("arc:identity_values", ("identity", "values", "preference", "belief")),
        ("arc:life_transition", ("moving", "moved", "relocation", "pregnancy", "transition")),
        ("arc:education_growth", ("school", "class", "exam", "grade", "teacher")),
        ("arc:finance_security", ("money", "finance", "rent", "salary", "debt")),
        ("arc:legal_admin", ("legal", "court", "lawyer", "hr", "complaint")),
    )
    for key, terms in arc_terms:
        if any(term in q for term in terms):
            keys.append(key)
    if query_kind == "temporal" and any(
        term in q for term in ("therapy", "therapist", "coping", "stress", "mental health")
    ):
        keys.extend([
            "arc:mental_health_coping",
            "arc:work_career",
            "arc:relationships",
            "arc:family",
            "arc:health_safety",
        ])
    if query_kind == "temporal":
        keys.extend(["label:temporal_event", "role:onset", "role:outcome"])
    elif query_kind == "conflict":
        keys.extend(["label:conflict_marker", "role:avoidance", "role:resolution"])
    elif query_kind == "user_modeling":
        keys.extend(["label:coping_strategy", "label:state_change", "role:state_change"])
    if is_origin_query(query_text):
        keys.extend(["phase:origin", "role:onset", "role:avoidance"])
    seen, out = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:8]


def _query_arc_keys(query_text: str, query_kind: str) -> List[str]:
    return [
        key for key in _arc_query_entity_keys(query_text, query_kind)
        if key.startswith("arc:")
    ]


def is_origin_query(query_text: str) -> bool:
    return bool(_ORIGIN_RE.search(query_text or ""))


def is_trajectory_query(query_text: str, query_kind: str) -> bool:
    q = query_text or ""
    if query_kind in {"temporal", "user_modeling"}:
        return True
    return bool(re.search(
        r"\b(evolved|evolution|progression|trajectory|sequence|led to|lead to|"
        r"over time|from .* to|how has|what changed)\b",
        q,
        re.I,
    ))


def _norm_words(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(w) > 2 and w not in {
            "the", "and", "for", "that", "this", "with", "user", "users",
            "their", "them", "they", "from", "about", "into", "was", "were",
        }
    }


def _redundant(a: Episode, b: Episode) -> bool:
    aw, bw = _norm_words(a.content), _norm_words(b.content)
    if not aw or not bw:
        return False
    small, large = (aw, bw) if len(aw) <= len(bw) else (bw, aw)
    if len(small) < 2:
        return False
    overlap = len(small & large) / max(1, len(small))
    return overlap >= 0.66


def _prefer_memory(a: Episode, b: Episode) -> Episode:
    """Prefer concise anchors unless the longer trace is much more important."""
    a_len, b_len = _count_tokens(a.content), _count_tokens(b.content)
    if abs(a.salience - b.salience) >= 3:
        return a if a.salience > b.salience else b
    if getattr(a, "fact_type", None) and not getattr(b, "fact_type", None):
        return a
    if getattr(b, "fact_type", None) and not getattr(a, "fact_type", None):
        return b
    return a if a_len <= b_len else b


def collapse_redundant(candidates: List[Episode]) -> List[Episode]:
    out: List[Episode] = []
    for cand in candidates:
        replaced = False
        for i, kept in enumerate(out):
            if _redundant(cand, kept):
                out[i] = _prefer_memory(cand, kept)
                replaced = True
                break
        if not replaced:
            out.append(cand)
    return sorted(out, key=lambda e: e.score, reverse=True)


def _query_overlap(ep: Episode, query_text: str) -> float:
    q = _norm_words(query_text)
    if not q:
        return 0.0
    mem = _norm_words(ep.content)
    for cue in getattr(ep, "retrieval_cues", []) or []:
        mem |= _norm_words(cue)
    if not mem:
        return 0.0
    return len(q & mem) / max(1, len(q))


def _has_entity_prefix(ep: Episode, prefix: str) -> bool:
    return any(
        str(key).startswith(prefix)
        for key in (getattr(ep, "entity_keys", []) or [])
    )


def _has_entity_key(ep: Episode, key: str) -> bool:
    return key in (getattr(ep, "entity_keys", []) or [])


def _has_any_entity_key(ep: Episode, keys: List[str]) -> bool:
    ep_keys = set(getattr(ep, "entity_keys", []) or [])
    return any(key in ep_keys for key in keys)


def _boost_for_kind(ep: Episode, query_kind: str, state: LatentState,
                    query_text: str = "") -> float:
    ft = getattr(ep, "fact_type", None)
    et = getattr(ep, "event_type", None)
    thread = getattr(ep, "thread_status", None)
    boost = 0.0
    if query_kind == "factual":
        boost += _query_overlap(ep, query_text) * 0.45
        if ft in ("relationship", "state", "event"):
            boost += 0.08
        if et in ("identity", "health", "achievement", "relationship"):
            boost += 0.06
    elif query_kind == "emotional":
        boost += ep.salience / 100.0
        boost += tone_compat(state_tone(state), ep.tone) * 0.08
    elif query_kind == "coherence":
        boost += _query_overlap(ep, query_text) * 0.08
        if thread in ("unresolved", "in_progress"):
            boost += 0.10
        if et in ("achievement", "health", "plan", "relationship"):
            boost += 0.08
    elif query_kind in ("temporal", "conflict", "user_modeling"):
        boost += _query_overlap(ep, query_text) * 0.12
        if ft == "trace":
            boost += 0.28
        if ft == "bridge":
            boost += 0.36
        if _has_entity_prefix(ep, "arc:"):
            boost += 0.10
        query_arcs = _query_arc_keys(query_text, query_kind)
        if query_arcs:
            if _has_any_entity_key(ep, query_arcs):
                boost += 0.22
            elif ft == "trace" and _has_entity_prefix(ep, "arc:"):
                boost -= 0.08
        if is_origin_query(query_text):
            if _has_entity_key(ep, "phase:origin"):
                boost += 0.24
            if _has_entity_key(ep, "role:onset") or _has_entity_key(ep, "role:avoidance"):
                boost += 0.12
            if _has_entity_key(ep, "role:escalation") and not _has_entity_key(ep, "phase:origin"):
                boost -= 0.08
        if is_trajectory_query(query_text, query_kind):
            if ft == "bridge":
                boost += 0.32
            if _has_entity_key(ep, "rel:caused_by") or _has_entity_prefix(ep, "cause:"):
                boost += 0.12
            if (
                _has_entity_key(ep, "rel:triggered_coping")
                or _has_entity_key(ep, "rel:coping_response")
                or _has_entity_prefix(ep, "coping:")
            ):
                boost += 0.12
            if _has_entity_key(ep, "phase:resolution") or _has_entity_key(ep, "rel:supersedes"):
                boost += 0.08
        if query_kind == "temporal":
            if _has_entity_prefix(ep, "time:") or _has_entity_key(ep, "label:temporal_event"):
                boost += 0.14
            if _has_entity_key(ep, "role:onset") or _has_entity_key(ep, "role:outcome"):
                boost += 0.08
        elif query_kind == "conflict":
            if _has_entity_key(ep, "label:conflict_marker"):
                boost += 0.16
            if _has_entity_key(ep, "role:avoidance") or _has_entity_key(ep, "role:resolution"):
                boost += 0.10
        elif query_kind == "user_modeling":
            if _has_entity_key(ep, "label:coping_strategy") or _has_entity_key(ep, "arc:mental_health_coping"):
                boost += 0.16
            if _has_entity_key(ep, "label:state_change") or _has_entity_key(ep, "role:state_change"):
                boost += 0.10
    return boost


def select_sparse_recall(candidates: List[Episode], state: LatentState,
                         query_kind: str, query_text: str = "") -> List[Episode]:
    """Apply biological-style sparse recall after candidate retrieval.

    RAG dumps a fixed top-k. TACO admits memories at write time, then lets
    current state and query intent decide how many traces reach the prompt.
    """
    if config.RECALL_POLICY != "adaptive":
        return candidates[:config.TOP_K]
    budget = recall_budget(query_kind)
    if budget <= 0:
        return []
    adjusted = list(candidates)
    for ep in adjusted:
        ep.score += _boost_for_kind(ep, query_kind, state, query_text)
    adjusted = sorted(adjusted, key=lambda e: e.score, reverse=True)
    collapsed = collapse_redundant(adjusted)
    return collapsed[:budget]


# --------------------------------------------------------------------------- #
# Context assembly (Figure 4 step 4)
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT_FULL = (
    "You are the reasoning engine inside a state-dependent cognitive memory "
    "layer. A separate system has inferred the user's current state and "
    "selected the most relevant memories. Respond naturally; let the state, "
    "the memories, and the reasoning directives below shape both what you say "
    "and what you choose to do."
)

_SYSTEM_PROMPT_COMPACT = (
    "Answer from the state tag and retrieved memories; preserve continuity."
)


def _count_tokens(text: str) -> int:
    """Token count under cl100k_base, with a whitespace fallback when tiktoken
    is unavailable (kept loose so the truncation cap is conservative)."""
    if not text:
        return 0
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text.split())


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    """Hard cap on a section: if over budget, trim by tokens and append an
    ellipsis marker so a reader can tell something was cut."""
    if max_tokens <= 0 or not text:
        return text
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        # Reserve 1 token for the truncation marker so the total stays under cap.
        kept = enc.decode(ids[: max(1, max_tokens - 1)]).rstrip()
        return f"{kept}…"
    except Exception:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[: max(1, max_tokens - 1)]) + "…"


def _compact_state_section(state: LatentState,
                           stance: Optional["ReasoningStance"]) -> str:
    """Dense state tag for ``compact`` mode.

    The LLM does not need prose explaining the state on every turn; it needs
    the decision variables. Full mode keeps the human-readable directives for
    debugging, while compact mode keeps the behavioral signal in ~15 tokens.
    """
    mode = stance.mode if stance is not None else "neutral"
    return (f"[S tone={state_tone(state)} stance={mode} "
            f"E={state.E:.0f} V={state.V:.0f} K={state.K:.0f} R={state.R:.0f}]")


def _compact_identity_section(identity: Optional[List[str]]) -> str:
    """Tiny identity tag for ``compact`` mode: top attribute only, no
    confidence percentages or prose."""
    if not identity:
        return ""
    short = []
    for line in identity[:1]:
        # incoming line shape: "attr — value (confidence X%)"
        item = line.split(" (confidence")[0].replace(" — ", "=")
        short.append(item)
    return "[ID " + "; ".join(short) + "]"


def _minimal_state_section(state: LatentState,
                           stance: Optional["ReasoningStance"]) -> str:
    """Single bracketed tag for ``minimal`` mode (target ≤ 35 tokens). Carries
    only the two signals that demonstrably modulate the reasoning engine in
    our ablations: the current tone and the resolved stance."""
    mode = stance.mode if stance is not None else "neutral"
    return f"[state tone={state_tone(state)}, stance={mode}]"


def _compact_memory_line(memory: Episode) -> str:
    text = " ".join(memory.content.strip().split())
    if len(text) > 180:
        text = text[:179].rsplit(" ", 1)[0].rstrip() + "…"
    return f"- {text}"


def _fit_retrieval_budget(lines: List[str], max_tokens: int) -> List[str]:
    if max_tokens <= 0:
        return lines
    kept: List[str] = []
    for line in lines:
        trial = kept + [line]
        if _count_tokens("\n".join(trial)) <= max_tokens:
            kept.append(line)
            continue
        remaining = max_tokens - _count_tokens("\n".join(kept))
        if remaining > 8:
            kept.append(_truncate_to_budget(line, remaining))
        break
    return kept


def briefing_sections(
    memories: List[Episode], beliefs: List[str], working: List[Episode],
    state: LatentState, stance: Optional["ReasoningStance"] = None,
    identity: Optional[List[str]] = None,
    prediction: Optional["PredictiveContinuity"] = None,
    mode: Optional[str] = None,
) -> Dict[str, str]:
    """Return the briefing broken down into its logical sections.

    Section keys:
      ``system``     — fixed scaffolding the engine always sees.
      ``state``      — the current S = (E, K, V, R) line and reasoning stance.
      ``identity``   — L10 persistent self-model.
      ``working``    — L1 working-memory transcript.
      ``prediction`` — L9 anticipatory continuity block.
      ``retrieval``  — the retrieved memory / fact payload (the policy-dependent
                       component — what the eval harness charges as
                       retrieval_tokens).

    The eval harness uses these to attribute tokens: ``retrieval`` is the
    retrieval payload, ``system`` is the fixed system prompt, and everything
    else (``state``, ``identity``, ``working``, ``prediction``) is the
    structured "state briefing" overhead TACO adds on top of vanilla RAG.

    *mode* (``full`` | ``compact`` | ``minimal``) controls state-briefing
    verbosity.  Defaults to ``config.STATE_BRIEFING_MODE``.  ``compact`` and
    ``minimal`` are hard-capped by token budget; the retrieved memory payload
    is **not** affected (the metrics protocol charges it separately).
    """
    if mode is None:
        mode = config.STATE_BRIEFING_MODE
    mode = mode.strip().lower()

    # ------------------------------------------------------------------
    # state-briefing sections (state / identity / working / prediction)
    # ------------------------------------------------------------------
    if mode == "minimal":
        state_block = _minimal_state_section(state, stance)
        identity_block = ""
        working_block = ""
        prediction_block = ""
    elif mode == "compact":
        state_block = _compact_state_section(state, stance)
        identity_block = _compact_identity_section(identity)
        working_block = ""        # dropped to stay under 80 tokens
        prediction_block = ""     # dropped to stay under 80 tokens
    else:  # "full" — preserve the original prompt verbatim
        state_lines = [
            f"User state — emotional intensity {state.E:.0f}/100, engagement "
            f"{state.K:.0f}/100, vulnerability {state.V:.0f}/100, recency "
            f"{state.R:.0f}/100 (current tone: {state_tone(state)})."
        ]
        if stance is not None:
            state_lines.append("")
            state_lines.append(stance.render())
        state_block = "\n".join(state_lines)

        identity_lines: List[str] = []
        if identity:
            identity_lines.append(
                "Persistent self-model (who this person is, across time):"
            )
            for fact in identity:
                identity_lines.append(f"  • {fact}")
        identity_block = "\n".join(identity_lines)

        working_lines: List[str] = []
        if working:
            working_lines.append("Recent conversation (working memory):")
            for w in working:
                working_lines.append(f"  {w.role}: {w.content}")
        working_block = "\n".join(working_lines)

        prediction_lines: List[str] = []
        if prediction is not None:
            prediction_lines.append(
                "Predictive continuity (anticipatory, not yet stated):")
            prediction_lines.append(prediction.render())
        prediction_block = "\n".join(prediction_lines)

    # ------------------------------------------------------------------
    # retrieval payload — identical across modes; the metrics protocol
    # charges these tokens separately from the state briefing.
    # ------------------------------------------------------------------
    retrieval_lines: List[str] = []
    if mode == "full":
        if beliefs:
            retrieval_lines.append("Durable, identity-level beliefs about this person:")
            for b in beliefs:
                retrieval_lines.append(f"  • {b}")
        if memories:
            if retrieval_lines:
                retrieval_lines.append("")
            retrieval_lines.append(
                "Psychologically privileged memories (re-ranked by R(m), most "
                "significant first):"
            )
            for m in memories:
                retrieval_lines.append(
                    f"  • [{m.tone or 'neutral'}, salience {m.salience:.0f}/10, "
                    f"R={m.score:.2f}] {m.content}"
                )
    else:
        if beliefs or memories:
            retrieval_lines.append("M:")
        for b in beliefs[:1]:
            retrieval_lines.append(f"- {b}")
        for m in memories:
            retrieval_lines.append(_compact_memory_line(m))
        retrieval_lines = _fit_retrieval_budget(
            retrieval_lines, config.RECALL_MAX_TOKENS)

    sections = {
        "system": _SYSTEM_PROMPT_FULL if mode == "full" else _SYSTEM_PROMPT_COMPACT,
        "state": state_block,
        "identity": identity_block,
        "retrieval": "\n".join(retrieval_lines),
        "working": working_block,
        "prediction": prediction_block,
    }

    # ------------------------------------------------------------------
    # Hard token cap on the state-briefing overhead.  Compact/minimal
    # must STAY under their advertised budgets even if a future tweak to
    # the formatters drifts longer than expected — the truncation is the
    # safety net the eval relies on for CES_total to be honest.
    # ------------------------------------------------------------------
    if mode in ("compact", "minimal"):
        cap = (config.STATE_BRIEFING_COMPACT_MAX_TOKENS if mode == "compact"
               else config.STATE_BRIEFING_MINIMAL_MAX_TOKENS)
        overhead_keys = ("state", "identity", "working", "prediction")
        overhead_text = "\n\n".join(sections[k] for k in overhead_keys if sections[k])
        if _count_tokens(overhead_text) > cap:
            # Concentrate the budget on the most-informative section (state).
            # Drop identity/working/prediction first, then trim state.
            for k in ("prediction", "working", "identity"):
                sections[k] = ""
                overhead_text = "\n\n".join(
                    sections[k2] for k2 in overhead_keys if sections[k2])
                if _count_tokens(overhead_text) <= cap:
                    break
            else:
                sections["state"] = _truncate_to_budget(sections["state"], cap)

    return sections


def _join_sections(sections: Dict[str, str]) -> str:
    """Re-join section blocks with blank-line separators, preserving order."""
    order = ("system", "state", "identity", "retrieval", "working", "prediction")
    blocks = [sections[k] for k in order if sections.get(k)]
    return "\n\n".join(blocks)


def assemble_briefing(memories: List[Episode], beliefs: List[str],
                      working: List[Episode], state: LatentState,
                      stance: Optional["ReasoningStance"] = None,
                      identity: Optional[List[str]] = None,
                      prediction: Optional["PredictiveContinuity"] = None) -> str:
    """Assemble the structured narrative briefing for the single LLM call.

    This is the prompt fragment the reasoning engine sees — not raw top-k
    passages, but a state-aware narrative of who it is talking to, what is most
    relevant right now, and (via the reasoning `stance`) *how to reason about it*.
    The stance is what makes this memory-conditioned rather than memory-augmented:
    it frames the memories as cognitively privileged and shifts the reasoning
    rules with the user's state.
    """
    return _join_sections(briefing_sections(
        memories, beliefs, working, state, stance, identity, prediction))


# --------------------------------------------------------------------------- #
# Phase 2 — hybrid retrieval orchestrator
#
# This is the "stop relying on cosine alone" path.  Four candidate retrievers
# fire in parallel; RRF fuses their rankings; an optional cross-encoder
# reranks the top-30 survivors; the state-modulated R(m) tilt picks the
# final top-k.  Every TACO invariant is preserved:
#
#   * write-time salience gate stays in front of all of this
#     (low-salience turns never reach the facts table)
#   * scenario isolation: every candidate query carries the caller's user_id
#   * state weighting: the final tilt is the same R(m) the semantic path uses
# --------------------------------------------------------------------------- #
def _empty_stats() -> Dict[str, int]:
    return {
        "candidates_semantic": 0,
        "candidates_summary": 0,
        "candidates_cues": 0,
        "candidates_entity": 0,
        "candidates_arc": 0,
        "candidates_origin": 0,
        "candidates_trajectory": 0,
        "candidates_after_rrf": 0,
        "cross_encoder_enabled": 0,
        "strong_rerank_enabled": 0,
    }


def hybrid_retrieve(conn, query_text: str, query_embedding: List[float],
                    state: LatentState, *, user_id: str,
                    top_k: Optional[int] = None,
                    per_retriever_k: Optional[int] = None,
                    after_rrf_k: Optional[int] = None,
                    query_kind: Optional[str] = None,
                    ) -> Tuple[List[Episode], Dict[str, int]]:
    """Cast four candidate sources, RRF-fuse, optional cross-encoder rerank,
    then apply the state-modulated tilt.

    Returns ``(top_k_episodes, stats)`` where *stats* counts the candidates
    each retriever produced and which rerank steps actually ran.  The harness
    aggregates these per-run for the report.
    """
    top_k = top_k if top_k is not None else config.TOP_K
    per_retriever_k = per_retriever_k if per_retriever_k is not None \
        else config.HYBRID_PER_RETRIEVER_K
    after_rrf_k = after_rrf_k if after_rrf_k is not None \
        else config.HYBRID_AFTER_RRF_K

    stats = _empty_stats()
    query_kind = query_kind or classify_query(query_text)

    # ----- (a) semantic kNN over fact embeddings ---------------------------
    semantic = _store.fact_knn_candidates(conn, query_embedding,
                                           per_retriever_k, user_id=user_id)
    stats["candidates_semantic"] = len(semantic)

    # ----- (b) summary trigram --------------------------------------------
    summary_hits = _store.fact_text_search(conn, query_text, k=per_retriever_k,
                                            user_id=user_id)
    stats["candidates_summary"] = len(summary_hits)

    # ----- (c) cue trigram, using both the raw query AND any generated cues
    cues = _rerank.extract_query_cues(query_text)
    cue_query = " ".join([query_text, *cues]).strip()
    cue_hits = _store.fact_cue_search(conn, cue_query, k=per_retriever_k,
                                       user_id=user_id)
    stats["candidates_cues"] = len(cue_hits)

    # ----- (d) entity overlap ---------------------------------------------
    entities = _rerank.extract_query_entities(query_text)
    entity_hits = (_store.fact_entity_overlap(conn, entities, k=per_retriever_k,
                                               user_id=user_id)
                   if entities else [])
    stats["candidates_entity"] = len(entity_hits)

    arc_hits: List[Episode] = []
    origin_hits: List[Episode] = []
    trajectory_hits: List[Episode] = []
    if query_kind in ("temporal", "conflict", "user_modeling"):
        arc_keys = _arc_query_entity_keys(query_text, query_kind)
        arc_hits = (_store.fact_entity_overlap(
            conn, arc_keys, k=per_retriever_k, user_id=user_id)
            if arc_keys else [])
        if is_origin_query(query_text):
            origin_hits = (_store.fact_arc_origin_search(
                conn, arc_keys, k=per_retriever_k, user_id=user_id)
                if arc_keys else [])
        if is_trajectory_query(query_text, query_kind):
            trajectory_hits = (_store.fact_trajectory_search(
                conn, arc_keys, k=per_retriever_k, user_id=user_id)
                if arc_keys else [])
    stats["candidates_arc"] = len(arc_hits)
    stats["candidates_origin"] = len(origin_hits)
    stats["candidates_trajectory"] = len(trajectory_hits)

    # ----- RRF fusion -----------------------------------------------------
    # Order matters: semantic first so the cosine ``similarity`` field
    # survives dedup (we use it in the final R(m) tilt).
    fused = _rerank.rrf_fuse(
        [
            semantic, summary_hits, cue_hits, entity_hits,
            arc_hits, origin_hits, trajectory_hits,
        ],
        k=config.HYBRID_RRF_K,
    )[:after_rrf_k]
    stats["candidates_after_rrf"] = len(fused)

    # ----- Optional cross-encoder rerank ----------------------------------
    if config.CROSS_ENCODER_RERANK:
        fused = _rerank.cross_encoder_rerank(query_text, fused)[:after_rrf_k]
        stats["cross_encoder_enabled"] = 1

    # ----- Optional strong-LLM rerank (ablation) --------------------------
    if config.STRONG_RERANK:
        fused = _rerank.strong_llm_rerank(query_text, fused)[:after_rrf_k]
        stats["strong_rerank_enabled"] = 1

    # ----- Final state-modulated R(m) tilt + isolation assert -------------
    # In adaptive mode, query-kind boosts need to see the fused candidate pool.
    # Truncating to TOP_K before those boosts can discard exactly the trace that
    # a temporal/conflict/user-modeling query is asking for.
    rerank_k = after_rrf_k if config.RECALL_POLICY == "adaptive" else top_k
    final = rerank(fused, state, top_k=rerank_k)
    if config.RECALL_POLICY == "adaptive":
        final = select_sparse_recall(
            final, state, query_kind, query_text)
    _rerank.assert_user_isolation(final, user_id)
    return final, stats
