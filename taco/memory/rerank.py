"""Phase 2 helpers: query-side entity / cue extraction, Reciprocal Rank Fusion,
and the optional cross-encoder + strong-LLM rerank passes.

These complement (but do not replace) the state-modulated R(m) rerank in
``retrieval.rerank``.  Cosine similarity is no longer the only retrieval
signal — RRF gives every candidate retriever an equal say in the fused
order, then TACO's state-aware ranker re-tilts the survivors.

Everything optional fails soft: missing spaCy → regex fallback; missing
fastembed → cross-encoder rerank skipped; the live benchmark does not
depend on either being installed.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config
from .episode import Episode

logger = logging.getLogger("taco.rerank")


# --------------------------------------------------------------------------- #
# Query-side entity extraction
# --------------------------------------------------------------------------- #
# Lightweight, deterministic patterns.  These cover the common probes the
# benchmark cares about (names, pets, jobs, places).  spaCy is consulted
# only when explicitly installed *and* the regex pass missed something —
# the goal is to never make the benchmark dependent on a heavyweight NLP
# install.
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
_LOWER_NAME_HINT_RE = re.compile(
    r"\bmy ([a-z]+) (?:is named|named|called)\s+([A-Z][a-z]+)"
)

_PET_WORDS = {
    "dog": "pet:dog", "puppy": "pet:dog", "labradoodle": "pet:dog",
    "cat": "pet:cat", "kitten": "pet:cat",
    "bird": "pet:bird", "parrot": "pet:bird",
    "hamster": "pet:hamster", "rabbit": "pet:rabbit", "fish": "pet:fish",
}

# Whole-phrase relationship triggers.  Order matters — longest first.
_RELATIONSHIP_TRIGGERS = (
    ("my partner", "relationship:partner"),
    ("my wife", "relationship:wife"),
    ("my husband", "relationship:husband"),
    ("my girlfriend", "relationship:girlfriend"),
    ("my boyfriend", "relationship:boyfriend"),
    ("my fiancee", "relationship:fiancee"),
    ("my fiance", "relationship:fiance"),
    ("my mom", "relationship:mother"),
    ("my mother", "relationship:mother"),
    ("my dad", "relationship:father"),
    ("my father", "relationship:father"),
    ("my sister", "relationship:sister"),
    ("my brother", "relationship:brother"),
    ("my son", "relationship:son"),
    ("my daughter", "relationship:daughter"),
    ("my therapist", "relationship:therapist"),
    ("my boss", "relationship:boss"),
    ("my manager", "relationship:manager"),
    ("my roommate", "relationship:roommate"),
    ("my friend", "relationship:friend"),
)

_NOISE_TOKENS = {
    "i", "i'm", "i'll", "i've", "you", "the", "and", "or", "but", "of", "to",
    "a", "an", "is", "was", "in", "on", "for", "that", "it", "what", "where",
    "who", "when", "how", "why", "this", "these", "those", "did", "do",
}


def _try_spacy():
    """Return a spaCy ``nlp`` callable if available, else ``None``.

    Loading the model is expensive — cache it on the function attribute so
    repeated calls don't re-import or re-download anything.
    """
    if getattr(_try_spacy, "_resolved", False):
        return getattr(_try_spacy, "_nlp", None)
    nlp = None
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # The package is installed but no model available — fall back to
            # the blank pipeline so we at least get NER off the rule-based
            # default. In practice we just skip.
            nlp = None
    except Exception:
        nlp = None
    _try_spacy._nlp = nlp  # type: ignore[attr-defined]
    _try_spacy._resolved = True  # type: ignore[attr-defined]
    return nlp


def extract_query_entities(query: str) -> List[str]:
    """Pull lightweight ``type:value`` entity keys out of *query*.

    Regex-only by default; uses spaCy as a soft enhancement when installed.
    Output keys mirror the format ``store.add_fact`` writes so RRF can match
    by string equality.  Deduplicated, lowercased values; capped at 8 keys.
    """
    if not query:
        return []
    q = query.strip()
    qlow = q.lower()

    keys: List[str] = []

    # Whole-phrase relationship triggers ("my partner Maya broke up with me").
    for needle, key in _RELATIONSHIP_TRIGGERS:
        if needle in qlow:
            keys.append(key)

    # Pet species — common in benchmark probes ("what's my dog's name?").
    for word, key in _PET_WORDS.items():
        if re.search(rf"\b{word}\b", qlow):
            keys.append(key)

    # Named entities by simple capitalization — proper nouns mid-sentence.
    # ``[A-Z][a-z]+`` only, so "USA" / "AI" / "I" don't match.
    for m in _PROPER_NOUN_RE.finditer(q):
        tok = m.group(1)
        if tok.lower() in _NOISE_TOKENS:
            continue
        # Sentence-initial "What", "Where", etc. — keep, but heuristically
        # filter common question words.
        if m.start() == 0 and tok.lower() in {"what", "where", "who", "when",
                                              "how", "why", "did", "does"}:
            continue
        keys.append(f"entity:{tok.lower()}")

    # Lower-cased noun hint ("my dog is named Max" → entity:max).
    for m in _LOWER_NAME_HINT_RE.finditer(q):
        keys.append(f"entity:{m.group(2).lower()}")

    # Optional spaCy enhancement (PERSON, ORG, GPE).
    nlp = _try_spacy()
    if nlp is not None:
        try:
            doc = nlp(q)
            for ent in doc.ents:
                low = ent.text.strip().lower()
                if not low or low in _NOISE_TOKENS:
                    continue
                if ent.label_ in ("PERSON",):
                    keys.append(f"person:{low}")
                elif ent.label_ in ("ORG",):
                    keys.append(f"org:{low}")
                elif ent.label_ in ("GPE", "LOC"):
                    keys.append(f"place:{low}")
        except Exception:
            pass

    # de-dup, preserve order, cap
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:8]


# --------------------------------------------------------------------------- #
# Query-side cue generation
# --------------------------------------------------------------------------- #
# Common probe → cue families. The mapping is deliberately narrow: we only
# add cues we are confident a fact's ``retrieval_cues`` might also carry.
_CUE_MAP: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    # Pet-name queries — Phase 2 benchmark depends on this matching cue_text.
    (("dog", "puppy", "labradoodle", "doggy"), ("dog name", "pet name")),
    (("cat", "kitten"), ("cat name", "pet name")),
    (("pet",), ("pet name",)),
    (("bird", "parrot"), ("bird name", "pet name")),
    # Grief / family loss — "I saw an old man" should be able to retrieve
    # facts whose cues include "old man" / "father loss".
    (("old man",), ("old man", "father loss", "grief")),
    (("old woman", "elderly woman"), ("old woman", "mother loss", "grief")),
    (("dad", "father", "papa"), ("father", "father loss")),
    (("mom", "mother", "mama"), ("mother", "mother loss")),
    (("died", "passed away", "death", "funeral"), ("loss", "grief", "bereavement")),
    (("breakup", "broke up", "split with"), ("breakup", "relationship end")),
    (("divorce",), ("divorce", "relationship end")),
    # Work
    (("hired", "got the job"), ("job", "career win", "new job")),
    (("fired", "laid off", "let go"), ("job", "job loss")),
    (("interview",), ("interview", "interview anxiety")),
    (("promotion", "raise"), ("promotion", "career win")),
    # Moves / places
    (("moved to", "relocate", "relocated"), ("city move", "relocation")),
    # Health
    (("diagnos",), ("diagnosis", "health")),
    (("medication", "meds", "prescription"), ("medication", "health")),
    (("cancer", "biopsy", "chemo"), ("cancer", "health")),
    # Emotional
    (("anxiety", "panic", "worried"), ("anxiety",)),
    (("depressed", "depression"), ("depression",)),
)


def extract_query_cues(query: str) -> List[str]:
    """Generate likely ``retrieval_cues`` phrases for *query*.

    Output is used both as a text input to ``store.fact_cue_search`` (so the
    trigram index does the real work) and as a stand-alone signal we
    deduplicate.  Capped at 6 to keep the cue query short.
    """
    if not query:
        return []
    q = query.lower()
    out: List[str] = []
    for needles, cues in _CUE_MAP:
        if any(n in q for n in needles):
            for c in cues:
                if c not in out:
                    out.append(c)
    return out[:6]


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(rankings: Sequence[Sequence[Episode]], k: int = 60) -> List[Episode]:
    """Reciprocal Rank Fusion across an arbitrary number of ranked lists.

    For each ranking, every item contributes ``1 / (k + rank)`` to its
    cumulative score.  Items are deduplicated by ``id``; the *first* Episode
    seen wins so callers can bias toward a preferred retriever (semantic
    first → its cosine ``similarity`` survives the fusion).

    Items without an ``id`` (defensive — store reads always populate it)
    fall back to using their ``content`` as the dedup key.
    """
    scores: Dict = defaultdict(float)
    fact_by_id: Dict = {}
    for ranking in rankings:
        if not ranking:
            continue
        for rank, fact in enumerate(ranking, 1):
            key = fact.id if getattr(fact, "id", None) is not None else fact.content
            scores[key] += 1.0 / (k + rank)
            if key not in fact_by_id:
                fact_by_id[key] = fact
    # Stable, descending by RRF score; ties break on original first-seen order.
    return sorted(
        fact_by_id.values(),
        key=lambda f: scores[f.id if getattr(f, "id", None) is not None else f.content],
        reverse=True,
    )


# --------------------------------------------------------------------------- #
# Optional cross-encoder rerank (BAAI/bge-reranker-v2-m3 via fastembed)
# --------------------------------------------------------------------------- #
def _try_cross_encoder():
    """Lazy-load a cross-encoder.  Cached on the function attribute.  Returns
    a ``(model, label)`` tuple or ``None`` on any import / load failure."""
    if getattr(_try_cross_encoder, "_resolved", False):
        return getattr(_try_cross_encoder, "_model", None)
    model = None
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore
        model = TextCrossEncoder("BAAI/bge-reranker-v2-m3")
    except Exception as e:
        logger.warning(
            "cross-encoder rerank unavailable (%s); falling back to RRF order",
            e,
        )
        model = None
    _try_cross_encoder._model = model  # type: ignore[attr-defined]
    _try_cross_encoder._resolved = True  # type: ignore[attr-defined]
    return model


def cross_encoder_rerank(query: str, candidates: List[Episode]) -> List[Episode]:
    """Re-order *candidates* by a cross-encoder query/document relevance score.

    No-op (returns *candidates* unchanged) when fastembed isn't installed or
    model load fails — the calling orchestrator is expected to log the
    fallback once at startup.
    """
    if not candidates:
        return candidates
    model = _try_cross_encoder()
    if model is None:
        return candidates
    try:
        docs = [ep.content for ep in candidates]
        scores = list(model.rerank(query, docs))
    except Exception as e:
        logger.warning("cross_encoder_rerank failed (%s); keeping RRF order", e)
        return candidates
    paired = sorted(zip(scores, candidates), key=lambda p: p[0], reverse=True)
    return [ep for _, ep in paired]


# --------------------------------------------------------------------------- #
# Optional strong-LLM rerank (intentionally separate from cross-encoder)
# --------------------------------------------------------------------------- #
def strong_llm_rerank(query: str, candidates: List[Episode],
                       judge_fn=None) -> List[Episode]:
    """Score every candidate 0-10 for its relevance to *query* using an LLM.

    ``judge_fn(query, candidate_text) -> int`` is injected for testability.
    When unset, falls back to ``taco.llm`` — and silently no-ops when no
    API key is available so the benchmark never crashes on this knob.
    Off by default (``config.STRONG_RERANK = 0``); the eval harness leaves
    it off unless an operator explicitly enables it.
    """
    if not candidates:
        return candidates
    if judge_fn is None:
        try:
            from .. import llm
            from ..retry import with_retries
            if not llm.available():
                return candidates

            def judge_fn(q: str, doc: str) -> int:  # type: ignore[no-redef]
                resp = with_retries(lambda: llm._client().chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content":
                         "Score 0-10 how relevant the candidate memory is to "
                         "answering the user's question. Reply JSON: {\"score\":0-10}"},
                        {"role": "user", "content":
                         f"QUERY: {q}\nCANDIDATE: {doc}"},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=12,
                    timeout=config.LLM_REQUEST_TIMEOUT_S,
                ), label="rerank.strong_llm")
                import json
                try:
                    return int(json.loads(resp.choices[0].message.content)
                               .get("score", 0))
                except Exception:
                    return 0
        except Exception:
            return candidates

    scored = []
    for ep in candidates:
        try:
            s = float(judge_fn(query, ep.content))
        except Exception:
            s = 0.0
        scored.append((s, ep))
    scored.sort(key=lambda p: p[0], reverse=True)
    return [ep for _, ep in scored]


# --------------------------------------------------------------------------- #
# Assertion helper — used by the harness and by hybrid_retrieve as a
# belt-and-suspenders check that no row from another user_id namespace
# slipped through.
# --------------------------------------------------------------------------- #
def assert_user_isolation(candidates: List[Episode], user_id: str) -> None:
    for ep in candidates:
        if getattr(ep, "user_id", user_id) != user_id:
            raise RuntimeError(
                f"hybrid retrieval leaked across user_id namespace: "
                f"got user_id={ep.user_id!r}, expected {user_id!r}, "
                f"content={ep.content[:120]!r}"
            )
