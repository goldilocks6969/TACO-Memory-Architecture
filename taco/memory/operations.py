"""Fact write operations (Phase 1).

After a fact is extracted, it is not blindly inserted. `decide_action` compares
it against its nearest existing neighbours and chooses one of:

    ADD     — genuinely new information
    UPDATE  — supersedes an existing fact (job change, moved city); the old fact
              is kept but marked outdated and linked via `superseded_by`
    MERGE   — same fact with extra detail; updated in place, lineage extended
    DELETE  — the user explicitly retracted/contradicted a prior fact (rare)
    NOOP    — an exact duplicate; nothing to do

This is what lets the briefing surface the *current* state (Case B in the brief)
instead of piling up contradictory near-duplicates the way naive RAG does.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

import psycopg

from .. import config, llm
from ..retry import with_retries
from ..state import LatentState
from . import store
from .fact import Fact

ADD, UPDATE, DELETE, MERGE, NOOP = "ADD", "UPDATE", "DELETE", "MERGE", "NOOP"
_VALID = {ADD, UPDATE, DELETE, MERGE, NOOP}

# Heuristic similarity bands (TACO_MOCK=1): how close a neighbour must be.
_DUP_SIM = 0.97     # ≥ this → exact duplicate → NOOP
_MERGE_SIM = 0.88   # ≥ this → same fact, extra detail → MERGE


@dataclass
class Action:
    kind: str
    target_id: Optional[int] = None
    content: Optional[str] = None


def _heuristic_action(new_fact: Fact, neighbors: List[Fact]) -> Action:
    top = neighbors[0]
    if top.similarity >= _DUP_SIM:
        return Action(NOOP, target_id=top.id)
    if top.similarity >= _MERGE_SIM:
        return Action(MERGE, target_id=top.id, content=new_fact.summary)
    return Action(ADD)


_SYS = (
    "You maintain a deduplicated long-term memory. Given a NEW candidate fact and "
    "its nearest EXISTING facts (with ids), decide the single best write action. "
    "Respond with strict JSON: {\"action\": \"ADD|UPDATE|DELETE|MERGE|NOOP\", "
    "\"target_id\": <id or null>, \"content\": \"<merged/updated summary or null>\"}.\n"
    "ADD: genuinely new. UPDATE: the new fact supersedes an existing one (the user's "
    "situation changed — job, city, status); set target_id to the superseded fact and "
    "content to the new summary. MERGE: same fact with extra detail; set target_id and "
    "a combined content. DELETE: the user explicitly retracted/contradicted an existing "
    "fact with no replacement; set target_id. NOOP: an exact duplicate. Prefer ADD when "
    "unsure. Only choose DELETE when the contradiction is explicit."
)


def decide_action(new_fact: Fact, neighbors: List[Fact]) -> Action:
    """Pick a write action for `new_fact` given its nearest neighbours.

    No neighbours → trivially ADD (skips the LLM call entirely — the common case
    for novel facts, which matters under rate limits). Otherwise heuristic under
    TACO_MOCK=1, else one LLM judgement.
    """
    if not neighbors:
        return Action(ADD)
    if not llm.available():
        return _heuristic_action(new_fact, neighbors)

    listing = "\n".join(
        f"  id={n.id} (sim={n.similarity:.2f}): {n.summary}" for n in neighbors
    )
    user = f"NEW: {new_fact.summary}\nEXISTING:\n{listing}"
    try:
        resp = with_retries(lambda: llm._client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        ), label="operations.decide_action")
        d = json.loads(resp.choices[0].message.content)
    except Exception:
        return _heuristic_action(new_fact, neighbors)

    kind = str(d.get("action", ADD)).upper()
    if kind not in _VALID:
        return Action(ADD)
    target = d.get("target_id")
    target_id = int(target) if isinstance(target, (int, float)) or (
        isinstance(target, str) and target.isdigit()) else None
    # actions that need a target but didn't get a valid one degrade to ADD
    if kind in (UPDATE, DELETE, MERGE) and target_id is None:
        return Action(ADD)
    content = d.get("content")
    return Action(kind, target_id=target_id,
                  content=str(content) if content else None)


def apply(conn: psycopg.Connection, action: Action, fact: Fact,
          embedding: List[float], source_episode_id: Optional[int],
          state: Optional[LatentState] = None,
          user_id: str = store.DEFAULT_USER_ID) -> Optional[int]:
    """Execute `action`. Returns the id of the resulting current fact (or None)."""
    if action.kind == NOOP:
        return action.target_id

    if action.kind == MERGE and action.target_id is not None:
        store.merge_fact(conn, action.target_id,
                         action.content or fact.summary, embedding,
                         source_episode_id, fact.retrieval_cues,
                         user_id=user_id)
        return action.target_id

    if action.kind == DELETE and action.target_id is not None:
        store.mark_fact_outdated(conn, action.target_id, user_id=user_id)
        return None

    if action.kind == UPDATE and action.target_id is not None:
        if action.content:
            fact.summary = action.content
        new_id = store.add_fact(conn, fact, embedding, source_episode_id,
                                state, user_id=user_id)
        store.supersede_fact(conn, action.target_id, new_id, user_id=user_id)
        return new_id

    # ADD (and any degraded case)
    return store.add_fact(conn, fact, embedding, source_episode_id, state,
                          user_id=user_id)
