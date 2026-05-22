"""The turn pipeline — the full hierarchical retrieval + action loop (Figure 4).

    incoming message
      → analyse (E, V, salience, tone)
      → infer state S
      → orchestrator resolves the six subsystems
      → cast 20 candidates (pgvector kNN) → re-rank by R(m) → top 4
      → assemble narrative briefing
      → single reasoning-engine invocation
      → write-time salience gate (store or drop)
      → state + memory write-back  (feedback updates S)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import psycopg

from . import config, embeddings, llm
from .memory import decay, retrieval, salience, store
from .memory.episode import Episode
from .state import ContentSignal, LatentState, infer_state
from .subsystems import orchestrator
from .subsystems.orchestrator import CognitivePlan


@dataclass
class TurnResult:
    response: str
    state: LatentState
    plan: CognitivePlan
    analysis: dict
    retrieved: List[Episode]
    stored: bool
    briefing: str


class Taco:
    """The cognitive layer. One instance per user/connection."""

    def __init__(self, conn: psycopg.Connection, working_window: int = 6):
        self.conn = conn
        self.working: deque = deque(maxlen=working_window)  # L1 working memory
        for ep in store.recent_episodes(conn, limit=working_window):
            self.working.append(ep)
        self.state = store.last_state(conn) or LatentState()

    # ------------------------------------------------------------------ #
    def _hours_since_last(self) -> float:
        last = store.last_contact_time(self.conn)
        if last is None:
            return 0.0
        now = datetime.now(timezone.utc)
        return max(0.0, (now - last).total_seconds() / 3600.0)

    def turn(self, user_message: str,
             hours_since_last: Optional[float] = None) -> TurnResult:
        # 1. interoception: read the message's affect + salience in one call
        analysis = llm.analyze(user_message)

        # 2. infer the new latent state S
        gap = self._hours_since_last() if hours_since_last is None else hours_since_last
        signal = ContentSignal(emotional=analysis["emotional"],
                               vulnerability=analysis["vulnerability"])
        self.state = infer_state(self.state, signal, gap)

        # 3. orchestrator resolves all six subsystems from S
        plan = orchestrator.plan(self.state)

        # 4. retrieval: cast → re-rank → top-k
        q_emb = embeddings.embed_list(user_message)
        candidates = store.knn_candidates(self.conn, q_emb, config.CANDIDATE_CAST)
        top = retrieval.rerank(candidates, self.state, top_k=config.TOP_K)
        store.touch_access(self.conn, [m.id for m in top if m.id])  # L7 reconsolidation
        beliefs = store.belief_candidates(self.conn, q_emb, k=2)

        # 5. assemble the narrative briefing
        briefing = retrieval.assemble_briefing(
            top, beliefs, list(self.working), self.state)

        # 6. single reasoning-engine invocation
        response = llm.respond(briefing, user_message, plan.planning_depth)

        # 7. write-time salience gate on the user's turn
        stored = salience.gate(analysis["salience"], self.state)
        if stored:
            ep = salience.build_episode("user", user_message, analysis)
            ep_id = store.add_episode(self.conn, ep, q_emb, self.state)
            store.add_emotional(self.conn, ep_id, self.state.E,
                                analysis.get("tone"), analysis["salience"])

        # 8. write-back: working memory + persisted state
        self.working.append(Episode(content=user_message, role="user",
                                    tone=analysis.get("tone")))
        self.working.append(Episode(content=response, role="assistant"))
        store.log_state(self.conn, self.state)

        return TurnResult(response=response, state=self.state, plan=plan,
                          analysis=analysis, retrieved=top, stored=stored,
                          briefing=briefing)

    # ------------------------------------------------------------------ #
    def probe(self, user_message: str) -> TurnResult:
        """Read-only turn: infer state and retrieve/respond WITHOUT writing back.

        Used for evaluation so probes don't contaminate memory or each other. The
        probe's own affect still transiently shapes S (state modulates retrieval),
        but nothing is persisted.
        """
        analysis = llm.analyze(user_message)
        signal = ContentSignal(emotional=analysis["emotional"],
                               vulnerability=analysis["vulnerability"])
        state = infer_state(self.state, signal, 0.5)  # transient, not persisted
        plan = orchestrator.plan(state)

        q_emb = embeddings.embed_list(user_message)
        candidates = store.knn_candidates(self.conn, q_emb, config.CANDIDATE_CAST)
        top = retrieval.rerank(candidates, state, top_k=config.TOP_K)
        beliefs = store.belief_candidates(self.conn, q_emb, k=2)
        briefing = retrieval.assemble_briefing(top, beliefs, list(self.working), state)
        response = llm.respond(briefing, user_message, plan.planning_depth)

        return TurnResult(response=response, state=state, plan=plan,
                          analysis=analysis, retrieved=top, stored=False,
                          briefing=briefing)

    def run_decay(self, extra_weeks: float = 0.0) -> decay.DecayReport:
        """Run the forgetting engine (tier decay + abstraction-before-pruning)."""
        return decay.run(self.conn, embeddings.embed_list, llm.abstract,
                         extra_weeks=extra_weeks)
