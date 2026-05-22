"""The turn pipeline — the full hierarchical retrieval + action loop (Figure 4).

    incoming message
      → analyse (E, V, salience, tone)
      → infer state S
      → orchestrator resolves the six subsystems + the reasoning stance
      → L9 predict trajectory (anticipatory prefetch when escalating)
      → cast 20 candidates (pgvector kNN) → re-rank by R(m) → top 4
      → L7 reconsolidate retrieved memories that are re-held from a changed state
      → assemble narrative briefing (stance + L10 self-model + L9 prediction)
      → single reasoning-engine invocation
      → write-time salience gate (store or drop)
      → L10 consolidate identity from significant moments
      → state + memory write-back  (feedback updates S)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import psycopg

from . import config, embeddings, llm
from .memory import decay, identity, reconsolidation, retrieval, salience, store
from .memory.episode import Episode
from .state import ContentSignal, LatentState, infer_state
from .subsystems import orchestrator, prediction
from .subsystems.orchestrator import CognitivePlan
from .subsystems.prediction import PredictiveContinuity


@dataclass
class TurnResult:
    response: str
    state: LatentState
    plan: CognitivePlan
    analysis: dict
    retrieved: List[Episode]
    stored: bool
    briefing: str
    prediction: Optional[PredictiveContinuity] = None
    reconsolidated: int = 0
    identity_updates: List[identity.IdentityUpdate] = field(default_factory=list)


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

        # 3. orchestrator resolves all six subsystems + the reasoning stance from S
        plan = orchestrator.plan(self.state)

        # 4. L9 predictive continuity: read the trajectory of S (incl. this turn)
        #    to anticipate where the user is heading and what to prefetch.
        history = store.recent_states(self.conn, config.PREDICT_HISTORY) + [self.state]
        recent_salient = store.recent_salient_episodes(self.conn)
        pred = prediction.predict(history, recent_salient)

        # 5. retrieval: cast → (state-gated anticipatory prefetch) → re-rank → top-k
        q_emb = embeddings.embed_list(user_message)
        candidates = store.knn_candidates(self.conn, q_emb, config.CANDIDATE_CAST)
        if pred.prefetch_query and pred.prefetch_query != user_message:
            seen = {c.id for c in candidates}
            pf_emb = embeddings.embed_list(pred.prefetch_query)
            for c in store.knn_candidates(self.conn, pf_emb, config.PREFETCH_K):
                if c.id not in seen:
                    candidates.append(c)
        top = retrieval.rerank(candidates, self.state, top_k=config.TOP_K)
        store.touch_access(self.conn, [m.id for m in top if m.id])

        # 6. L7 reconsolidation: re-remembering a memory from a changed state
        #    rewrites its meaning and attenuates its charge (before beliefs are read).
        current_tone = retrieval.state_tone(self.state)
        recon = reconsolidation.reconsolidate(
            self.conn, top, self.state, current_tone,
            llm.reconsolidate, embeddings.embed_list)

        beliefs = store.belief_candidates(self.conn, q_emb, k=2)
        identity_lines = identity.snapshot_lines(self.conn)  # L10 self-model

        # 7. assemble the narrative briefing (stance + self-model + prediction)
        briefing = retrieval.assemble_briefing(
            top, beliefs, list(self.working), self.state, plan.stance,
            identity=identity_lines, prediction=pred)

        # 8. single reasoning-engine invocation
        response = llm.respond(briefing, user_message, plan.planning_depth)

        # 9. write-time salience gate on the user's turn
        stored = salience.gate(analysis["salience"], self.state)
        identity_updates: List[identity.IdentityUpdate] = []
        if stored:
            ep = salience.build_episode("user", user_message, analysis)
            ep_id = store.add_episode(self.conn, ep, q_emb, self.state)
            store.add_emotional(self.conn, ep_id, self.state.E,
                                analysis.get("tone"), analysis["salience"])
            # L10: consolidate identity from significant moments only.
            identity_updates = identity.consolidate(
                self.conn, user_message, analysis["salience"], self.state,
                llm.extract_identity)

        # 10. write-back: working memory + persisted state
        self.working.append(Episode(content=user_message, role="user",
                                    tone=analysis.get("tone")))
        self.working.append(Episode(content=response, role="assistant"))
        store.log_state(self.conn, self.state)

        return TurnResult(response=response, state=self.state, plan=plan,
                          analysis=analysis, retrieved=top, stored=stored,
                          briefing=briefing, prediction=pred,
                          reconsolidated=recon.count,
                          identity_updates=identity_updates)

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

        history = store.recent_states(self.conn, config.PREDICT_HISTORY) + [state]
        pred = prediction.predict(history, store.recent_salient_episodes(self.conn))

        q_emb = embeddings.embed_list(user_message)
        candidates = store.knn_candidates(self.conn, q_emb, config.CANDIDATE_CAST)
        top = retrieval.rerank(candidates, state, top_k=config.TOP_K)
        beliefs = store.belief_candidates(self.conn, q_emb, k=2)
        identity_lines = identity.snapshot_lines(self.conn)
        briefing = retrieval.assemble_briefing(
            top, beliefs, list(self.working), state, plan.stance,
            identity=identity_lines, prediction=pred)
        response = llm.respond(briefing, user_message, plan.planning_depth)

        return TurnResult(response=response, state=state, plan=plan,
                          analysis=analysis, retrieved=top, stored=False,
                          briefing=briefing, prediction=pred)

    def run_decay(self, extra_weeks: float = 0.0) -> decay.DecayReport:
        """Run the forgetting engine (tier decay + abstraction-before-pruning)."""
        return decay.run(self.conn, embeddings.embed_list, llm.abstract,
                         extra_weeks=extra_weeks)
