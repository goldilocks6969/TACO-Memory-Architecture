"""The turn pipeline — the full hierarchical retrieval + action loop (Figure 4).

    incoming message
      → analyse (E, V, salience, tone)
      → infer state S
      → orchestrator resolves the six subsystems + the reasoning stance
      → L9 predict trajectory (anticipatory prefetch when escalating)
      → cast episode candidates → re-rank → L7 reconsolidate the re-held episodes
      → cast FACT candidates → re-rank by R(m) → top-k (+ always-on open threads)
      → assemble narrative briefing (stance + L10 self-model + L9 prediction)
      → single reasoning-engine invocation
      → write path: episodes gated by θ(S); facts gated by θ_facts(S), then
        ADD/UPDATE/MERGE/DELETE/NOOP reconciliation against existing facts
      → L10 consolidate identity from significant moments
      → state + memory write-back  (feedback updates S)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import psycopg

from . import config, embeddings, llm
from .memory import (decay, extract, identity, operations, reconsolidation,
                     retrieval, salience, store)
from .memory.episode import Episode
from .retry import CallTimeout
from .state import ContentSignal, LatentState, infer_state
from .subsystems import orchestrator, prediction
from .subsystems.orchestrator import CognitivePlan
from .subsystems.prediction import PredictiveContinuity


def _new_extraction_stats() -> Dict[str, int]:
    """A fresh per-Taco counter dict. ``extraction_mode`` is captured live by
    the harness so callers can scrape an aggregate run summary."""
    return {
        "light_facts_created": 0,
        "full_extract_attempts": 0,
        "full_extract_successes": 0,
        "full_extract_timeouts": 0,
        "full_extract_failures": 0,
        "full_extract_fallbacks": 0,
    }


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
    recon_beliefs: List[str] = field(default_factory=list)
    identity_updates: List[identity.IdentityUpdate] = field(default_factory=list)
    briefing_sections: Dict[str, str] = field(default_factory=dict)


class Taco:
    """The cognitive layer. One instance per user/connection.

    Pass a distinct ``user_id`` per logical user (or, in the eval harness, per
    scenario) — every store read and write is scoped to it, so personas can
    share a database without bleeding memories into each other.
    """

    def __init__(self, conn: psycopg.Connection, working_window: int = 6,
                 user_id: str = store.DEFAULT_USER_ID):
        self.conn = conn
        self.user_id = user_id
        self.working: deque = deque(maxlen=working_window)  # L1 working memory
        for ep in store.recent_episodes(conn, limit=working_window,
                                        user_id=user_id):
            self.working.append(ep)
        self.state = store.last_state(conn, user_id=user_id) or LatentState()
        # Per-instance write-path counters; the eval harness sums these across
        # scenarios to populate the run-level extraction summary.
        self.extraction_stats: Dict[str, int] = _new_extraction_stats()

    # ------------------------------------------------------------------ #
    def _hours_since_last(self) -> float:
        last = store.last_contact_time(self.conn, user_id=self.user_id)
        if last is None:
            return 0.0
        now = datetime.now(timezone.utc)
        return max(0.0, (now - last).total_seconds() / 3600.0)

    def turn(self, user_message: str,
             hours_since_last: Optional[float] = None) -> TurnResult:
        # 1. interoception + light extraction: affect + entities + cues in one call
        light = extract.light_extract(user_message)
        analysis = {"emotional": light.emotional, "vulnerability": light.vulnerability,
                    "salience": light.salience, "tone": light.tone}

        # 2. infer the new latent state S
        gap = self._hours_since_last() if hours_since_last is None else hours_since_last
        signal = ContentSignal(emotional=light.emotional,
                               vulnerability=light.vulnerability)
        self.state = infer_state(self.state, signal, gap)

        # 3. orchestrator resolves all six subsystems + the reasoning stance from S
        plan = orchestrator.plan(self.state)

        # 4. L9 predictive continuity: read the trajectory of S (incl. this turn)
        #    to anticipate where the user is heading and what to prefetch.
        history = (store.recent_states(self.conn, config.PREDICT_HISTORY,
                                       user_id=self.user_id)
                   + [self.state])
        recent_salient = store.recent_salient_episodes(self.conn,
                                                       user_id=self.user_id)
        pred = prediction.predict(history, recent_salient)

        # 5. retrieval. Episodes still feed L7 reconsolidation (they are the raw
        #    affective record); FACTS are the retrieval target for the briefing.
        q_emb = embeddings.embed_list(user_message)
        epi_cands = store.knn_candidates(self.conn, q_emb, config.CANDIDATE_CAST,
                                         user_id=self.user_id)
        top_epi = retrieval.rerank(epi_cands, self.state, top_k=config.TOP_K)
        store.touch_access(self.conn, [m.id for m in top_epi if m.id],
                           user_id=self.user_id)

        # 6. L7 reconsolidation on the re-held episodes (before beliefs are read).
        current_tone = retrieval.state_tone(self.state)
        recon = reconsolidation.reconsolidate(
            self.conn, top_epi, self.state, current_tone,
            llm.reconsolidate, embeddings.embed_list, user_id=self.user_id)

        # 6b. fact retrieval (+ state-gated anticipatory prefetch) → re-rank → top-k.
        fact_cands = store.fact_knn_candidates(
            self.conn, q_emb, config.CANDIDATE_CAST, user_id=self.user_id)
        if pred.prefetch_query and pred.prefetch_query != user_message:
            seen = {c.id for c in fact_cands}
            pf_emb = embeddings.embed_list(pred.prefetch_query)
            for c in store.fact_knn_candidates(
                    self.conn, pf_emb, config.PREFETCH_K, user_id=self.user_id):
                if c.id not in seen:
                    fact_cands.append(c)
        top = retrieval.rerank(fact_cands, self.state, top_k=config.TOP_K)
        top = self._surface_threads(top)  # open threads always included (Phase 1.6)

        beliefs = store.belief_candidates(self.conn, q_emb, k=2,
                                          user_id=self.user_id)
        identity_lines = identity.snapshot_lines(self.conn, user_id=self.user_id)

        # 7. assemble the narrative briefing (stance + self-model + prediction)
        sections = retrieval.briefing_sections(
            top, beliefs, list(self.working), self.state, plan.stance,
            identity=identity_lines, prediction=pred)
        briefing = retrieval._join_sections(sections)

        # 8. single reasoning-engine invocation
        response = llm.respond(briefing, user_message, plan.planning_depth)

        # 9. write path. Episodes are gated by the emotional threshold θ(S); facts
        #    by the lower informational gate θ_facts(S), so paraphrasable content
        #    survives even when it carries little emotional charge.
        stored = salience.gate(analysis["salience"], self.state)
        identity_updates: List[identity.IdentityUpdate] = []
        ep_id: Optional[int] = None
        if stored:
            ep = salience.build_episode("user", user_message, analysis)
            ep_id = store.add_episode(self.conn, ep, q_emb, self.state,
                                      user_id=self.user_id)
            store.add_emotional(self.conn, ep_id, self.state.E,
                                analysis.get("tone"), analysis["salience"],
                                user_id=self.user_id)
            # L10: consolidate identity from significant moments only.
            identity_updates = identity.consolidate(
                self.conn, user_message, analysis["salience"], self.state,
                llm.extract_identity, user_id=self.user_id)

        # 9b. extract + reconcile structured facts (ADD/UPDATE/MERGE/DELETE/NOOP).
        if light.salience >= self.state.theta_facts():
            self._write_facts(user_message, light, ep_id)

        # 10. write-back: working memory + persisted state
        self.working.append(Episode(content=user_message, role="user",
                                    tone=analysis.get("tone"),
                                    user_id=self.user_id))
        self.working.append(Episode(content=response, role="assistant",
                                    user_id=self.user_id))
        store.log_state(self.conn, self.state, user_id=self.user_id)

        return TurnResult(response=response, state=self.state, plan=plan,
                          analysis=analysis, retrieved=top, stored=stored,
                          briefing=briefing, prediction=pred,
                          reconsolidated=recon.count,
                          recon_beliefs=[b for _, b in recon.rewrites],
                          identity_updates=identity_updates,
                          briefing_sections=sections)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Fact extraction write path — three modes, picked live from
    # ``config.EXTRACTION_MODE`` so the eval harness can switch without
    # re-importing.
    # ------------------------------------------------------------------ #
    def _store_light_fact(self, user_message: str,
                          light: "extract.LightExtract",
                          source_episode_id: Optional[int]) -> int:
        """Synthesize a light fact (no LLM) and insert it. Returns the new id."""
        fact = extract.light_fact(user_message, light)
        f_emb = embeddings.embed_list(fact.summary)
        fid = store.add_fact(self.conn, fact, f_emb, source_episode_id,
                             self.state, user_id=self.user_id)
        self.extraction_stats["light_facts_created"] += 1
        print("[extract] light fact created", flush=True)
        return fid

    def _apply_full_facts(self, full: "extract.FullExtract",
                          source_episode_id: Optional[int]) -> None:
        """Run the rich facts through dedup (``decide_action``) + ``apply``.

        Used by the FULL mode and by AUTO mode's post-light upgrade step.
        A fact whose summary is close to the light fact merges into it
        (``operations.apply`` MERGE branch); a distinct fact ADDs alongside.
        """
        for fact in full.facts:
            f_emb = embeddings.embed_list(fact.summary)
            neighbors = store.fact_neighbors(self.conn, f_emb, k=3, min_sim=0.7,
                                             user_id=self.user_id)
            action = operations.decide_action(fact, neighbors)
            operations.apply(self.conn, action, fact, f_emb,
                             source_episode_id, self.state,
                             user_id=self.user_id)

    def _write_facts(self, user_message: str, light: "extract.LightExtract",
                     source_episode_id: Optional[int]) -> None:
        """Persist the structured fact(s) for one above-threshold turn.

        The caller (``turn``) has already enforced the salience gate
        (``light.salience >= self.state.theta_facts()``), so trivial turns
        never reach this method — that satisfies "do not store trivial
        turns as facts" without a duplicate check here.
        """
        mode = config.EXTRACTION_MODE

        if mode == "light":
            # One light fact, period. No full_extract, no decide_action — the
            # cheap, hang-free write path the first real benchmark uses.
            self._store_light_fact(user_message, light, source_episode_id)
            return

        if mode == "auto":
            # Light fact first so we never have *no* fact, even if the
            # subsequent rich extraction times out.
            self._store_light_fact(user_message, light, source_episode_id)
            if (light.salience < config.AUTO_FULL_MIN_SALIENCE
                    or len(user_message) >= config.AUTO_FULL_MAX_CHARS):
                return  # not worth the rich-extract spend / hang risk
            self.extraction_stats["full_extract_attempts"] += 1
            try:
                full = extract.full_extract(user_message, light)
            except CallTimeout as e:
                self.extraction_stats["full_extract_timeouts"] += 1
                self.extraction_stats["full_extract_fallbacks"] += 1
                print(f"[extract] full_extract TIMED OUT; keeping light "
                      f"fact: {e}", flush=True)
                return
            except Exception as e:  # noqa: BLE001 — provider errors vary
                self.extraction_stats["full_extract_failures"] += 1
                self.extraction_stats["full_extract_fallbacks"] += 1
                print(f"[extract] full_extract FAILED; keeping light fact: "
                      f"{type(e).__name__}: {e}", flush=True)
                return
            self.extraction_stats["full_extract_successes"] += 1
            try:
                self._apply_full_facts(full, source_episode_id)
                print(f"[extract] full extraction produced {len(full.facts)} "
                      "fact(s); light fact upgraded/merged", flush=True)
            except CallTimeout as e:
                # decide_action timed out — light fact still stands.
                self.extraction_stats["full_extract_timeouts"] += 1
                self.extraction_stats["full_extract_fallbacks"] += 1
                print(f"[extract] decide_action TIMED OUT; keeping light "
                      f"fact: {e}", flush=True)
            return

        # mode == "full" — original behaviour.  A hang here will raise
        # CallTimeout and abort the current turn, which is the failure mode
        # this whole option set exists to give the user a way out of.
        full = extract.full_extract(user_message, light)
        self.extraction_stats["full_extract_attempts"] += 1
        self.extraction_stats["full_extract_successes"] += 1
        self._apply_full_facts(full, source_episode_id)

    def _surface_threads(self, top: List[Episode]) -> List[Episode]:
        """Always include open threads due within 7 days, regardless of score."""
        threads = store.due_threads(self.conn, within_days=7,
                                    user_id=self.user_id)
        if not threads:
            return top
        have = {m.id for m in top}
        return [t for t in threads if t.id not in have] + top

    # ------------------------------------------------------------------ #
    def probe(self, user_message: str) -> TurnResult:
        """Read-only turn: infer state and retrieve/respond WITHOUT writing back.

        Used for evaluation so probes don't contaminate memory or each other. The
        probe's own affect still transiently shapes S (state modulates retrieval),
        but nothing is persisted.
        """
        light = extract.light_extract(user_message)
        analysis = {"emotional": light.emotional, "vulnerability": light.vulnerability,
                    "salience": light.salience, "tone": light.tone}
        signal = ContentSignal(emotional=light.emotional,
                               vulnerability=light.vulnerability)
        state = infer_state(self.state, signal, 0.5)  # transient, not persisted
        plan = orchestrator.plan(state)

        history = (store.recent_states(self.conn, config.PREDICT_HISTORY,
                                       user_id=self.user_id)
                   + [state])
        pred = prediction.predict(
            history, store.recent_salient_episodes(self.conn, user_id=self.user_id))

        q_emb = embeddings.embed_list(user_message)
        fact_cands = store.fact_knn_candidates(
            self.conn, q_emb, config.CANDIDATE_CAST, user_id=self.user_id)
        top = self._surface_threads(retrieval.rerank(fact_cands, state, top_k=config.TOP_K))
        beliefs = store.belief_candidates(self.conn, q_emb, k=2,
                                          user_id=self.user_id)
        identity_lines = identity.snapshot_lines(self.conn, user_id=self.user_id)
        sections = retrieval.briefing_sections(
            top, beliefs, list(self.working), state, plan.stance,
            identity=identity_lines, prediction=pred)
        briefing = retrieval._join_sections(sections)
        response = llm.respond(briefing, user_message, plan.planning_depth)

        return TurnResult(response=response, state=state, plan=plan,
                          analysis=analysis, retrieved=top, stored=False,
                          briefing=briefing, prediction=pred,
                          briefing_sections=sections)

    def run_decay(self, extra_weeks: float = 0.0) -> decay.DecayReport:
        """Run the forgetting engine (tier decay + abstraction-before-pruning)."""
        return decay.run(self.conn, embeddings.embed_list, llm.abstract,
                         extra_weeks=extra_weeks, user_id=self.user_id)
