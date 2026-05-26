"""Run the continuity benchmark: ingest each scenario through both pipelines,
probe both, judge the responses, and record rigorous per-probe measurements.

Debuggability:
  • Every major step prints a flushed progress line so a hang is visible.
  • Every probe row is appended to ``eval/out/results_partial.jsonl`` *as soon
    as it is scored*, so a crash or timeout still preserves the work-so-far.
  • Every LLM/embedding call is bounded by a per-attempt timeout (see
    ``taco/retry.py``); a hang raises ``CallTimeout`` naming the call.
  • ``TACO_MOCK=1`` completes the whole eval without any external API call.

For every probe and condition we record (per professor's review):
  • score                  — LLM-judged continuity (0-100)
  • retrieval_tokens       — tokens of the retrieved memory/fact payload only
  • state_briefing_tokens  — tokens of TACO's structured state briefing only
                              (state line + stance + identity + working + prediction)
  • memory_context_tokens  — retrieval_tokens + state_briefing_tokens
  • system_prompt_tokens   — fixed scaffolding (the "You are…" preamble)
  • user_query_tokens      — the probe text
  • total_context_tokens   — system + memory_context + user_query
  • answer_tokens          — tokens of the generated answer
  • judge_input_tokens     — rubric + judge user prompt
  • judge_output_tokens    — judge's reply
  • n_memories             — number of retrieved items
  • retrieved              — the retrieved texts (qualitative inspection)

Two efficiency metrics are reported (downstream, in metrics.py):
  CES_retrieval = Q / avg_retrieval_tokens * 1000  (memory-policy efficiency)
  CES_total     = Q / avg_total_context_tokens * 1000  (full-prompt efficiency)

Isolation: Taco uses a dedicated `taco_eval` database, truncated between
scenarios so histories never bleed across personas.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from taco import config, db
from taco.pipeline import Taco
from taco.retry import CallTimeout

from . import judge
from .baseline import NaiveRAG
from .dataset import SCENARIOS, Scenario
from .tokens import counter

EVAL_DSN = os.getenv("TACO_EVAL_DSN", "postgresql://localhost:5432/taco_eval")
RAG_TOPK = 5  # naive RAG retrieves top-5 session chunks

# Defense-in-depth: even with per-scenario user_id namespacing, every memory
# table is truncated between scenarios so a leak (or a future namespace
# regression) cannot silently survive into the next persona.  ``facts`` and
# ``reflections`` were missing from this list previously, which is how grief
# memories were retrieved in the job_loss / breakup / pregnancy probes.
_TABLES = ("episodes", "facts", "semantic_beliefs", "emotional_timeline",
           "reflections", "procedural", "identity", "state_log")

# Token attribution: which Taco briefing sections count as state-briefing overhead
# (everything except the fixed system prompt and the retrieved memory payload).
_STATE_SECTIONS = ("state", "identity", "working", "prediction")

# Per-probe rows stream to this file as they are scored, so a hang/crash never
# loses the prior work. The full harness still returns the in-memory list too.
OUT_DIR = Path(__file__).resolve().parent / "out"
PARTIAL_PATH = OUT_DIR / "results_partial.jsonl"


def _log(msg: str) -> None:
    """Single-line, flushed progress log. Timestamps are wall-clock seconds
    since the harness started, so a stuck step is obvious in the transcript."""
    print(f"[harness +{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def _timed(label: str, fn):
    """Run *fn*; log a before/after line bracketing it with elapsed seconds."""
    _log(f"→ {label} ...")
    t0 = time.monotonic()
    try:
        return fn()
    finally:
        _log(f"← {label} done ({time.monotonic() - t0:.2f}s)")


def _append_partial(row: Dict) -> None:
    """Append one probe row to results_partial.jsonl atomically (one line)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PARTIAL_PATH.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _reset_partial() -> None:
    """Truncate the partial file at the start of a new run."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL_PATH.write_text("")


def _reset(conn) -> None:
    conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY")


def _assert_no_cross_scenario_leak(retrieved, scenario_name: str) -> None:
    """Catch any cross-scenario contamination at the source.

    Every item returned by ``Taco.probe`` carries the ``user_id`` it was
    stored under (via ``Episode.user_id``).  If any item disagrees with the
    current scenario, fail loud — a silent leak invalidates the benchmark.
    """
    for ep in retrieved:
        u = getattr(ep, "user_id", None)
        if u and u != scenario_name:
            raise RuntimeError(
                "Cross-scenario memory leak detected: retrieved item with "
                f"user_id={u!r} during scenario={scenario_name!r} "
                f"(content={getattr(ep, 'content', '?')[:120]!r})"
            )


def _ingest(scenario: Scenario, taco: Taco, rag: NaiveRAG) -> int:
    """Ingest sessions. RAG stores one raw multi-turn chunk per session; Taco
    processes each turn through its write-time salience gate. Returns total turns.

    When ``config.SKIP_RESPOND_DURING_INGEST`` is set, Taco's turn loop runs
    with ``generate_response=False`` — the response LLM call (the most common
    hang during long ingests) is skipped, while state inference, episode /
    fact / identity writes, and state logging all still execute.  The probe
    phase still invokes respond normally so the benchmark numbers are unaffected.
    """
    skip_respond = config.SKIP_RESPOND_DURING_INGEST
    if skip_respond:
        _log("[ingest] response generation skipped "
             "(TACO_EVAL_SKIP_RESPOND_DURING_INGEST=1)")
    turns = 0
    for si, session in enumerate(scenario.sessions, 1):
        rag.ingest_chunk("  ".join(t.text for t in session))  # raw multi-turn chunk
        for ti, turn in enumerate(session, 1):
            taco.turn(turn.text, hours_since_last=turn.gap_hours,
                      generate_response=not skip_respond)
            turns += 1
        _log(f"  ingested session {si}/{len(scenario.sessions)} "
             f"({len(session)} turns, total {turns})")
    return turns


def _taco_corpus_tokens(conn, ntok) -> int:
    rows = conn.execute("SELECT content FROM episodes").fetchall()
    beliefs = conn.execute("SELECT belief FROM semantic_beliefs").fetchall()
    return sum(ntok(r[0]) for r in rows) + sum(ntok(b[0]) for b in beliefs)


def _taco_token_breakdown(t, probe_text: str, response: str, ntok) -> Dict[str, int]:
    """Attribute Taco's tokens to retrieval / state-briefing / system / query / answer."""
    sections = t.briefing_sections or {}
    retrieval_tokens = ntok(sections.get("retrieval", ""))
    state_briefing_tokens = sum(ntok(sections.get(k, "")) for k in _STATE_SECTIONS)
    system_prompt_tokens = ntok(sections.get("system", ""))
    user_query_tokens = ntok(probe_text)
    memory_context_tokens = retrieval_tokens + state_briefing_tokens
    return {
        "retrieval_tokens": retrieval_tokens,
        "state_briefing_tokens": state_briefing_tokens,
        "memory_context_tokens": memory_context_tokens,
        "system_prompt_tokens": system_prompt_tokens,
        "user_query_tokens": user_query_tokens,
        "total_context_tokens": (system_prompt_tokens + memory_context_tokens
                                 + user_query_tokens),
        "answer_tokens": ntok(response),
    }


def _rag_token_breakdown(sections: Dict[str, str], retrieval_tokens: int,
                         probe_text: str, response: str, ntok) -> Dict[str, int]:
    """Same breakdown for RAG. State-briefing is always 0 (no structured state)."""
    system_prompt_tokens = ntok(sections.get("system", ""))
    state_briefing_tokens = ntok(sections.get("state", ""))  # always 0 for RAG
    user_query_tokens = ntok(probe_text)
    memory_context_tokens = retrieval_tokens + state_briefing_tokens
    return {
        "retrieval_tokens": retrieval_tokens,
        "state_briefing_tokens": state_briefing_tokens,
        "memory_context_tokens": memory_context_tokens,
        "system_prompt_tokens": system_prompt_tokens,
        "user_query_tokens": user_query_tokens,
        "total_context_tokens": (system_prompt_tokens + memory_context_tokens
                                 + user_query_tokens),
        "answer_tokens": ntok(response),
    }


# Wall-clock anchor used by _log; set by run() at the top.
_T0 = time.monotonic()


def run() -> Dict:
    """Execute the benchmark and return the in-memory result.

    Mock mode (``TACO_MOCK=1``) is supported and exercises every code path
    without any external API call — useful for plumbing checks and CI.
    """
    global _T0
    _T0 = time.monotonic()
    _reset_partial()

    # Eval defaults to the *compact* state briefing (≤80 tokens of overhead per
    # turn) so CES_total reflects a realistic deployment.  An explicit
    # TACO_STATE_BRIEFING_MODE env var (full | compact | minimal) overrides
    # this — set TACO_STATE_BRIEFING_MODE=full for a debug run.
    if not os.getenv("TACO_STATE_BRIEFING_MODE"):
        config.STATE_BRIEFING_MODE = "compact"

    # Eval defaults to the LIGHT extraction mode so a hung full_extract cannot
    # wedge an entire scenario ingest.  ``TACO_EVAL_EXTRACTION_MODE`` (light |
    # auto | full) overrides for product-fidelity or debug runs.
    if not os.getenv("TACO_EVAL_EXTRACTION_MODE"):
        config.EXTRACTION_MODE = "light"

    # Eval defaults to skipping ``llm.respond`` during ingest — the per-turn
    # response is not part of the benchmark target (probes are), and a single
    # wedged respond call would otherwise abort a 50-turn scenario replay.
    # ``TACO_EVAL_SKIP_RESPOND_DURING_INGEST=0`` re-enables it.
    if "TACO_EVAL_SKIP_RESPOND_DURING_INGEST" not in os.environ:
        config.SKIP_RESPOND_DURING_INGEST = True

    # Eval defaults to the LOCAL heuristic light extractor — even when an API
    # key is configured.  The LLM-backed light tier can wedge during ingest in
    # the same way ``respond`` can; the heuristic carries every field the
    # downstream light_fact builder needs (salience, tone, vulnerability,
    # retrieval_cues, entities).  ``TACO_EVAL_LIGHT_EXTRACT_LOCAL=0`` opts back
    # in to the LLM call.
    if "TACO_EVAL_LIGHT_EXTRACT_LOCAL" not in os.environ:
        config.LIGHT_EXTRACT_LOCAL = True

    # Eval defaults to fast responses: a short "answer concisely from memory"
    # system prompt + max_tokens cap, so a single runaway generation cannot
    # wedge the benchmark.  TACO_EVAL_FAST_RESPONSES=0 falls back to the full
    # cognitive-layer briefing prompt.
    if "TACO_EVAL_FAST_RESPONSES" not in os.environ:
        config.FAST_RESPONSES = True

    # FAST_LIVE — the master "live benchmark" flag.  Implies FAST_RESPONSES,
    # forces temperature=0 in respond, and switches the judge to its short
    # rubric.  Defaults ON for ``eval.run`` so the live benchmark is fast and
    # deterministic by default; ``TACO_EVAL_FAST_LIVE=0`` re-enables the
    # verbose prompts for a fidelity ablation.
    if "TACO_EVAL_FAST_LIVE" not in os.environ:
        config.FAST_LIVE = True

    # Phase 2 hybrid retrieval — the eval default.  Combines semantic kNN,
    # summary trigram, cue trigram, and entity overlap via RRF.  Product /
    # CLI users can stay on the simpler ``semantic`` path by leaving the
    # env unset; the harness opts in for everyone unless TACO_RETRIEVAL_MODE
    # is explicitly set to "semantic" (e.g. for an ablation run).
    if "TACO_RETRIEVAL_MODE" not in os.environ:
        config.RETRIEVAL_MODE = "hybrid"

    mode = "MOCK" if config.MOCK else "LIVE"
    _log(f"starting harness.run() · mode={mode} · model={config.LLM_MODEL} "
         f"· response_model={config.RESPONSE_MODEL} · "
         f"judge_model={config.JUDGE_MODEL} "
         f"· retrieval_mode={config.RETRIEVAL_MODE} "
         f"· state_briefing_mode={config.STATE_BRIEFING_MODE} "
         f"· extraction_mode={config.EXTRACTION_MODE} "
         f"· skip_respond_during_ingest={int(config.SKIP_RESPOND_DURING_INGEST)} "
         f"· light_extract_local={int(config.LIGHT_EXTRACT_LOCAL)} "
         f"· fast_responses={int(config.FAST_RESPONSES)} "
         f"· fast_live={int(config.FAST_LIVE)} "
         f"· cross_encoder={int(config.CROSS_ENCODER_RERANK)} "
         f"· strong_rerank={int(config.STRONG_RERANK)} "
         f"· max_tokens response/judge={config.RESPONSE_MAX_TOKENS}/"
         f"{config.JUDGE_MAX_TOKENS} · sdk_timeout={config.LLM_REQUEST_TIMEOUT_S:.0f}s")
    if config.LIMIT_SCENARIOS is not None or config.LIMIT_PROBES is not None:
        _log(f"smoke limits: scenarios≤{config.LIMIT_SCENARIOS} · "
             f"probes/scenario≤{config.LIMIT_PROBES}")
    _log(f"loading {len(SCENARIOS)} scenarios from eval.dataset")
    ntok = counter(config.LLM_MODEL)

    _log(f"connecting to Postgres at {EVAL_DSN} (init_db + connect)")
    db.init_db(EVAL_DSN)
    conn = db.connect(EVAL_DSN)
    _log("Postgres ready")

    # Truncate every memory table once at the top, in case a previous abandoned
    # run left rows behind (the per-scenario truncate inside the loop would not
    # touch a leftover user_id that no current scenario claims).
    conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY")
    _log(f"truncated tables: {', '.join(_TABLES)}")

    rows: List[Dict] = []
    scale: List[Dict] = []
    extraction_totals: Dict[str, int] = {
        "light_facts_created": 0,
        "full_extract_attempts": 0,
        "full_extract_successes": 0,
        "full_extract_timeouts": 0,
        "full_extract_failures": 0,
        "full_extract_fallbacks": 0,
    }
    # Phase 2 hybrid retrieval — totals across every scenario the run touches.
    retrieval_totals: Dict[str, int] = {
        "retrieval_calls": 0,
        "candidates_semantic": 0,
        "candidates_summary": 0,
        "candidates_cues": 0,
        "candidates_entity": 0,
        "candidates_after_rrf": 0,
        "cross_encoder_calls": 0,
        "strong_rerank_calls": 0,
    }
    # Per-stage timeout counters.  A single hung respond / answer / judge
    # call during one probe must not abort the whole benchmark: we record an
    # error row, bump the counter, and continue.
    timeout_counts: Dict[str, int] = {
        "taco_probe": 0, "rag_answer": 0,
        "judge_taco": 0, "judge_rag": 0,
        "total": 0,
    }

    # Apply the smoke-mode limits (if set) once, at the top of the loop.
    scenarios_to_run = (SCENARIOS if config.LIMIT_SCENARIOS is None
                        else SCENARIOS[: config.LIMIT_SCENARIOS])
    if len(scenarios_to_run) != len(SCENARIOS):
        _log(f"running {len(scenarios_to_run)}/{len(SCENARIOS)} scenarios "
             f"(TACO_EVAL_LIMIT_SCENARIOS={config.LIMIT_SCENARIOS})")

    for si, scenario in enumerate(scenarios_to_run, 1):
        _log(f"=== scenario {si}/{len(scenarios_to_run)}: {scenario.name} (start) ===")
        _reset(conn)
        # The scenario name is the user_id for the duration of this scenario.
        # All Taco writes/reads filter on it, so even if the truncate above
        # were skipped, retrieval would still not see another scenario.
        taco = Taco(conn, user_id=scenario.name)
        rag = NaiveRAG(k=RAG_TOPK)
        n_turns = _timed(
            f"ingest scenario {scenario.name}",
            lambda: _ingest(scenario, taco, rag),
        )
        history_tokens = sum(ntok(t.text) for s in scenario.sessions for t in s)
        _log(f"scenario {scenario.name}: ingested {n_turns} turns in "
             f"{len(scenario.sessions)} sessions ({history_tokens} tok)")

        scale.append(dict(
            scenario=scenario.name,
            n_turns=n_turns,
            n_sessions=len(scenario.sessions),
            history_tokens=history_tokens,
            rag_corpus_tokens=rag.corpus_tokens(ntok),
            taco_corpus_tokens=_taco_corpus_tokens(conn, ntok),
        ))

        probes_to_run = (scenario.probes if config.LIMIT_PROBES is None
                         else scenario.probes[: config.LIMIT_PROBES])
        if len(probes_to_run) != len(scenario.probes):
            _log(f"  running {len(probes_to_run)}/{len(scenario.probes)} probes "
                 f"(TACO_EVAL_LIMIT_PROBES={config.LIMIT_PROBES})")

        for pi, probe in enumerate(probes_to_run, 1):
            _log(f"  probe {pi}/{len(probes_to_run)} [{probe.kind}] "
                 f"start: {probe.text[:80]!r}")

            base = dict(scenario=scenario.name, probe=probe.text,
                        kind=probe.kind, salience=probe.salience,
                        ground_truth=probe.ground_truth)

            def _timeout_row(condition: str, stage: str, exc: BaseException) -> Dict:
                """A minimal partial row recorded when an LLM stage hangs.
                The metrics aggregator skips rows where ``error`` is set, so
                these don't contaminate CES means.  The row carries the model,
                SDK timeout, and max-tokens that were active when the call
                hung — invaluable when triaging why the live benchmark stalls.
                """
                timeout_counts[stage] += 1
                timeout_counts["total"] += 1
                # Stage → which knob was in effect.  ``taco_probe`` /
                # ``rag_answer`` run through ``llm.respond``; the two judge
                # stages run through ``judge.judge``.
                if stage in ("taco_probe", "rag_answer"):
                    stage_model = config.RESPONSE_MODEL
                    stage_max_tokens = config.RESPONSE_MAX_TOKENS
                else:
                    stage_model = config.JUDGE_MODEL
                    stage_max_tokens = config.JUDGE_MAX_TOKENS
                row = {**base, "condition": condition, "error": "timeout",
                       "error_stage": stage, "error_detail": str(exc)[:200],
                       "error_model": stage_model,
                       "error_timeout_s": config.LLM_REQUEST_TIMEOUT_S,
                       "error_max_tokens": stage_max_tokens,
                       "score": None, "n_memories": 0,
                       "response": "", "retrieved": []}
                _append_partial(row)
                _log(f"  ! probe {pi}/{len(probes_to_run)} {condition} "
                     f"TIMED OUT in {stage} · model={stage_model} · "
                     f"timeout={config.LLM_REQUEST_TIMEOUT_S:.0f}s · "
                     f"max_tokens={stage_max_tokens}; "
                     f"recorded error row and continuing")
                return row

            # --- TACO response ----------------------------------------------
            try:
                t = _timed(f"taco.probe (probe {pi})",
                           lambda: taco.probe(probe.text))
            except CallTimeout as e:
                rows.append(_timeout_row("taco", "taco_probe", e))
                continue  # neither side can score without a response either
            _assert_no_cross_scenario_leak(t.retrieved, scenario.name)
            t_retr = [ep.content for ep in t.retrieved]
            t_resp = t.response

            # --- RAG response -----------------------------------------------
            try:
                r_resp, r_sections, r_retr_tok, r_n, r_passages = _timed(
                    f"rag.answer (probe {pi})",
                    lambda: rag.answer(probe.text, ntok),
                )
            except CallTimeout as e:
                # We have a TACO response but no RAG counterpart — record both
                # so the partial JSONL is consistent: a real TACO row plus a
                # RAG error row.  The TACO judge still runs below.
                rag_row = _timeout_row("rag", "rag_answer", e)
                rows.append(rag_row)
                r_resp = ""
                r_sections = {"system": "", "state": "", "retrieval": ""}
                r_retr_tok = 0
                r_n = 0
                r_passages = []
                rag_done = False
            else:
                rag_done = True

            # --- judge TACO -------------------------------------------------
            try:
                t_judge = _timed(
                    f"judge taco (probe {pi}, kind={probe.kind})",
                    lambda: judge.judge(probe.text, probe.ground_truth,
                                        probe.kind, t_resp),
                )
            except CallTimeout as e:
                rows.append(_timeout_row("taco", "judge_taco", e))
                if rag_done:
                    # try the rag judge anyway so we still get half a probe
                    try:
                        r_judge = _timed(
                            f"judge rag  (probe {pi}, kind={probe.kind})",
                            lambda: judge.judge(probe.text, probe.ground_truth,
                                                probe.kind, r_resp),
                        )
                    except CallTimeout as e2:
                        rows.append(_timeout_row("rag", "judge_rag", e2))
                        continue
                    r_tok = _rag_token_breakdown(r_sections, r_retr_tok,
                                                 probe.text, r_resp, ntok)
                    rag_row = {**base, "condition": "rag",
                               "score": r_judge["score"], **r_tok,
                               "total_tokens": r_tok["total_context_tokens"],
                               "judge_input_tokens": ntok(r_judge.get("input_text", "")),
                               "judge_output_tokens": ntok(r_judge.get("output_text", "")),
                               "n_memories": r_n, "response": r_resp,
                               "retrieved": r_passages}
                    rows.append(rag_row)
                    _append_partial(rag_row)
                continue

            # --- judge RAG (only if rag actually answered) -------------------
            if rag_done:
                try:
                    r_judge = _timed(
                        f"judge rag  (probe {pi}, kind={probe.kind})",
                        lambda: judge.judge(probe.text, probe.ground_truth,
                                            probe.kind, r_resp),
                    )
                except CallTimeout as e:
                    rows.append(_timeout_row("rag", "judge_rag", e))
                    # still record TACO's side below
                    r_judge = None
            else:
                r_judge = None

            # --- token accounting + persist real rows ------------------------
            t_tok = _taco_token_breakdown(t, probe.text, t_resp, ntok)
            taco_row = {**base, "condition": "taco", "score": t_judge["score"],
                        **t_tok,
                        "total_tokens": t_tok["total_context_tokens"],
                        "judge_input_tokens": ntok(t_judge.get("input_text", "")),
                        "judge_output_tokens": ntok(t_judge.get("output_text", "")),
                        "n_memories": len(t_retr),
                        "response": t_resp, "retrieved": t_retr}
            rows.append(taco_row)
            _append_partial(taco_row)

            if rag_done and r_judge is not None:
                r_tok = _rag_token_breakdown(r_sections, r_retr_tok,
                                             probe.text, r_resp, ntok)
                rag_row = {**base, "condition": "rag",
                           "score": r_judge["score"], **r_tok,
                           "total_tokens": r_tok["total_context_tokens"],
                           "judge_input_tokens": ntok(r_judge.get("input_text", "")),
                           "judge_output_tokens": ntok(r_judge.get("output_text", "")),
                           "n_memories": r_n, "response": r_resp,
                           "retrieved": r_passages}
                rows.append(rag_row)
                _append_partial(rag_row)
            _log(f"  probe {pi}/{len(probes_to_run)} done: "
                 f"taco={t_judge['score']} "
                 f"rag={'-' if r_judge is None else r_judge['score']} "
                 f"(written to {PARTIAL_PATH.name})")

        # Roll up this scenario's write-path counters into the run total.
        for k in extraction_totals:
            extraction_totals[k] += taco.extraction_stats.get(k, 0)
        # Same for the Phase-2 retrieval counters.
        for k in retrieval_totals:
            retrieval_totals[k] += taco.retrieval_stats.get(k, 0)
        _log(f"=== scenario {si}/{len(scenarios_to_run)}: {scenario.name} (done, "
             f"{len(probes_to_run)} probes scored) ===")

    conn.close()
    attempts = extraction_totals["full_extract_attempts"]
    successes = extraction_totals["full_extract_successes"]
    rich_rate = round(successes / attempts, 3) if attempts else None
    _log(f"harness.run() complete · extraction={config.EXTRACTION_MODE} · "
         f"light_facts={extraction_totals['light_facts_created']} · "
         f"full_extract attempts/successes/timeouts/fallbacks="
         f"{attempts}/{successes}/{extraction_totals['full_extract_timeouts']}/"
         f"{extraction_totals['full_extract_fallbacks']}")
    if timeout_counts["total"]:
        _log(f"probe-stage timeouts: total={timeout_counts['total']} · "
             f"taco_probe={timeout_counts['taco_probe']} · "
             f"rag_answer={timeout_counts['rag_answer']} · "
             f"judge_taco={timeout_counts['judge_taco']} · "
             f"judge_rag={timeout_counts['judge_rag']}")
    _log("closing DB connection")
    return {"rows": rows, "scale": scale,
            "meta": {"model": config.LLM_MODEL, "rag_topk": RAG_TOPK,
                     "taco_topk": config.TOP_K, "mode": mode,
                     "state_briefing_mode": config.STATE_BRIEFING_MODE,
                     "extraction_mode": config.EXTRACTION_MODE,
                     "response_model": config.RESPONSE_MODEL,
                     "judge_model": config.JUDGE_MODEL,
                     "response_max_tokens": config.RESPONSE_MAX_TOKENS,
                     "judge_max_tokens": config.JUDGE_MAX_TOKENS,
                     "llm_request_timeout_s": config.LLM_REQUEST_TIMEOUT_S,
                     "fast_responses": config.FAST_RESPONSES,
                     "fast_live": config.FAST_LIVE,
                     "limit_scenarios": config.LIMIT_SCENARIOS,
                     "limit_probes": config.LIMIT_PROBES,
                     "scenarios_run": len(scenarios_to_run),
                     "retrieval_mode": config.RETRIEVAL_MODE,
                     "cross_encoder_enabled": bool(config.CROSS_ENCODER_RERANK),
                     "strong_rerank_enabled": bool(config.STRONG_RERANK),
                     "retrieval_stats": dict(retrieval_totals),
                     "timeouts": timeout_counts["total"],
                     "timeouts_by_stage": dict(timeout_counts),
                     "extraction_stats": {
                         **extraction_totals,
                         "rich_extraction_success_rate": rich_rate,
                     }}}
