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
    """
    turns = 0
    for si, session in enumerate(scenario.sessions, 1):
        rag.ingest_chunk("  ".join(t.text for t in session))  # raw multi-turn chunk
        for ti, turn in enumerate(session, 1):
            taco.turn(turn.text, hours_since_last=turn.gap_hours)
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

    mode = "MOCK" if config.MOCK else "LIVE"
    _log(f"starting harness.run() · mode={mode} · model={config.LLM_MODEL} "
         f"· state_briefing_mode={config.STATE_BRIEFING_MODE}")
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

    for si, scenario in enumerate(SCENARIOS, 1):
        _log(f"=== scenario {si}/{len(SCENARIOS)}: {scenario.name} (start) ===")
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

        for pi, probe in enumerate(scenario.probes, 1):
            _log(f"  probe {pi}/{len(scenario.probes)} [{probe.kind}] "
                 f"start: {probe.text[:80]!r}")

            t = _timed(f"taco.probe (probe {pi})", lambda: taco.probe(probe.text))
            _assert_no_cross_scenario_leak(t.retrieved, scenario.name)
            t_retr = [ep.content for ep in t.retrieved]
            t_resp = t.response

            r_resp, r_sections, r_retr_tok, r_n, r_passages = _timed(
                f"rag.answer (probe {pi})",
                lambda: rag.answer(probe.text, ntok),
            )

            t_judge = _timed(
                f"judge taco (probe {pi}, kind={probe.kind})",
                lambda: judge.judge(probe.text, probe.ground_truth,
                                    probe.kind, t_resp),
            )
            r_judge = _timed(
                f"judge rag  (probe {pi}, kind={probe.kind})",
                lambda: judge.judge(probe.text, probe.ground_truth,
                                    probe.kind, r_resp),
            )

            t_tok = _taco_token_breakdown(t, probe.text, t_resp, ntok)
            r_tok = _rag_token_breakdown(r_sections, r_retr_tok, probe.text,
                                         r_resp, ntok)

            base = dict(scenario=scenario.name, probe=probe.text, kind=probe.kind,
                        salience=probe.salience, ground_truth=probe.ground_truth)

            taco_row = {**base, "condition": "taco", "score": t_judge["score"],
                        **t_tok,
                        "total_tokens": t_tok["total_context_tokens"],
                        "judge_input_tokens": ntok(t_judge.get("input_text", "")),
                        "judge_output_tokens": ntok(t_judge.get("output_text", "")),
                        "n_memories": len(t_retr),
                        "response": t_resp, "retrieved": t_retr}
            rag_row = {**base, "condition": "rag", "score": r_judge["score"],
                       **r_tok,
                       "total_tokens": r_tok["total_context_tokens"],
                       "judge_input_tokens": ntok(r_judge.get("input_text", "")),
                       "judge_output_tokens": ntok(r_judge.get("output_text", "")),
                       "n_memories": r_n,
                       "response": r_resp, "retrieved": r_passages}
            rows.append(taco_row)
            rows.append(rag_row)
            _append_partial(taco_row)
            _append_partial(rag_row)
            _log(f"  probe {pi}/{len(scenario.probes)} done: "
                 f"taco={t_judge['score']} rag={r_judge['score']} "
                 f"(written to {PARTIAL_PATH.name})")

        _log(f"=== scenario {si}/{len(SCENARIOS)}: {scenario.name} (done, "
             f"{len(scenario.probes)} probes scored) ===")

    conn.close()
    _log("harness.run() complete — closing DB connection")
    return {"rows": rows, "scale": scale,
            "meta": {"model": config.LLM_MODEL, "rag_topk": RAG_TOPK,
                     "taco_topk": config.TOP_K, "mode": mode,
                     "state_briefing_mode": config.STATE_BRIEFING_MODE}}
