"""Run the continuity benchmark: ingest each scenario through both pipelines,
probe both, judge the responses, and record rigorous per-probe measurements.

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

The split lets us see when TACO is retrieval-efficient but not yet
total-prompt-efficient: TACO may add a larger structured state briefing, so the
total injected context can exceed naive RAG even when the *retrieval payload* is
much smaller and bounded.

Isolation: Taco uses a dedicated `taco_eval` database, truncated between
scenarios so histories never bleed across personas.
"""
from __future__ import annotations

import os
from typing import Dict, List

from taco import config, db
from taco.pipeline import Taco

from . import judge
from .baseline import NaiveRAG
from .dataset import SCENARIOS, Scenario
from .tokens import counter

EVAL_DSN = os.getenv("TACO_EVAL_DSN", "postgresql://localhost:5432/taco_eval")
RAG_TOPK = 5  # naive RAG retrieves top-5 session chunks

_TABLES = ("episodes", "semantic_beliefs", "emotional_timeline", "reflections",
           "procedural", "identity", "state_log")

# Token attribution: which Taco briefing sections count as state-briefing overhead
# (everything except the fixed system prompt and the retrieved memory payload).
_STATE_SECTIONS = ("state", "identity", "working", "prediction")


def _reset(conn) -> None:
    conn.execute(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY")


def _ingest(scenario: Scenario, taco: Taco, rag: NaiveRAG) -> int:
    """Ingest sessions. RAG stores one raw multi-turn chunk per session; Taco
    processes each turn through its write-time salience gate. Returns total turns.
    """
    turns = 0
    for session in scenario.sessions:
        rag.ingest_chunk("  ".join(t.text for t in session))  # raw multi-turn chunk
        for turn in session:
            taco.turn(turn.text, hours_since_last=turn.gap_hours)
            turns += 1
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


def run() -> Dict:
    assert not config.MOCK, "Evaluation must run with real LLMs"
    ntok = counter(config.LLM_MODEL)
    db.init_db(EVAL_DSN)
    conn = db.connect(EVAL_DSN)

    rows: List[Dict] = []
    scale: List[Dict] = []

    for si, scenario in enumerate(SCENARIOS, 1):
        _reset(conn)
        taco = Taco(conn)
        rag = NaiveRAG(k=RAG_TOPK)
        n_turns = _ingest(scenario, taco, rag)
        history_tokens = sum(ntok(t.text) for s in scenario.sessions for t in s)
        print(f"[{si}/{len(SCENARIOS)}] {scenario.name}: ingested {n_turns} turns "
              f"in {len(scenario.sessions)} sessions ({history_tokens} tok)...")

        scale.append(dict(
            scenario=scenario.name,
            n_turns=n_turns,
            n_sessions=len(scenario.sessions),
            history_tokens=history_tokens,
            rag_corpus_tokens=rag.corpus_tokens(ntok),
            taco_corpus_tokens=_taco_corpus_tokens(conn, ntok),
        ))

        for probe in scenario.probes:
            t = taco.probe(probe.text)
            t_retr = [ep.content for ep in t.retrieved]
            t_resp = t.response

            r_resp, r_sections, r_retr_tok, r_n, r_passages = rag.answer(probe.text, ntok)

            t_judge = judge.judge(probe.text, probe.ground_truth, probe.kind, t_resp)
            r_judge = judge.judge(probe.text, probe.ground_truth, probe.kind, r_resp)

            t_tok = _taco_token_breakdown(t, probe.text, t_resp, ntok)
            r_tok = _rag_token_breakdown(r_sections, r_retr_tok, probe.text, r_resp, ntok)

            base = dict(scenario=scenario.name, probe=probe.text, kind=probe.kind,
                        salience=probe.salience, ground_truth=probe.ground_truth)

            rows.append({**base, "condition": "taco", "score": t_judge["score"],
                         **t_tok,
                         "total_tokens": t_tok["total_context_tokens"],  # back-compat
                         "judge_input_tokens": ntok(t_judge.get("input_text", "")),
                         "judge_output_tokens": ntok(t_judge.get("output_text", "")),
                         "n_memories": len(t_retr),
                         "response": t_resp, "retrieved": t_retr})
            rows.append({**base, "condition": "rag", "score": r_judge["score"],
                         **r_tok,
                         "total_tokens": r_tok["total_context_tokens"],  # back-compat
                         "judge_input_tokens": ntok(r_judge.get("input_text", "")),
                         "judge_output_tokens": ntok(r_judge.get("output_text", "")),
                         "n_memories": r_n,
                         "response": r_resp, "retrieved": r_passages})
        print(f"    scored {len(scenario.probes)} probes")

    conn.close()
    return {"rows": rows, "scale": scale,
            "meta": {"model": config.LLM_MODEL, "rag_topk": RAG_TOPK,
                     "taco_topk": config.TOP_K}}
