"""Run the continuity benchmark: ingest each scenario through both pipelines,
probe both, judge the responses, and record rigorous per-probe measurements.

For every probe and condition we record:
  • score            — LLM-judged continuity (0-100)
  • retrieval_tokens — tokens of the retrieved MEMORY payload only (scales with policy)
  • total_tokens     — tokens of the full injected context (payload + scaffolding)
  • n_memories       — number of retrieved items
  • retrieved        — the retrieved texts (for qualitative inspection)

Per scenario we also record corpus sizes (for the compression ratio) and history
length (for the scale analysis).

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

EVAL_DSN = os.getenv("TACO_EVAL_DSN", "postgresql://localhost:5432/taco_eval")
RAG_TOPK = 5  # naive RAG retrieves top-5 session chunks

_TABLES = ("episodes", "semantic_beliefs", "emotional_timeline", "reflections",
           "procedural", "identity", "state_log")


def _token_counter():
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(config.LLM_MODEL)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return lambda s: len(enc.encode(s or ""))


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


def run() -> Dict:
    ntok = _token_counter()
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

            r_resp, r_ctx, r_retr_tok, r_n, r_passages = rag.answer(probe.text, ntok)

            t_score = judge.judge(probe.text, probe.ground_truth, probe.kind, t_resp)
            r_score = judge.judge(probe.text, probe.ground_truth, probe.kind, r_resp)

            base = dict(scenario=scenario.name, probe=probe.text, kind=probe.kind,
                        salience=probe.salience, ground_truth=probe.ground_truth)
            rows.append({**base, "condition": "taco", "score": t_score["score"],
                         "retrieval_tokens": sum(ntok(x) for x in t_retr),
                         "total_tokens": ntok(t.briefing),
                         "n_memories": len(t_retr),
                         "response": t_resp, "retrieved": t_retr})
            rows.append({**base, "condition": "rag", "score": r_score["score"],
                         "retrieval_tokens": r_retr_tok,
                         "total_tokens": ntok(r_ctx),
                         "n_memories": r_n,
                         "response": r_resp, "retrieved": r_passages})
        print(f"    scored {len(scenario.probes)} probes")

    conn.close()
    return {"rows": rows, "scale": scale,
            "meta": {"model": config.LLM_MODEL, "rag_topk": RAG_TOPK,
                     "taco_topk": config.TOP_K}}
