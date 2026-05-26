"""Aggregate per-probe rows into continuity dimensions and token-efficiency metrics.

Metric definitions
------------------
  retrieval_tokens       R_tok : tokens of the retrieved MEMORY payload per turn
                                 (excludes scaffolding and the query). Scales
                                 with the memory policy.
  state_briefing_tokens        : tokens of TACO's structured state briefing per
                                 turn (state line + stance + identity + working +
                                 prediction). RAG = 0.
  memory_context_tokens        : retrieval + state_briefing.
  total_context_tokens   T_tok : system_prompt + memory_context + user_query.
  answer_tokens                : tokens of the generated answer.
  continuity Q                 : mean judged score over the meaningful dimensions
                                 (factual, emotional, coherence), 0-100.

  CES_retrieval               : Q / avg_retrieval_tokens × 1000.  Memory-policy
                                efficiency — how much continuity each unit of
                                *retrieval* overhead delivers. This is TACO's
                                primary claim.
  CES_total                   : Q / avg_total_context_tokens × 1000.  Full-prompt
                                efficiency — how much continuity each unit of
                                *total* injected context delivers.  This is the
                                fairer system-level comparison, because TACO's
                                state briefing makes the full prompt larger than
                                naive RAG's even when retrieval is much smaller.

  compression ratio CR        : taco_corpus_tokens / rag_corpus_tokens (lower =
                                Taco stores less, via salience gating + abstraction).

We keep CES_retrieval (the old CES, renamed for clarity) and add CES_total so a
reviewer can see both numbers side by side.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

CONDITIONS = ("rag", "taco")
CONDITION_LABELS = {"rag": "Base LLM + naive RAG", "taco": "Base LLM + Taco"}

# Token components surfaced per condition (drives summary.csv + plots).
_TOKEN_FIELDS = (
    "retrieval_tokens",
    "state_briefing_tokens",
    "memory_context_tokens",
    "system_prompt_tokens",
    "user_query_tokens",
    "total_context_tokens",
    "answer_tokens",
    "judge_input_tokens",
    "judge_output_tokens",
)


def _rows(rows, condition, kind=None):
    out = [r for r in rows if r["condition"] == condition]
    if kind:
        out = [r for r in out if r["kind"] == kind]
    return out


def _mean(rows, condition, field, kind=None) -> float:
    rs = _rows(rows, condition, kind)
    if not rs:
        return 0.0
    vals = [r.get(field, 0) for r in rs]
    return round(mean(vals), 1) if vals else 0.0


def _mean_score(rows, condition, kind=None) -> float:
    return _mean(rows, condition, "score", kind)


def _salience_weighted_recall(rows, condition) -> float:
    rs = [r for r in _rows(rows, condition) if r["kind"] in ("factual", "emotional")]
    num = sum(r["salience"] * r["score"] for r in rs)
    den = sum(r["salience"] for r in rs)
    return round(num / den, 1) if den else 0.0


def salience_bucket(s: int) -> str:
    if s >= 8:
        return "high (8-10)"
    if s >= 4:
        return "mid (4-7)"
    return "low (1-3)"


def recall_by_salience(rows, condition) -> Dict[str, float]:
    out = {}
    for bucket in ("low (1-3)", "mid (4-7)", "high (8-10)"):
        rs = [r for r in _rows(rows, condition)
              if salience_bucket(r["salience"]) == bucket]
        out[bucket] = round(mean(r["score"] for r in rs), 1) if rs else 0.0
    return out


def _overall(rows, condition) -> float:
    dims = [_mean_score(rows, condition, k)
            for k in ("factual", "emotional", "coherence")]
    return round(mean(dims), 1)


def _safe_ces(quality: float, tokens: float) -> Optional[float]:
    """CES = quality / tokens * 1000.  Returns None (and warns once) when
    *tokens* is zero — better than silently dividing by 1.0 like the old code."""
    if not tokens or tokens <= 0:
        print("[metrics] WARNING: zero token denominator; CES set to null",
              file=sys.stderr)
        return None
    return round(quality / tokens * 1000.0, 1)


def _pct_savings(baseline: float, candidate: float) -> Optional[float]:
    """Percent reduction of *candidate* vs *baseline* (positive = candidate is
    cheaper). Returns None if the baseline is zero (no signal to compare to)."""
    if not baseline or baseline <= 0:
        return None
    return round((baseline - candidate) / baseline * 100.0, 1)


def aggregate(result: Dict) -> Dict:
    rows = result["rows"]
    scale = result["scale"]
    out: Dict = {
        "dimensions": {},
        "tokens": {},                    # per-condition: every token field, mean
        "retrieval_tokens": {},          # back-compat: same as tokens[c]["retrieval_tokens"]
        "total_tokens": {},              # back-compat: same as tokens[c]["total_context_tokens"]
        "n_memories": {},
        "recall_by_salience": {},
        "overall": {},
        "ces_retrieval": {},
        "ces_total": {},
        "ces": {},                       # back-compat alias for ces_retrieval
        "compression": {},
        "scale": scale,
        "meta": result.get("meta", {}),
        "n_probes": len({(r["scenario"], r["probe"]) for r in rows}),
        "savings": {},                   # taco-vs-rag percent savings
        "warnings": [],
    }

    for c in CONDITIONS:
        out["dimensions"][c] = {
            "Factual recall": _mean_score(rows, c, "factual"),
            "Salience-weighted recall": _salience_weighted_recall(rows, c),
            "Emotional continuity": _mean_score(rows, c, "emotional"),
            "Long-horizon coherence": _mean_score(rows, c, "coherence"),
            "Filler recall": _mean_score(rows, c, "filler"),
        }
        out["tokens"][c] = {f: _mean(rows, c, f) for f in _TOKEN_FIELDS}
        out["retrieval_tokens"][c] = out["tokens"][c]["retrieval_tokens"]
        out["total_tokens"][c] = out["tokens"][c]["total_context_tokens"]
        out["n_memories"][c] = _mean(rows, c, "n_memories")
        out["recall_by_salience"][c] = recall_by_salience(rows, c)
        q = _overall(rows, c)
        out["overall"][c] = q
        out["ces_retrieval"][c] = _safe_ces(q, out["tokens"][c]["retrieval_tokens"])
        out["ces_total"][c] = _safe_ces(q, out["tokens"][c]["total_context_tokens"])
        # Back-compat: the original report wrote `ces[c]`; keep it pointed at CES_retrieval.
        out["ces"][c] = out["ces_retrieval"][c]

    # Taco-vs-RAG savings (positive = Taco is smaller)
    rag_tok = out["tokens"]["rag"]
    taco_tok = out["tokens"]["taco"]
    out["savings"] = {
        "retrieval_token_savings_vs_rag_pct":
            _pct_savings(rag_tok["retrieval_tokens"], taco_tok["retrieval_tokens"]),
        "total_context_savings_vs_rag_pct":
            _pct_savings(rag_tok["total_context_tokens"], taco_tok["total_context_tokens"]),
    }

    # Honest-comparison warnings the professor asked for.
    rag_ces_r = out["ces_retrieval"]["rag"]
    taco_ces_r = out["ces_retrieval"]["taco"]
    rag_ces_t = out["ces_total"]["rag"]
    taco_ces_t = out["ces_total"]["taco"]
    if (taco_ces_r is not None and rag_ces_r is not None
            and taco_ces_t is not None and rag_ces_t is not None
            and taco_ces_r > rag_ces_r and taco_ces_t < rag_ces_t):
        msg = "TACO is retrieval-efficient but not total-context-efficient in this run."
        out["warnings"].append(msg)
        print(f"[metrics] {msg}", file=sys.stderr)
    if taco_tok["total_context_tokens"] > rag_tok["total_context_tokens"]:
        msg = "TACO uses larger total injected context; optimize state briefing compression."
        out["warnings"].append(msg)
        print(f"[metrics] {msg}", file=sys.stderr)

    # enrich each scenario with its per-condition avg retrieval payload (scale plot)
    for s in scale:
        sc = s["scenario"]
        for c in CONDITIONS:
            rs = [r for r in rows if r["scenario"] == sc and r["condition"] == c]
            s[f"{c}_retr_tokens"] = round(mean(r["retrieval_tokens"] for r in rs), 1) \
                if rs else 0.0

    # corpus compression (Taco vs baseline), averaged across scenarios
    ratios = [s["taco_corpus_tokens"] / s["rag_corpus_tokens"]
              for s in scale if s["rag_corpus_tokens"]]
    out["compression"] = {
        "ratio": round(mean(ratios), 3) if ratios else None,
        "rag_corpus_tokens_avg": round(mean(s["rag_corpus_tokens"] for s in scale), 1),
        "taco_corpus_tokens_avg": round(mean(s["taco_corpus_tokens"] for s in scale), 1),
    }
    return out


# --------------------------------------------------------------------------- #
# summary.csv — one row per condition, columns the professor asked for.
# --------------------------------------------------------------------------- #
_SUMMARY_FIELDS = (
    "condition",
    "overall_quality",
    "avg_retrieval_tokens",
    "avg_state_briefing_tokens",
    "avg_memory_context_tokens",
    "avg_total_context_tokens",
    "avg_answer_tokens",
    "ces_retrieval",
    "ces_total",
    "retrieval_token_savings_vs_rag_pct",
    "total_context_savings_vs_rag_pct",
)


def _fmt(v):
    return "" if v is None else v


def write_summary_csv(agg: Dict, path: Path) -> Path:
    """Write the per-condition headline numbers to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()
        for c in CONDITIONS:
            tok = agg["tokens"][c]
            # Savings are only meaningful for the candidate; the baseline row gets 0.
            if c == "rag":
                r_sav = 0.0
                t_sav = 0.0
            else:
                r_sav = agg["savings"]["retrieval_token_savings_vs_rag_pct"]
                t_sav = agg["savings"]["total_context_savings_vs_rag_pct"]
            w.writerow({
                "condition": c,
                "overall_quality": agg["overall"][c],
                "avg_retrieval_tokens": tok["retrieval_tokens"],
                "avg_state_briefing_tokens": tok["state_briefing_tokens"],
                "avg_memory_context_tokens": tok["memory_context_tokens"],
                "avg_total_context_tokens": tok["total_context_tokens"],
                "avg_answer_tokens": tok["answer_tokens"],
                "ces_retrieval": _fmt(agg["ces_retrieval"][c]),
                "ces_total": _fmt(agg["ces_total"][c]),
                "retrieval_token_savings_vs_rag_pct": _fmt(r_sav),
                "total_context_savings_vs_rag_pct": _fmt(t_sav),
            })
    return path
