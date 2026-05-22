"""Aggregate per-probe rows into continuity dimensions and token-efficiency metrics.

Metric definitions
------------------
  retrieval_tokens  R_tok : tokens of the retrieved MEMORY payload injected per
                            turn (excludes fixed scaffolding and the query). This
                            is the component that scales with the memory policy.
  total_tokens      T_tok : tokens of the full injected context per turn.
  continuity Q            : mean judged score over the meaningful dimensions
                            (factual, emotional, coherence), 0-100.
  CES                     : Continuity Efficiency Score = Q / R_tok, reported per
                            1,000 retrieval tokens (continuity delivered per unit
                            of retrieval overhead). Primary comparison metric.
  compression ratio CR    : taco_corpus_tokens / baseline_corpus_tokens (lower =
                            Taco stores less, via salience gating + abstraction).
"""
from __future__ import annotations

from statistics import mean
from typing import Dict, List

CONDITIONS = ("rag", "taco")
CONDITION_LABELS = {"rag": "Base LLM + naive RAG", "taco": "Base LLM + Taco"}


def _rows(rows, condition, kind=None):
    out = [r for r in rows if r["condition"] == condition]
    if kind:
        out = [r for r in out if r["kind"] == kind]
    return out


def _mean(rows, condition, field, kind=None) -> float:
    rs = _rows(rows, condition, kind)
    return round(mean(r[field] for r in rs), 1) if rs else 0.0


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


def aggregate(result: Dict) -> Dict:
    rows = result["rows"]
    scale = result["scale"]
    out = {"dimensions": {}, "retrieval_tokens": {}, "total_tokens": {},
           "n_memories": {}, "recall_by_salience": {}, "overall": {}, "ces": {},
           "compression": {}, "scale": scale, "meta": result.get("meta", {}),
           "n_probes": len({(r["scenario"], r["probe"]) for r in rows})}

    for c in CONDITIONS:
        out["dimensions"][c] = {
            "Factual recall": _mean_score(rows, c, "factual"),
            "Salience-weighted recall": _salience_weighted_recall(rows, c),
            "Emotional continuity": _mean_score(rows, c, "emotional"),
            "Long-horizon coherence": _mean_score(rows, c, "coherence"),
            "Filler recall": _mean_score(rows, c, "filler"),
        }
        out["retrieval_tokens"][c] = _mean(rows, c, "retrieval_tokens")
        out["total_tokens"][c] = _mean(rows, c, "total_tokens")
        out["n_memories"][c] = _mean(rows, c, "n_memories")
        out["recall_by_salience"][c] = recall_by_salience(rows, c)
        q = _overall(rows, c)
        out["overall"][c] = q
        rtok = out["retrieval_tokens"][c] or 1.0
        # CES per 1,000 retrieval tokens
        out["ces"][c] = round(q / rtok * 1000.0, 1)

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
