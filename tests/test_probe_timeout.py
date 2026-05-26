"""Per-probe timeout fallback — error rows are recorded but don't pollute the
metrics aggregator.  This is the contract that lets the harness continue
through a slow probe instead of crashing the whole benchmark.
"""
from __future__ import annotations

import pytest

from eval import metrics


def _row(condition, kind, score, rt, sbt, spt, uqt, at, *, error=None):
    """Build a synthetic per-probe row in the shape the harness emits."""
    if error:
        return {"condition": condition, "kind": kind, "scenario": "x",
                "probe": "p", "salience": 5, "ground_truth": "gt",
                "score": None, "n_memories": 0, "response": "",
                "retrieved": [], "error": error}
    mct = rt + sbt
    tct = spt + mct + uqt
    return {"condition": condition, "kind": kind, "scenario": "x",
            "probe": "p", "salience": 5, "ground_truth": "gt", "score": score,
            "retrieval_tokens": rt, "state_briefing_tokens": sbt,
            "memory_context_tokens": mct, "system_prompt_tokens": spt,
            "user_query_tokens": uqt, "total_context_tokens": tct,
            "total_tokens": tct, "answer_tokens": at,
            "judge_input_tokens": 200, "judge_output_tokens": 30,
            "n_memories": 3, "response": "r", "retrieved": []}


@pytest.fixture()
def result_with_timeouts():
    """A result where one probe timed out on the TACO side.  The other
    healthy rows should drive the aggregate; the error row must be invisible
    to mean / CES computations."""
    rows = []
    # 3 healthy probes (per dimension TACO=80, RAG=60)
    for kind in ("factual", "emotional", "coherence"):
        rows.append(_row("taco", kind, 80, rt=100, sbt=80, spt=40, uqt=10, at=50))
        rows.append(_row("rag",  kind, 60, rt=800, sbt=0,  spt=40, uqt=10, at=50))
    # 1 timed-out probe on the TACO side — score=None, no token counts
    rows.append(_row("taco", "factual", 0, 0, 0, 0, 0, 0, error="timeout"))
    return {"rows": rows, "scale":
            [{"scenario": "x", "history_tokens": 100,
              "rag_corpus_tokens": 100, "taco_corpus_tokens": 50}],
            "meta": {"timeouts": 1,
                     "timeouts_by_stage": {"taco_probe": 1, "total": 1}}}


def test_error_rows_excluded_from_overall(result_with_timeouts):
    """The overall continuity score must not include the timed-out probe."""
    agg = metrics.aggregate(result_with_timeouts)
    # Healthy TACO probes were all 80 → overall is 80, NOT mixed with a 0 row.
    assert agg["overall"]["taco"] == 80.0


def test_error_rows_excluded_from_token_means(result_with_timeouts):
    """Average retrieval / total context tokens must reflect only healthy rows."""
    agg = metrics.aggregate(result_with_timeouts)
    # Healthy rows have retrieval_tokens=100 for TACO; the error row's 0 would
    # otherwise drag the mean down.
    assert agg["retrieval_tokens"]["taco"] == 100.0
    # And CES_retrieval = 80 / 100 * 1000 = 800.
    assert agg["ces_retrieval"]["taco"] == 800.0


def test_meta_timeouts_preserved(result_with_timeouts):
    """``meta.timeouts`` must round-trip through ``aggregate`` unchanged so
    the report can surface "N probes timed out"."""
    agg = metrics.aggregate(result_with_timeouts)
    assert agg["meta"]["timeouts"] == 1
    assert agg["meta"]["timeouts_by_stage"]["taco_probe"] == 1


def test_no_timeouts_still_works():
    """Sanity: aggregate on a clean result still produces normal output."""
    rows = [
        _row("taco", k, 70, rt=120, sbt=80, spt=40, uqt=10, at=50)
        for k in ("factual", "emotional", "coherence")
    ] + [
        _row("rag", k, 50, rt=800, sbt=0, spt=40, uqt=10, at=50)
        for k in ("factual", "emotional", "coherence")
    ]
    agg = metrics.aggregate(
        {"rows": rows, "scale": [{"scenario": "x", "history_tokens": 1,
                                  "rag_corpus_tokens": 100, "taco_corpus_tokens": 50}],
         "meta": {}})
    assert agg["overall"]["taco"] == 70.0
    assert agg["overall"]["rag"] == 50.0
