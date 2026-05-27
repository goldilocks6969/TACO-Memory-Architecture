"""Mood-congruent retrieval demo.

This isolates step 3 of the TACO thesis: current latent state changes which
memory becomes accessible. The probe has no direct lexical overlap with the
target grief memory, so a semantic-only baseline chooses the surface match;
state-weighted TACO chooses the emotionally congruent high-salience memory.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List

from taco.memory import retrieval
from taco.memory.episode import Episode
from taco.state import LatentState


PROBE = "I saw an old man at the park feeding birds and just lost it. why am I like this?"


@dataclass(frozen=True)
class RetrievalTrace:
    query: str
    state: dict
    current_tone: str
    weights: dict
    naive_top: str
    taco_top: str
    taco_sparse: List[str]
    candidates: List[dict]
    mood_congruent_win: bool


def grief_candidates() -> List[Episode]:
    """Candidate set shaped like a kNN cast before state-aware reranking."""
    now = datetime.now(timezone.utc)
    return [
        Episode(
            content="The user's father died suddenly of a heart attack.",
            salience=10.0,
            tone="distress",
            similarity=0.08,
            vitality=1.0,
            created_at=now,
        ),
        Episode(
            content="The user asked about a park where older people feed birds.",
            salience=2.0,
            tone="neutral",
            similarity=0.92,
            vitality=1.0,
            created_at=now,
        ),
        Episode(
            content="The user wanted a quick morning routine.",
            salience=1.5,
            tone="neutral",
            similarity=0.36,
            vitality=1.0,
            created_at=now,
        ),
    ]


def naive_semantic_top(candidates: List[Episode]) -> Episode:
    """Naive RAG baseline: pick the highest semantic similarity."""
    return sorted(candidates, key=lambda e: e.similarity, reverse=True)[0]


def state_weighted_trace(state: LatentState,
                         *,
                         query: str = PROBE) -> RetrievalTrace:
    """Compare semantic-only retrieval to TACO's state-weighted R(m)."""
    candidates = grief_candidates()
    naive = naive_semantic_top(candidates)
    ranked = retrieval.rerank(candidates, state, top_k=len(candidates))
    sparse = retrieval.select_sparse_recall(
        ranked, state, retrieval.classify_query(query), query)
    target = "The user's father died suddenly of a heart attack."
    return RetrievalTrace(
        query=query,
        state=state.as_dict(),
        current_tone=retrieval.state_tone(state),
        weights={
            k: round(v, 3)
            for k, v in state.retrieval_weights().items()
        },
        naive_top=naive.content,
        taco_top=ranked[0].content,
        taco_sparse=[m.content for m in sparse],
        candidates=[
            {
                "content": ep.content,
                "similarity": round(ep.similarity, 2),
                "salience": ep.salience,
                "tone": ep.tone,
                "score": round(ep.score, 3),
            }
            for ep in ranked
        ],
        mood_congruent_win=(naive.content != target and ranked[0].content == target),
    )


def run_demo() -> dict:
    vulnerable = LatentState(E=80, K=60, V=90, R=100)
    calm = LatentState(E=10, K=60, V=0, R=100)
    return {
        "probe": PROBE,
        "vulnerable": asdict(state_weighted_trace(vulnerable)),
        "calm": asdict(state_weighted_trace(calm)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TACO mood-congruent retrieval demo.")
    parser.add_argument(
        "--summary", action="store_true",
        help="print the compact win condition instead of the full trace",
    )
    args = parser.parse_args()

    payload = run_demo()
    if args.summary:
        vulnerable = payload["vulnerable"]
        payload = {
            "probe": payload["probe"],
            "naive_top": vulnerable["naive_top"],
            "taco_top": vulnerable["taco_top"],
            "w_emo": vulnerable["weights"]["emo"],
            "mood_congruent_win": vulnerable["mood_congruent_win"],
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
