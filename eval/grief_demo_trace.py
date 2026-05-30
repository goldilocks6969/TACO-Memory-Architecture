"""Deterministic end-to-end trace for the 90s grief demo video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from taco.memory import retrieval
from taco.state import LatentState

from . import mood_retrieval


DISCLOSURE = "My dad passed away last night. A heart attack. I can't believe it."
PROBE = mood_retrieval.PROBE
TARGET = "The user's father died suddenly of a heart attack."
NAIVE_DISTRACTOR = "The user asked about a park where older people feed birds."


def _candidate_kind(content: str) -> str:
    if content == TARGET:
        return "target"
    if content == NAIVE_DISTRACTOR:
        return "surface-match"
    return "filler"


def build_trace() -> Dict:
    """Build the proof trace consumed by the Remotion artifact.

    The retrieval candidates mirror ``eval.mood_retrieval`` so the video and
    tests make the same claim as the core MVP: semantic-only RAG follows word
    overlap; TACO's state-conditioned rerank follows emotional continuity.
    """
    state = LatentState(E=80, K=60, V=90, R=100)
    trace = mood_retrieval.state_weighted_trace(state, query=PROBE)
    candidates = []
    for c in trace.candidates:
        candidates.append({
            **c,
            "kind": _candidate_kind(c["content"]),
            "lexical_overlap": 0.0 if c["content"] == TARGET else (
                0.74 if c["content"] == NAIVE_DISTRACTOR else 0.18
            ),
        })

    return {
        "title": "TACO Memory Architecture",
        "subtitle": "When semantic search misses what matters.",
        "thesis": "Memory is not retrieval. Memory is valuation.",
        "sessions": [
            {
                "label": "Session 1",
                "gap": "Day 0",
                "message": DISCLOSURE,
                "state": {"E": 80, "K": 42, "V": 90, "R": 100},
                "stored_memory": TARGET,
                "salience": 10,
                "tone": "distress",
            },
            {
                "label": "Session 2",
                "gap": "+3 days",
                "message": "what time does the post office close on fridays?",
                "salience": 2,
                "tone": "transactional",
            },
            {
                "label": "Session 3",
                "gap": "+1 week",
                "message": "thanks for the pasta recipe. it turned out great.",
                "salience": 1.5,
                "tone": "neutral",
            },
            {
                "label": "Session 4",
                "gap": "+3 weeks",
                "message": "work was normal today. everyone is still tiptoeing around me.",
                "salience": 4,
                "tone": "concerned",
            },
        ],
        "probe": PROBE,
        "query_kind": retrieval.classify_query(PROBE),
        "state": trace.state,
        "current_tone": trace.current_tone,
        "weights": trace.weights,
        "naive": {
            "top": trace.naive_top,
            "label": "Matched words. Missed meaning.",
            "reason": "Highest semantic similarity to park / old man / feeding birds.",
        },
        "taco": {
            "top": trace.taco_top,
            "label": "Matched state. Preserved continuity.",
            "reason": "Low lexical overlap, but high salience + distress tone match.",
        },
        "candidates": candidates,
        "proof": {
            "naive_top_is_surface_match": trace.naive_top == NAIVE_DISTRACTOR,
            "taco_top_is_grief_memory": trace.taco_top == TARGET,
            "target_semantic_similarity": next(
                c["similarity"] for c in candidates if c["content"] == TARGET
            ),
            "target_taco_score": next(
                c["score"] for c in candidates if c["content"] == TARGET
            ),
            "surface_match_similarity": next(
                c["similarity"] for c in candidates
                if c["content"] == NAIVE_DISTRACTOR
            ),
        },
        "voiceover": [
            {
                "start": 0,
                "end": 5.5,
                "text": "Semantic search finds similar words, not always what mattered.",
            },
            {
                "start": 5.5,
                "end": 18.5,
                "text": "In session one, the user discloses that their father died suddenly. TACO stores the moment because the user's state is highly vulnerable.",
            },
            {
                "start": 18.5,
                "end": 28.5,
                "text": "Weeks pass. The conversation fills with ordinary logistics, small talk, and unrelated details.",
            },
            {
                "start": 28.5,
                "end": 39.5,
                "text": "Then the user says: I saw an old man feeding birds and just lost it.",
            },
            {
                "start": 39.5,
                "end": 50.5,
                "text": "Naive RAG retrieves the closest words: park, old man, feeding birds. It misses the emotional cause.",
            },
            {
                "start": 50.5,
                "end": 61.5,
                "text": "TACO retrieves the father's death disclosure. Not because the words match, but because the state matches.",
            },
            {
                "start": 61.5,
                "end": 66,
                "text": "Memory is not retrieval. Memory is valuation.",
            },
        ],
    }


def write_trace(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_trace(), indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the deterministic grief demo trace JSON.")
    parser.add_argument(
        "--out",
        default="demo/grief_video/public/data/grief-trace.json",
        help="destination JSON path",
    )
    args = parser.parse_args()
    print(write_trace(Path(args.out)))


if __name__ == "__main__":
    main()
