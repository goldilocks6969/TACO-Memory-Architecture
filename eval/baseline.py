"""Naive RAG baseline — a realistic stand-in for standard long-context/RAG memory.

This mirrors how production retrieval memory typically behaves:

  • the conversation is logged as raw multi-turn CHUNKS (here, one chunk per
    session — several turns of unedited text), not single sentences;
  • every chunk is kept (no salience filtering, no abstraction, no decay);
  • at query time the top-k chunks are retrieved by cosine similarity and their
    raw text is concatenated into the prompt.

Consequently the retrieved payload is large and grows with history length, and
retrieval is purely semantic (surface similarity), with no notion of which
memories mattered. Same base LLM and embedding model as Taco — the only
difference is the memory policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from taco import embeddings, llm


@dataclass
class NaiveRAG:
    k: int = 5
    min_chunk_tokens: int = 3
    eps: float = 1e-8
    _chunks: List[str] = field(default_factory=list)
    _embs: List[np.ndarray] = field(default_factory=list)
    dropped_chunks: int = 0

    def ingest_chunk(self, text: str) -> None:
        """Store one raw multi-turn chunk (no filtering)."""
        text = text.strip()
        if len(re.findall(r"\S+", text)) < self.min_chunk_tokens:
            self.dropped_chunks += 1
            return

        emb = np.asarray(embeddings.embed_list(text), dtype=np.float64)
        if not np.all(np.isfinite(emb)):
            self.dropped_chunks += 1
            return
        norm = float(np.linalg.norm(emb))
        if norm <= self.eps:
            self.dropped_chunks += 1
            return

        self._chunks.append(text)
        self._embs.append(emb / norm)

    def corpus_tokens(self, ntok: Callable[[str], int]) -> int:
        return sum(ntok(c) for c in self._chunks)

    def retrieve(self, query: str) -> List[str]:
        if not self._chunks:
            return []
        q = np.asarray(embeddings.embed_list(query), dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
        if not np.all(np.isfinite(q)) or q_norm <= self.eps:
            raise RuntimeError(
                "NaiveRAG received a non-finite or zero-norm query embedding"
            )
        q = q / q_norm
        mat = np.vstack(self._embs)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            try:
                sims = np.einsum("ij,j->i", mat, q, optimize=False)
            except FloatingPointError as exc:
                raise RuntimeError(
                    "NaiveRAG similarity produced a floating-point error; "
                    "check chunk embedding norms and finite values"
                ) from exc
        if not np.all(np.isfinite(sims)):
            raise RuntimeError("NaiveRAG similarity produced non-finite scores")
        idx = np.argsort(-sims)[: self.k]
        return [self._chunks[i] for i in idx]

    _SYSTEM_PROMPT = (
        "You are a helpful assistant. The following are raw excerpts "
        "from earlier conversation, retrieved by similarity. Use any "
        "that are relevant."
    )

    def build_sections(self, passages: List[str]) -> Dict[str, str]:
        """Return the RAG prompt broken into ``system`` and ``retrieval`` sections.

        The state-briefing slot is intentionally empty: a naive RAG injects no
        structured state. The harness reads these to attribute tokens the same
        way it does for Taco.
        """
        body = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages)) or "(none)"
        retrieval_block = f"Retrieved conversation log:\n{body}"
        return {"system": self._SYSTEM_PROMPT, "state": "", "retrieval": retrieval_block}

    def build_context(self, passages: List[str]) -> str:
        s = self.build_sections(passages)
        return f"{s['system']}\n\n{s['retrieval']}"

    def answer(self, query: str, ntok: Callable[[str], int]) -> Tuple:
        """Return (response, sections, retrieval_tokens, n_memories, passages)."""
        passages = self.retrieve(query)
        sections = self.build_sections(passages)
        context = f"{sections['system']}\n\n{sections['retrieval']}"
        response = llm.respond(context, query, planning_depth=1)
        retrieval_tokens = sum(ntok(p) for p in passages)
        return response, sections, retrieval_tokens, len(passages), passages
