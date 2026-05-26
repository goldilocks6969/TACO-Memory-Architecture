"""Gen-2 naive RAG baseline — a realistic modern assistant memory stack.

Compared to the gen-1 baseline this is deliberately *stronger and heavier*, to
mirror how production assistants/agents actually implement memory today:

  • a rolling raw chat-history window (the last W turns, always injected);
  • semantic top-k retrieval over large raw episodic chunks (one chunk per
    session, several turns of unedited text);
  • k that SCALES with history size (more stored → retrieve more), so the
    retrieval payload grows with the corpus;
  • everything kept — no salience gating, no abstraction, no decay, no
    state-conditioning, no identity model, no reconsolidation, no prediction.

It uses the SAME base LLM and embedding model as TACO. The only thing that varies
is the memory policy. Each chunk carries provenance tags so retrieval relevance /
noise can be scored deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Set, Tuple

import numpy as np

from taco import embeddings, llm

REASONING_HEADER = (
    "You are a helpful assistant with access to memory of earlier conversations. "
    "The following are raw excerpts retrieved by semantic similarity, plus the "
    "most recent messages. Use anything relevant to answer naturally."
)


@dataclass
class Chunk:
    text: str
    tags: Set[str]
    emb: np.ndarray


@dataclass
class NaiveRAG2:
    k_base: int = 6              # bigger top-k than gen-1
    k_per_chunks: int = 6        # +1 to k for every k_per_chunks stored chunks
    k_max: int = 16             # cap so it stays "retrieval", not full-context
    window_turns: int = 8        # rolling raw chat-history window
    _chunks: List[Chunk] = field(default_factory=list)
    _recent: List[Tuple[str, Set[str]]] = field(default_factory=list)  # (text, tags)

    # -- ingestion ---------------------------------------------------------- #
    def ingest_session(self, turns: List) -> None:
        """Store one raw multi-turn chunk per session (no filtering) + roll window."""
        text = "  ".join(t.text for t in turns)
        tags = {t.tag for t in turns}
        self._chunks.append(Chunk(text, tags,
                                  np.asarray(embeddings.embed_list(text), dtype=np.float32)))
        for t in turns:
            self._recent.append((t.text, {t.tag}))

    def corpus_tokens(self, ntok: Callable[[str], int]) -> int:
        return sum(ntok(c.text) for c in self._chunks)

    def n_chunks(self) -> int:
        return len(self._chunks)

    # -- retrieval ---------------------------------------------------------- #
    def _k(self) -> int:
        return min(self.k_max, self.k_base + len(self._chunks) // self.k_per_chunks)

    def retrieve(self, query: str) -> List[Chunk]:
        if not self._chunks:
            return []
        q = np.asarray(embeddings.embed_list(query), dtype=np.float32)
        mat = np.vstack([c.emb for c in self._chunks])
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
        idx = np.argsort(-sims)[: self._k()]
        return [self._chunks[i] for i in idx]

    def _window(self) -> List[Tuple[str, Set[str]]]:
        return self._recent[-self.window_turns:]

    def retrieved_items(self, query: str) -> List[Tuple[str, Set[str]]]:
        """All injected memory items (retrieved chunks + rolling window), with tags.
        Used for relevance / noise scoring."""
        items = [(c.text, c.tags) for c in self.retrieve(query)]
        items += self._window()
        return items

    def build_context(self, chunks: List[Chunk]) -> str:
        body = "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks)) or "(none)"
        window = "\n".join(f"- {t}" for t, _ in self._window()) or "(none)"
        return (f"{REASONING_HEADER}\n\nRetrieved conversation log:\n{body}\n\n"
                f"Recent messages:\n{window}")

    def answer(self, query: str, ntok: Callable[[str], int]) -> Tuple:
        """Return (response, full_context, retrieval_tokens, n_items, items_with_tags)."""
        chunks = self.retrieve(query)
        context = self.build_context(chunks)
        response = llm.respond(context, query, planning_depth=1)
        items = [(c.text, c.tags) for c in chunks] + self._window()
        retrieval_tokens = sum(ntok(c.text) for c in chunks) + \
            sum(ntok(t) for t, _ in self._window())
        return response, context, retrieval_tokens, len(items), items
