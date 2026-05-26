"""The Fact record — the structured retrieval unit (Phase 1).

Episodes ([`episode.py`](episode.py)) remain the raw conversational record that
feeds the L4 emotional timeline and L7 reconsolidation. Facts are *extracted*
from those turns: deduplicated, paraphrase-retrievable units carrying rich
emotional and entity metadata. From Phase 1 on, facts — not episodes — are the
primary thing retrieval surfaces into the briefing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Fact:
    """One structured memory. Mirrors a row of the `facts` table."""

    summary: str
    fact_type: Optional[str] = None        # preference|event|relationship|state|thread
    event_type: Optional[str] = None       # conflict|goal|fear|plan|achievement|...
    emotional_tone: Optional[str] = None
    emotional_cause: Optional[str] = None   # TACO-specific: why it carries charge
    user_belief: Optional[str] = None       # TACO-specific: the self-belief it implies
    salience: float = 1.0
    retrieval_cues: List[str] = field(default_factory=list)   # alt phrasings (≤3)
    entity_keys: List[str] = field(default_factory=list)      # "person:maya", ...
    validity: str = "current"               # current|outdated|uncertain|resolved
    thread_status: Optional[str] = None     # unresolved|in_progress|resolved
    due_date: Optional[datetime] = None
    confidence: float = 0.7

    # populated only when read back from storage
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    similarity: float = 0.0   # cosine similarity from a candidate / neighbour query
