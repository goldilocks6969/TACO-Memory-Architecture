"""Embeddings via OpenAI, with a deterministic mock for keyless runs."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

import numpy as np

from . import config


def _mock_embed(text: str) -> List[float]:
    """Deterministic pseudo-embedding so the pipeline runs without an API key.

    Seeded by the text hash → stable across runs; normalised so cosine
    similarity behaves. Not semantically meaningful, only structurally valid.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(config.EMBED_DIM)
    v /= np.linalg.norm(v) + 1e-12
    return v.astype(float).tolist()


@lru_cache(maxsize=2048)
def embed(text: str) -> tuple:
    """Return an embedding for `text` as a tuple (hashable/cacheable)."""
    if config.MOCK or not config.OPENAI_API_KEY:
        return tuple(_mock_embed(text))

    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=text)
    return tuple(resp.data[0].embedding)


def embed_list(text: str) -> List[float]:
    return list(embed(text))
