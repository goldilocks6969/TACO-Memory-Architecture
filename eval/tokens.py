"""Token counting helper shared by the harness, metrics, and plots.

Uses tiktoken; falls back to cl100k_base when the requested model has no
dedicated encoding, and to a whitespace approximation if tiktoken itself is
unavailable (so unit tests that don't ship the wheel still pass).
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=8)
def _encoding(model: str):
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return the number of tokens in *text* under the given model's encoding.

    If model-specific encoding fails, falls back to cl100k_base. If tiktoken is
    unavailable, falls back to a whitespace word count (last-resort approximation).
    """
    if not text:
        return 0
    enc = _encoding(model)
    if enc is None:
        return len(text.split())
    return len(enc.encode(text))


def counter(model: str = "gpt-4o-mini"):
    """Return a closure equivalent to ``lambda s: count_tokens(s, model)``."""
    enc = _encoding(model)
    if enc is None:
        return lambda s: len((s or "").split())
    return lambda s: len(enc.encode(s or ""))
