"""Small retry helper for transient provider errors (rate limits, propagation
lag right after an Azure deployment is created, brief network blips).
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retries(fn: Callable[[], T], attempts: int = 6,
                 base_delay: float = 1.5, factor: float = 2.0,
                 max_delay: float = 20.0) -> T:
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — provider errors vary widely
            last = e
            if i == attempts - 1:
                break
            time.sleep(min(max_delay, base_delay * (factor ** i)))
    raise last  # type: ignore[misc]
