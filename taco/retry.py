"""Small retry helper for transient provider errors (rate limits, propagation
lag right after an Azure deployment is created, brief network blips).

Each attempt is run in a daemon thread so a blocking provider SDK that wedges
on a socket can be force-aborted; without this the eval harness can hang
silently for the duration of the OS-level TCP timeout.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Default per-attempt timeout. The eval harness assumes any single LLM /
# embedding round-trip that exceeds this is dead, regardless of provider.
DEFAULT_TIMEOUT_S = float(os.getenv("TACO_LLM_TIMEOUT_S", "60"))


class CallTimeout(RuntimeError):
    """A single LLM/embedding attempt did not return within its time budget."""


def call_with_timeout(fn: Callable[[], T], *, label: str,
                       timeout: float = DEFAULT_TIMEOUT_S) -> T:
    """Run *fn* in a daemon thread; raise ``CallTimeout`` if it exceeds *timeout*.

    The daemon flag means the worker thread never blocks process exit, so a
    truly wedged provider socket cannot keep the harness alive.
    """
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 — propagate everything
            box["error"] = e

    t = threading.Thread(target=runner, daemon=True, name=f"call[{label}]")
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise CallTimeout(
            f"LLM call hung (>{timeout:.0f}s) — label={label!r}; "
            "set TACO_LLM_TIMEOUT_S=<seconds> to extend or investigate the provider"
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def with_retries(fn: Callable[[], T], attempts: int = 6,
                 base_delay: float = 1.5, factor: float = 2.0,
                 max_delay: float = 20.0,
                 *, label: str = "LLM call",
                 timeout_per_attempt: Optional[float] = None) -> T:
    """Retry *fn* on exceptions, with exponential backoff between attempts.

    Each attempt is bounded by *timeout_per_attempt* (defaults to
    ``DEFAULT_TIMEOUT_S``); a timeout raises ``CallTimeout`` immediately —
    retrying a hung socket is pointless and would multiply the wait.
    """
    timeout = DEFAULT_TIMEOUT_S if timeout_per_attempt is None else timeout_per_attempt
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return call_with_timeout(fn, label=label, timeout=timeout)
        except CallTimeout:
            raise  # don't retry hangs
        except Exception as e:  # noqa: BLE001 — provider errors vary widely
            last = e
            if i == attempts - 1:
                break
            time.sleep(min(max_delay, base_delay * (factor ** i)))
    raise last  # type: ignore[misc]
