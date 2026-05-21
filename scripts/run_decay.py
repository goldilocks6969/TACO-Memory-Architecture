#!/usr/bin/env python3
"""Weekly forgetting job: tier-based decay + abstraction-before-pruning.

Run from cron, e.g.  0 4 * * 0  python scripts/run_decay.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bubbles import db, embeddings, llm          # noqa: E402
from bubbles.memory import decay                  # noqa: E402


def main() -> None:
    extra_weeks = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    conn = db.connect()
    report = decay.run(conn, embeddings.embed_list, llm.abstract,
                       extra_weeks=extra_weeks)
    print(f"decayed={report.decayed} abstracted={report.abstracted}")
    for b in report.beliefs:
        print(f"  → {b}")
    conn.close()


if __name__ == "__main__":
    main()
