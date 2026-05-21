"""Interactive REPL for talking to Bubbles, with introspection commands."""
from __future__ import annotations

import argparse
import sys

from . import config, db
from .pipeline import Bubbles

BANNER = """\
Bubbles — State-Dependent Cognitive Orchestration
  Type a message to talk. Commands:
    /state         show the latent state S = (E, K, V, R)
    /plan          show the six state-dependent subsystem decisions
    /mem           show the most relevant memories from the last turn
    /why           show the narrative briefing sent to the reasoning engine
    /decay [wks]   run the forgetting engine (optionally fast-forward N weeks)
    /quit          exit
"""


def _print_turn(result) -> None:
    s = result.state
    gate = "STORED" if result.stored else "dropped"
    print(f"\nbubbles> {result.response}")
    print(
        f"  [S: E={s.E:.0f} K={s.K:.0f} V={s.V:.0f} R={s.R:.0f} | "
        f"salience {result.analysis['salience']:.1f} vs θ {result.plan.theta:.2f} "
        f"→ {gate} | tone {result.analysis['tone']}]"
    )
    if result.plan.proactive.initiate:
        p = result.plan.proactive
        print(f"  [proactive: would {p.kind} — {p.reason}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bubbles cognitive layer")
    parser.add_argument("--init", action="store_true",
                        help="create the schema then exit")
    args = parser.parse_args()

    if args.init:
        db.init_db()
        print("schema initialised.")
        return

    db.init_db()  # idempotent — ensures tables exist
    conn = db.connect()
    bubbles = Bubbles(conn)

    mode = "MOCK" if (config.MOCK or not config.OPENAI_API_KEY) else config.LLM_MODEL
    print(BANNER)
    print(f"  (reasoning engine: {mode})\n")

    last = None
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        if line == "/state":
            print(f"  {bubbles.state}")
            continue
        if line == "/plan":
            if last:
                print(last.plan.summary())
            else:
                print("  (no turn yet)")
            continue
        if line == "/mem":
            if last and last.retrieved:
                for m in last.retrieved:
                    print(f"  • R={m.score:.2f} [{m.tone}] {m.content}")
            else:
                print("  (no memories retrieved last turn)")
            continue
        if line == "/why":
            print(last.briefing if last else "  (no turn yet)")
            continue
        if line.startswith("/decay"):
            parts = line.split()
            weeks = float(parts[1]) if len(parts) > 1 else 0.0
            report = bubbles.run_decay(extra_weeks=weeks)
            print(f"  decayed {report.decayed} episodes; "
                  f"abstracted {report.abstracted} into beliefs:")
            for b in report.beliefs:
                print(f"    → {b}")
            continue

        last = bubbles.turn(line)
        _print_turn(last)

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
