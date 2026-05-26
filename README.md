# Taco AI OS — State-Dependent Cognitive Orchestration

A memory + action layer that sits **between the user and an LLM**. A continuously
inferred four-dimensional latent state `S = (E, K, V, R)` modulates not only the
agent's *tone* but six distinct cognitive subsystems: write-time salience
gating, retrieval weighting, proactive initiation, planning depth, interruption
timing, and reinforcement rate.

This is a working implementation of the architecture in the whitepaper
*"State-Dependent Cognitive Orchestration: A New Paradigm for AI Memory
Architecture"* (Research Note, May 2026). The LLM is never fine-tuned —
it is the reasoning engine, and this layer is the prefrontal cortex around it.

> The novel claim: in every other companion system, internal state is used for
> exactly one thing — changing the tone of voice. Here, **state changes the
> system's cognition itself** — what it stores, what it retrieves, when it acts,
> and how deep it plans.

## The latent state `S = (E, K, V, R)`

Inferred every turn (never declared by the user), each dimension bounded `[0,100]`:

| Dim | Name | Driven up by | Decayed by |
|-----|------|--------------|------------|
| `E` | Emotional intensity | disclosure of fear, grief, conflict, excitement | neutral content |
| `K` | Engagement streak | sustained back-and-forth, depth, frequency | long gaps |
| `V` | Vulnerability | self-disclosure, expressed need, signs of distress | guarded/transactional turns |
| `R` | Recency of contact | recent meaningful interaction | time (half-life) |

`S` is read by every subsystem *before* it makes a decision. See
[`taco/state.py`](taco/state.py).

## The six state-dependent subsystems (§4)

| # | Subsystem | Modulation | Code |
|---|-----------|-----------|------|
| 1 | **Salience gate** `θ(S)` | `θ ∈ [2.1, 4.3]`, falls with E/V → captures more in emotional moments | [`memory/salience.py`](taco/memory/salience.py) |
| 2 | **Retrieval weights** `R(m)` | `w_emo(V) ∈ [0.20, 0.48]`; `w_rec` elevated in crisis | [`memory/retrieval.py`](taco/memory/retrieval.py) |
| 3 | **Proactive initiation** | crisis + low recency → unprompted check-in | [`subsystems/proactive.py`](taco/subsystems/proactive.py) |
| 4 | **Planning depth** | scales with engagement streak + trust (1→4 steps) | [`state.py`](taco/state.py) |
| 5 | **Interruption timing** | high E+V → deliver within the hour; low → hold | [`state.py`](taco/state.py) |
| 6 | **Reinforcement rate** | high V × high K → learn faster from moments that matter | [`state.py`](taco/state.py) |

All six are resolved together by the [orchestrator](taco/subsystems/orchestrator.py).

### 7. Memory-conditioned reasoning — the reasoning stance

The six subsystems decide *what* is stored, *what* is retrieved, *when* to act,
and *how deep* to plan. None of them change how the reasoning engine **treats**
the memories once they are in hand — by default an LLM reads them as ordinary
extra context. That is the difference between *smart retrieval* and a
*state-conditioned reasoning architecture*, and it is closed by a seventh
decision the orchestrator resolves from `S`: the **reasoning stance**
([`subsystems/reasoning.py`](taco/subsystems/reasoning.py)).

The stance is injected into the briefing *before* the memories, framing them as
**psychologically privileged information** and shifting the reasoning rules with
state. The same retrieved set is reasoned about differently:

| Stance | Region of `S` | Reasoning rule |
|--------|---------------|----------------|
| `distress` / `crisis` | high `E` | continuity **over** semantic similarity; connect identity beliefs to the present; maintain longitudinal coherence |
| `vulnerable` | high `V` | lead with emotional resonance; reference specific shared history gently |
| `concerned` | moderate `E` | weave history in; watch for escalation |
| `engaged` | high `K` | reason *across* the relationship; build on open threads, anticipate |
| `transactional` | low `E`,`V`,`K` | memories are factual reference only; don't force continuity |

> The LLM is still never fine-tuned. The stance makes the *reasoning engine
> itself* state-dependent — the same model, told to weigh the same memories
> differently depending on who it is talking to and how they are right now.

## The retrieval scoring function

```
R(m) = w_sem·sem(m) + w_sal·sal(m) + w_emo·emo(m) + w_rec·rec(m) + w_decay·decay(m)
```

Defaults `{sem 0.35, sal 0.30, emo 0.20, rec 0.10, decay 0.05}`; weights are
re-normalised after state modulation. `emo(m)` adds a `+0.30` boost when a
memory's tone matches the user's current tone. The pipeline casts 20 candidates
via pgvector kNN, re-ranks by `R(m)`, selects the top 4, and assembles a
narrative briefing for a single LLM invocation (Figure 4).

## The ten-layer cognitive stack → storage

| Layer | Role | Where |
|-------|------|-------|
| L1 Working memory | short-term context window | in-process deque ([`pipeline.py`](taco/pipeline.py)) |
| L2 Episodic | event-based history | `episodes` table |
| L3 Semantic | beliefs abstracted from decayed episodes | `semantic_beliefs` |
| L4 Emotional | salience-weighted independent timeline | `emotional_timeline` |
| L5 Procedural | learned workflows / habits | `procedural` |
| L6 Reflective | self-generated abstractions | `reflections` |
| L7 Reconsolidation | re-remembering rewrites meaning + attenuates charge | [`memory/reconsolidation.py`](taco/memory/reconsolidation.py) |
| L8 Forgetting | tier decay + abstraction-before-pruning | [`memory/decay.py`](taco/memory/decay.py) |
| L9 Predictive prefetch | trajectory projection + anticipatory retrieval | [`subsystems/prediction.py`](taco/subsystems/prediction.py) |
| L10 Identity graph | persistent self-model, confidence-weighted | [`memory/identity.py`](taco/memory/identity.py) |

## Memory decay (§4.4, Figure 6)

Each memory's vitality starts at 1.0 and is multiplied weekly by its
tier-specific rate. Below `0.15`, the episode is **abstracted into a semantic
belief, then pruned** — meaning survives, detail is forgotten.

| Salience | Example | Weekly decay |
|----------|---------|--------------|
| 9–10 | breakups · deaths · identity revelations | ×0.99 (near-permanent) |
| 7–8 | conflict · significant fear | ×0.95 |
| 5–6 | plans · moderate stress | ×0.88 |
| 3–4 | light personal sharing | ×0.75 |
| 1–2 | greetings · small talk | ×0.60 (pruned in days) |

## Memory that evolves: identity, reconsolidation, prediction

The layers above make retrieval *state-aware*. These three make memory itself
**alive** — it consolidates into a self-model, rewrites itself on recall, and
reaches forward in time. All three are wired into every `turn()` and run keyless
(heuristic fallbacks) or LLM-backed.

**L10 — Identity abstraction (the self-model).** Episodes are isolated events;
identity is *who the person is across time*. From significant moments (tier 5+),
TACO extracts durable facts — relationships, roles, values, ongoing struggles —
into the `identity` graph. Repeated, consistent evidence raises confidence
asymptotically toward 1.0; contradicting evidence triggers **belief revision**
(`identity.reconcile`). The confident self-model is asserted at the top of every
briefing, so the engine reasons from a stable picture of the person, not just the
last few messages.

**L7 — Reconsolidation (memory rewrites itself).** Humans don't replay static
memories; recall re-encodes them in the present state. When a memory encoded in a
hot state (`s_e ≥ 55`) is retrieved while the user is now substantially calmer
(`ΔE ≥ 30`), TACO **reconsolidates** it: the semantic abstraction is rewritten
toward an integrated belief (*"I failed"* → *"that failure made me stronger"*),
the emotional charge attenuates on the L4 timeline (`×0.6`), and the trace is
timestamped (cooldown-gated, bounded to one rewrite/turn). The event is never
altered — only its meaning and weight.

**L9 — Predictive continuity (memory reaches forward).** Reactive subsystems read
the *current* `S`; this one reads its *trajectory*. It extrapolates the next state
from the recent state log (damped, bounded), classifies the trend
(`escalating` / `recovering` / `stable`), and surfaces **emergent themes** —
tones recurring across recent significant episodes, i.e. what is likely to become
persistent. When the trajectory is escalating, it issues a state-gated
**anticipatory prefetch** so the right memory is already in hand before the need
is spoken.

> Net effect: identical retrieval is no longer the ceiling. The same model now
> reasons from a persistent identity, watches memories *change* as the user
> changes, and gets ahead of where they're heading.

## Setup

Requires Postgres + pgvector (already installed via Homebrew on this machine):

```bash
brew services start postgresql@17        # if not already running
createdb taco                             # if not already created
python3 -m pip install --user -r requirements.txt
cp .env.example .env                       # add your OPENAI_API_KEY
python3 -m taco.cli --init             # create the schema
```

## Run

```bash
python3 -m taco.cli                     # interactive REPL
```

In the REPL: `/state`, `/plan`, `/mem`, `/why`, `/decay [weeks]`, `/quit`.

**No API key?** Run with deterministic heuristics instead of OpenAI:

```bash
TACO_MOCK=1 python3 -m taco.cli
```

The weekly forgetting job (for cron):

```bash
python3 scripts/run_decay.py               # 0 4 * * 0  in crontab
```

## Tests

```bash
python3 -m pytest -q                       # 24 tests, no DB or API key needed
```

The tests validate the pure cognitive logic: the modulation curves and their
endpoints, the state-dependent salience gate, `R(m)` scoring and re-ranking, and
the tier decay schedule against the paper's stated values.

## Evaluation: two efficiency metrics, not one

The continuity benchmark in [`eval/`](eval/) compares Taco against a naive RAG
baseline under an identical base LLM. To stay honest about *where* Taco wins,
the harness reports **two** efficiency numbers per system:

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **CES_retrieval** | `Q / avg_retrieval_tokens × 1000` | Memory-policy efficiency — continuity per 1,000 *retrieval* tokens. |
| **CES_total**     | `Q / avg_total_context_tokens × 1000` | Full-prompt efficiency — continuity per 1,000 *total* injected context tokens (system + state briefing + retrieval + query). |

The split matters because Taco injects a structured state briefing on top of
its retrieved memories, so the **total** prompt sent to the LLM can be larger
than naive RAG's even when the retrieved payload is much smaller and bounded.
Quoting only `CES_retrieval` would hide that overhead.

Taco's current design target is **retrieval quality + bounded retrieval
overhead**, i.e. winning on `CES_retrieval`. Compressing the state briefing so
`CES_total` also leads is a separate optimization goal we have not yet pursued.
The harness prints two warnings to make this explicit:

- *"TACO is retrieval-efficient but not total-context-efficient in this run."*
  — when Taco leads on `CES_retrieval` but loses on `CES_total`.
- *"TACO uses larger total injected context; optimize state briefing compression."*
  — when Taco's average `total_context_tokens` exceeds RAG's.

The legacy single-number `CES` is retained as an alias for `CES_retrieval` so
older reports still resolve.

### What the harness records per probe

`retrieval_tokens`, `state_briefing_tokens`, `memory_context_tokens`,
`system_prompt_tokens`, `user_query_tokens`, `total_context_tokens`,
`answer_tokens`, `judge_input_tokens`, `judge_output_tokens`. Token counts use
[tiktoken](https://github.com/openai/tiktoken) with `cl100k_base` as a
fallback. `eval/out/summary.csv` aggregates these into the per-system headline
numbers (`overall_quality`, the average of each token component, both CES
scores, and the percent savings vs. RAG).

```bash
python3 -m eval.run                          # writes results.json, summary.csv, *.png
```

## Project layout

```
taco/
  config.py            every constant from the paper (tiers, weights, endpoints)
  state.py             S=(E,K,V,R), inference, and all S→parameter curves
  db.py                Postgres + pgvector connection
  embeddings.py        OpenAI embeddings (+ deterministic mock)
  llm.py               turn analysis, reasoning response, abstraction (+ mock)
  memory/
    episode.py         the Episode record
    store.py           CRUD + pgvector kNN candidate cast
    salience.py        write-time salience gate (L4)
    retrieval.py       R(m) scoring + narrative briefing (Figure 4)
    decay.py           tier decay + abstraction-before-pruning (L8→L3)
  subsystems/
    proactive.py       proactive-initiation policy
    orchestrator.py    resolves all six subsystems from S
  pipeline.py          the full turn loop (Figure 4)
  cli.py               interactive REPL
schema.sql             the ten-layer stack as tables
scripts/run_decay.py   weekly forgetting job
tests/                 pure-logic unit tests
```
