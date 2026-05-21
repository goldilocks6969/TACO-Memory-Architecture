# Bubbles — State-Dependent Cognitive Orchestration

A memory + action layer that sits **between the user and an LLM**. A continuously
inferred four-dimensional latent state `S = (E, K, V, R)` modulates not only the
agent's *tone* but six distinct cognitive subsystems: write-time salience
gating, retrieval weighting, proactive initiation, planning depth, interruption
timing, and reinforcement rate.

This is a working implementation of the architecture in the whitepaper
*"State-Dependent Cognitive Orchestration: A New Paradigm for AI Memory
Architecture"* (Bubbles Research Note, May 2026). The LLM is never fine-tuned —
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
[`bubbles/state.py`](bubbles/state.py).

## The six state-dependent subsystems (§4)

| # | Subsystem | Modulation | Code |
|---|-----------|-----------|------|
| 1 | **Salience gate** `θ(S)` | `θ ∈ [2.1, 4.3]`, falls with E/V → captures more in emotional moments | [`memory/salience.py`](bubbles/memory/salience.py) |
| 2 | **Retrieval weights** `R(m)` | `w_emo(V) ∈ [0.20, 0.48]`; `w_rec` elevated in crisis | [`memory/retrieval.py`](bubbles/memory/retrieval.py) |
| 3 | **Proactive initiation** | crisis + low recency → unprompted check-in | [`subsystems/proactive.py`](bubbles/subsystems/proactive.py) |
| 4 | **Planning depth** | scales with engagement streak + trust (1→4 steps) | [`state.py`](bubbles/state.py) |
| 5 | **Interruption timing** | high E+V → deliver within the hour; low → hold | [`state.py`](bubbles/state.py) |
| 6 | **Reinforcement rate** | high V × high K → learn faster from moments that matter | [`state.py`](bubbles/state.py) |

All six are resolved together by the [orchestrator](bubbles/subsystems/orchestrator.py).

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
| L1 Working memory | short-term context window | in-process deque ([`pipeline.py`](bubbles/pipeline.py)) |
| L2 Episodic | event-based history | `episodes` table |
| L3 Semantic | beliefs abstracted from decayed episodes | `semantic_beliefs` |
| L4 Emotional | salience-weighted independent timeline | `emotional_timeline` |
| L5 Procedural | learned workflows / habits | `procedural` |
| L6 Reflective | self-generated abstractions | `reflections` |
| L7 Reconsolidation | reads mutate traces (`last_access` touch) | acts on `episodes` |
| L8 Forgetting | tier decay + abstraction-before-pruning | [`memory/decay.py`](bubbles/memory/decay.py) |
| L9 Predictive prefetch | anticipatory retrieval | (scaffolded) |
| L10 Identity graph | persistent self-model | `identity` |

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

## Setup

Requires Postgres + pgvector (already installed via Homebrew on this machine):

```bash
brew services start postgresql@17        # if not already running
createdb bubbles                          # if not already created
python3 -m pip install --user -r requirements.txt
cp .env.example .env                       # add your OPENAI_API_KEY
python3 -m bubbles.cli --init             # create the schema
```

## Run

```bash
python3 -m bubbles.cli                     # interactive REPL
```

In the REPL: `/state`, `/plan`, `/mem`, `/why`, `/decay [weeks]`, `/quit`.

**No API key?** Run with deterministic heuristics instead of OpenAI:

```bash
BUBBLES_MOCK=1 python3 -m bubbles.cli
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

## Project layout

```
bubbles/
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
