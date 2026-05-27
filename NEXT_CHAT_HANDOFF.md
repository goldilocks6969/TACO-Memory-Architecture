# TACO Memory Infra Handoff

Date: 2026-05-27

## Current Goal

Build TACO into a real memory-layer infra product whose differentiator is:

> State-aware memory formation for emotionally and contextually persistent agents.

The thesis is not "better RAG" generically. The defensible novelty is:

> TACO makes salience/state a first-class write-time control signal that decides memory admission, memory form, and later retrieval weighting/budget.

Mem0-style systems extract useful memories and retrieve relevant memories. TACO's distinction should be framed as:

- write-time salience/state admission,
- state-conditioned sparse recall,
- emotional/contextual continuity,
- biologically inspired encoding and recall, not fixed top-k retrieval.

## What Was Implemented

### 1. Anchor facts / write-time factual anchors

Problem: TACO was compressing too hard and dropping important factual/coherence anchors such as:

- "kiddo"
- Halcyon job outcome
- A1C progress
- Sam reaching out

Implemented deterministic anchor facts in `taco/memory/extract.py`.

Examples now preserved:

- nicknames / "called me"
- career outcomes
- health progress markers
- baby/name/gender anchors
- relationship unresolved threads
- housing updates
- academic grades

Pipeline now writes facts when either:

- `theta_facts(S)` passes, or
- `has_memory_anchor(...)` detects a durable continuity anchor.

### 2. Compact state briefing

Problem: TACO had high `CES_total` overhead because state briefing was verbose.

Compact mode now uses:

```text
[S tone=distress stance=distress E=72 V=58 K=40 R=80]
[ID relationship:partner=Maya]
```

Compact/minimal mode also uses a short system prompt:

```text
Answer from the state tag and retrieved memories; preserve continuity.
```

Full mode still keeps verbose reasoning directives for debug / qualitative inspection.

### 3. Adaptive sparse retrieval

Problem: After anchor facts, retrieval payload inflated. TACO was injecting too many verbose/redundant memories.

Implemented adaptive sparse recall:

- deterministic query classifier:
  - `factual`
  - `emotional`
  - `coherence`
  - `filler`
  - `general`
- query-kind-specific recall budgets:
  - filler: 0 by default
  - factual: 2
  - emotional: 2
  - coherence: 3
  - general: 2
- redundancy collapse after rerank
- compact retrieval rendering
- direct lexical/cue overlap boost for factual/coherence queries
- emotional boost for salience + tone match

New config knobs:

```bash
TACO_RECALL_POLICY=adaptive
TACO_RECALL_MAX_TOKENS=80
TACO_RECALL_FACTUAL_K=2
TACO_RECALL_EMOTIONAL_K=2
TACO_RECALL_COHERENCE_K=3
TACO_RECALL_GENERAL_K=2
TACO_RECALL_FILLER_K=0
```

### 4. Benchmark stability

Azure endpoint was working but slow for benchmark-shaped prompts.

Implemented:

- response/judge outer timeout now honors `TACO_LLM_REQUEST_TIMEOUT_S`
- judge re-raises `CallTimeout` instead of silently returning score 0
- eval ingest can skip identity extraction:

```bash
TACO_EVAL_SKIP_IDENTITY_DURING_INGEST=1
```

This makes ingest much faster while preserving normal product behavior.

## Current Test Status

Latest full test command:

```bash
python3 -m pytest -q
```

Latest result:

```text
127 passed
```

## Latest Mock Eval Result

Command:

```bash
TACO_MOCK=1 \
TACO_EVAL_LIMIT_SCENARIOS=4 \
TACO_EVAL_LIMIT_PROBES=5 \
MPLCONFIGDIR=/private/tmp/mplconfig \
python3 -m eval.run
```

Latest mock summary:

```text
overall continuity : RAG 1.4  TACO 34.2
CES_retrieval      : RAG 19.0 TACO 1191.6
CES_total          : RAG 12.7 TACO 469.8
retrieval tok/turn : RAG 73.8 TACO 28.7
total tok/turn     : RAG 109.9 TACO 72.8
probes             : 20
timeouts           : 0
```

Important caveat:

This is a mock eval. It is useful for token/plumbing validation only. It does **not** prove equal-quality live performance.

The real takeaway from mock:

- retrieval payload inflation is fixed structurally,
- TACO now recalls sparsely,
- filler probes retrieve no memory by default,
- `CES_total` moved in the right direction.

## Azure Live Eval Setup

Secrets are in `.env` locally. Do not paste keys into future chat.

The important non-secret config shape:

```bash
BASE_LLM_MODEL=gpt-4.1-mini
BASE_EMBED_MODEL=text-embedding-3-small
OPENAI_BASE_URL=https://hackathons6969-resource.services.ai.azure.com/openai/v1
AZURE_EMBED_ENDPOINT=https://hackathons6969-resource.services.ai.azure.com
AZURE_API_VERSION=2024-10-21

TACO_RESPONSE_MAX_TOKENS=80
TACO_JUDGE_MAX_TOKENS=60
TACO_LLM_REQUEST_TIMEOUT_S=90
TACO_EVAL_SKIP_IDENTITY_DURING_INGEST=1
```

Verified:

- chat path works
- embedding path works
- embeddings return 1536 dimensions

Important Azure caveat:

The endpoint sometimes takes about 60 seconds for benchmark-shaped prompt/judge calls. Use small smoke runs first.

## Recommended Live Eval Commands

Smoke:

```bash
TACO_EVAL_LIMIT_SCENARIOS=1 \
TACO_EVAL_LIMIT_PROBES=3 \
MPLCONFIGDIR=/private/tmp/mplconfig \
python3 -m eval.run
```

Current main slice:

```bash
TACO_EVAL_LIMIT_SCENARIOS=4 \
TACO_EVAL_LIMIT_PROBES=5 \
MPLCONFIGDIR=/private/tmp/mplconfig \
python3 -m eval.run
```

Inspect summary:

```bash
cat eval/out/summary.csv
```

Detailed check:

```bash
jq '{
  overall,
  ces_retrieval,
  ces_total,
  retrieval_tokens,
  total_tokens,
  n_probes,
  meta: {
    timeouts: .meta.timeouts,
    timeouts_by_stage: .meta.timeouts_by_stage
  }
}' eval/out/summary.json
```

Live eval acceptance criteria:

- TACO overall quality >= RAG
- TACO retrieval tokens < RAG
- TACO total context tokens <= or near RAG
- TACO CES_retrieval > RAG
- TACO CES_total > RAG or materially improved
- timeouts = 0 or very low
- no major factual/coherence regression from sparse recall

## Files Changed In This Session

Core:

- `taco/memory/extract.py`
- `taco/memory/retrieval.py`
- `taco/memory/store.py`
- `taco/pipeline.py`
- `taco/config.py`
- `taco/llm.py`

Eval:

- `eval/harness.py`
- `eval/judge.py`
- `eval/run.py`

Tests:

- `tests/test_extract.py`
- `tests/test_sparse_recall.py`
- `tests/test_briefing_mode.py`
- `tests/test_ingest_only.py`
- `tests/test_response_hardening.py`

## Next Technical Step

Run a clean live eval with Azure.

If live quality regresses, inspect per-probe rows:

```bash
jq -r '.[] | select(.condition=="taco") |
"\\(.scenario) | \\(.kind) | score=\\(.score) rtok=\\(.retrieval_tokens) n=\\(.n_memories) | \\(.probe) | \\(.retrieved|join(" ; "))"'
eval/out/results.json
```

Likely next fixes depending on live result:

1. If factual recall is low:
   - improve direct query overlap and retrieval cues,
   - add more anchor patterns,
   - ensure exact facts like medication/company/class/name beat salient but unrelated progress facts.

2. If emotional recall is low:
   - tune emotional query classifier,
   - raise salience/tone boost,
   - ensure emotional queries get origin event, not only latest outcome.

3. If coherence is low:
   - add supersession/resolution logic:
     - interview tomorrow -> got job
     - diagnosis overwhelmed -> A1C improved
     - breakup -> Sam reached out

4. If retrieval tokens are still high:
   - lower `TACO_RECALL_MAX_TOKENS`,
   - reduce coherence K from 3 to 2,
   - compact memory text more aggressively.

## Product Direction Reminder

Do not package as production infra until a clean live benchmark proves the core claim.

Near-term proof target:

- TACO beats Naive RAG on overall continuity.
- TACO beats Naive RAG on retrieval tokens.
- TACO has strong emotional-continuity advantage.
- TACO preserves factual/coherence recall after sparse retrieval.

Then productize as:

```text
State-aware memory formation and sparse recall for emotionally persistent AI agents.
```

