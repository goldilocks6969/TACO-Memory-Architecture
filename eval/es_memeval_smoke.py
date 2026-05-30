"""Small ES-MemEval live smoke run for TACO vs session-level naive RAG.

This adapter intentionally avoids the ES-MemEval repo's heavier reproduction
stack. It uses the public EvoEmo JSON, samples a tiny QA subset, and keeps the
evaluation cheap enough to run locally with real API calls.

Default:
    python3 -m eval.es_memeval_smoke

Outputs:
    eval/out/es_memeval_smoke/results.jsonl
    eval/out/es_memeval_smoke/summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from taco import config, db
from taco.llm import _client
from taco.pipeline import Taco
from taco.retry import with_retries

from .baseline import NaiveRAG
from .harness import EVAL_DSN
from .tokens import counter


DATA_URL = (
    "https://raw.githubusercontent.com/slptongji/ES-MemEval/main/"
    "data/evo_emo.json"
)
OUT_DIR = Path(__file__).resolve().parent / "out" / "es_memeval_smoke"
DATA_PATH = OUT_DIR / "evo_emo.json"
RESULTS_PATH = OUT_DIR / "results.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
CACHE_DIR = OUT_DIR / "ingest_cache"
MEM0_DIR = OUT_DIR / "mem0"
TABLES = (
    "episodes", "facts", "semantic_beliefs", "emotional_timeline",
    "reflections", "procedural", "identity", "state_log",
)
USER_TABLES = tuple(t for t in TABLES if t != "procedural")

ANSWER_SYSTEM = """## Task Description
You are given a user question and a set of retrieved memory fragments.
Filter and summarize the relevant information from the memory fragments and
generate a concise, accurate answer to the user's question based on the most
pertinent details. If the question cannot be answered with the available
information, return "Unknown."

## Output Format
Answer: <concise answer>"""

JUDGE_SYSTEM = "You are a strict evaluator."


def _log(msg: str) -> None:
    print(f"[es-memeval +{time.monotonic() - _T0:6.1f}s] {msg}", flush=True)


def _download_data() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return
    _log(f"downloading EvoEmo data from {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        DATA_PATH.write_bytes(response.read())


def _load_data() -> List[Dict]:
    _download_data()
    return json.loads(DATA_PATH.read_text())


def _reset(conn) -> None:
    conn.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY")


def _delete_user(conn, user_id: str) -> None:
    for table in USER_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))


def _user_row_count(conn, user_id: str) -> int:
    total = 0
    for table in USER_TABLES:
        total += conn.execute(
            f"SELECT count(*) FROM {table} WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
    return total


def _cache_key(seeker_id: str, args: argparse.Namespace) -> str:
    payload = {
        "dataset": "ES-MemEval EvoEmo",
        "seeker": seeker_id,
        "max_sessions": args.max_sessions,
        "max_user_turns": args.max_user_turns,
        "embedding_model": config.EMBED_MODEL,
        "extraction_mode": config.EXTRACTION_MODE,
        "light_extract_local": config.LIGHT_EXTRACT_LOCAL,
        "skip_identity": config.SKIP_IDENTITY_DURING_INGEST,
        "schema": 1,
    }
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{seeker_id}_{digest}"


def _cache_meta_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def _mem0_key(seeker_id: str, args: argparse.Namespace) -> str:
    payload = {
        "dataset": "ES-MemEval EvoEmo",
        "seeker": seeker_id,
        "max_sessions": args.max_sessions,
        "max_user_turns": args.max_user_turns,
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBED_MODEL,
        "mem0_ingest": "session_messages_infer_true",
        "schema": 1,
    }
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{seeker_id}_{digest}"


def _mem0_meta_path(mem0_key: str) -> Path:
    return MEM0_DIR / f"{mem0_key}.json"


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _gap_hours(prev: date | None, current: date | None) -> float:
    if prev is None or current is None:
        return 0.5
    return max(0.5, float((current - prev).days * 24))


def _session_text(session: Dict, include_summary: bool = False) -> str:
    lines = [f"[{session.get('timestamp', '?')}] session {session.get('id', '?')}"]
    if include_summary and session.get("summary"):
        lines.append(f"Summary: {session['summary']}")
    for turn in session.get("dialogue", []):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        idx = turn.get("idx", "?")
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines)


def _dialogue_line(turn: Dict) -> str:
    role = turn.get("role", "unknown")
    content = str(turn.get("content", "")).strip()
    idx = turn.get("idx", "?")
    return f"{idx}. {role}: {content}"


def _rag_chunks(session: Dict, args: argparse.Namespace) -> List[str]:
    """Build explicit RAG chunks for ES-MemEval.

    Default: sliding windows of 8 dialogue turns with overlap 2. That preserves
    local conversational context without embedding whole-session blobs or using
    dataset-authored summaries as oracle memory.
    """
    if args.rag_chunking == "session":
        return [_session_text(session, include_summary=args.rag_include_summary)]

    turns = [
        _dialogue_line(turn)
        for turn in session.get("dialogue", [])
        if str(turn.get("content", "")).strip()
    ]
    if args.rag_chunking == "turn":
        windows = [[turn] for turn in turns]
    else:
        size = max(1, args.rag_window_turns)
        overlap = min(max(0, args.rag_window_overlap), size - 1)
        step = max(1, size - overlap)
        windows = [turns[i:i + size] for i in range(0, len(turns), step)]

    chunks: List[str] = []
    for window in windows:
        if not window:
            continue
        header = f"[{session.get('timestamp', '?')}] session {session.get('id', '?')}"
        chunks.append("\n".join([header, *window]))
    return chunks


def _seeker_turns(session: Dict) -> Iterable[str]:
    timestamp = session.get("timestamp", "?")
    sid = session.get("id", "?")
    for turn in session.get("dialogue", []):
        if turn.get("role") != "seeker":
            continue
        content = str(turn.get("content", "")).strip()
        if content:
            yield f"[{timestamp} session {sid} turn {turn.get('idx', '?')}] {content}"


def _session_messages(session: Dict) -> List[Dict[str, str]]:
    timestamp = session.get("timestamp", "?")
    sid = session.get("id", "?")
    messages: List[Dict[str, str]] = []
    for turn in session.get("dialogue", []):
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        role = "user" if turn.get("role") == "seeker" else "assistant"
        messages.append({
            "role": role,
            "content": f"[{timestamp} session {sid} turn {turn.get('idx', '?')}] {content}",
        })
    return messages


def _flatten_questions(seeker: Dict) -> List[Dict]:
    out: List[Dict] = []
    for group in seeker.get("questions", []):
        for q in group.get("questions", []):
            out.append({**q, "group": group.get("id")})
    return out


def _sample_questions(seeker: Dict, per_capability: int, max_questions: int) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for q in _flatten_questions(seeker):
        grouped[q["capability"]].append(q)

    preferred = [
        "information extraction",
        "temporal reasoning",
        "conflict detection",
        "abstention",
        "user modeling",
    ]
    sampled: List[Dict] = []
    for capability in preferred:
        sampled.extend(grouped.get(capability, [])[:per_capability])
    return sampled[:max_questions]


def _hard_questions(seeker: Dict, max_questions: int) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for q in _flatten_questions(seeker):
        grouped[q["capability"]].append(q)

    order = [
        ("abstention", 0),
        ("conflict detection", 0),
        ("user modeling", 0),
        ("temporal reasoning", 0),
        ("conflict detection", 5),
        ("abstention", 4),
        ("user modeling", 7),
        ("temporal reasoning", 6),
        ("information extraction", 0),
    ]
    selected: List[Dict] = []
    seen = set()
    for capability, index in order:
        items = grouped.get(capability, [])
        if index >= len(items):
            continue
        q = items[index]
        key = (q["capability"], q["idx"], q["question"])
        if key in seen:
            continue
        selected.append(q)
        seen.add(key)
        if len(selected) >= max_questions:
            break
    return selected


def _configure_fast_eval() -> None:
    if not os.getenv("TACO_STATE_BRIEFING_MODE"):
        config.STATE_BRIEFING_MODE = "compact"
    if not os.getenv("TACO_EVAL_EXTRACTION_MODE"):
        config.EXTRACTION_MODE = "light"
    if "TACO_EVAL_SKIP_RESPOND_DURING_INGEST" not in os.environ:
        config.SKIP_RESPOND_DURING_INGEST = True
    if "TACO_EVAL_SKIP_IDENTITY_DURING_INGEST" not in os.environ:
        config.SKIP_IDENTITY_DURING_INGEST = True
    if "TACO_EVAL_LIGHT_EXTRACT_LOCAL" not in os.environ:
        config.LIGHT_EXTRACT_LOCAL = True
    if "TACO_EVAL_FAST_RESPONSES" not in os.environ:
        config.FAST_RESPONSES = True
    if "TACO_EVAL_FAST_LIVE" not in os.environ:
        config.FAST_LIVE = True
    if "TACO_RETRIEVAL_MODE" not in os.environ:
        config.RETRIEVAL_MODE = "hybrid"


def _openai_answer(memory_context: str, question: str) -> str:
    user = f"Question: {question}\nRelevant Memory:\n{memory_context or '(none)'}"
    if config.MOCK:
        return "Unknown"
    resp = with_retries(lambda: _client().chat.completions.create(
        model=config.RESPONSE_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=config.RESPONSE_MAX_TOKENS,
        timeout=config.LLM_REQUEST_TIMEOUT_S,
    ), label="es_memeval.answer",
        timeout_per_attempt=config.LLM_REQUEST_TIMEOUT_S)
    text = resp.choices[0].message.content.strip()
    return re.sub(r"^Answer:\s*", "", text, flags=re.I).strip()


class Mem0Baseline:
    def __init__(self, mem0_key: str, user_id: str):
        # Keep Mem0 self-contained for benchmark runs. Telemetry creates a
        # shared local Qdrant store under ~/.mem0 by default, which can lock
        # across runs and is irrelevant to benchmark behavior.
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        os.environ.setdefault("MEM0_DIR", str(MEM0_DIR / "home"))
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError(
                "mem0ai is not installed. Run `python3 -m pip install mem0ai`."
            ) from exc

        self.user_id = user_id
        base = MEM0_DIR / mem0_key
        base.mkdir(parents=True, exist_ok=True)
        llm_config = {
            "model": config.LLM_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": 0,
            # Mem0 uses this model for extraction/update JSON, not final user
            # answers. Keep it separate from TACO_RESPONSE_MAX_TOKENS, which is
            # intentionally tiny for benchmark answers.
            "max_tokens": 1000,
        }
        embed_config = {
            "model": config.EMBED_MODEL,
            "api_key": config.EMBED_API_KEY,
            "embedding_dims": config.EMBED_DIM,
        }
        if config.OPENAI_BASE_URL:
            llm_config["openai_base_url"] = config.OPENAI_BASE_URL
        if config.EMBED_BASE_URL:
            embed_config["openai_base_url"] = config.EMBED_BASE_URL
        self.memory = Memory.from_config({
            "llm": {"provider": "openai", "config": llm_config},
            "embedder": {"provider": "openai", "config": embed_config},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "embedding_model_dims": config.EMBED_DIM,
                    "path": str(base / "qdrant"),
                },
            },
            "history_db_path": str(base / "history.db"),
            "version": "v1.1",
        })

    def ingest_session(self, session: Dict) -> None:
        messages = _session_messages(session)
        if messages:
            self.memory.add(
                messages,
                user_id=self.user_id,
                metadata={
                    "date": session.get("timestamp"),
                    "session_id": session.get("id"),
                },
                infer=True,
            )

    def search(self, query: str, top_k: int) -> List[str]:
        result = self.memory.search(
            query, top_k=top_k, filters={"user_id": self.user_id}, threshold=0.0
        )
        memories = result.get("results", result) if isinstance(result, dict) else result
        out: List[str] = []
        for item in memories or []:
            if isinstance(item, dict):
                text = item.get("memory") or item.get("text") or item.get("data")
            else:
                text = getattr(item, "memory", None) or getattr(item, "text", None)
            if text:
                out.append(str(text))
        return out

    def close(self) -> None:
        close = getattr(self.memory, "close", None)
        if close:
            close()


def _prepare_mem0(
    seeker: Dict,
    sessions: Sequence[Dict],
    args: argparse.Namespace,
) -> tuple[Mem0Baseline, int, bool, str]:
    sid = seeker["id"]
    mem0_key = _mem0_key(sid, args)
    user_id = f"es_memeval_mem0_{mem0_key}"
    meta_path = _mem0_meta_path(mem0_key)
    MEM0_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear_mem0_cache:
        import shutil

        shutil.rmtree(MEM0_DIR / mem0_key, ignore_errors=True)
        if meta_path.exists():
            meta_path.unlink()

    baseline = Mem0Baseline(mem0_key, user_id)
    if args.mem0_cache and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        _log(
            f"  mem0 ingest cache hit user_id={user_id} "
            f"sessions={meta.get('sessions_ingested', '?')}"
        )
        return baseline, int(meta.get("sessions_ingested", 0)), True, user_id

    # Recreate the local store for deterministic rebuilds on cache miss.
    if not args.clear_mem0_cache:
        import shutil

        baseline.close()
        shutil.rmtree(MEM0_DIR / mem0_key, ignore_errors=True)
        baseline = Mem0Baseline(mem0_key, user_id)

    ingested = 0
    for session in sessions:
        baseline.ingest_session(session)
        ingested += 1
        _log(f"  mem0 ingested session {ingested}/{len(sessions)}")
    meta_path.write_text(json.dumps({
        "user_id": user_id,
        "seeker": sid,
        "sessions_ingested": ingested,
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBED_MODEL,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    return baseline, ingested, False, user_id


def _judge(gold: str, prediction: str, question: str) -> Dict:
    prompt = f"""You are an impartial evaluator.
Your task is to score a model's answer to a given question against a gold
(reference) answer.

Scoring criteria:
- 0: Completely wrong or irrelevant
- 1: Partially correct but incomplete, vague, or missing key information
- 2: Completely correct and contextually accurate

Question: {question}
Gold Answer: {gold}
Model Answer: {prediction}

Output only one line in the exact format: Score: X"""
    if config.MOCK:
        return {"score": 0, "raw": "Score: 0"}
    resp = with_retries(lambda: _client().chat.completions.create(
        model=config.JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=20,
        timeout=config.LLM_REQUEST_TIMEOUT_S,
    ), label="es_memeval.judge",
        timeout_per_attempt=config.LLM_REQUEST_TIMEOUT_S)
    raw = resp.choices[0].message.content.strip()
    m = re.search(r"[0-2]", raw)
    return {"score": int(m.group(0)) if m else 0, "raw": raw}


def _f1(gold: str, prediction: str) -> float:
    def normalize(text: str) -> List[str]:
        return re.sub(r"\W+", " ", text.lower()).strip().split()

    gold_tokens = normalize(gold)
    pred_tokens = normalize(prediction)
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = set(gold_tokens) & set(pred_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _summarize(rows: List[Dict], meta: Dict) -> Dict:
    by_condition: Dict[str, List[Dict]] = defaultdict(list)
    by_capability: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_capability[row["capability"]][row["condition"]].append(row)

    summary = {
        "meta": meta,
        "overall": {},
        "by_capability": {},
    }
    for condition, condition_rows in by_condition.items():
        summary["overall"][condition] = {
            "n": len(condition_rows),
            "f1": _mean([r["f1"] for r in condition_rows]),
            "judge_0_2": _mean([r["judge_score"] for r in condition_rows]),
            "retrieval_tokens": _mean([r["retrieval_tokens"] for r in condition_rows]),
            "total_context_tokens": _mean([r["total_context_tokens"] for r in condition_rows]),
        }
    for capability, conditions in by_capability.items():
        summary["by_capability"][capability] = {}
        for condition, condition_rows in conditions.items():
            summary["by_capability"][capability][condition] = {
                "n": len(condition_rows),
                "f1": _mean([r["f1"] for r in condition_rows]),
                "judge_0_2": _mean([r["judge_score"] for r in condition_rows]),
            }
    return summary


def _ingest_taco(
    conn,
    taco: Taco,
    seeker: Dict,
    sessions: Sequence[Dict],
    args: argparse.Namespace,
) -> int:
    prev_day: date | None = None
    n_taco_turns = 0
    stop_ingest = False
    for session in sessions:
        current_day = _parse_day(session.get("timestamp"))
        first = True
        for text in _seeker_turns(session):
            if args.max_user_turns is not None and n_taco_turns >= args.max_user_turns:
                stop_ingest = True
                break
            gap = _gap_hours(prev_day, current_day) if first else 0.2
            taco.turn(text, hours_since_last=gap, generate_response=False)
            n_taco_turns += 1
            first = False
        if current_day is not None:
            prev_day = current_day
        if stop_ingest:
            break
    return n_taco_turns


def _prepare_taco(
    conn,
    seeker: Dict,
    sessions: Sequence[Dict],
    args: argparse.Namespace,
) -> tuple[Taco, int, bool, str]:
    sid = seeker["id"]
    cache_key = _cache_key(sid, args)
    user_id = f"es_memeval_{cache_key}"
    cache_path = _cache_meta_path(cache_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear_ingest_cache:
        if cache_path.exists():
            cache_path.unlink()
        _delete_user(conn, user_id)

    if args.ingest_cache and cache_path.exists():
        row_count = _user_row_count(conn, user_id)
        if row_count:
            meta = json.loads(cache_path.read_text())
            _log(
                f"  taco ingest cache hit user_id={user_id} "
                f"rows={row_count} turns={meta.get('taco_user_turns', '?')}"
            )
            return Taco(conn, user_id=user_id), int(meta.get("taco_user_turns", 0)), True, user_id
        _log(f"  taco ingest cache marker existed but DB rows were missing; rebuilding {user_id}")

    _delete_user(conn, user_id)
    taco = Taco(conn, user_id=user_id)
    n_taco_turns = _ingest_taco(conn, taco, seeker, sessions, args)
    cache_path.write_text(json.dumps({
        "user_id": user_id,
        "seeker": sid,
        "taco_user_turns": n_taco_turns,
        "sessions": len(sessions),
        "max_sessions": args.max_sessions,
        "max_user_turns": args.max_user_turns,
        "embedding_model": config.EMBED_MODEL,
        "extraction_mode": config.EXTRACTION_MODE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    _log(f"  taco ingest cache stored user_id={user_id} turns={n_taco_turns}")
    return taco, n_taco_turns, False, user_id


def run(args: argparse.Namespace) -> Dict:
    _configure_fast_eval()
    data = _load_data()
    seekers = data[: args.seekers]
    ntok = counter(config.LLM_MODEL)

    if not config.OPENAI_API_KEY and not config.MOCK:
        raise RuntimeError("Set OPENAI_API_KEY or TACO_MOCK=1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("")

    _log(
        f"starting ES-MemEval smoke · seekers={len(seekers)} · "
        f"per_capability={args.per_capability} · max_questions={args.max_questions} · "
        f"model={config.LLM_MODEL} · response={config.RESPONSE_MODEL} · "
        f"judge={config.JUDGE_MODEL}"
    )

    db.init_db(EVAL_DSN)
    conn = db.connect(EVAL_DSN)

    rows: List[Dict] = []
    mem0_baselines: List[Mem0Baseline] = []
    try:
        for si, seeker in enumerate(seekers, 1):
            sid = seeker["id"]
            name = seeker.get("basic_info", {}).get("name", sid)
            questions = (
                _sample_questions(seeker, args.per_capability, args.max_questions)
                if args.question_set == "dataset-order"
                else _hard_questions(seeker, args.max_questions)
            )
            sessions = seeker.get("dialog_history", [])
            if args.max_sessions is not None:
                sessions = sessions[: args.max_sessions]
            _log(
                f"seeker {si}/{len(seekers)} {sid} ({name}) · "
                f"sessions={len(sessions)}/{len(seeker.get('dialog_history', []))} · "
                f"questions={len(questions)}"
            )

            rag = NaiveRAG(k=args.rag_topk)
            for session in sessions:
                for chunk in _rag_chunks(session, args):
                    rag.ingest_chunk(chunk)
            taco, n_taco_turns, cache_hit, taco_user_id = _prepare_taco(
                conn, seeker, sessions, args
            )
            mem0_baseline = None
            mem0_sessions = 0
            mem0_cache_hit = False
            if args.include_mem0:
                mem0_baseline, mem0_sessions, mem0_cache_hit, mem0_user_id = _prepare_mem0(
                    seeker, sessions, args
                )
                mem0_baselines.append(mem0_baseline)
            _log(
                f"  ready rag_chunks={len(rag._chunks)} "
                f"rag_dropped={rag.dropped_chunks} "
                f"taco_user_turns={n_taco_turns} cache_hit={int(cache_hit)} "
                f"mem0_sessions={mem0_sessions} mem0_cache_hit={int(mem0_cache_hit)}"
            )

            for qi, q in enumerate(questions, 1):
                question = q["question"]
                gold = q["answer"]
                capability = q["capability"]
                _log(f"  q {qi}/{len(questions)} [{capability}] {question[:90]!r}")

                taco_result = taco.probe(question)
                taco_native_answer = taco_result.response
                taco_retrieved = [ep.content for ep in taco_result.retrieved]
                taco_retrieval_text = "\n".join(
                    f"{i}. {passage}" for i, passage in enumerate(taco_retrieved, 1)
                )
                if args.taco_answer_mode == "native":
                    taco_answer = taco_native_answer
                else:
                    taco_answer = _openai_answer(taco_retrieval_text, question)
                taco_judge = _judge(gold, taco_answer, question)
                taco_row = {
                    "seeker": sid,
                    "name": name,
                    "condition": "taco",
                    "capability": capability,
                    "question": question,
                    "gold": gold,
                    "answer": taco_answer,
                    "native_answer": taco_native_answer,
                    "answer_mode": args.taco_answer_mode,
                    "f1": _f1(gold, taco_answer),
                    "judge_score": taco_judge["score"],
                    "judge_raw": taco_judge["raw"],
                    "retrieval_tokens": ntok(taco_retrieval_text),
                    "total_context_tokens": ntok(taco_result.briefing) + ntok(question),
                    "n_memories": len(taco_retrieved),
                    "retrieved": taco_retrieved,
                }
                rows.append(taco_row)
                with RESULTS_PATH.open("a") as f:
                    f.write(json.dumps(taco_row, ensure_ascii=False) + "\n")

                rag_passages = rag.retrieve(question)
                rag_memory = "\n".join(
                    f"{i}. {passage}" for i, passage in enumerate(rag_passages, 1)
                )
                rag_answer = _openai_answer(rag_memory, question)
                rag_judge = _judge(gold, rag_answer, question)
                rag_row = {
                    "seeker": sid,
                    "name": name,
                    "condition": "rag",
                    "capability": capability,
                    "question": question,
                    "gold": gold,
                    "answer": rag_answer,
                    "f1": _f1(gold, rag_answer),
                    "judge_score": rag_judge["score"],
                    "judge_raw": rag_judge["raw"],
                    "retrieval_tokens": ntok(rag_memory),
                    "total_context_tokens": ntok(ANSWER_SYSTEM) + ntok(rag_memory) + ntok(question),
                    "n_memories": len(rag_passages),
                    "retrieved": rag_passages,
                }
                rows.append(rag_row)
                with RESULTS_PATH.open("a") as f:
                    f.write(json.dumps(rag_row, ensure_ascii=False) + "\n")

                if mem0_baseline is not None:
                    mem0_passages = mem0_baseline.search(question, args.mem0_topk)
                    mem0_memory = "\n".join(
                        f"{i}. {passage}" for i, passage in enumerate(mem0_passages, 1)
                    )
                    mem0_answer = _openai_answer(mem0_memory, question)
                    mem0_judge = _judge(gold, mem0_answer, question)
                    mem0_row = {
                        "seeker": sid,
                        "name": name,
                        "condition": "mem0",
                        "capability": capability,
                        "question": question,
                        "gold": gold,
                        "answer": mem0_answer,
                        "f1": _f1(gold, mem0_answer),
                        "judge_score": mem0_judge["score"],
                        "judge_raw": mem0_judge["raw"],
                        "retrieval_tokens": ntok(mem0_memory),
                        "total_context_tokens": ntok(ANSWER_SYSTEM) + ntok(mem0_memory) + ntok(question),
                        "n_memories": len(mem0_passages),
                        "retrieved": mem0_passages,
                    }
                    rows.append(mem0_row)
                    with RESULTS_PATH.open("a") as f:
                        f.write(json.dumps(mem0_row, ensure_ascii=False) + "\n")

                _log(
                    f"    scores taco={taco_row['judge_score']} "
                    f"rag={rag_row['judge_score']} "
                    f"mem0={'-' if mem0_baseline is None else mem0_row['judge_score']} · "
                    f"f1 taco={taco_row['f1']:.2f} rag={rag_row['f1']:.2f}"
                )
    finally:
        for baseline in mem0_baselines:
            baseline.close()
        conn.close()

    meta = {
        "dataset": "ES-MemEval EvoEmo",
        "dataset_url": DATA_URL,
        "model": config.LLM_MODEL,
        "response_model": config.RESPONSE_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "embedding_model": config.EMBED_MODEL,
        "seekers": len(seekers),
        "rag_topk": args.rag_topk,
        "rag_chunking": args.rag_chunking,
        "rag_window_turns": args.rag_window_turns,
        "rag_window_overlap": args.rag_window_overlap,
        "rag_include_summary": args.rag_include_summary,
        "question_set": args.question_set,
        "include_mem0": args.include_mem0,
        "mem0_topk": args.mem0_topk,
        "state_briefing_mode": config.STATE_BRIEFING_MODE,
        "extraction_mode": config.EXTRACTION_MODE,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "fast_live": config.FAST_LIVE,
        "ingest_cache": args.ingest_cache,
    }
    summary = _summarize(rows, meta)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _log(f"wrote {RESULTS_PATH}")
    _log(f"wrote {SUMMARY_PATH}")
    return summary


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seekers", type=int, default=1)
    p.add_argument("--per-capability", type=int, default=1)
    p.add_argument("--max-questions", type=int, default=5)
    p.add_argument("--question-set", choices=("hard", "dataset-order"),
                   default="hard")
    p.add_argument("--rag-topk", type=int, default=8)
    p.add_argument("--rag-chunking", choices=("window", "turn", "session"),
                   default="window")
    p.add_argument("--rag-window-turns", type=int, default=8)
    p.add_argument("--rag-window-overlap", type=int, default=2)
    p.add_argument("--rag-include-summary", action="store_true")
    p.add_argument("--include-mem0", action="store_true")
    p.add_argument("--mem0-topk", type=int, default=8)
    p.add_argument("--no-mem0-cache", dest="mem0_cache",
                   action="store_false", default=True)
    p.add_argument("--clear-mem0-cache", action="store_true")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--max-user-turns", type=int, default=None)
    p.add_argument("--taco-answer-mode", choices=("es-prompt", "native"),
                   default="es-prompt")
    p.add_argument("--no-ingest-cache", dest="ingest_cache",
                   action="store_false", default=True)
    p.add_argument("--clear-ingest-cache", action="store_true")
    return p


_T0 = time.monotonic()


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(summary["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
