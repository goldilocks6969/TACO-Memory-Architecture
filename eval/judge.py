"""LLM-as-judge: scores how well a response recalls/uses a probe's ground truth.

Returns a 0-100 continuity score per probe. Has a deterministic mock fallback so
the harness can be validated end-to-end without an API key (mock scores are
lexical-overlap proxies, useful only for plumbing, not for the real comparison).
"""
from __future__ import annotations

import json
import re
from typing import Dict

from taco import config

_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "with", "their", "they", "user",
    "users", "is", "was", "in", "on", "for", "that", "it", "connects", "trivial",
    "notes", "reflects", "the", "about", "after", "over",
}


def _mock_score(response: str, ground_truth: str) -> int:
    """Lexical-overlap proxy. Validation only — not a real judgement."""
    gt_terms = {w for w in re.findall(r"[a-z0-9']+", ground_truth.lower())
                if w not in _STOP and len(w) > 2}
    if not gt_terms:
        return 0
    resp = response.lower()
    hits = sum(1 for w in gt_terms if w in resp)
    return int(round(100 * hits / len(gt_terms)))


_RUBRIC = (
    "You are a strict evaluator of conversational MEMORY and CONTINUITY. "
    "Given a PROBE the user sent, the GROUND TRUTH a system should recall, and a "
    "candidate RESPONSE, score 0-100 how well the response demonstrates correct, "
    "relevant recall/use of the ground truth.\n"
    "100 = accurately and naturally reflects the ground truth; "
    "50 = partial, vague, or hedged; "
    "0 = absent, generic, asks what the user means, or contradicts it.\n"
    "Judge ONLY memory/continuity, not writing style. For 'filler' probes, score "
    "literal recall of the trivial fact. Respond with strict JSON: "
    '{"score": <0-100>, "reason": "<short>"}'
)


def judge(probe_text: str, ground_truth: str, kind: str, response: str) -> Dict:
    """Score a response. Always returns ``input_text`` / ``output_text`` (the raw
    prompt and the raw judge reply) so the eval harness can charge judge tokens.
    """
    user = (
        f"PROBE: {probe_text}\nGROUND TRUTH: {ground_truth}\n"
        f"PROBE KIND: {kind}\nRESPONSE: {response}"
    )
    input_text = _RUBRIC + "\n" + user

    if config.MOCK:
        score = _mock_score(response, ground_truth)
        return {"score": score, "reason": "mock",
                "input_text": input_text,
                "output_text": json.dumps({"score": score, "reason": "mock"})}
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "No API key configured — set OPENAI_API_KEY or TACO_MOCK=1 explicitly"
        )

    from taco.llm import _client
    from taco.retry import with_retries

    raw_output = ""
    try:
        resp = with_retries(lambda: _client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": _RUBRIC},
                      {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        ))
        raw_output = resp.choices[0].message.content
        data = json.loads(raw_output)
        score = int(max(0, min(100, float(data.get("score", 0)))))
        return {"score": score, "reason": str(data.get("reason", ""))[:200],
                "input_text": input_text, "output_text": raw_output}
    except Exception as e:  # robust to endpoints lacking json mode, etc.
        m = re.search(r'"?score"?\s*[:=]\s*(\d+)', raw_output)
        return {"score": int(m.group(1)) if m else 0,
                "reason": f"parse-fallback: {e}"[:120],
                "input_text": input_text, "output_text": raw_output}
