"""LLM-backed cognition: turn analysis, the reasoning response, and abstraction.

Each function has a deterministic heuristic fallback (used when BUBBLES_MOCK=1
or no API key is present) so the whole architecture runs without external calls.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from . import config

# --------------------------------------------------------------------------- #
# Heuristic lexicons for the keyless fallback (and rule-based mode, §3.1)
# --------------------------------------------------------------------------- #
_HIGH_AROUSAL = {
    "died", "death", "dead", "passed away", "breakup", "broke up", "divorce",
    "fired", "laid off", "scared", "terrified", "panic", "anxious", "anxiety",
    "furious", "devastated", "crying", "cried", "suicidal", "hopeless",
    "emergency", "crisis", "hate", "miscarriage", "diagnosed", "cancer",
}
_MED_AROUSAL = {
    "worried", "stressed", "stress", "nervous", "upset", "angry", "sad",
    "frustrated", "argument", "fight", "conflict", "overwhelmed", "afraid",
    "lonely", "hurt", "guilty", "ashamed", "exhausted",
}
_VULN_MARKERS = {
    "i feel", "i'm struggling", "i am struggling", "i don't know what to do",
    "i need", "help me", "i've never told", "i have never told", "honestly",
    "to be honest", "i'm scared", "i am scared", "i can't", "i cannot",
    "i'm not okay", "i am not okay", "confess", "admit", "vulnerable",
}
_TRANSACTIONAL = {
    "what time", "schedule", "remind me", "how do i", "thanks", "thank you",
    "hi", "hello", "hey", "ok", "okay", "cool", "got it", "weather",
}


def _heuristic_analyze(text: str) -> Dict:
    t = text.lower().strip()
    emotional = 0.0
    vulnerability = 0.0

    for kw in _HIGH_AROUSAL:
        if kw in t:
            emotional += 60
            vulnerability += 18
    for kw in _MED_AROUSAL:
        if kw in t:
            emotional += 24
    for kw in _VULN_MARKERS:
        if kw in t:
            vulnerability += 30
            emotional += 8

    # surface cues
    emotional += min(20, text.count("!") * 7)
    if re.search(r"[A-Z]{4,}", text):
        emotional += 10
    if "..." in text:
        vulnerability += 8
    # first-person disclosure density nudges vulnerability
    vulnerability += min(20, len(re.findall(r"\bi\b|\bme\b|\bmy\b", t)) * 4)

    transactional = any(t.startswith(k) or t == k for k in _TRANSACTIONAL)
    if transactional and emotional < 15:
        emotional = max(0, emotional - 10)

    emotional = max(0.0, min(100.0, emotional))
    vulnerability = max(0.0, min(100.0, vulnerability))

    # salience 1..10 derived from the affective read
    drive = max(emotional, 0.7 * emotional + 0.6 * vulnerability)
    salience = 1 + (drive / 100.0) * 9
    salience = max(1.0, min(10.0, round(salience, 1)))

    if emotional >= 60:
        tone = "distress" if vulnerability >= 40 else "intense"
    elif emotional >= 30:
        tone = "tender" if vulnerability >= 30 else "concerned"
    elif transactional:
        tone = "transactional"
    else:
        tone = "neutral"

    return {
        "emotional": emotional,
        "vulnerability": vulnerability,
        "salience": salience,
        "tone": tone,
    }


def _client():
    from openai import OpenAI

    return OpenAI(api_key=config.OPENAI_API_KEY)


def analyze(text: str) -> Dict:
    """Return {emotional, vulnerability, salience, tone} for one turn.

    A single call does double duty: it feeds state inference (E, V) AND the
    write-time salience gate (1–10), which is more token-efficient than two
    separate judge calls (§6.1 token-efficiency argument).
    """
    if config.MOCK or not config.OPENAI_API_KEY:
        return _heuristic_analyze(text)

    sys = (
        "You are the interoceptive sensor of a cognitive memory layer. Read the "
        "user's message and rate it. Respond with strict JSON only:\n"
        '{"emotional": <0-100>, "vulnerability": <0-100>, '
        '"salience": <1-10>, "tone": "<one word>"}\n'
        "emotional = affective arousal in the message. vulnerability = openness, "
        "fragility, self-disclosure, expressed need. salience = how worth "
        "remembering long-term this is (10 = breakup/death/identity revelation, "
        "1 = greeting/small talk). tone = a single lowercase word."
    )
    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": text}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "emotional": float(data.get("emotional", 0)),
        "vulnerability": float(data.get("vulnerability", 0)),
        "salience": max(1.0, min(10.0, float(data.get("salience", 1)))),
        "tone": str(data.get("tone", "neutral")),
    }


def respond(system_brief: str, user_message: str, planning_depth: int) -> str:
    """The single reasoning-engine invocation (Figure 4 step 5)."""
    if config.MOCK or not config.OPENAI_API_KEY:
        return _mock_respond(system_brief, user_message, planning_depth)

    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system_brief},
                  {"role": "user", "content": user_message}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def _mock_respond(system_brief: str, user_message: str, planning_depth: int) -> str:
    plan = "" if planning_depth <= 1 else f" [planning {planning_depth} steps ahead]"
    return (
        f"(mock reasoning engine{plan}) I hear you. "
        f"Given what I remember, here's my response to: \"{user_message[:80]}\""
    )


def abstract(texts: List[str]) -> str:
    """Roll a set of decaying episodes into one semantic belief (§4.4, L3)."""
    if config.MOCK or not config.OPENAI_API_KEY:
        joined = " ".join(texts)[:160]
        return f"This person has shown a recurring pattern around: {joined}"

    sys = (
        "Compress these fading episodic memories into a single durable belief "
        "about the user — the meaning that should survive after the specific "
        "events are forgotten. One sentence, present tense, starts with 'This "
        "person'."
    )
    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": "\n".join(f"- {t}" for t in texts)}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
