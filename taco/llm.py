"""LLM-backed cognition: turn analysis, the reasoning response, and abstraction.

Each function has a deterministic heuristic fallback (used when TACO_MOCK=1
or no API key is present) so the whole architecture runs without external calls.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from . import config
from .retry import with_retries

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

    return OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


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
    resp = with_retries(lambda: _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": text}],
        temperature=0,
        response_format={"type": "json_object"},
    ))
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

    resp = with_retries(lambda: _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system_brief},
                  {"role": "user", "content": user_message}],
        temperature=0.7,
    ))
    return resp.choices[0].message.content.strip()


def _stance_mode(system_brief: str) -> str:
    m = re.search(r"\[stance:\s*([a-z]+)\]", system_brief)
    return m.group(1) if m else "neutral"


def _top_memory(system_brief: str) -> str:
    """Pull the most privileged remembered episode out of the briefing block.

    Lets the keyless mock actually *demonstrate* memory-conditioned recall (and
    makes the continuity benchmark reflect retrieval quality, not just plumbing).
    """
    in_block = False
    for line in system_brief.splitlines():
        if line.startswith("Psychologically privileged memories"):
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if stripped.startswith("•"):
                # strip the "• [tone, salience N/10, R=x] " metadata prefix
                return re.sub(r"^•\s*\[[^\]]*\]\s*", "", stripped).strip()
            if stripped == "":
                break
    return ""


def _mock_respond(system_brief: str, user_message: str, planning_depth: int) -> str:
    plan = "" if planning_depth <= 1 else f" [planning {planning_depth} steps ahead]"
    mode = _stance_mode(system_brief)
    mem = _top_memory(system_brief)
    # In privileged stances the mock leads with continuity; in transactional ones
    # it stays terse and only uses memory if it surfaced — mirroring the directives.
    if mem and mode in ("distress", "crisis", "vulnerable", "concerned"):
        recall = f" Staying with what matters: {mem}"
    elif mem:
        recall = f" For reference: {mem}"
    else:
        recall = ""
    return (
        f"(mock reasoning engine, stance={mode}{plan}) I hear you.{recall} "
        f"Here's my response to: \"{user_message[:80]}\""
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
    resp = with_retries(lambda: _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": "\n".join(f"- {t}" for t in texts)}],
        temperature=0.3,
    ))
    return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# L10 — identity extraction (the persistent self-model)
# --------------------------------------------------------------------------- #
_REL_ROLES = (
    "partner", "wife", "husband", "girlfriend", "boyfriend", "fiancé", "fiancee",
    "mother", "mom", "father", "dad", "son", "daughter", "sister", "brother",
    "friend", "boss", "manager", "therapist", "doctor", "roommate",
)


def _heuristic_identity(text: str) -> List[Dict]:
    """Cheap, deterministic identity extraction for the keyless path.

    Pulls named relationships ("my partner Maya") and a coarse occupation. The
    real LLM extractor below generalises well beyond these patterns.
    """
    out: List[Dict] = []
    roles = "|".join(_REL_ROLES)
    for m in re.finditer(rf"\bmy ({roles})(?:[, ]+(?:named|called)\s+)?\s+([A-Z][a-z]+)", text):
        out.append({"attribute": f"relationship:{m.group(1).lower()}",
                    "value": m.group(2), "confidence": 0.6})
    job = re.search(r"\bI(?:'m| am)? (?:a|an) ([a-z]+(?: [a-z]+)?)\b", text)
    if job and job.group(1) not in ("bit", "little", "lot", "few"):
        out.append({"attribute": "occupation", "value": job.group(1), "confidence": 0.5})
    return out


def extract_identity(text: str, existing: List[tuple] | None = None) -> List[Dict]:
    """Return durable identity facts [{attribute, value, confidence}] from a turn.

    `existing` is the current self-model (attribute, value, confidence) so the
    model can revise rather than duplicate. Identity = who the person *is* across
    time (values, relationships, ongoing struggles), not transient events.
    """
    if config.MOCK or not config.OPENAI_API_KEY:
        return _heuristic_identity(text)

    known = "; ".join(f"{a}={v}" for a, v, _ in (existing or [])) or "(none yet)"
    sys = (
        "You maintain a persistent self-model of a user. From the message, extract "
        "ONLY durable, identity-level facts — relationships, values, roles, ongoing "
        "struggles, stable preferences — not transient events or moods. Revise "
        "existing facts instead of duplicating them. Respond with strict JSON: "
        '{"facts": [{"attribute": "<stable kebab/colon key>", "value": "<short>", '
        '"confidence": <0-1>}]}. Return an empty list if nothing identity-level is present.'
    )
    try:
        resp = with_retries(lambda: _client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": f"KNOWN: {known}\nMESSAGE: {text}"}],
            temperature=0,
            response_format={"type": "json_object"},
        ))
        facts = json.loads(resp.choices[0].message.content).get("facts", [])
        clean: List[Dict] = []
        for f in facts:
            if f.get("attribute") and f.get("value"):
                clean.append({
                    "attribute": str(f["attribute"])[:64],
                    "value": str(f["value"])[:120],
                    "confidence": max(0.0, min(1.0, float(f.get("confidence", 0.5)))),
                })
        return clean
    except Exception:
        return _heuristic_identity(text)


# --------------------------------------------------------------------------- #
# L7 — reconsolidation (a re-remembered memory is rewritten)
# --------------------------------------------------------------------------- #
def reconsolidate(original: str, original_tone: str, current_tone: str) -> str:
    """Rewrite a memory's meaning when it is re-experienced from a changed state.

    The event is unchanged; its *abstraction* integrates the new perspective —
    e.g. an episode encoded in distress, revisited from calm, becomes a belief
    about growth rather than a raw wound (§4.3, L7)."""
    if config.MOCK or not config.OPENAI_API_KEY:
        stem = original.strip().rstrip(".")
        return (f"This person has integrated an earlier {original_tone or 'difficult'} "
                f"experience — \"{stem[:120]}\" — and now holds it from a "
                f"{current_tone or 'steadier'} place.")

    sys = (
        "A memory is being recalled from a changed emotional state — this is "
        "reconsolidation: the event stays fixed but its meaning evolves. Given the "
        "ORIGINAL memory (and the tone it was encoded in) and the user's CURRENT "
        "tone, write ONE present-tense belief that integrates the original "
        "experience with the new perspective. Start with 'This person'."
    )
    try:
        resp = with_retries(lambda: _client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": (
                          f"ORIGINAL (tone={original_tone}): {original}\n"
                          f"CURRENT tone: {current_tone}")}],
            temperature=0.4,
        ))
        return resp.choices[0].message.content.strip()
    except Exception:
        stem = original.strip().rstrip(".")
        return f"This person has reframed \"{stem[:120]}\" from a {current_tone} place."
