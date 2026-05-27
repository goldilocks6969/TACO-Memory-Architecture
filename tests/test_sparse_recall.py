"""Adaptive sparse recall: query intent → memory budget → compact payload."""
from taco import config
from taco.memory import retrieval
from taco.memory.episode import Episode
from taco.state import LatentState


def _ep(content, **kw):
    base = dict(salience=5.0, vitality=1.0, tone="neutral", similarity=0.5,
                score=0.5)
    base.update(kw)
    return Episode(content=content, **base)


def test_query_classification_examples():
    assert retrieval.classify_query("what medication did my doctor put me on?") == "factual"
    assert retrieval.classify_query(
        "I'm at a birthday party feeling alienated. why?") == "emotional"
    assert retrieval.classify_query("am I making any progress health-wise?") == "coherence"
    assert retrieval.classify_query(
        "what did I ask you to recommend in our first chat?") == "filler"


def test_filler_retrieves_nothing_by_default(monkeypatch):
    monkeypatch.setattr(config, "RECALL_POLICY", "adaptive")
    monkeypatch.setattr(config, "RECALL_FILLER_K", 0)
    cands = [_ep("User's father died.", salience=10, score=0.9)]
    assert retrieval.select_sparse_recall(cands, LatentState(), "filler") == []


def test_factual_budget_caps_to_two(monkeypatch):
    monkeypatch.setattr(config, "RECALL_POLICY", "adaptive")
    monkeypatch.setattr(config, "RECALL_FACTUAL_K", 2)
    cands = [
        _ep("metformin medication", score=1.0),
        _ep("type diabetes diagnosis", score=0.9),
        _ep("halcyon job outcome", score=0.8),
        _ep("sam relationship", score=0.7),
    ]
    assert len(retrieval.select_sparse_recall(cands, LatentState(), "factual")) == 2


def test_factual_prefers_direct_query_overlap(monkeypatch):
    monkeypatch.setattr(config, "RECALL_POLICY", "adaptive")
    monkeypatch.setattr(config, "RECALL_FACTUAL_K", 1)
    progress = _ep("The user's A1C dropped at a recheck.", salience=8, score=0.7)
    meds = _ep("The doctor said diet and metformin.", salience=7, score=0.65)
    top = retrieval.select_sparse_recall(
        [progress, meds], LatentState(), "factual",
        "what medication did my doctor put me on?")
    assert top[0].content == "The doctor said diet and metformin."


def test_emotional_prefers_salient_tone_match(monkeypatch):
    monkeypatch.setattr(config, "RECALL_POLICY", "adaptive")
    monkeypatch.setattr(config, "RECALL_EMOTIONAL_K", 1)
    state = LatentState(E=80, V=70)
    salient = _ep("father died", salience=10, tone="distress", score=0.50)
    filler = _ep("post office hours", salience=1, tone="neutral", score=0.55)
    top = retrieval.select_sparse_recall([filler, salient], state, "emotional")
    assert top[0].content == "father died"


def test_coherence_keeps_outcome_and_origin(monkeypatch):
    monkeypatch.setattr(config, "RECALL_POLICY", "adaptive")
    monkeypatch.setattr(config, "RECALL_COHERENCE_K", 3)
    origin = _ep("User was laid off from Nimbus.", salience=8, score=0.6)
    outcome = _ep("User got hired at Halcyon.", salience=8, score=0.59)
    outcome.event_type = "achievement"
    filler = _ep("User asked about Australia.", salience=1, score=0.58)
    top = retrieval.select_sparse_recall([origin, outcome, filler],
                                         LatentState(), "coherence")
    contents = [m.content for m in top]
    assert "User got hired at Halcyon." in contents
    assert "User was laid off from Nimbus." in contents


def test_redundancy_collapse_prefers_short_anchor():
    raw = _ep("he always called me 'kiddo'", salience=7, score=0.8)
    anchor = _ep("The user was called 'kiddo' by someone important to them.",
                 salience=7, score=0.7)
    anchor.fact_type = "relationship"
    out = retrieval.collapse_redundant([raw, anchor])
    assert len(out) == 1
    assert out[0].content == anchor.content


def test_compact_retrieval_rendering_is_token_capped(monkeypatch):
    monkeypatch.setattr(config, "RECALL_MAX_TOKENS", 24)
    memories = [
        _ep("The user was called kiddo by their father.", salience=7, score=0.9),
        _ep("The user's father died of a heart attack.", salience=10, score=0.8),
    ]
    sections = retrieval.briefing_sections(
        memories, [], [], LatentState(), None, mode="compact")
    assert "salience" not in sections["retrieval"]
    assert retrieval._count_tokens(sections["retrieval"]) <= 24
