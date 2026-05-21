"""Tests for R(m) scoring and re-ranking (§4.2, Figure 4)."""
from datetime import datetime, timedelta, timezone

from taco import config
from taco.memory import retrieval
from taco.memory.episode import Episode
from taco.state import LatentState


def _ep(**kw) -> Episode:
    base = dict(content="x", salience=5.0, vitality=1.0, tone="neutral",
                similarity=0.5)
    base.update(kw)
    return Episode(**base)


def test_components_in_unit_range():
    s = LatentState(E=50, V=50)
    c = retrieval.components(_ep(), s, retrieval.state_tone(s))
    for k, v in c.items():
        assert 0.0 <= v <= 1.0, k


def test_tone_match_boost():
    # exact tone match scores higher than a mismatch
    match = retrieval.tone_compat("distress", "distress")
    mismatch = retrieval.tone_compat("distress", "transactional")
    assert match >= mismatch + config.EMO_TONE_MATCH_BOOST - 1e-9
    assert match <= 1.0


def test_recency_decays_with_age():
    s = LatentState()
    tone = retrieval.state_tone(s)
    fresh = _ep(created_at=datetime.now(timezone.utc))
    old = _ep(created_at=datetime.now(timezone.utc) - timedelta(days=29))
    c_fresh = retrieval.components(fresh, s, tone)
    c_old = retrieval.components(old, s, tone)
    assert c_fresh["rec"] > c_old["rec"]


def test_rerank_selects_top_k_by_score():
    s = LatentState(E=70, K=50, V=80, R=20)  # vulnerable → w_emo high
    cands = [
        _ep(content="low sim low sal", similarity=0.1, salience=2, tone="neutral"),
        _ep(content="high sim high sal", similarity=0.9, salience=9, tone="distress"),
        _ep(content="mid", similarity=0.5, salience=5, tone="concerned"),
        _ep(content="emo match", similarity=0.3, salience=6, tone="distress"),
        _ep(content="filler", similarity=0.2, salience=1, tone="neutral"),
    ]
    top = retrieval.rerank(cands, s, top_k=config.TOP_K)
    assert len(top) == config.TOP_K
    # scores are sorted descending
    assert all(top[i].score >= top[i + 1].score for i in range(len(top) - 1))
    # the strongest candidate wins
    assert top[0].content == "high sim high sal"


def test_vulnerability_raises_emotional_contribution():
    """Higher V should make a tone-matched memory rank relatively higher."""
    emo_mem = _ep(content="emo", similarity=0.3, salience=5, tone="distress")
    sem_mem = _ep(content="sem", similarity=0.8, salience=5, tone="neutral")

    calm = LatentState(E=60, K=50, V=0, R=50)
    vuln = LatentState(E=60, K=50, V=100, R=50)

    retrieval.rerank([emo_mem, sem_mem], calm)
    gap_calm = sem_mem.score - emo_mem.score
    retrieval.rerank([emo_mem, sem_mem], vuln)
    gap_vuln = sem_mem.score - emo_mem.score
    # emotional memory closes the gap on the semantic one as V rises
    assert gap_vuln < gap_calm
