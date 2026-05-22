"""Tests for L9 predictive continuity: state extrapolation + theme detection."""
from taco import config
from taco.memory.episode import Episode
from taco.state import LatentState
from taco.subsystems import prediction


def _hist(es, vs=None):
    vs = vs or [0] * len(es)
    return [LatentState(E=e, K=50, V=v, R=80) for e, v in zip(es, vs)]


def test_rising_trajectory_projects_higher_and_escalates():
    p = prediction.predict(_hist([10, 40, 70]), [])
    assert p.projected.E > 70           # trend carries forward
    assert p.trend == "escalating"


def test_falling_trajectory_recovers():
    p = prediction.predict(_hist([80, 50, 20]), [])
    assert p.projected.E < 20
    assert p.trend == "recovering"


def test_flat_trajectory_is_stable_with_no_prefetch():
    p = prediction.predict(_hist([40, 40, 40]), [])
    assert p.trend == "stable"
    assert p.prefetch_query is None


def test_projection_is_bounded():
    p = prediction.project_state(_hist([60, 80, 100]))   # would overshoot 100
    assert 0.0 <= p.E <= config.STATE_MAX


def test_single_state_history_has_no_trend():
    p = prediction.project_state([LatentState(E=55, K=10, V=20, R=90)])
    assert p.E == 55 and p.V == 20


def test_escalating_trajectory_sets_prefetch_from_hottest_memory():
    hot = Episode(content="my dad is in the ICU", tone="distress", salience=9.0, s_e=88.0)
    cool = Episode(content="changed my coffee order", tone="neutral", salience=6.0, s_e=5.0)
    p = prediction.predict(_hist([10, 40, 70]), [cool, hot])
    assert p.prefetch_query == "my dad is in the ICU"


def test_emergent_themes_need_recurrence_and_exclude_neutral():
    eps = [
        Episode(content="panic before the review", tone="distress", salience=8.0),
        Episode(content="couldn't sleep again", tone="distress", salience=9.0),
        Episode(content="what time is the post office open", tone="neutral", salience=6.0),
    ]
    themes = prediction.emergent_themes(eps)
    assert any("distress thread" in t for t in themes)
    assert not any("neutral" in t for t in themes)

    # a tone that appears only once is not yet "emergent"
    assert prediction.emergent_themes([eps[0], eps[2]]) == []
