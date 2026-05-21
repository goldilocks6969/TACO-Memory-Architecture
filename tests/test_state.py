"""Tests for the latent state and its modulation curves (§3.1, §4)."""
import math

from taco import config
from taco.state import ContentSignal, LatentState, infer_state, recency_score


def test_theta_endpoints():
    # Calm, engaged, transactional → maximally selective threshold.
    calm = LatentState(E=0, K=100, V=0, R=100)
    assert calm.theta() == config.THETA_MAX
    # Peak emotional intensity → most permissive threshold.
    crisis = LatentState(E=100, K=100, V=100, R=100)
    assert crisis.theta() == config.THETA_MIN
    # Threshold always stays within the paper's stated band.
    for e in range(0, 101, 10):
        for v in range(0, 101, 50):
            t = LatentState(E=e, K=50, V=v).theta()
            assert config.THETA_MIN <= t <= config.THETA_MAX


def test_theta_monotonic_in_intensity():
    prev = LatentState(E=0, K=50, V=0).theta()
    for e in range(10, 101, 10):
        cur = LatentState(E=e, K=50, V=0).theta()
        assert cur <= prev  # rising intensity lowers the threshold
        prev = cur


def test_w_emo_endpoints():
    assert math.isclose(LatentState(V=0).w_emo(), config.W_EMO_MIN)
    assert math.isclose(LatentState(V=100).w_emo(), config.W_EMO_MAX)
    # rises monotonically with vulnerability
    assert LatentState(V=30).w_emo() < LatentState(V=70).w_emo()


def test_retrieval_weights_normalised():
    for v in (0, 50, 100):
        w = LatentState(E=80, K=50, V=v, R=10).retrieval_weights()
        assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
        assert set(w) == set(config.DEFAULT_WEIGHTS)


def test_planning_depth_scales_with_engagement_and_trust():
    new_user = LatentState(E=0, K=0, V=0)
    deep = LatentState(E=0, K=100, V=100)
    assert new_user.planning_depth() == 1
    assert deep.planning_depth() == 4
    assert new_user.planning_depth() < deep.planning_depth()


def test_interruption_policy_bands():
    assert LatentState(E=0, V=0).interruption_policy() == "hold"
    assert LatentState(E=90, V=90).interruption_policy() == "within_hour"


def test_reinforcement_rate_higher_in_meaningful_states():
    routine = LatentState(K=10, V=10).reinforcement_rate()
    meaningful = LatentState(K=90, V=90).reinforcement_rate()
    assert meaningful > routine


def test_recency_halflife():
    assert recency_score(0) == 100.0
    assert math.isclose(recency_score(config.RECENCY_HALFLIFE_HOURS), 50.0, rel_tol=1e-6)


def test_neutral_content_decays_emotional_intensity():
    s = LatentState(E=80, K=50, V=40, R=100)
    neutral = ContentSignal(emotional=0, vulnerability=0)
    s2 = infer_state(s, neutral, hours_since_last=1.0)
    assert s2.E < s.E  # arousal relaxes toward the neutral signal


def test_engagement_streak_erodes_over_long_gap():
    s = LatentState(E=10, K=80, V=10, R=100)
    away = ContentSignal(emotional=10, vulnerability=10)
    s2 = infer_state(s, away, hours_since_last=24 * 30)  # ~a month away
    assert s2.K < s.K
