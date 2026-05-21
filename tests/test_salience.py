"""Tests for the write-time salience gate and tier table (§4.1, §4.3, Fig 5)."""
from bubbles import config
from bubbles.memory import salience
from bubbles.state import LatentState


def test_tier_table_matches_paper():
    # score → (band, weekly decay) from Figure 5
    assert config.tier_for_score(10).weekly_decay == 0.99
    assert config.tier_for_score(9).weekly_decay == 0.99
    assert config.tier_for_score(8).weekly_decay == 0.95
    assert config.tier_for_score(6).weekly_decay == 0.88
    assert config.tier_for_score(4).weekly_decay == 0.75
    assert config.tier_for_score(1).weekly_decay == 0.60


def test_tier_clamps_out_of_range():
    assert config.tier_for_score(0).low == 1
    assert config.tier_for_score(99).high == 10


def test_build_episode_assigns_tier():
    ep = salience.build_episode("user", "my dad died",
                                {"salience": 9.5, "tone": "distress"})
    assert ep.tier_low == 9 and ep.tier_high == 10
    assert ep.weekly_decay == 0.99
    assert ep.vitality == config.INITIAL_VITALITY


def test_gate_is_state_dependent():
    # A borderline episode (salience inside the [2.1, 4.3] modulation band) is
    # stored in a vulnerable state but dropped when calm — the same content,
    # different fate, decided by S.
    vulnerable = LatentState(E=80, K=50, V=80)   # low threshold → captures more
    calm = LatentState(E=0, K=100, V=0)          # high threshold → selective
    assert calm.theta() > vulnerable.theta()
    assert salience.gate(3.0, vulnerable) is True
    assert salience.gate(3.0, calm) is False


def test_small_talk_dropped_in_calm_state():
    calm = LatentState(E=0, K=100, V=0)
    assert salience.gate(1.0, calm) is False
