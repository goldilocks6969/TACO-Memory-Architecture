"""Tests for the tier-based decay schedule (§4.4, Figure 6).

These validate the decay *math* (vitality = weekly_decay ** weeks) and the
abstraction threshold without touching the database.
"""
from taco import config
from taco.memory import decay


def vitality(weekly_decay: float, weeks: float) -> float:
    return decay.vitality_after_weeks(weekly_decay, weeks)


def test_high_salience_is_near_permanent():
    # 9–10 tier (×0.99/week) stays well above the abstraction threshold for a year.
    rate = config.tier_for_score(10).weekly_decay
    assert vitality(rate, 52) > config.ABSTRACTION_THRESHOLD
    assert vitality(rate, 52) > 0.5  # Figure 6: still ~0.6 at 52 weeks


def test_small_talk_is_pruned_quickly():
    # 1–2 tier (×0.60/week) crosses the abstraction threshold within weeks.
    rate = config.tier_for_score(1).weekly_decay
    weeks_to_threshold = next(
        w for w in range(1, 53) if vitality(rate, w) < config.ABSTRACTION_THRESHOLD
    )
    assert weeks_to_threshold <= 4


def test_decay_is_monotonic_and_ordered_by_tier():
    rates = [config.tier_for_score(s).weekly_decay for s in (10, 8, 6, 4, 1)]
    # higher tiers decay slower
    assert rates == sorted(rates, reverse=True)
    # each curve is monotonically decreasing in time
    for rate in rates:
        assert vitality(rate, 5) > vitality(rate, 10)


def test_abstraction_threshold_value():
    assert config.ABSTRACTION_THRESHOLD == 0.15


def test_crosses_abstraction_threshold_helper():
    rate = config.tier_for_score(3).weekly_decay
    assert decay.crosses_abstraction_threshold(rate, 1) is False
    assert decay.crosses_abstraction_threshold(rate, 10) is True
