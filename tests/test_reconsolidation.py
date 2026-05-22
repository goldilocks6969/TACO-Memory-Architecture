"""Tests for L7 reconsolidation candidacy + emotional-charge attenuation."""
from datetime import datetime, timedelta, timezone

from taco import config
from taco.memory.reconsolidation import attenuated_intensity, is_candidate


def test_hot_memory_revisited_from_calm_is_a_candidate():
    # encoded at E=80, now at E=20 → big drop, never reconsolidated.
    assert is_candidate(encoded_e=80.0, current_e=20.0, reconsolidated_at=None)


def test_coolly_encoded_memory_is_not_a_candidate():
    assert not is_candidate(encoded_e=30.0, current_e=5.0, reconsolidated_at=None)


def test_small_emotional_drop_is_not_a_candidate():
    # still hot now → nothing to integrate.
    assert not is_candidate(encoded_e=80.0, current_e=70.0, reconsolidated_at=None)


def test_missing_encoding_snapshot_is_not_a_candidate():
    assert not is_candidate(encoded_e=None, current_e=10.0, reconsolidated_at=None)


def test_cooldown_blocks_then_allows():
    now = datetime.now(timezone.utc)
    recent = now - timedelta(days=config.RECON_COOLDOWN_DAYS / 2)
    old = now - timedelta(days=config.RECON_COOLDOWN_DAYS + 1)
    assert not is_candidate(80.0, 20.0, recent, now=now)   # too soon to re-write
    assert is_candidate(80.0, 20.0, old, now=now)          # off cooldown


def test_attenuation_reduces_charge():
    assert attenuated_intensity(80.0) == 80.0 * config.RECON_ATTENUATION
    assert attenuated_intensity(80.0) < 80.0
