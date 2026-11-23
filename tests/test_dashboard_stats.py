import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.dashboard.stats import wilson_interval  # noqa: E402


def test_wilson_interval_none_for_zero_n():
    assert wilson_interval(0, 0) is None


def test_wilson_interval_bounds_within_0_and_1():
    low, high = wilson_interval(3, 10)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_narrows_with_more_data_at_same_rate():
    low_small, high_small = wilson_interval(3, 10)
    low_large, high_large = wilson_interval(30, 100)
    assert (high_large - low_large) < (high_small - low_small)


def test_wilson_interval_all_successes_does_not_exceed_one():
    low, high = wilson_interval(10, 10)
    assert high <= 1.0
    assert low > 0.0


def test_wilson_interval_rejects_successes_greater_than_n():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
