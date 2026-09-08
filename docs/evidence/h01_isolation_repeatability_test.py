"""Tests for the isolation decision limits and contrast tiers."""

import pytest

from docs.evidence.h01_isolation_repeatability import (
    branch_summary, combined_limit, contrast_rows, first_spike_vector, positions,
    repeat_limits, subthreshold_contrast, tier, tree)


def _row(spike, **values):
    base = {"spike": spike, "threshold_ms": 10.+spike, "threshold_mv": -60., "peak_mv": 20.,
            "max_rise_v_s": 600., "max_fall_v_s": -300., "minimum_mv": -78., "cycle_ms": 10.}
    base.update(values)
    return base


def test_first_spike_vector_carries_latency_and_handles_empty_trains():
    vector = first_spike_vector([_row(1)], 5.)
    assert vector["latency_ms"] == pytest.approx(6.)
    assert "cycle_ms" not in vector
    assert first_spike_vector([], 5.) is None
    assert first_spike_vector([_row(1, threshold_ms=None)], 5.)["latency_ms"] is None


@pytest.mark.parametrize("count, factor", [(2, 2.95), (3, 1.81), (4, 1.47), (7, 1.47)])
def test_repeat_limits_apply_the_dixon_factor_with_a_floor(count, factor):
    vectors = [{"peak_mv": 20.+i} for i in range(count)]
    limits = repeat_limits(vectors)
    assert limits["factor"] == factor
    assert limits["limits"]["peak_mv"] == pytest.approx((count-1)*factor)


def test_repeat_limits_skip_none_and_need_two_repeats():
    limits = repeat_limits([{"peak_mv": 1., "latency_ms": None}, None, {"peak_mv": 2., "latency_ms": 3.}])
    assert limits["repeats"] == 2 and "latency_ms" not in limits["limits"]
    with pytest.raises(ValueError):
        repeat_limits([{"peak_mv": 1.}])


def test_positions_take_first_and_last_three_without_overlap():
    assert positions(12) == [("early", 0), ("early", 1), ("early", 2), ("late", 9), ("late", 10), ("late", 11)]
    assert positions(4) == [("early", 0), ("early", 1), ("early", 2), ("late", 3)]
    assert positions(2) == [("early", 0), ("early", 1)]


def test_combined_limit_is_root_sum_square_and_names_its_branches():
    assert combined_limit("a", {"a": 3.}, {"a": 4.}) == (pytest.approx(5.), "human and model")
    assert combined_limit("a", {}, {"a": 4.}) == (pytest.approx(4.), "model")
    assert combined_limit("a", {}, {}) == (None, "none")


def test_subthreshold_contrast_rows_carry_their_limit_source():
    rows = subthreshold_contrast({"plateau_mv": -60., "return_mv": -80.}, {"plateau_mv": -58., "return_mv": -78.5},
                                 {"plateau_mv": .5}, {"plateau_mv": .1, "return_mv": .2})
    assert rows[0]["limit_source"] == "human and model" and rows[0]["tier"] == "real"
    assert rows[1]["limit_source"] == "model" and rows[1]["tier"] == "search"


def test_tier_thresholds():
    assert tier(None) == "unlimited" and tier(6.) == "search"
    assert tier(2.) == "real" and tier(.5) == "parked"


def test_contrast_rows_align_late_spikes_from_the_end_and_tier_them():
    human = [_row(i) for i in range(1, 13)]
    model = [_row(i, peak_mv=45.) for i in range(1, 12)]
    limits = {"peak_mv": 2., "cycle_ms": 1.}
    rows = contrast_rows(human, model, limits, {})
    late = [r for r in rows if r["phase"] == "late" and r["key"] == "peak_mv"]
    assert [r["spike"] for r in late] == [10, 11, 12]
    assert all(r["tier"] == "search" and r["contrast"] == pytest.approx(25.) for r in late)
    summary = branch_summary(rows)
    assert "peak_mv" in summary["search_landmarks"]
    assert "cycle_ms" in summary["parked_landmarks"]
    assert "threshold_mv" not in summary["parked_landmarks"]
    assert "parked, inside repeatability" in tree("I", summary)
