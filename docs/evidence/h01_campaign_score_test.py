"""Row selection, normalised RSS, and sparsity ranking of the campaign scorer."""

import pytest

from docs.evidence.h01_campaign_score import (cell_verdicts, control_names, member_effects, normalized_rss, rank,
                                              select_rows, tree)


def test_cell_verdicts_pass_only_when_every_row_of_the_group_passes():
    vector = {"a": {"kind": "minimum_voltage_mv", "verdict": "pass"}, "b": {"kind": "minimum_voltage_mv", "verdict": "fail"},
              "c": {"kind": "interval_ms", "verdict": "pass"}, "d": {"kind": "count", "verdict": "unavailable"}}
    assert cell_verdicts(vector) == {"minima": "fail", "intervals": "pass", "count": "unavailable"}


def test_member_effects_average_over_pairs_and_report_the_ratio():
    def vec(minimum, interval):
        return {"s:m1_voltage_mv": {"residual": minimum, "kind": "minimum_voltage_mv", "verdict": "fail"},
                "s:i2_ms": {"residual": interval, "kind": "interval_ms", "verdict": "fail"}}
    cells = {"00": vec(-4., 0.), "10": vec(-2., 10.), "01": vec(-3., 0.), "11": vec(-1., 10.)}
    effects = member_effects(cells, ["A", "B"], lambda k: k.endswith("_voltage_mv"), "s:i2_ms")
    assert effects["A"]["minima_change_mv"] == pytest.approx(2.) and effects["A"]["interval_change_ms"] == pytest.approx(10.)
    assert effects["A"]["ratio_mv_per_ms"] == pytest.approx(.2) and effects["A"]["pairs"] == [("00", "10"), ("01", "11")]
    assert effects["B"]["interval_change_ms"] == pytest.approx(0.) and effects["B"]["ratio_mv_per_ms"] is None


def test_control_names_alias_source_candidate_and_repeats_by_identity():
    manifest = {"common_settings": {"nseg_factor": 9}, "source_flags": {}, "candidate_flags": {"a": 1},
                "candidates": [{"name": "s", "stage": "0", "settings": {"nseg_factor": 9}, "flags": {}},
                               {"name": "c", "stage": "0", "settings": {"nseg_factor": 9}, "flags": {"a": 1}},
                               {"name": "c27", "stage": "0", "settings": {"nseg_factor": 27}, "flags": {"a": 1}},
                               {"name": "x", "stage": "A", "settings": {"nseg_factor": 9}, "flags": {"a": 1}}]}
    assert control_names(manifest) == {"source": "s", "candidate": "c", "c27": "c27"}
    manifest["candidates"].pop(1)
    with pytest.raises(ValueError):
        control_names(manifest)


def _row(residual, verdict="fail", kind="interval_ms"):
    return {"residual": residual, "verdict": verdict, "kind": kind}


def test_select_rows_requires_contrast_beyond_five_limits_and_a_failure():
    source = {"a": _row(10.), "b": _row(1., "pass"), "c": _row(None), "d": _row(5.)}
    candidate = {"a": _row(0.), "b": _row(1.2, "pass"), "c": _row(1.), "d": _row(4.9)}
    limits = {"a": 1., "b": .01, "c": .1, "d": .001}
    assert select_rows(source, candidate, limits) == ["a", "d"]


def test_normalized_rss_divides_by_limits_and_skips_zero_limits():
    base = {"a": _row(0.), "b": _row(0.)}
    swapped = {"a": _row(3.), "b": _row(4.)}
    assert normalized_rss(base, swapped, ["a", "b"], {"a": 1., "b": 1.}) == pytest.approx(5.)
    assert normalized_rss(base, swapped, ["a", "b"], {"a": 1., "b": 0.}) == pytest.approx(3.)
    with pytest.raises(ValueError):
        normalized_rss(base, {"a": _row(None), "b": _row(1.)}, ["a"], {"a": 1.})


def test_rank_names_a_steep_x_only_when_it_dominates_in_quadrature():
    vectors = {"source": {"k": _row(0.)}, "candidate": {"k": _row(10.)},
               "a-into-X": {"k": _row(9.)}, "a-out-X": {"k": _row(1.)},
               "a-into-Y": {"k": _row(1.)}, "a-out-Y": {"k": _row(9.)}}
    ranking = rank(vectors, ["X", "Y"], ["k"], {"k": 1.})
    assert ranking["order"] == ["X", "Y"] and ranking["steep_x"] == "X"
    vectors["a-into-Y"]["k"] = _row(9.)
    vectors["a-out-Y"]["k"] = _row(1.)
    assert rank(vectors, ["X", "Y"], ["k"], {"k": 1.})["steep_x"] is None
    incomplete = rank({**vectors, "a-out-Y": {"k": _row(None)}}, ["X", "Y"], ["k"], {"k": 1.})
    assert incomplete["subsystems"]["Y"]["combined"] is None and incomplete["steep_x"] is None


def test_tree_lists_groups_and_marks_the_steep_x():
    groups = {"minima": {"keys": ["k"], "order": ["X", "Y"], "steep_x": "X",
                         "subsystems": {"X": {"combined": 9.}, "Y": {"combined": None}}},
              "empty": {"keys": [], "order": [], "steep_x": None}}
    text = tree(groups)
    assert "minima_X[X: RSS 9.0; Steep X]" in text and "incomplete" in text and "empty" not in text
