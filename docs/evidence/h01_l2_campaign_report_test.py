"""Table rendering and the literal Stage B prediction check of the campaign report."""
from docs.evidence import h01_l2_campaign_report as report


def _vectors(into_events=5, out_events=6):
    base = {"i1": -8., "i2": 50., "i3": 0., "i4": 20., "sub_1120_v": .39, "m1_v": 0., "m2_v": 0., "e1_rise_to_peak": 0., "e1_peak_v": 0.}
    return {"source": {**base, "event_count_model": 7, "i2": -90., "i3": -90., "i4": -70., "sub_1120_v": 1.76},
            "candidate": {**base, "event_count_model": 5},
            "ab-into-F3F5": {**base, "event_count_model": into_events, "sub_1120_v": .41},
            "ab-out-F3F5": {**base, "event_count_model": out_events, "i2": -80., "i3": -60., "i4": -40., "sub_1120_v": 1.74}}


def test_prediction_check_holds_only_for_the_stated_pattern():
    holds = report.prediction_check(_vectors(), None, "F3F5", "text")
    assert holds["evaluated"] and holds["holds"] and all(holds["checks"].values())
    broken = report.prediction_check(_vectors(into_events=6), None, "F3F5", "text")
    assert not broken["holds"] and not broken["checks"]["into_has_five_events"]
    assert report.prediction_check({"source": {}, "candidate": {}}, None, "F3F5", "text") == {"evaluated": False}


def test_tables_render_every_available_run_and_group():
    vectors = _vectors()
    table = report.residual_table(vectors, ["source", "candidate", "absent"])
    assert table.count("\n| ") == 3 and "absent" not in table and "missing" in report.fmt(float("nan"))
    groups = {"event_count": {"families": {"F3": {"into_source": -1, "out_of_candidate": 1}}, "source": 7, "candidate": 5},
              "intervals": {"families": {"F3": {"combined": 12.5}}, "keys": ["i1"], "steep_x": "F3"}}
    text = report.group_table(groups, ["F3"])
    assert "-1 / +1" in text and "| intervals | 1 | 12.5 | F3 |" in text


def test_stage_c_names_exceeding_keys_and_contrast_ratio(tmp_path, monkeypatch):
    import json
    vectors = _vectors()
    vectors["c-out-F3F5-atol11"] = {**vectors["ab-out-F3F5"], "i2": -80.05, "e1_rise_to_peak": .001}
    ranking = {"keys": ["i2", "e1_rise_to_peak", "sub_1120_v"], "vectors": vectors, "families": {}, "order": [], "steep_x": None, "groups": {}}
    folder = tmp_path/"runs"; folder.mkdir()
    (folder/"stage-a-ranking.json").write_text(json.dumps(ranking))
    manifest = {"output_dir": "runs", "families": [], "candidates": []}
    (tmp_path/"m.json").write_text(json.dumps(manifest))
    for name in ("h01-l2-density130-tolerance-result.json", "h01-l2-density130-spatial-result.json",
                 "h01-l2-density130-minima-numerical-review.json"):
        (tmp_path/name).write_text("{}")
    monkeypatch.setattr(report, "FOLDER", tmp_path)
    monkeypatch.setattr(report.score, "numerical_limits", lambda *a: {"fields": {"rise_crossing_ms": .0357, "peak_sample_voltage_mv": .01, "time_above_threshold_ms": .0004}, "phase_ms": .0004, "minimum_voltage_mv": .0013})
    monkeypatch.setattr(report.score, "load_datums", lambda: {})
    import sys
    monkeypatch.setattr(sys, "argv", ["report", "--manifest", str(tmp_path/"m.json")])
    report.main()
    out = json.loads((folder/"stage-b-prediction-check.json").read_text())["stage_c"]["out"]
    assert not out["within_limits"] and set(out["exceeding"]) == {"i2", "e1_rise_to_peak"}
    assert out["exceeding"]["i2"]["claimed_contrast"] == -130. and out["max_change_over_claimed_contrast"] < .01
