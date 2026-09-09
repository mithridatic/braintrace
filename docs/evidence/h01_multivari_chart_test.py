"""Multivari chart: nesting order, family ranges, and the files it writes."""

import json

import h01_multivari_chart as chart


def _row(machine, inp, repeat, cycle, width):
    return {"cell": "I", "machine": machine, "input": inp, "repeat": repeat, "cycle": cycle,
            "width_ms": width, "ahp_mv": -70.+cycle, "cycle_ms": 10.*cycle, "rise_v_per_s": 300.}


ROWS = [_row("human", "0.19 nA", None, 1, .28), _row("human", "0.19 nA", None, 2, .29),
        _row("human", "repeat", 40, 1, .27), _row("human", "repeat", 41, 1, .30),
        _row("model", "0.19 nA", None, 1, .22), _row("model", "0.19 nA", None, 2, .22)]


def test_input_key_orders_drives_numerically_and_repeats_last():
    assert chart.input_key("0.19 nA") < chart.input_key("250 pA") < chart.input_key("repeat")


def test_columns_put_human_before_model_and_separate_machines_with_a_heavy_rule():
    x, dividers, *_ = chart.columns(ROWS)
    assert x[("human", "0.19 nA", None, 1)] < x[("model", "0.19 nA", None, 1)]
    assert any(weight == 2. for _, weight in dividers)


def test_family_ranges_name_the_machine_split_when_the_model_is_biased():
    ranges = chart.family_ranges(ROWS, "width_ms")
    assert max(ranges, key=ranges.get) == "machine (human vs model)"


def test_main_writes_figure_and_answers(tmp_path):
    data = tmp_path/"data.json"
    data.write_text(json.dumps({"rows": ROWS, "missing": [{"cell": "I", "input": "0.23 nA"}]}))
    answers = chart.main(data, tmp_path)
    assert (tmp_path/"h01-multivari-i.png").exists() and "E" not in answers
    record = json.loads((tmp_path/"h01-multivari-chart.json").read_text())
    assert record["missing_model_traces"] and record["answers"]["I"]["width_ms"]["varies_most"]
