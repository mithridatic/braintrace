"""Youden contract plot: rows are read, pass and fail counted, files written."""

import json

import h01_youden_contract as youden


def _usable(tmp_path, rows):
    path = tmp_path/"usable.json"
    path.write_text(json.dumps({"cell": "X", "model": "toy", "rows": rows}))
    return path


def _row(metric, human, model, verdict, inp="1 nA", spread=None):
    return {"input": inp, "row": metric, "cycle": 1, "human": human, "model": model,
            "limit": .5, "spread": spread, "verdict": verdict}


def test_rows_grouped_by_metric_and_missing_values_dropped(tmp_path):
    path = _usable(tmp_path, [_row("rate_hz", 10., 12., "fail"), _row("width_ms", 1., 1.1, "pass"),
                              {"input": "1 nA", "row": "ahp_mv", "cycle": 1, "human": None,
                               "model": -70., "limit": 2., "spread": None, "verdict": "unresolvable"}])
    _, by_metric = youden.load_rows(path)
    assert [len(by_metric[m]) for m, _ in youden.METRICS] == [1, 0, 1, 0]


def test_draw_cell_counts_and_writes_png_and_svg(tmp_path):
    path = _usable(tmp_path, [_row("rate_hz", 10., 12., "fail"), _row("rate_hz", 20., 20.2, "pass", "2 nA", .3),
                              _row("width_ms", 1., 1.05, "pass")])
    counts = youden.draw_cell("X", path, tmp_path/"out")
    assert counts["rate_hz"] == (1, 2) and counts["width_ms"] == (1, 1) and counts["ahp_mv"] == (0, 0)
    assert (tmp_path/"out.png").exists() and (tmp_path/"out.svg").exists()


def test_main_on_committed_sources_reports_both_cells(tmp_path):
    summary = youden.main(tmp_path)
    assert set(summary) == {"E", "I"}
    assert summary["E"]["rate_hz"][1] == 3 and summary["I"]["rate_hz"][1] == 3
    written = json.loads((tmp_path/"h01-youden-contract.json").read_text())
    assert written["pass_counts"]["I"]["width_ms"][0] < written["pass_counts"]["I"]["width_ms"][1]
