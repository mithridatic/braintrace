"""Tests for the keep/drop chain generator."""

import json
import sys

from docs.evidence import h01_keep_drop_chain as chain

L2, L4, PV, SST = ("l2-pyramidal-allen-541563728", "l4-pyramidal-allen-527952884",
                   "l5-pv-basket-hl5bn1", "l3-sst-interneuron-hl5mn1")


def test_protocol_matches_the_spec():
    p = chain.PROTOCOL
    assert (p[L2]["current_na"], p[L4]["current_na"], p[PV]["current_na"], p[SST]["current_na"]) == (.31, .09, .19, .10)
    assert (p[L2]["registered_count"], p[L4]["registered_count"], p[PV]["registered_count"],
            p[SST]["registered_count"]) == (10, 12, 12, 14)
    assert (p[L2]["donor_model_count"], p[L4]["donor_model_count"], p[PV]["donor_model_count"],
            p[SST]["donor_model_count"]) == (10, 8, 14, 16)
    assert p[L2]["duration_ms"] == p[L4]["duration_ms"] == 2300 and p[L2]["pulse_on_ms"] == 1020
    assert p[PV]["duration_ms"] == p[SST]["duration_ms"] == 1500 and p[PV]["pulse_on_ms"] == 270
    assert p[SST]["repeat_counts"] == (14, 14, 13, 12) and all(p[k]["repeat_counts"] is None for k in (L2, L4, PV))
    assert all(p[k]["pulse_ms"] == 1000 for k in p)


def test_primary_command_for_an_e_cell():
    line = chain.command("955432427", L2)
    assert line.startswith("$R transfer-all-955432427 1800 -- --cell 955432427 --donor " + L2 + " --polarity E")
    assert "--pulse-on-ms 1020 --pulse-ms 1000 --duration-ms 2300" in line
    assert "--current-na 0.31 --registered-count 10 --donor-model-count 10" in line
    assert "--dt-ms" not in line and "--repeat-counts" not in line


def test_dthalf_command_label_cap_and_dt():
    line = chain.command("3761379470", L4, dt_half=True)
    assert line.startswith("$R transfer-all-3761379470-dthalf 3600 --")
    assert line.endswith("--donor-model-count 8 --dt-ms 0.0025")


def test_sst_command_carries_repeat_counts():
    line = chain.command("4420044370", SST)
    assert "--registered-count 14 --repeat-counts 14 14 13 12 --donor-model-count 16" in line
    assert "--polarity I --pulse-on-ms 270 --pulse-ms 1000 --duration-ms 1500 --current-na 0.1 " in line


def test_chains_deal_sorted_cells_round_robin():
    rows = [(str(1000+i*7), L2 if i % 2 else PV) for i in range(104)]
    dealt = chain.chains(rows, 4)
    assert [len(c) for c in dealt] == [26, 26, 26, 26]
    flat = [cell for c in dealt for cell, _ in c]
    assert len(set(flat)) == 104
    ordered = sorted(rows, key=lambda pair: pair[0])
    assert dealt[0][0] == ordered[0] and dealt[1][0] == ordered[1] and dealt[0][1] == ordered[4]


def test_write_chains_primary_and_dthalf(tmp_path):
    rows = [("a1", L2), ("a2", L4), ("a3", PV), ("a4", SST), ("a5", L2)]
    paths = chain.write_chains(rows, tmp_path, "primary", 4)
    assert [p.name for p in paths] == [f"keep-chain-{k}.sh" for k in (1, 2, 3, 4)]
    text = paths[0].read_bytes().decode()
    assert "\r" not in text
    lines = text.splitlines()
    assert lines[:5] == ["#!/usr/bin/env bash", f"# Keep/drop primary chain 1 of 4; rule and protocol in {chain.SPEC}.",
                         "cd /workspace/braintrace", f"R={chain.RUNNER}", "export CPUSET=0-31 MEMFRAC=.2"]
    assert len(lines) == 5+2+1 and lines[-1] == "echo CHAIN-DONE > var/h01-driven/keep-chain-1.done"
    assert paths[3].read_text().splitlines()[4] == "export CPUSET=96-127 MEMFRAC=.2"
    assert len(paths[1].read_text().splitlines()) == 5+1+1

    decision = tmp_path/"decision.json"
    decision.write_text(json.dumps({"candidates": ["a2", "a4"]}))
    types = tmp_path/"types.json"
    types.write_text(json.dumps({"rows": [{"cell_id": c, "donor_key": d} for c, d in rows]}))
    filtered = chain.rows_from_types(types, json.loads(decision.read_text())["candidates"])
    assert filtered == [("a2", L4), ("a4", SST)]
    half = chain.write_chains(filtered, tmp_path/"half", "dthalf", 4)
    assert [p.name for p in half] == [f"keep-chain-{k}-dthalf.sh" for k in (1, 2, 3, 4)]
    first = half[0].read_text().splitlines()
    assert first[5].startswith("$R transfer-all-a2-dthalf 3600 --") and first[5].endswith("--dt-ms 0.0025")
    assert first[-1] == "echo CHAIN-DONE > var/h01-driven/keep-chain-1-dthalf.done"
    assert len(half[2].read_text().splitlines()) == 5+0+1


def test_currents_for_maps_each_cell_to_its_donor_primary_input():
    rows = [("a", "l2-pyramidal-allen-541563728"), ("b", "l4-pyramidal-allen-527952884"),
            ("c", "l5-pv-basket-hl5bn1"), ("d", "l3-sst-interneuron-hl5mn1")]
    assert chain.currents_for(rows) == {"a": .31, "b": .09, "c": .19, "d": .10}


def test_currents_phase_writes_kept_cells_only(tmp_path, monkeypatch):
    types = tmp_path/"types.json"
    types.write_text(json.dumps(dict(rows=[dict(cell_id="a", donor_key="l2-pyramidal-allen-541563728"),
                                          dict(cell_id="b", donor_key="l4-pyramidal-allen-527952884")])))
    decision = tmp_path/"decision.json"
    decision.write_text(json.dumps(dict(kept=["b"])))
    out = tmp_path/"currents.json"
    monkeypatch.setattr(sys, "argv", ["x", "--types", str(types), "--phase", "currents",
                                      "--candidates", str(decision), "--currents-out", str(out)])
    chain.main()
    assert json.loads(out.read_text()) == {"b": .09}


def test_priority_order_and_skip_started(tmp_path):
    types = tmp_path/"types.json"
    types.write_text(json.dumps(dict(rows=[dict(cell_id="9", donor_key=L2), dict(cell_id="5", donor_key=L4),
                                          dict(cell_id="7", donor_key=SST), dict(cell_id="1", donor_key=PV),
                                          dict(cell_id="3", donor_key=L4)])))
    (tmp_path/"transfer-all-3").mkdir()
    (tmp_path/"transfer-all-3"/"launch.json").write_text("{}")
    rows = chain.rows_from_types(types, skip_done=tmp_path, priority=True)
    assert rows == [("5", L4), ("1", PV), ("7", SST), ("9", L2)]
    assert chain.chains(rows, 2, keep_order=True) == [[("5", L4), ("7", SST)], [("1", PV), ("9", L2)]]
    assert chain.chains(rows, 2) == [[("1", PV), ("7", SST)], [("5", L4), ("9", L2)]]


def test_ramp_command_drives_three_times_the_donor_input_without_a_count():
    line = chain.command("42", "l5-pv-basket-hl5bn1", ramp=True)
    assert line.startswith("$R transfer-all-42-ramp 1800 -- --cell 42 --donor l5-pv-basket-hl5bn1 --polarity I")
    assert "--ramp-na 0.57" in line and "--current-na" not in line and "--registered-count" not in line
    assert "--dt-ms" not in line and "--donor-model-count" not in line
    sst = chain.command("7", "l3-sst-interneuron-hl5mn1", ramp=True, extra="--archive c3.zip --tags interneuron")
    assert "--ramp-na 0.3" in sst and "--repeat-counts" not in sst and sst.endswith("--archive c3.zip --tags interneuron")


def test_write_chains_ramp_phase_and_workdir(tmp_path):
    rows = [("1", "l4-pyramidal-allen-527952884"), ("2", "l2-pyramidal-allen-541563728")]
    paths = chain.write_chains(rows, tmp_path, "ramp", n=1, extra="--components c3.json", workdir="/workspace/braintrace-c3")
    text = paths[0].read_text()
    assert paths[0].name == "keep-chain-1-ramp.sh" and "cd /workspace/braintrace-c3" in text
    assert "transfer-all-1-ramp 1800 --" in text and "--ramp-na 0.27" in text and "--ramp-na 0.93" in text
    assert text.count("--components c3.json") == 2 and "keep-chain-1-ramp.done" in text
