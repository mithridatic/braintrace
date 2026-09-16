"""Tests for the C3 test-to-failure chain writer."""

import json

from docs.evidence import h01_c3_keep_chain as chain
from docs.evidence.h01_keep_drop_chain import CAP_S

L2, L4, PV, SST = ("l2-pyramidal-allen-541563728", "l4-pyramidal-allen-527952884", "l5-pv-basket-hl5bn1",
                   "l3-sst-interneuron-hl5mn1")
EXTRA = chain.candidate_extra("a.zip", "a"*64, "comp.json", "table.json", "t"*64)
CANDIDATES = [("21", PV), ("22", SST), ("23", PV), ("11", L4), ("12", L2)]
KEPT = [("7", L2), ("8", L4), ("9", L2)]


def test_candidate_extra_names_the_archive_inventory_and_tag_table():
    assert EXTRA == ("--archive a.zip --archive-sha256 "+"a"*64+" --components comp.json --cell-table table.json "
                     "--cell-table-sha256 "+"t"*64)
    assert chain.candidate_extra("a.zip", "a"*64, "comp.json", "table.json").endswith("--cell-table table.json")


def test_jobs_pair_primary_and_ramp_per_candidate_and_ramp_only_per_kept_cell():
    job_list = chain.jobs(CANDIDATES, KEPT, EXTRA)
    assert [(j[0], j[2], len(j[3])) for j in job_list] == [("21", "candidate", 2), ("22", "candidate", 2),
                                                           ("23", "candidate", 2), ("11", "candidate", 2),
                                                           ("12", "candidate", 2), ("7", "kept", 1), ("8", "kept", 1),
                                                           ("9", "kept", 1)]
    primary, ramp = job_list[0][3]
    assert primary.startswith(f"$R transfer-all-21 {CAP_S['primary']} -- --cell 21 --donor {PV} --polarity I")
    assert "--current-na 0.19 --registered-count 12" in primary and primary.endswith(EXTRA)
    assert ramp.startswith(f"$R transfer-all-21-ramp {CAP_S['ramp']} -- --cell 21 --donor {PV} --polarity I")
    assert "--ramp-na 0.57" in ramp and "--current-na" not in ramp and ramp.endswith(EXTRA)
    kept_ramp = job_list[5][3][0]
    assert "--ramp-na 0.93" in kept_ramp and "--archive" not in kept_ramp and "--cell-table" not in kept_ramp
    assert "--ramp-na 0.27" in job_list[6][3][0] and "--ramp-na 0.3 " in job_list[1][3][1]


def test_jobs_skip_runs_already_launched(tmp_path):
    for label in ("transfer-all-21", "transfer-all-22", "transfer-all-22-ramp", "transfer-all-7-ramp"):
        (tmp_path/label).mkdir()
        (tmp_path/label/"launch.json").write_text("{}")
    job_list = chain.jobs(CANDIDATES, KEPT, EXTRA, skip_done=tmp_path)
    assert [j[0] for j in job_list] == ["21", "23", "11", "12", "8", "9"]
    assert len(job_list[0][3]) == 1 and "--ramp-na" in job_list[0][3][0]   # 21: primary launched, ramp left


def test_deal_keeps_relative_order_and_puts_both_runs_of_a_cell_in_one_chain():
    job_list = chain.jobs(CANDIDATES, KEPT, EXTRA)
    one, two = chain.deal(job_list, 2)
    assert [j[0] for j in one] == ["21", "23", "12", "8"] and [j[0] for j in two] == ["22", "11", "7", "9"]
    assert sum(len(j[3]) for j in one) == 7 and sum(len(j[3]) for j in two) == 6


def test_write_chains_scripts_header_lines_and_done_file(tmp_path):
    job_list = chain.jobs(CANDIDATES, KEPT, EXTRA)
    paths = chain.write_chains(job_list, tmp_path/"chains", 2, "/workspace/braintrace-c3-keep",
                               "docs/evidence/h01-c3-keep-drop/run_transfer.sh", "var/c3-keep")
    assert [p.name for p in paths] == ["keep-chain-1.sh", "keep-chain-2.sh"]
    text = paths[0].read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/usr/bin/env bash" and "7 runs on 4 cells; receipts under var/c3-keep/" in lines[2]
    assert lines[3] == "cd /workspace/braintrace-c3-keep" and lines[4] == "R=docs/evidence/h01-c3-keep-drop/run_transfer.sh"
    assert lines[5] == "export CPUSET=0-31 MEMFRAC=.2"
    assert lines[6].startswith("$R transfer-all-21 ") and lines[7].startswith("$R transfer-all-21-ramp ")
    assert lines[-1] == "echo CHAIN-DONE > var/c3-keep/keep-chain-1.done"
    assert "\r" not in text and paths[1].read_text().splitlines()[5] == "export CPUSET=32-63 MEMFRAC=.2"
    assert "/workspace/braintrace/" not in text and "var/h01-driven" not in text


def test_dry_run_report_counts_runs_and_quotes_only_the_earlier_campaign_number():
    report = chain.dry_run_report(chain.jobs(CANDIDATES, KEPT, EXTRA), 2)
    assert report["runs_total"] == 13 and report["cells_total"] == 8
    assert report["runs_by_kind"] == dict(candidate_primary=5, candidate_ramp=5, kept_ramp=3)
    assert report["seconds_per_run_estimate"] == 190. and report["wall_estimate_s"] == 13*190.
    assert "estimate" in report["estimate_basis"] and "launch-bound" in report["estimate_basis"]
    assert [c["runs"] for c in report["chains"]] == [7, 6] and len(report["chains"][0]["first_commands"]) == 3
    assert report["chains"][0]["first_commands"][0].startswith("$R transfer-all-21 ")


def test_main_dry_run_on_the_registered_files_writes_no_chain(tmp_path, capsys):
    report = tmp_path/"report.json"
    chain.main(["--dry-run", "--report", str(report), "--out-dir", str(tmp_path/"chains")])
    assert not (tmp_path/"chains").exists()
    written = json.loads(report.read_text())
    assert written["runs_total"] == 97 and written["cells_total"] == 57
    assert written["runs_by_kind"] == dict(candidate_primary=40, candidate_ramp=40, kept_ramp=17)
    assert [c["runs"] for c in written["chains"]] == [49, 48]
    first = written["chains"][0]["first_commands"][0]
    assert first.startswith("$R transfer-all-3504577656 1800 -- --cell 3504577656 --donor l5-pv-basket-hl5bn1 --polarity I")
    assert first.endswith("--cell-table-sha256 "+chain.DEFAULTS["cell_table_sha256"])
    assert "--archive .cache/h01/c3-candidates-20260916.zip --archive-sha256 "+chain.DEFAULTS["archive_sha256"] in first
    out = capsys.readouterr().out
    assert "total 97 runs before dt-half (40 candidate primary + 40 candidate ramp + 17 kept ramp) on 57 cells" in out
    assert "estimate: 97 x 190 s = 18430 s (5.1 h)" in out


def test_main_writes_the_two_chains(tmp_path):
    chain.main(["--out-dir", str(tmp_path/"chains"), "--chains", "2"])
    one = (tmp_path/"chains"/"keep-chain-1.sh").read_text().splitlines()
    two = (tmp_path/"chains"/"keep-chain-2.sh").read_text().splitlines()
    runs = [l for l in one+two if l.startswith("$R ")]
    assert len(runs) == 97 and len({l.split()[1] for l in runs}) == 97
    interneurons = [l for l in runs if "--polarity I" in l]
    assert len(interneurons) == 40 and all("--ramp-na 0.57" in l or "--ramp-na 0.3 " in l or "--current-na" in l
                                           for l in interneurons)
    kept = [l for l in runs if "--archive" not in l]
    assert len(kept) == 17 and all("-ramp " in l for l in kept)
    # order inside each chain: interneuron candidates, pyramidal candidates, kept ramps
    for lines in (one, two):
        kinds = ["I" if "--polarity I" in l else ("kept" if "--archive" not in l else "E") for l in lines if l.startswith("$R ")]
        assert kinds == sorted(kinds, key=["I", "E", "kept"].index)


def test_write_chains_emits_the_follower_with_the_same_extra_args(tmp_path):
    job_list = chain.jobs(CANDIDATES, KEPT, EXTRA)
    paths = chain.write_chains(job_list, tmp_path, 2, "/workspace/braintrace-c3-keep",
                               "docs/evidence/h01-c3-keep-drop/run_transfer.sh", "var/c3-keep",
                               types="docs/evidence/h01-c3-keep-drop/types.json", extra=EXTRA)
    assert paths[-1].name == "follower.sh"
    text = paths[-1].read_text()
    assert "--gate failure --datums docs/evidence/h01-c3-keep-drop/human-datums.json --chains 2" in text
    assert "--folder var/c3-keep --types docs/evidence/h01-c3-keep-drop/types.json" in text
    assert f'--extra-args "{EXTRA}"' in text and "--runner docs/evidence/h01-c3-keep-drop/run_transfer.sh" in text
    assert "kept-types" not in text   # the kept cells have their dt-half receipts already
    assert len(chain.write_chains(job_list, tmp_path/"plain", 2, "w", "r", "v")) == 2
