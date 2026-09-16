"""Tests for the C3 keep-test type tables."""

import json

import pytest

from docs.evidence import h01_c3_keep_types as types_module

L2, L4, PV, SST = ("l2-pyramidal-allen-541563728", "l4-pyramidal-allen-527952884", "l5-pv-basket-hl5bn1",
                   "l3-sst-interneuron-hl5mn1")


def _candidate(c3_id, polarity, layer, donor_key, match="matched", axon=1, nsi=100, cell_class=None):
    return dict(c3_id=c3_id, polarity=polarity, layer=layer, donor_key=donor_key, donor_match=match,
                cell_class=cell_class or ("interneuron" if polarity == "I" else "pyramidal"), modifiers=[],
                donor_profile="profile-"+donor_key, synapses_total_axon=axon, nsi=nsi)


CANDIDATES = [_candidate(11, "E", "L4", L4, axon=4), _candidate(12, "E", "L3", L2, match="layer", axon=2),
              _candidate(21, "I", "L2", PV, match="layer", axon=11), _candidate(22, "I", "L3", SST, axon=3)]
COMPONENTS = dict(archive="c3-candidates-20260916.zip",
                  cells=[dict(cell_id=str(c), largest_component=k) for k, c in enumerate((11, 12, 21, 22))])


def test_candidate_rows_put_interneurons_first_in_rank_order_and_keep_the_assigned_donor():
    rows = types_module.candidate_rows(CANDIDATES, COMPONENTS)
    assert [r["cell_id"] for r in rows] == ["21", "22", "11", "12"]
    assert [r["rank"] for r in rows] == [1, 2, 1, 2]
    assert [r["donor_key"] for r in rows] == [PV, SST, L4, L2]
    assert rows[0]["match"] == rows[0]["donor_match"] == "layer" and rows[0]["note"].startswith("layer-only")
    assert rows[1]["match"] == "matched" and rows[1]["note"] == ""
    assert all(r["source"] == "c3 non-proofread" for r in rows)
    assert [r["component"] for r in rows] == [2, 3, 0, 1]
    assert rows[0]["donor"] == "profile-"+PV and rows[0]["synapses_total_axon"] == 11
    without_inventory = types_module.candidate_rows(CANDIDATES)
    assert "component" not in without_inventory[0] and len(without_inventory) == 4


@pytest.mark.parametrize("bad, message", [
    (_candidate(31, "E", "L2", "unknown-donor"), "not registered"),
    (_candidate(31, "I", "L2", L2), "is E, candidate is I"),
    (_candidate(11, "E", "L4", L4), "duplicate"),
    (_candidate(99, "E", "L2", L2), "not in the components inventory"),
])
def test_candidate_rows_reject_bad_donors_duplicates_and_unknown_cells(bad, message):
    with pytest.raises(ValueError, match=message):
        types_module.candidate_rows(CANDIDATES+[bad], COMPONENTS)


def test_kept_rows_follow_the_decision_and_mark_the_proofread_source():
    population = dict(rows=[dict(cell_id="7", donor_key=L2, layer="L2", cell_class="pyramidal", polarity="E", match="matched"),
                            dict(cell_id="8", donor_key=L4, layer="L4", cell_class="pyramidal", polarity="E", match="matched"),
                            dict(cell_id="9", donor_key=PV, layer="L5", cell_class="interneuron", polarity="I", match="matched")])
    rows = types_module.kept_rows(population, dict(kept=["8", "7"]))
    assert [r["cell_id"] for r in rows] == ["8", "7"] and all(r["source"] == "proofread_104" for r in rows)
    with pytest.raises(ValueError, match="without a type row"):
        types_module.kept_rows(population, dict(kept=["7", "77"]))


def test_build_and_cli_write_both_documents(tmp_path):
    population = dict(archive_sha256="p"*64,
                      rows=[dict(cell_id="7", donor_key=L2, layer="L2", cell_class="pyramidal", polarity="E", match="matched")])
    candidates_doc, kept_doc = types_module.build(dict(candidates=CANDIDATES, archive_sha256="a"*64), COMPONENTS,
                                                  population, dict(kept=["7"]))
    assert candidates_doc["summary"] == dict(cells=4, by_donor={PV: 1, SST: 1, L4: 1, L2: 1},
                                             by_match={"layer": 2, "matched": 2}, by_polarity={"I": 2, "E": 2},
                                             by_layer_class={"L2 interneuron": 1, "L3 interneuron": 1,
                                                             "L4 pyramidal": 1, "L3 pyramidal": 1})
    assert candidates_doc["archive"] == "c3-candidates-20260916.zip" and candidates_doc["archive_sha256"] == "a"*64
    assert kept_doc["summary"]["cells"] == 1 and kept_doc["archive_sha256"] == "p"*64
    paths = {}
    for name, document in (("candidates.json", dict(candidates=CANDIDATES)), ("components.json", COMPONENTS),
                           ("population.json", population), ("decision.json", dict(kept=["7"]))):
        paths[name] = tmp_path/name
        paths[name].write_text(json.dumps(document))
    types_module.main(["--candidates", str(paths["candidates.json"]), "--components", str(paths["components.json"]),
                       "--population-types", str(paths["population.json"]), "--decision", str(paths["decision.json"]),
                       "--archive-sha256", "b"*64, "--out-dir", str(tmp_path/"out")])
    written = json.loads((tmp_path/"out"/"types.json").read_text())
    assert [r["cell_id"] for r in written["rows"]] == ["21", "22", "11", "12"] and written["archive_sha256"] == "b"*64
    assert json.loads((tmp_path/"out"/"kept-types.json").read_text())["rows"][0]["cell_id"] == "7"


def test_real_candidates_file_yields_the_registered_donor_split():
    """The 40 candidates: 19 PV + 1 SST interneurons, 18 L2-donor + 2 L4-donor pyramidal cells."""
    evidence = types_module.EVIDENCE
    candidates = json.loads((evidence/"h01-c3-candidates.json").read_text(encoding="utf-8"))
    components = json.loads((evidence/"h01-c3-candidates-components.json").read_text(encoding="utf-8"))
    rows = types_module.candidate_rows(candidates["candidates"], components)
    assert len(rows) == 40 and [r["polarity"] for r in rows] == ["I"]*20+["E"]*20
    summary = types_module.summarize(rows)
    assert summary["by_donor"] == {PV: 19, SST: 1, L2: 18, L4: 2}
    assert summary["by_match"]["layer"] == 20 and sum(1 for r in rows if r["donor_key"] == L2 and r["layer"] == "L3") == 5
    assert rows[0]["cell_id"] == "3504577656" and rows[20]["cell_id"] == "3790464726"   # top-ranked I, top-ranked E
