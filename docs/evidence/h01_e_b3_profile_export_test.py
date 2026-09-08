"""Density-by-density equality between the exported B3 profile and the NEURON report."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from braintrace.datasets import _h01_ei_parameters as parameters
from docs.evidence import h01_e_b3_profile_export as export

FOLDER = Path(__file__).parent
REPORT = json.loads((FOLDER/export.DEFAULT_REPORT).read_text())


def _load_generated(path):
    spec = importlib.util.spec_from_file_location("generated_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_applied_density_appears_once_in_the_region_tuples():
    regions = {row[0]: row for row in export.profile_from_report(REPORT)["regions"]}
    for genome in REPORT["applied_genome"]:
        family, cm, leak, channels, calcium = regions[genome["section"]]
        if genome["name"] == "g_pas":
            assert leak == genome["value"]
        elif genome["name"].startswith("gbar_"):
            assert dict(channels)[genome["name"][5:]] == genome["value"]
        else:
            assert calcium[0 if genome["name"].startswith("decay") else 1] == genome["value"]
    assert sum(len(row[3]) for row in regions.values()) == sum(
        1 for g in REPORT["applied_genome"] if g["name"].startswith("gbar_"))


def test_stated_b3_levers_match_the_report():
    profile = export.profile_from_report(REPORT)
    regions = {row[0]: row for row in profile["regions"]}
    source_soma = dict(dict((r[0], r) for r in parameters.E_SOURCE)["soma"][3])
    soma = dict(regions["soma"][3])
    assert soma["NaTs"] == pytest.approx(source_soma["NaTs"]*.9, rel=1e-12)
    assert soma["SK"] == pytest.approx(source_soma["SK"]*.35, rel=1e-12)
    assert dict(regions["axon"][3]) == {"NaTs": 3.814} and regions["axon"][4] is None
    assert regions["soma"][4][0] == pytest.approx(494.01955262344603)  # source decay: factor 1.0, not the 657 ms candidate
    assert regions["soma"][4][1] == pytest.approx(8.762096311710155e-4)
    distributed = REPORT["ih_distribution"]["uniform_density_s_cm2"]
    assert all(dict(regions[r][3])["Ih"] == distributed for r in ("soma", "dend", "apic"))
    total_area_cm2 = sum(REPORT["ih_distribution"]["areas_um2"].values())*1e-8
    assert distributed*total_area_cm2 == pytest.approx(REPORT["ih_distribution"]["source_maximum_conductance_s"], rel=1e-9)
    assert profile["reversal_mv"] == pytest.approx(REPORT["initial_mv"]-4.)
    assert profile["controls"] == {"NaTs": {"m_open": 2., "h_recovery": 1.}, "Kv3_1": {"m_close": .9}}
    assert profile["axial_ohm_cm"] == pytest.approx(94.62299222737664)
    assert [row[1] for row in profile["regions"]] == [1., 1., 2.3031548608821226, 2.3031548608821226]


def test_unknown_channel_is_rejected():
    report = json.loads(json.dumps(REPORT))
    report["applied_genome"].append({"section": "soma", "name": "gbar_Kv7", "value": 1., "mechanism": "Kv7"})
    with pytest.raises(ValueError, match="Kv7"):
        export.region_tuple(report, "soma")


def test_generated_file_is_current_and_loads_as_the_same_literal(tmp_path):
    out = tmp_path/"profile.py"
    export.main(["--report", str(FOLDER/export.DEFAULT_REPORT), "--output", str(out)])
    generated = _load_generated(out)
    committed = _load_generated(FOLDER/export.DEFAULT_OUTPUT)
    profile = export.profile_from_report(REPORT)
    assert generated.E_B3_EXPERIMENTAL == profile["regions"] == committed.E_B3_EXPERIMENTAL
    assert committed.E_B3_EXPERIMENTAL_CONTROLS == profile["controls"]
    assert committed.E_B3_EXPERIMENTAL_REVERSAL_MV == profile["reversal_mv"]
    assert np.isfinite([committed.E_B3_EXPERIMENTAL_AXIAL_OHM_CM, committed.E_B3_EXPERIMENTAL_INITIAL_MV]).all()
    assert out.read_text() == (FOLDER/export.DEFAULT_OUTPUT).read_text()
