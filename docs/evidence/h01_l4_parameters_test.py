"""The committed L4 region tuples equal the Allen fit, density by density; the generator reproduces E_SOURCE."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_l4_parameters as parse  # noqa: E402
from braintrace.datasets import _h01_ei_parameters as parameters  # noqa: E402

SYNTHETIC = {
    "passive": [{"ra": 15., "e_pas": -80., "cm": [{"section": "soma", "cm": 1.}, {"section": "axon", "cm": 1.},
                                                  {"section": "dend", "cm": 1.5}, {"section": "apic", "cm": 1.5}]}],
    "fitting": [{"junction_potential": -14., "sweeps": [69]}],
    "conditions": [{"celsius": 34., "v_init": -80., "erev": [{"section": "soma", "ena": 53., "ek": -107.}]}],
    "genome": [{"section": "soma", "name": "gbar_Im", "value": 1e-4, "mechanism": "Im"},
               {"section": "soma", "name": "gbar_NaTs", "value": 2., "mechanism": "NaTs"},
               {"section": "soma", "name": "gamma_CaDynamics", "value": 7e-5, "mechanism": "CaDynamics"},
               {"section": "soma", "name": "decay_CaDynamics", "value": 175., "mechanism": "CaDynamics"},
               {"section": "soma", "name": "g_pas", "value": 4e-4, "mechanism": ""},
               {"section": "axon", "name": "g_pas", "value": 2e-4, "mechanism": ""},
               {"section": "dend", "name": "g_pas", "value": 1e-5, "mechanism": ""},
               {"section": "apic", "name": "g_pas", "value": 1e-7, "mechanism": ""}],
}


def test_synthetic_fit_parses_capacitance_leak_channels_and_calcium():
    regions = parse.region_tuples(SYNTHETIC)
    assert regions[0] == ("soma", 1., 4e-4, (("Im", 1e-4), ("NaTs", 2.)), (175., 7e-5))
    assert regions[1] == ("axon", 1., 2e-4, (), None)
    assert regions[2] == ("dend", 1.5, 1e-5, (), None)
    assert regions[3] == ("apic", 1.5, 1e-7, (), None)
    assert parse.physiology(SYNTHETIC) == {"initial_mv": -80., "e_pas_mv": -80., "ra_ohm_cm": 15., "celsius": 34.,
                                           "ena_mv": 53., "ek_mv": -107., "junction_potential_mv": -14.}


def test_missing_region_or_unsupported_row_raise():
    missing = json.loads(json.dumps(SYNTHETIC))
    missing["genome"] = [row for row in missing["genome"] if not (row["section"] == "apic" and row["name"] == "g_pas")]
    with pytest.raises(ValueError, match="No cm or g_pas for apic"):
        parse.region_tuples(missing)
    odd = json.loads(json.dumps(SYNTHETIC))
    odd["genome"].append({"section": "soma", "name": "vshift_NaTs", "value": 1., "mechanism": "NaTs"})
    with pytest.raises(ValueError, match="Unsupported genome row vshift_NaTs"):
        parse.region_tuples(odd)


def test_committed_constant_facts():
    regions = {r[0]: r for r in parameters.L4_ALLEN_527952884_SOURCE}
    assert list(r[0] for r in parameters.L4_ALLEN_527952884_SOURCE) == ["soma", "axon", "dend", "apic"]
    soma = dict(regions["soma"][3])
    assert set(soma) == {"Im", "Ih", "NaTs", "Nap", "K_P", "K_T", "SK", "Kv3_1", "Ca_HVA", "Ca_LVA"}
    assert soma["NaTs"] == pytest.approx(2.0538, abs=1e-4) and soma["Ih"] == pytest.approx(5.766e-3, abs=1e-6)
    assert regions["soma"][4] == pytest.approx((175.17, 7.43e-5), rel=1e-3)
    assert regions["soma"][2] == pytest.approx(4.325e-4, rel=1e-3) and regions["axon"][2] == pytest.approx(2.68e-4, rel=1e-3)
    assert regions["dend"][1] == regions["apic"][1] == pytest.approx(1.5826, abs=1e-4)
    assert all(regions[f][3] == () and regions[f][4] is None for f in ("axon", "dend", "apic"))
    assert parameters.DONOR_REGIONS["l4-pyramidal-allen-527952884"] == {
        "candidate": parameters.L4_ALLEN_527952884_SOURCE, "source": parameters.L4_ALLEN_527952884_SOURCE}


@pytest.mark.skipif(not parse.FIT.exists(), reason="donor cache not fetched")
def test_committed_constant_equals_the_cached_fit_density_by_density():
    fit = json.loads(parse.FIT.read_text())
    generated = parse.region_tuples(fit)
    assert generated == parameters.L4_ALLEN_527952884_SOURCE
    for made, kept in zip(generated, parameters.L4_ALLEN_527952884_SOURCE):
        assert made[0] == kept[0] and made[1] == kept[1] and made[2] == kept[2]
        assert dict(made[3]) == dict(kept[3]) and made[4] == kept[4]
    assert parse.physiology(fit)["initial_mv"] == -80.81838607788086
    assert parse.physiology(fit)["ra_ohm_cm"] == 14.9970627156


@pytest.mark.skipif(not parse.L2_FIT.exists(), reason="sibling L2 cache absent")
def test_generator_reproduces_the_committed_l2_source_from_its_fit():
    """The same function applied to 541563728_fit.json yields E_SOURCE, so the generator is not L4-specific."""
    fit = json.loads(parse.L2_FIT.read_text())
    assert parse.region_tuples(fit) == parameters.E_SOURCE
    assert parse.physiology(fit)["initial_mv"] == -83.97993469238281


def test_main_prints_the_constant_and_physiology_for_a_given_fit(tmp_path, capsys):
    path = tmp_path/"fit.json"
    path.write_text(json.dumps(SYNTHETIC))
    parse.main([str(path)])
    out = capsys.readouterr().out
    assert out.startswith("L4_ALLEN_527952884_SOURCE = (('soma',") and "'ra_ohm_cm': 15.0" in out
