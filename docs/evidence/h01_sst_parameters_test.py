"""The committed SST region tuples equal the hoc, density by density."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_sst_parameters as parse  # noqa: E402
from braintrace.datasets import _h01_ei_parameters as parameters  # noqa: E402

SYNTHETIC = """proc biophys_X(){
\tforsec $o1.all {
\t\tinsert pas
\t\tRa = 100
\t\tcm = 1
\t\te_pas = -81.5
\t\tg_pas = 0.0000232
\t\tinsert Ih
\t}
\tforsec $o1.somatic {
\t\tinsert NaTg
\t\tinsert CaDynamics
\t\tgamma_CaDynamics = 0.0005
\t\tgbar_Ih = 0.0000431
\t}
\tforsec $o1.basal {
\t\tcm = 2
\t\tgbar_Ih = 0.0000949
\t}
\tforsec $o1.axonal {
\t\tinsert Nap
\t}
\t$o1.distribute_channels("soma","decay_CaDynamics",0,1.000000,0.000000,0.000000,0.000000,465.0000000000)
\t$o1.distribute_channels("soma","gbar_NaTg",0,1.000000,0.000000,0.000000,0.000000,0.1270000000)
\t$o1.distribute_channels("axon","gbar_Nap",0,1.000000,0.000000,0.000000,0.000000,0.0004440000)
}
"""


def test_synthetic_hoc_parses_defaults_overrides_and_calcium():
    regions = parse.region_tuples(SYNTHETIC)
    assert regions[0] == ("soma", 1., 2.32e-5, (("NaTg", .127), ("Ih", 4.31e-5)), (465., 5e-4))
    assert regions[1] == ("axon", 1., 2.32e-5, (("Nap", 4.44e-4), ("Ih", 1e-5)), None)
    assert regions[2] == ("dend", 2., 2.32e-5, (("Ih", 9.49e-5),), None)
    assert regions[3] == ("apic", 1., 2.32e-5, (("Ih", 1e-5),), None)
    assert parse.passive(SYNTHETIC) == {"e_pas_mv": -81.5, "ra_ohm_cm": 100.}


def test_unsupported_distribution_or_missing_density_raise():
    graded = SYNTHETIC.replace('"gbar_NaTg",0,1.000000,0.000000', '"gbar_NaTg",2,1.000000,0.000000')
    with pytest.raises(ValueError, match="Unsupported distribution"):
        parse.region_tuples(graded)
    missing = SYNTHETIC.replace('\t$o1.distribute_channels("axon","gbar_Nap",0,1.000000,0.000000,0.000000,0.000000,0.0004440000)\n', "")
    with pytest.raises(ValueError, match="No density for Nap"):
        parse.region_tuples(missing)


@pytest.mark.skipif(not parse.HOC.exists(), reason="donor cache not fetched")
def test_committed_constant_equals_the_cached_hoc():
    text = parse.HOC.read_text()
    assert parse.region_tuples(text) == parameters.SST_L3_HL5MN1_SOURCE
    assert parse.passive(text) == {"e_pas_mv": -81.5, "ra_ohm_cm": 100.}


def test_committed_constant_facts():
    regions = {r[0]: r for r in parameters.SST_L3_HL5MN1_SOURCE}
    assert set(regions) == {"soma", "axon", "dend", "apic"}
    assert all(r[1] == 1. and r[2] == 2.32e-5 for r in regions.values())
    assert dict(regions["soma"][3])["NaTg"] == .127 and "Nap" not in dict(regions["soma"][3])
    assert dict(regions["axon"][3])["Nap"] == 4.44e-4 and regions["axon"][4] == (469., 5e-4)
    assert parameters.DONOR_REGIONS["l3-sst-interneuron-hl5mn1"] == {
        "candidate": parameters.SST_L3_HL5MN1_SOURCE, "source": parameters.SST_L3_HL5MN1_SOURCE}
