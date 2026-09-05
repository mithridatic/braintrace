"""Verify total conductance, source isolation, and invalid locations."""

import copy

import pytest

from docs.evidence.h01_l2_ih_location import distribute_ih


@pytest.mark.parametrize("case", ["valid", "missing_area", "bad_area", "missing_target", "duplicate", "wrong_region", "bad_density"])
def test_distribution(case):
    source = {"genome": [dict(section="soma", mechanism="Ih", name="gbar_Ih", value=.0001)]}
    areas = dict(soma=100., dend=300., apic=600.)
    if case == "missing_area":
        del areas["apic"]
    elif case == "bad_area":
        areas["dend"] = float("nan")
    elif case == "missing_target":
        source["genome"] = []
    elif case == "duplicate":
        source["genome"] *= 2
    elif case == "wrong_region":
        source["genome"][0]["section"] = "axon"
    elif case == "bad_density":
        source["genome"][0]["value"] = -1.
    saved = copy.deepcopy(source)
    if case != "valid":
        with pytest.raises(ValueError):
            distribute_ih(source, areas)
    else:
        result, report = distribute_ih(source, areas)
        assert source == saved
        assert {r["section"] for r in result["genome"]} == set(areas)
        assert all(r["value"] == pytest.approx(.00001) for r in result["genome"])
        assert report["source_maximum_conductance_s"] == pytest.approx(report["distributed_maximum_conductance_s"], rel=1e-12)
