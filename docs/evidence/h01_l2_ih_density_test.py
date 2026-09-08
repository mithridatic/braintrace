"""Verify intervention isolation, source preservation, and invalid inputs."""

import copy

import pytest

from docs.evidence.h01_l2_ih_density import ih_density_fit


def fit():
    return {"genome": [dict(section="soma", mechanism="Ih", name="gbar_Ih", value=2.),
                       dict(section="axon", mechanism="Ih", name="gbar_Ih", value=3.)]}


def test_isolation_and_default():
    source = fit()
    original = copy.deepcopy(source)
    assert ih_density_fit(source) == source
    result = ih_density_fit(source, .9)
    assert result["genome"][0]["value"] == 1.8
    assert result["genome"][1] == source["genome"][1]
    assert source == original
    assert result["genome"][1] is not source["genome"][1]


@pytest.mark.parametrize("factor", [0., -1., float("nan"), float("inf")])
def test_invalid_factor(factor):
    with pytest.raises(ValueError):
        ih_density_fit(fit(), factor)


@pytest.mark.parametrize("case", ["missing", "duplicate", "negative", "nan", "overflow"])
def test_invalid_target(case):
    source = fit()
    if case == "missing":
        source["genome"].pop(0)
    elif case == "duplicate":
        source["genome"].append(dict(source["genome"][0]))
    else:
        source["genome"][0]["value"] = {"negative": -1., "nan": float("nan"), "overflow": 1e308}[case]
    with pytest.raises(ValueError):
        ih_density_fit(source, 10. if case == "overflow" else 1.)
