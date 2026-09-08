"""Verify isolation of the calcium-removal intervention."""

import copy

import numpy as np
import pytest

from docs.evidence.h01_l2_calcium import calcium_removal_fit


@pytest.fixture
def fit():
    return {"genome": [
        {"section": "soma", "mechanism": "CaDynamics", "name": "decay_CaDynamics", "value": 494.},
        {"section": "soma", "mechanism": "CaDynamics", "name": "gamma_CaDynamics", "value": .0008},
        {"section": "axon", "mechanism": "CaDynamics", "name": "decay_CaDynamics", "value": 200.},
        {"section": "soma", "mechanism": "SK", "name": "gbar_SK", "value": .003}],
        "conditions": [{"celsius": 34.}]}


def test_default_identity_and_independent_copy(fit):
    result = calcium_removal_fit(fit)
    assert result == fit
    result["conditions"][0]["celsius"] = 1.
    assert fit["conditions"][0]["celsius"] == 34.


def test_only_somatic_removal_changes(fit):
    before = copy.deepcopy(fit)
    result = calcium_removal_fit(fit, 2.)
    assert result["genome"][0]["value"] == 988.
    result["genome"][0]["value"] = 494.
    assert result == before == fit


@pytest.mark.parametrize("factor", [0., -1., np.nan, np.inf])
def test_invalid_factor(fit, factor):
    with pytest.raises(ValueError):
        calcium_removal_fit(fit, factor)


@pytest.mark.parametrize("case", ["missing", "duplicate", "zero", "negative", "nan", "overflow"])
def test_invalid_target(fit, case):
    if case == "missing":
        fit["genome"].pop(0)
    elif case == "duplicate":
        fit["genome"].append(copy.deepcopy(fit["genome"][0]))
    else:
        fit["genome"][0]["value"] = {"zero": 0., "negative": -1., "nan": np.nan, "overflow": 1e308}[case]
    with pytest.raises(ValueError):
        calcium_removal_fit(fit, 2.)
