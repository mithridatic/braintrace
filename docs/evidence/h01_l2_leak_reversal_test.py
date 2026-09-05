"""Verify reversal isolation and ambiguous-source rejection."""

import copy

import pytest

from docs.evidence.h01_l2_leak_reversal import shift_leak_reversal


@pytest.mark.parametrize("case", ["valid", "nan_shift", "multiple", "missing", "override", "nan_source", "overflow"])
def test_shift(case):
    source = {"passive": [dict(e_pas=-84., ra=100.)], "genome": [dict(name="g_pas", value=.001)]}
    shift = -4.
    if case == "nan_shift":
        shift = float("nan")
    elif case == "multiple":
        source["passive"] *= 2
    elif case == "missing":
        del source["passive"][0]["e_pas"]
    elif case == "override":
        source["genome"].append(dict(name="e_pas", value=-80.))
    elif case == "nan_source":
        source["passive"][0]["e_pas"] = float("nan")
    elif case == "overflow":
        source["passive"][0]["e_pas"] = shift = 1e308
    if case != "valid":
        with pytest.raises(ValueError):
            shift_leak_reversal(source, shift)
    else:
        saved = copy.deepcopy(source)
        assert shift_leak_reversal(source) == source
        result = shift_leak_reversal(source, shift)
        assert result["passive"] == [dict(e_pas=-88., ra=100.)]
        assert result["genome"] == source["genome"]
        assert source == saved
