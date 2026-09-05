"""Check source preservation and regional leak intervention boundaries."""

import copy

import pytest

from docs.evidence.h01_l2_leak import leak_fit


def fit():
    return {"genome": [dict(section=s, mechanism="", name="g_pas", value=float(i + 1))
                       for i, s in enumerate(("soma", "axon", "dend", "apic"))],
            "passive": [{"e_pas": -84.}]}


def test_isolation_and_default():
    source = fit()
    saved = copy.deepcopy(source)
    assert leak_fit(source) == source
    result = leak_fit(source, 1.5)
    assert [r["value"] for r in result["genome"]] == [1.5, 3., 4.5, 6.]
    assert result["passive"] == source["passive"]
    assert result["passive"] is not source["passive"]
    assert source == saved


@pytest.mark.parametrize("factor", [0., -1., float("nan"), float("inf")])
def test_invalid_factor(factor):
    with pytest.raises(ValueError):
        leak_fit(fit(), factor)


@pytest.mark.parametrize("case", ["missing", "duplicate", "region", "mechanism", "nan", "negative", "overflow"])
def test_invalid_target(case):
    source = fit()
    row = source["genome"][0]
    if case == "missing":
        source["genome"].pop()
    elif case == "duplicate":
        source["genome"].append(dict(row))
    elif case == "region":
        row["section"] = "foreign"
    elif case == "mechanism":
        row["mechanism"] = "NaTs"
    else:
        row["value"] = {"nan": float("nan"), "negative": -1., "overflow": 1e308}[case]
    with pytest.raises(ValueError):
        leak_fit(source, 10. if case == "overflow" else 1.)
