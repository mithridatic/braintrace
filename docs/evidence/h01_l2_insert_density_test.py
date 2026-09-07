import pytest

from docs.evidence.h01_l2_insert_density import insert_density_fit, parse_insert_density

FIT = {"genome": [{"section": "soma", "mechanism": "NaTs", "name": "gbar_NaTs", "value": 2.9},
                  {"section": "axon", "mechanism": "", "name": "g_pas", "value": 2.8e-4}],
       "conditions": [{"erev": [{"section": "soma", "ena": 53., "ek": -107.}], "v_init": -80.}]}


def test_parse_accepts_absolute_density():
    assert parse_insert_density("NaTs:axon:3.81") == {"mechanism": "NaTs", "region": "axon", "value": 3.81}


@pytest.mark.parametrize("text", ["NaTs:axon", "NaTs:apex:1", "NaTs:all:1", "NaTs:axon:-1", "NaTs:axon:nan", "NaTs:axon:inf"])
def test_parse_rejects_malformed_or_invalid(text):
    with pytest.raises(ValueError):
        parse_insert_density(text)


def test_insert_adds_density_and_reversal_rows_without_touching_the_source():
    result = insert_density_fit(FIT, "NaTs", "axon", 3.81)
    assert result["genome"][-1] == {"section": "axon", "mechanism": "NaTs", "name": "gbar_NaTs", "value": 3.81}
    assert result["conditions"][0]["erev"][-1] == {"section": "axon", "ena": 53., "ek": -107.}
    assert len(FIT["genome"]) == 2 and len(FIT["conditions"][0]["erev"]) == 1


def test_insert_keeps_an_existing_reversal_row_for_the_region():
    fit = {"genome": [], "conditions": {"erev": [{"section": "soma", "ena": 53., "ek": -107.},
                                                 {"section": "axon", "ena": 50., "ek": -100.}]}}
    result = insert_density_fit(fit, "NaTs", "axon", 1.)
    assert [r["section"] for r in result["conditions"]["erev"]] == ["soma", "axon"]
    assert result["conditions"]["erev"][1]["ena"] == 50.


def test_insert_without_conditions_block():
    assert insert_density_fit({"genome": []}, "NaTs", "dend", 0.)["genome"][0]["value"] == 0.
    assert insert_density_fit({"genome": [], "conditions": []}, "NaTs", "dend", 0.)["conditions"] == []


def test_insert_refuses_a_region_that_already_has_the_row():
    with pytest.raises(ValueError, match="scale it instead"):
        insert_density_fit(FIT, "NaTs", "soma", 1.)


@pytest.mark.parametrize("region, value", [("apex", 1.), ("axon", -1.), ("axon", float("nan"))])
def test_insert_rejects_bad_region_or_value(region, value):
    with pytest.raises(ValueError):
        insert_density_fit(FIT, "NaTs", region, value)
