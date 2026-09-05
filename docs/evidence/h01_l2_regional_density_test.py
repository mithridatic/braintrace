"""Regional density parsing and scaling."""
import pytest
from docs.evidence.h01_l2_regional_density import parse_regional_density, regional_density_fit

FIT = {"genome": [{"section": "soma", "mechanism": "NaTs", "name": "gbar_NaTs", "value": 2.},
                  {"section": "axon", "mechanism": "NaTs", "name": "gbar_NaTs", "value": 4.},
                  {"section": "soma", "mechanism": "Kv3_1", "name": "gbar_Kv3_1", "value": 1.},
                  {"section": "soma", "mechanism": "NaTs", "name": "vshift_NaTs", "value": 9.}]}


def test_parse_accepts_valid_and_rejects_invalid_text():
    assert parse_regional_density("NaTs:axon:1.3") == {"mechanism": "NaTs", "region": "axon", "factor": 1.3}
    for bad in ("NaTs:axon", "NaTs:apex:1", "NaTs:soma:0", "NaTs:soma:nan", "NaTs:all:-1"):
        with pytest.raises(ValueError):
            parse_regional_density(bad)


def test_scaling_touches_only_selected_density_rows():
    scaled = regional_density_fit(FIT, "NaTs", "axon", 1.5)
    assert [r["value"] for r in scaled["genome"]] == [2., 6., 1., 9.]
    assert [r["value"] for r in FIT["genome"]] == [2., 4., 1., 9.]
    everywhere = regional_density_fit(FIT, "NaTs", "all", .5)
    assert [r["value"] for r in everywhere["genome"]] == [1., 2., 1., 9.]


def test_missing_row_or_invalid_factor_fails():
    with pytest.raises(ValueError, match="No SK density"):
        regional_density_fit(FIT, "SK", "soma", 1.1)
    with pytest.raises(ValueError, match="positive and finite"):
        regional_density_fit(FIT, "NaTs", "soma", 0.)
