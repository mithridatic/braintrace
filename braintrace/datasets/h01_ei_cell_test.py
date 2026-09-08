"""Explicit H01 region transfer and original anatomical identity."""
from types import SimpleNamespace
import brainstate
import brainunit as u
import numpy as np
import pytest
from braincell.filter import AllRegion
from .h01_anatomy_test import imported
from .h01_ei_cell import make_h01_ei_cell


def test_e_candidate_runs_on_h01_and_retains_identity(imported):
    annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=("L2", "pyramidal")))
    with brainstate.environ.context(precision=64):
        cell, evidence = make_h01_ei_cell(imported, annotations, polarity="E",
            regions={"soma": AllRegion()}, region_basis="Explicit whole-component soma surrogate for transfer test.")
        result = cell.run(dt=.001*u.ms, duration=.02*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
    assert evidence["source_tags"] == ["L2", "pyramidal"]
    assert evidence["borrowed_dynamics"]["name"].endswith("candidate:v1")
    assert evidence["measured_anatomy"]["label_counts"] == imported.anatomy().label_counts


from .h01_anatomy import _Region, _geometry_signature
from .h01_ei_cell import _validate_regions


def partition(imported):
    anatomy = imported.anatomy()
    soma = anatomy.cable_neighborhood(30, radius_um=2.)
    axon = AllRegion()-soma
    return {"soma": soma, "axon": axon}


@pytest.mark.parametrize("role,mode", [("I", "candidate"), ("I", "source"), ("E", "source")])
def test_region_partition_runs_and_explicit_role_does_not_relabel(imported, role, mode):
    annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=("pyramidal",)))
    with brainstate.environ.context(precision=64):
        cell, evidence = make_h01_ei_cell(imported, annotations, polarity=role,
            regions=partition(imported), region_basis="Counterfactual test partition.", mode=mode, pop_size=(1,))
        result = cell.run(dt=.001*u.ms, duration=.02*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
    assert evidence["source_tags"] == ["pyramidal"]
    assert evidence["modeled_polarity"] == role
    assert evidence["borrowed_dynamics"]["mode"] == mode


@pytest.mark.parametrize("kind", ["missing", "overlap", "gap", "unknown", "bounds"])
def test_reject_ambiguous_regions(imported, kind):
    regions = partition(imported)
    if kind == "missing": regions = {"soma": AllRegion()}
    if kind == "overlap": regions["dend"] = AllRegion()
    if kind == "gap": regions["axon"] = regions["soma"]
    if kind == "unknown": regions["bad"] = AllRegion()
    if kind == "bounds":
        regions["axon"] = _Region(_geometry_signature(imported.morphology), ((-1, 0., 1.),), "test")
    with pytest.raises(ValueError):
        _validate_regions(imported.morphology, regions, "I")


@pytest.mark.parametrize("kwargs", [{"region_basis": ""}, {"current_na": np.nan},
    {"duration_ms": 0.}, {"delay_ms": -1.}, {"max_cv_length_um": 0.}])
def test_invalid_settings_fail_before_construction(imported, kwargs):
    arguments = dict(polarity="E", regions={"soma": AllRegion()}, region_basis="test")
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        make_h01_ei_cell(imported, None, **arguments)


def test_reject_map_that_excludes_measured_soma(imported):
    regions = partition(imported)
    annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=()))
    with pytest.raises(ValueError, match="soma probe"):
        make_h01_ei_cell(imported, annotations, polarity="I", region_basis="reversed map",
                         regions={"soma": regions["axon"], "axon": regions["soma"]})


def test_reject_uncovered_branch_tail(imported):
    soma = imported.anatomy().cable_neighborhood(30, radius_um=2.)
    with pytest.raises(ValueError, match="gaps"):
        _validate_regions(imported.morphology, {"soma": soma}, "E")


def test_donor_key_selects_profile_and_is_recorded(imported):
    annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=("L5", "interneuron")))
    with brainstate.environ.context(precision=64):
        cell, evidence = make_h01_ei_cell(imported, annotations, polarity="I", donor="l5-pv-basket-hl5bn1",
            regions=partition(imported), region_basis="Donor key test.", pop_size=(1,))
        result = cell.run(dt=.001*u.ms, duration=.02*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
    assert evidence["donor"] == "l5-pv-basket-hl5bn1"
    assert evidence["borrowed_dynamics"]["channel_prefix"] == "H01PV"
    assert evidence["borrowed_dynamics"]["sodium_reversal_mv"] == 50.
    _, default = make_h01_ei_cell(imported, annotations, polarity="E", regions={"soma": AllRegion()},
                                  region_basis="Default donor test.")
    assert default["donor"] == "l2-pyramidal-allen-541563728"
    assert default["borrowed_dynamics"]["name"] == "h01-l2-kv3-ninety-ca133:candidate:v1"


@pytest.mark.parametrize("donor,polarity", [("l5-pv-basket-hl5bn1", "E"), ("no-such-donor", "E")])
def test_donor_polarity_mismatch_or_unknown_key_fails(imported, donor, polarity):
    with pytest.raises(ValueError):
        make_h01_ei_cell(imported, None, polarity=polarity, donor=donor, regions={"soma": AllRegion()},
                         region_basis="test")


from dataclasses import replace
import braincell
from .h01_ei_cell import _paint_profile
from .h01_ei_profiles import get_ei_profile


def _paint_spy(monkeypatch):
    painted, original = [], braincell.Cell.paint

    def spy(self, region, *mechanisms):
        painted.append((region, tuple((getattr(m, "class_name", type(m).__name__), getattr(m, "name", None))
                                      for m in mechanisms)))
        return original(self, region, *mechanisms)
    monkeypatch.setattr(braincell.Cell, "paint", spy)
    return painted


def _mechanisms(painted, region):
    return [m for r, ms in painted if r is region for m in ms]


def test_b3_axon_with_channels_but_no_calcium_carries_sodium_and_potassium_ions(imported, monkeypatch):
    regions = partition(imported)
    annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=("L2", "pyramidal")))
    painted = _paint_spy(monkeypatch)
    with brainstate.environ.context(precision=64):
        cell, evidence = make_h01_ei_cell(imported, annotations, polarity="E", mode="b3", regions=regions,
                                          region_basis="B3 axon ion test.", pop_size=(1,))
        result = cell.run(dt=.001*u.ms, duration=.02*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
    assert evidence["borrowed_dynamics"]["mode"] == "b3"
    axon = _mechanisms(painted, regions["axon"])
    for mechanism in (("SodiumFixed", "sodium"), ("PotassiumFixed", "potassium"), ("H01L2_NaTs", "pv_NaTs")):
        assert mechanism in axon
    assert ("H01PV_Calcium", "calcium") not in axon
    assert ("H01PV_Calcium", "calcium") in _mechanisms(painted, regions["soma"])


@pytest.mark.parametrize("mode", ["candidate", "source"])
def test_candidate_and_source_axon_rows_without_channels_get_no_ions(imported, monkeypatch, mode):
    regions = partition(imported)
    painted = _paint_spy(monkeypatch)
    make_h01_ei_cell(imported, SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=())), polarity="E",
                     mode=mode, regions=regions, region_basis="Unchanged paint list.")
    assert {m[0] for m in _mechanisms(painted, regions["axon"])} == {"CableProperty", "IL"}


def test_calcium_mechanism_without_a_calcium_tuple_is_rejected():
    profile = replace(get_ei_profile("E"), regions=(("soma", 1., 4e-4, (("SK", 1e-3),), None),))
    with pytest.raises(ValueError, match="calcium"):
        _paint_profile(SimpleNamespace(paint=lambda *a, **k: None), profile, {"soma": AllRegion()})


def test_rounded_interval_ends_snap_to_branch_bounds(imported):
    regions = partition(imported)
    rows = tuple((b, -1e-12 if lo == 0. else lo, 1.+2e-16 if hi == 1. else hi)
                 for b, lo, hi in regions["axon"].evaluate(imported.morphology).intervals)
    assert any(hi > 1. for _, _, hi in rows)  # The 12-cell build failure: 2451406889 branch 162 soma hi 1+2e-16.
    regions["axon"] = _Region(_geometry_signature(imported.morphology), rows, "test")
    intervals = _validate_regions(imported.morphology, regions, "I")
    assert all(0. <= lo < hi <= 1. for _, lo, hi in intervals["axon"])
    regions["axon"] = _Region(_geometry_signature(imported.morphology), tuple((b, lo, hi+1e-6) for b, lo, hi in rows), "test")
    with pytest.raises(ValueError):
        _validate_regions(imported.morphology, regions, "I")
