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
