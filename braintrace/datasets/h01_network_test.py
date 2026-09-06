"""Dale signs, contact placement, and compiled H01 network execution."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import brainstate
import brainunit as u
import numpy as np
import pytest

from .h01_anatomy_test import imported
from .h01_connectivity_test import annotations, edge, report
from .h01_connectivity import select_connectivity
from .h01_network import make_h01_network


@pytest.fixture
def arguments(imported, annotations):
    topology = select_connectivity(report([edge(), edge("13", "14", "2", 1)]), annotations)
    point = np.array([6.4, 0., 0.])
    projection = imported.anatomy().project([point], max_distance_um=1.)[0]
    for contact in topology["contacts"]:
        contact["construction_ready"] = True
        for side in ("pre", "post"):
            contact[side+"_placement"] = dict(component=0, source_swc_sha256=imported.source_sha256,
                position_um=point.tolist(), cable_location=list(projection.location.evaluate(imported.morphology).points[0]))
    topology["max_distance_um"] = 1.
    archive = SimpleNamespace(load=lambda identity, component: replace(imported, neuron_id=identity, component_id=component))
    return dict(topology=topology, archive=archive, annotations=annotations)


@pytest.mark.parametrize("disconnected", [False, True])
def test_construct_and_run_with_source_signed_receptors(arguments, disconnected):
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_network(**arguments, disconnected=disconnected)
        assert len(network.projections) == (0 if disconnected else 2)
        assert evidence["simulated_cell_ids"] == ["12", "13", "14"]
        e, i = evidence["contacts"]
        assert (e["reversal_mv"], i["reversal_mv"]) == (0., -80.)
        assert e["signed_weight_us"] > 0 and i["signed_weight_us"] < 0
        assert e["weight_us"] > 0 and i["weight_us"] > 0
        result = network.run(dt=.001*u.ms, duration=.003*u.ms)
        for traces in result.traces.values():
            assert np.isfinite(traces["voltage"].to_decimal(u.mV)).all()


def test_fragment_contact_retained_but_not_simulated(arguments):
    edge = arguments["topology"]["contacts"][1]
    edge.update(construction_ready=False, blockers=["soma-free component"])
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_network(**arguments)
    assert len(network.projections) == 1
    assert len(evidence["nodes"]) == 3 and evidence["simulated_cell_ids"] == ["12", "13"]
    assert evidence["blocked_contacts"][0]["blockers"] == ["soma-free component"]


@pytest.mark.parametrize("key,value", [("excitatory_weight_us", -1.), ("inhibitory_weight_us", np.nan),
    ("delay_ms", -1.), ("max_cv_length_um", 0.), ("currents_na", {"99": 1.}),
    ("currents_na", {"12": np.inf})])
def test_bad_parameters_rejected(arguments, key, value):
    with pytest.raises(ValueError):
        make_h01_network(**arguments, **{key: value})


@pytest.mark.parametrize("case", ["archive", "nodes", "empty", "sign", "type", "verify", "duplicate", "label",
    "coordinate", "component", "hash", "location", "projection", "multisite"])
def test_invalid_contact_evidence_rejected(arguments, case):
    topology = arguments["topology"]
    item = topology["contacts"][0]
    if case == "archive": topology["archive_sha256"] = "bad"
    if case == "nodes": topology["nodes"].append(topology["nodes"][0])
    if case == "empty": topology["contacts"] = []
    if case == "sign": topology["nodes"][0]["dale_sign"] = -1
    if case == "type": item["type"] = 1
    if case == "verify": item["endpoints_verified"] = False
    if case == "duplicate": topology["contacts"].append(deepcopy(item))
    if case == "label": item["pre_proofread_label"] = "999"
    if case == "coordinate": item["pre_placement"]["position_um"] = [1., 0., 0.]
    if case == "component": topology["contacts"][1]["pre_placement"]["component"] = 1
    if case == "hash": item["pre_placement"]["source_swc_sha256"] = "bad"
    if case == "location": item["pre_placement"]["cable_location"][1] += .1
    if case == "projection":
        item["pre_voxel"] = [100000, 0, 0]
        item["pre_placement"]["position_um"] = [800., 0., 0.]
    if case == "multisite":
        other = deepcopy(item)
        other["annotation_id"] = "3"
        other["pre_voxel"] = [400, 0, 0]
        other["pre_placement"]["position_um"] = [3.2, 0., 0.]
        imported = arguments["archive"].load("12", component=0)
        projection = imported.anatomy().project([[3.2, 0., 0.]], max_distance_um=1.)[0]
        other["pre_placement"]["cable_location"] = list(projection.location.evaluate(imported.morphology).points[0])
        topology["contacts"].append(other)
    with brainstate.environ.context(precision=64), pytest.raises(ValueError):
        make_h01_network(**arguments)


def test_e_event_delivers_to_i_only_after_delay(arguments):
    traces = {}
    with brainstate.environ.context(precision=64):
        for disconnected in (False, True):
            network, _ = make_h01_network(**arguments, disconnected=disconnected,
                excitatory_weight_us=.0001, delay_ms=.1, currents_na={"12": 1.})
            result = network.run(dt=.001*u.ms, duration=4.*u.ms, spike_recording="population")
            traces[disconnected] = (
                np.asarray(result.spikes["cell_12"]).ravel(),
                np.asarray(result.traces["cell_13"]["syn_1_g"].to_decimal(u.uS)).ravel(),
                np.asarray(result.traces["cell_13"]["syn_1_voltage"].to_decimal(u.mV)).ravel())
    events, conductance, voltage = traces[False]
    emitted, delivered = np.flatnonzero(events), np.flatnonzero(conductance > 0)
    assert len(emitted) and len(delivered)
    assert delivered[0]-emitted[0] in (100, 101)
    np.testing.assert_array_equal(events, traces[True][0])
    np.testing.assert_array_equal(voltage[:delivered[0]], traces[True][2][:delivered[0]])
    assert not traces[True][1].any()
    assert voltage[delivered[0]+1] > traces[True][2][delivered[0]+1]
