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
from .h01_connectivity import select_connectivity, cell_sign
from .h01_network import _nearest_cv, make_h01_network, plan_h01_cells
from .h01_cell_types import DEFAULT_DONOR_KEYS


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
    messages = []
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_network(**arguments, disconnected=disconnected, progress=messages.append)
        assert len(network.projections) == (0 if disconnected else 2)
        assert evidence["simulated_cell_ids"] == ["12", "13", "14"]
        e, i = evidence["contacts"]
        assert (e["reversal_mv"], i["reversal_mv"]) == (0., -80.)
        assert e["signed_weight_us"] > 0 and i["signed_weight_us"] < 0
        assert e["weight_us"] > 0 and i["weight_us"] > 0
        assert messages[0].startswith("Loading cell")
        assert messages[-1].startswith("Construction complete: 3 cells")
        assert all(record["n_compartments"] > 0 for record in evidence["cells"].values())
        assert evidence["donors"] == {identity: DEFAULT_DONOR_KEYS["E" if cell_sign(arguments["annotations"].metadata(identity).tags) == 1 else "I"]
                                      for identity in evidence["simulated_cell_ids"]}
        assert all(evidence["cells"][k]["donor"] == v for k, v in evidence["donors"].items())
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
    ("currents_na", {"12": np.inf}), ("pulse_delay_ms", -1.), ("pulse_duration_ms", 0.),
    ("pulse_duration_ms", np.nan)])
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


def _components(**rows):
    base = dict(components=3, nodes=100, largest_component=0, largest_share=.8, largest_has_soma=True)
    return {"cells": [dict(base, cell_id=identity, **row) for identity, row in rows.items()]}


@pytest.mark.parametrize("control,expected", [("ei", ["1", "2"]), ("e_only", ["1"]), ("i_only", ["2"]), ("disconnected", [])])
def test_control_selects_projections_by_presynaptic_sign(arguments, control, expected):
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_network(**arguments, control=control)
    assert len(network.projections) == len(expected)
    assert evidence["control"] == control
    assert evidence["enabled_contacts"] == expected
    assert evidence["removed_contacts"] == [e for e in ("1", "2") if e not in expected]
    assert [c["enabled"] for c in evidence["contacts"]] == [c["annotation_id"] in expected for c in evidence["contacts"]]
    assert evidence["disconnected"] is (control == "disconnected")
    assert evidence["isolated_cells"] == {} and evidence["cell_order"] == ["12", "13", "14"]


def test_disconnected_alias_and_conflicts(arguments):
    with brainstate.environ.context(precision=64):
        _, evidence = make_h01_network(**arguments, disconnected=True)
    assert evidence["control"] == "disconnected" and evidence["enabled_contacts"] == []
    with pytest.raises(ValueError):
        make_h01_network(**arguments, disconnected=True, control="e_only")
    with pytest.raises(ValueError):
        make_h01_network(**arguments, control="both")


def test_isolated_cell_built_from_largest_soma_component(arguments):
    arguments["topology"]["contacts"].pop()  # Leave 12 -> 13; cell 14 is isolated.
    components = _components(**{"12": {}, "13": {}, "14": dict(largest_component=2, nodes=1000, largest_share=.4,
                                                              components=5)})
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_network(**arguments, include_isolated=True, components=components,
                                             currents_na={"14": .5})
    assert evidence["simulated_cell_ids"] == ["12", "13", "14"] and evidence["cell_order"] == ["12", "13", "14"]
    assert len(network.projections) == 1 and set(network.populations) == {"cell_12", "cell_13", "cell_14"}
    record = evidence["isolated_cells"]["14"]
    assert record["summary"] == "isolated, largest component 2 of 400 nodes, fraction 0.400"
    assert (record["component"], record["component_nodes"], record["cell_nodes"], record["components"]) == (2, 400, 1000, 5)
    assert evidence["cells"]["14"]["input_na"] == .5 and evidence["cells"]["14"]["donor"] == evidence["donors"]["14"]
    assert evidence["cells"]["14"]["n_compartments"] > 0


def test_isolated_ordering_by_component_nodes_then_id_and_prefix(arguments):
    arguments["topology"]["contacts"].pop()
    nodes = arguments["topology"]["nodes"]
    nodes.append(dict(nodes[2], cell_id="9", index=3))
    nodes.append(dict(nodes[2], cell_id="15", index=4))
    arguments["annotations"].metadata = lambda identity: SimpleNamespace(tags=("pyramidal",) if identity in ("12", "9", "15", "14") else ("interneuron",))
    components = _components(**{"14": dict(nodes=100, largest_share=.5), "9": dict(nodes=100, largest_share=.5),
                                "15": dict(nodes=20, largest_share=.5)})
    with brainstate.environ.context(precision=64):
        _, evidence = make_h01_network(**arguments, include_isolated=True, components=components, cells=3)
    assert evidence["cell_order"] == ["12", "13", "15", "9", "14"]
    assert evidence["simulated_cell_ids"] == ["12", "13", "15"]
    assert set(evidence["isolated_cells"]) == {"15"}
    for count in (1, 6):
        with pytest.raises(ValueError):
            make_h01_network(**arguments, include_isolated=True, components=components, cells=count)


@pytest.mark.parametrize("case", ["missing", "no_soma", "no_components"])
def test_isolated_cell_requires_soma_bearing_largest_component(arguments, case):
    arguments["topology"]["contacts"].pop()
    components = {"missing": _components(**{"12": {}}), "no_soma": _components(**{"14": dict(largest_has_soma=False)}),
                  "no_components": None}[case]
    with pytest.raises(ValueError):
        make_h01_network(**arguments, include_isolated=True, components=components)


def test_all_isolated_population_without_contacts(arguments):
    for contact in arguments["topology"]["contacts"]:
        contact["construction_ready"] = False
    components = {row["cell_id"]: row for row in _components(**{"12": {}, "13": {}, "14": {}})["cells"]}
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_network(**arguments, include_isolated=True, components=components)
    assert len(network.projections) == 0 and len(evidence["blocked_contacts"]) == 2
    assert evidence["simulated_cell_ids"] == ["12", "13", "14"] and set(evidence["isolated_cells"]) == {"12", "13", "14"}
    assert evidence["contacts"] == [] and evidence["enabled_contacts"] == []
    assert all(r["output_site"] for r in evidence["cells"].values())


def test_plan_matches_builder_without_loading(arguments):
    arguments["topology"]["contacts"].pop()
    components = _components(**{"14": {}})
    plan = plan_h01_cells(arguments["topology"], include_isolated=True, components=components, cells=3)
    assert plan["incident_cell_ids"] == ["12", "13"] and plan["cell_order"] == ["12", "13", "14"]
    assert plan["simulated_cell_ids"] == ["12", "13", "14"] and set(plan["isolated_cells"]) == {"14"}
    assert plan_h01_cells(arguments["topology"])["simulated_cell_ids"] == ["12", "13"]


def test_per_cell_init_progress_then_run_is_idempotent(arguments):
    from .h01_network_init import init_h01_network_states
    messages = []
    with brainstate.environ.context(precision=64):
        network, _ = make_h01_network(**arguments)
        init = init_h01_network_states(network, progress=messages.append, heartbeat_seconds=60.)
        assert init["initialized_populations"] == ["cell_12", "cell_13", "cell_14"]
        assert [m for m in messages if m.startswith("Initializing cell_")] == [
            f"Initializing cell_{c} ({i}/3, {network.populations['cell_'+c].cell.n_cv} compartments)"
            for i, c in enumerate(("12", "13", "14"), start=1)]
        assert init["init_state_seconds"] > 0. and init["peak_rss_mb"] > 0.
        again = init_h01_network_states(network, progress=messages.append)
        assert again["initialized_populations"] == []
        result = network.run(dt=.001*u.ms, duration=.003*u.ms)
        for traces in result.traces.values():
            assert np.isfinite(traces["voltage"].to_decimal(u.mV)).all()

def test_explicit_component_selection_overrides_largest_without_mutating_inventory():
    from braintrace.datasets.h01_network import _isolated_record
    row = dict(largest_component=0, largest_has_soma=True, largest_share=.8,
               nodes=100, components=3, selected_component=dict(
                   component=2, nodes=10, has_soma=True, source_sha256='a'*64,
                   evidence='source-anchor.json'))
    original = deepcopy(row)
    result = _isolated_record('7', {'7': row})
    assert result['component'] == 2
    assert result['component_nodes'] == 10
    assert result['fraction'] == .1 and result['omitted_fraction'] == .9
    assert result['selection_evidence'] == 'source-anchor.json'
    assert row == original

@pytest.mark.parametrize('change', [None, {}, {'component': -1}, {'component': True},
    {'nodes': 0}, {'nodes': 101}, {'nodes': 1.5}, {'has_soma': False},
    {'source_sha256': 'a'}, {'source_sha256': 'g'*64}, {'evidence': ''}])
def test_invalid_explicit_selection_rejected(change):
    from braintrace.datasets.h01_network import _isolated_record
    selected = dict(component=2, nodes=10, has_soma=True, source_sha256='a'*64, evidence='anchor.json')
    if change is None:
        selected = None
    elif change == {}:
        selected = {}
    else:
        selected.update(change)
    with pytest.raises(ValueError, match='Explicit component'):
        _isolated_record('7', {'7': dict(nodes=100, components=3, selected_component=selected)})


@pytest.mark.parametrize('mismatch', ['none', 'hash', 'component', 'nodes'])
def test_explicit_selection_checked_against_loaded_source(arguments, mismatch):
    arguments['topology']['contacts'].pop()
    source = arguments['archive'].load('14', 2)
    selected = dict(component=2, nodes=len(source.source_rows), has_soma=True,
                    source_sha256=source.source_sha256, evidence='anchor.json')
    if mismatch == 'hash': selected['source_sha256'] = 'f'*64
    if mismatch == 'nodes': selected['nodes'] += 1
    if mismatch == 'component':
        loader = arguments['archive'].load
        arguments['archive'].load = lambda identity, component: loader(identity, 1 if identity == '14' else component)
    components = _components(**{'14': dict(nodes=10000, selected_component=selected)})
    if mismatch != 'none':
        with pytest.raises(ValueError, match='does not match loaded source'):
            make_h01_network(**arguments, include_isolated=True, components=components)
    else:
        with brainstate.environ.context(precision=64):
            _, evidence = make_h01_network(**arguments, include_isolated=True, components=components)
        assert evidence['isolated_cells']['14']['component'] == 2
        assert evidence['cells']['14']['measured_anatomy']['member'] == '14.2.swc'


def test_explicit_isolated_selection_does_not_override_contact_component(arguments):
    from braintrace.datasets.h01_network import plan_h01_cells
    components = _components(**{'12': dict(selected_component=None)})
    plan = plan_h01_cells(arguments['topology'], include_isolated=True, components=components)
    assert plan['isolated_cells'] == {}


def test_nearest_cv_resolves_platform_ties_to_the_lowest_index():
    # Real branch fractions of H01 cell 4157825456: the soma sample is a CV boundary, so the
    # two adjacent midpoints are equidistant up to 1e-13 and the bare rule flipped across platforms.
    midpoints = [0.20454545454548817, 0.3409090909091248]
    assert _nearest_cv(midpoints, 0.27272727272731756) == 0   # Vast rounding
    assert _nearest_cv(midpoints, 0.27272727272732095) == 0   # Windows rounding chose 1 before the fix
    assert _nearest_cv(midpoints, 0.30) == 1                   # a real nearest still wins
    assert _nearest_cv([0.1, 0.5, 0.9], 0.9) == 2
    assert _nearest_cv([0.25], 0.7) == 0
