"""Circuit contacts, source identities, and compiled event delivery."""
from dataclasses import replace
from types import SimpleNamespace
import brainstate
import brainunit as u
import numpy as np
import pytest
from braincell.filter import AllRegion
from .h01_anatomy_test import imported
from .h01_ei_circuit import make_h01_ei_circuit


def arguments(imported):
    soma = imported.anatomy().cable_neighborhood(30, radius_um=2.)
    return dict(connectivity="illustrative", components={"E": imported, "I": replace(imported, neuron_id="13")},
        annotations=SimpleNamespace(metadata=lambda nid: SimpleNamespace(tags=("pyramidal",) if nid == "12" else ("interneuron",))),
        regions={r: {"soma": soma, "axon": AllRegion()-soma} for r in ("E", "I")},
        region_basis={r: "Synthetic fixture partition, not a measured circuit." for r in ("E", "I")})


@pytest.mark.parametrize("control,count", [("ei",2),("e_only",1),("i_only",1),("disconnected",0)])
def test_controls_change_only_projections(imported, control, count):
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_ei_circuit(**arguments(imported), control=control)
        assert len(network.projections) == count
        assert sum(c["enabled"] for c in evidence["inferred_contacts"]) == count
        for contact in evidence["inferred_contacts"]:
            assert contact["pre_index"] == contact["post_index"] == 0
            assert contact["weight_us"] > 0 and contact["delay_ms"] == .5
        assert evidence["cells"]["E"]["incoming_synapse"]["reversal_mv"] == -80.
        assert evidence["cells"]["I"]["incoming_synapse"]["reversal_mv"] == 0.
        result = network.run(dt=.001*u.ms, duration=.003*u.ms)
        for role in ("E", "I"):
            assert np.isfinite(result.traces[role]["voltage"].to_decimal(u.mV)).all()
            np.testing.assert_array_equal(result.traces[role]["synaptic_conductance"].to_decimal(u.uS), 0.)


@pytest.mark.parametrize("case", ["identical", "roles", "control", "weight", "delay", "currents", "tags"])
def test_invalid_circuit_inputs_fail(imported, case):
    args = arguments(imported)
    if case == "identical": args["components"]["I"] = imported
    if case == "roles": del args["regions"]["I"]
    if case == "control": args["control"] = "bad"
    if case == "weight": args["inhibitory_weight_us"] = -.1
    if case == "delay": args["delay_ms"] = np.nan
    if case == "currents": args["currents_na"] = {"E": 0.}
    if case == "tags": args["annotations"] = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=()))
    with pytest.raises(ValueError):
        make_h01_ei_circuit(**args)


def test_e_event_drives_i_conductance_and_voltage_after_delay(imported):
    traces = {}
    with brainstate.environ.context(precision=64):
        for control in ("disconnected", "e_only"):
            network, _ = make_h01_ei_circuit(**arguments(imported), control=control,
                excitatory_weight_us=.0001, delay_ms=.1)
            result = network.run(dt=.001*u.ms, duration=4.*u.ms, spike_recording="population")
            traces[control] = {"v": np.asarray(result.traces["I"]["voltage"].to_decimal(u.mV)).ravel(),
                "g": np.asarray(result.traces["I"]["synaptic_conductance"].to_decimal(u.uS)).ravel(),
                "pre": np.asarray(result.spikes["E"]).ravel()}
    connected, isolated = traces["e_only"], traces["disconnected"]
    events = np.flatnonzero(connected["pre"])
    delivery = np.flatnonzero(connected["g"] > 0)
    assert len(events) > 0 and len(delivery) > 0
    assert delivery[0]-events[0] in (100, 101)
    np.testing.assert_array_equal(connected["pre"], isolated["pre"])
    np.testing.assert_array_equal(connected["v"][:delivery[0]], isolated["v"][:delivery[0]])
    assert connected["v"][delivery[0]+1] > isolated["v"][delivery[0]+1]
    assert not isolated["g"].any()


@pytest.mark.parametrize("connectivity", ["illustrative", "measured"])
def test_i_event_changes_e_voltage_with_reversal_driving_force(imported, monkeypatch, connectivity):
    traces = {}
    args = arguments(imported)
    args["connectivity"] = connectivity
    if connectivity == "measured":
        from . import h01_ei_circuit as module
        location = imported.anatomy().location(30)
        monkeypatch.setattr(module, "measured_ie_contact", lambda _: (location, location, {"annotation_id": "fixture"}))
    with brainstate.environ.context(precision=64):
        for control in ("disconnected", "i_only"):
            network, _ = make_h01_ei_circuit(**args, control=control,
                currents_na={"E": 1., "I": 1.}, inhibitory_weight_us=.0001, delay_ms=.1)
            result = network.run(dt=.001*u.ms, duration=4.*u.ms, spike_recording="population")
            traces[control] = {"v": np.asarray(result.traces["E"]["voltage"].to_decimal(u.mV)).ravel(),
                "g": np.asarray(result.traces["E"]["synaptic_conductance"].to_decimal(u.uS)).ravel(),
                "pre": np.asarray(result.spikes["I"]).ravel()}
    connected, isolated = traces["i_only"], traces["disconnected"]
    events, delivery = np.flatnonzero(connected["pre"]), np.flatnonzero(connected["g"] > 0)
    assert len(events) > 0 and len(delivery) > 0
    first = delivery[0]
    assert first-events[0] in (100, 101)
    np.testing.assert_array_equal(connected["pre"], isolated["pre"])
    np.testing.assert_array_equal(connected["v"][:first], isolated["v"][:first])
    assert isolated["v"][first] > -80., "This suppression test requires voltage above I reversal."
    assert connected["v"][first+1] < isolated["v"][first+1]
    assert not isolated["g"].any()


def test_measured_default_rejects_unrelated_components(imported):
    args = arguments(imported)
    del args['connectivity']
    with pytest.raises(ValueError, match='pinned'):
        make_h01_ei_circuit(**args)


def test_incoming_voltage_selects_receptor_site_not_soma(imported, monkeypatch):
    from . import h01_ei_circuit as module
    from braincell._multi_compartment.probes import _representative_cv_id
    from braincell.mech import StateProbe, MechanismProbe
    args = arguments(imported)
    args["connectivity"] = "measured"
    soma, receptor = imported.anatomy().location(30), imported.anatomy().location(50)
    monkeypatch.setattr(module, "measured_ie_contact", lambda _: (soma, receptor, {"annotation_id": "fixture"}))
    with brainstate.environ.context(precision=64):
        network, _ = make_h01_ei_circuit(**args)
        cell = network.populations["E"].cell
        cell.init_state()
        points = {}
        for layout in cell.runtime.layouts:
            declaration = cell.runtime.get_layout_mechanism(layout.id)
            if isinstance(declaration, (StateProbe, MechanismProbe)):
                points[declaration.name] = int(layout.point_index[0])
        assert points["incoming_voltage"] == points["synaptic_conductance"]
        assert points["incoming_voltage"] != points["voltage"]
        cv = _representative_cv_id(cell.runtime, point_id=points["incoming_voltage"])
        cell.V.value = u.math.arange(cell.n_cv)[None, :]*u.mV
        assert float(np.asarray(cell.sample_probe("incoming_voltage").to_decimal(u.mV)).ravel()[0]) == cv


@pytest.mark.parametrize("settings", [
    {"pulse_delays_ms": {"E": 2.}},
    {"pulse_durations_ms": {"E": 3., "X": 3.}},
    {"pulse_delays_ms": {"E": -1., "I": 2.}},
    {"pulse_delays_ms": {"E": np.nan, "I": 2.}},
    {"pulse_durations_ms": {"E": 0., "I": 3.}},
    {"pulse_durations_ms": {"E": -1., "I": 3.}},
    {"pulse_durations_ms": {"E": 3., "I": np.inf}},
])
def test_invalid_pulse_timing_fails_before_network_construction(imported, monkeypatch, settings):
    from . import h01_ei_circuit as module
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid timing reached network construction")
    monkeypatch.setattr(module.braincell, "Network", unexpected)
    with pytest.raises(ValueError, match="pulse delays"):
        make_h01_ei_circuit(**arguments(imported), **settings)


def test_explicit_defaults_and_delayed_E_drive(imported):
    results = {}
    configs = {"default": {}, "explicit": {"pulse_delays_ms": {"E": 2., "I": 2.},
        "pulse_durations_ms": {"E": 3., "I": 3.}}, "later": {"pulse_delays_ms": {"E": 3., "I": 2.},
        "pulse_durations_ms": {"E": 1., "I": 3.}}, "short": {"pulse_durations_ms": {"E": .25, "I": 3.}}}
    with brainstate.environ.context(precision=64):
        for name, settings in configs.items():
            network, evidence = make_h01_ei_circuit(**arguments(imported), control="disconnected",
                solver="h01_staggered_scan", currents_na={"E": 1., "I": 0.}, **settings)
            results[name] = network.run(dt=.001*u.ms, duration=2.5*u.ms)
            assert evidence["cells"]["E"]["pulse_delay_ms"] == (3. if name == "later" else 2.)
            assert evidence["cells"]["E"]["pulse_duration_ms"] == {"later": 1., "short": .25}.get(name, 3.)
    for role in ("E", "I"):
        for key in results["default"].traces[role]:
            np.testing.assert_array_equal(u.get_mantissa(results["default"].traces[role][key]),
                                          u.get_mantissa(results["explicit"].traces[role][key]))
    first = results["default"].traces["E"]["voltage"].to_decimal(u.mV).ravel()
    later = results["later"].traces["E"]["voltage"].to_decimal(u.mV).ravel()
    np.testing.assert_array_equal(first[:2000], later[:2000])
    assert np.max(np.abs(first[2000:]-later[2000:])) > 1e-3
    np.testing.assert_array_equal(results["default"].traces["I"]["voltage"].to_decimal(u.mV),
                                  results["later"].traces["I"]["voltage"].to_decimal(u.mV))
    short = results["short"].traces["E"]["voltage"].to_decimal(u.mV).ravel()
    np.testing.assert_array_equal(first[:2250], short[:2250])
    assert np.max(np.abs(first[2250:]-short[2250:])) > 1e-3
    np.testing.assert_array_equal(results["default"].traces["I"]["voltage"].to_decimal(u.mV),
                                  results["short"].traces["I"]["voltage"].to_decimal(u.mV))


@pytest.mark.parametrize('control,count', [('ei', 1), ('i_only', 1), ('e_only', 0), ('disconnected', 0)])
def test_measured_topology_has_only_source_supported_direction(imported, monkeypatch, control, count):
    from . import h01_ei_circuit as module
    args = arguments(imported)
    args['connectivity'] = 'measured'
    location = imported.anatomy().location(30)
    monkeypatch.setattr(module, 'measured_ie_contact', lambda _: (location, location, {'annotation_id': 'fixture'}))
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_ei_circuit(**args, control=control)
        assert len(network.projections) == count
        assert evidence['inferred_contacts'] == []
        assert len(evidence['measured_contacts']) == 1
        contact = evidence['measured_contacts'][0]
        assert (contact['pre'], contact['post']) == ('I', 'E')
        assert contact['enabled'] == bool(count)
        result = network.run(dt=.001*u.ms, duration=.003*u.ms)
        assert np.isfinite(result.traces['E']['voltage'].to_decimal(u.mV)).all()


def test_inhibitory_receptor_parameters_are_recorded(imported):
    with brainstate.environ.context(precision=64):
        _, evidence = make_h01_ei_circuit(**arguments(imported), inhibitory_reversal_mv=-75., inhibitory_tau_ms=4.18)
    assert evidence["cells"]["E"]["incoming_synapse"] == {"name": "inh", "reversal_mv": -75., "tau_ms": 4.18}
    assert evidence["cells"]["I"]["incoming_synapse"]["tau_ms"] == 2.
