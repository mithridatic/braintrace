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
    return dict(components={"E": imported, "I": replace(imported, neuron_id="13")},
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


def test_i_event_changes_e_voltage_with_reversal_driving_force(imported):
    traces = {}
    with brainstate.environ.context(precision=64):
        for control in ("disconnected", "i_only"):
            network, _ = make_h01_ei_circuit(**arguments(imported), control=control,
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
