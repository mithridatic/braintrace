"""Direct-voltage recording must preserve the circuit's I response."""
import json
from types import SimpleNamespace

import brainstate
import brainunit as u
import numpy as np
import pytest
from braintrace.datasets.h01_anatomy_test import imported
from braintrace.datasets.h01_ei_circuit_test import arguments
from braintrace.datasets.h01_ei_circuit import make_h01_ei_circuit
from docs.evidence import h01_i_all_cv as recorder


def test_full_cv_recorder_matches_network_and_additional_probes(imported):
    with brainstate.environ.context(precision=64):
        args = dict(**arguments(imported), control="i_only", solver="h01_staggered_scan",
                    currents_na={"E": 1., "I": 1.})
        network, _ = make_h01_ei_circuit(**args)
        reference = network.run(dt=.001*u.ms, duration=4.*u.ms, spike_recording="population")
        network, evidence = make_h01_ei_circuit(**args)
        cell = network.populations["I"].cell
        recorder._add_gate_probes(cell, imported.anatomy().soma_location())
        times, actual = recorder._record(cell, dt_ms=.001, duration_ms=4.)
        np.testing.assert_allclose(times.to_decimal(u.ms), reference.time.to_decimal(u.ms)+.001, rtol=0, atol=1e-14)
        for name, unit in (("voltage", u.mV), ("output_voltage", u.mV), ("synaptic_conductance", u.uS)):
            np.testing.assert_allclose(actual[name].to_decimal(unit), reference.traces["I"][name].to_decimal(unit), rtol=1e-10, atol=1e-9)
        assert np.asarray(actual["events"]).any()
        np.testing.assert_array_equal(actual["events"], reference.spikes["I"])
        cv = evidence["cells"]["I"]["output_site"]["output_cv_id"]
        np.testing.assert_array_equal(actual["all_voltage"][..., cv].to_decimal(u.mV).ravel(), actual["output_voltage"].to_decimal(u.mV).ravel())
        assert actual["all_voltage"].shape == (4000, 1, cell.n_cv)
        for gate in ("NaTg_m", "NaTg_h"):
            assert np.isfinite(actual[gate]).all()
            assert np.asarray(actual[gate]).min() >= 0 and np.asarray(actual[gate]).max() <= 1
        assert float(cell.current_time.to_decimal(u.ms)) == 4.


@pytest.mark.parametrize("dt,duration", [(0,1), (-1,1), (1,0), (np.nan,1), (1,np.inf)])
def test_reject_invalid_time_before_initialization(dt,duration):
    with pytest.raises(ValueError, match="positive and finite"):
        recorder._record(None, dt_ms=dt, duration_ms=duration)


@pytest.mark.parametrize("restore", [False, True])
def test_cli_exports_unit_scaled_arrays_and_cv_mapping(imported, monkeypatch, tmp_path, restore):
    from braintrace.datasets import h01, h01_annotations, h01_ei_circuit
    from examples import h01_ei_candidates
    monkeypatch.setattr(h01, "H01Archive", lambda _: SimpleNamespace(load=lambda *a, **k: imported))
    monkeypatch.setattr(h01_annotations, "H01Annotations", lambda _: None)
    monkeypatch.setattr(h01_ei_candidates, "label_partition", lambda _: (arguments(imported)["regions"]["I"], "fixture"))
    def build(*a, **k):
        return make_h01_ei_circuit(**arguments(imported), control="i_only", solver="h01_staggered_scan")
    monkeypatch.setattr(h01_ei_circuit, "make_h01_ei_circuit", build)
    prefix = tmp_path/"direct"
    monkeypatch.setattr("sys.argv", ["recorder", "--output", str(prefix), "--duration-ms", ".02", "--dt-ms", ".001"] + (["--restore-closing"] if restore else []))
    recorder.main()
    with np.load(prefix.with_suffix(".npz")) as arrays:
        record = json.loads(prefix.with_suffix(".json").read_text())
        assert arrays["all_voltage"].shape == (20, 1, len(record["cv_columns"]))
        assert arrays["all_voltage"].max() < -50
        assert arrays["time_ms"][-1] == .02
        assert set(arrays.files) == {"all_voltage", "voltage", "output_voltage", "synaptic_conductance", "events", "NaTg_m", "NaTg_h", "time_ms"}
        assert ("diagnostic_override" in record["cells"]["I"]) == restore


def test_closing_override_leaves_opening_unchanged_and_restores_on_error():
    from braintrace.datasets.h01_pv_channels import _CHANNELS
    from braintrace.datasets.h01_pv_channels_test import _ions
    cls = _CHANNELS["NaTg"]
    original = cls.f_h_tau
    with brainstate.environ.context(precision=64):
        channel = cls(size=1, g_max=1.*u.mS/u.cm**2, h_close=.15, h_slope=5.)
        voltage = u.math.asarray([-20.])*u.mV
        ions = _ions("NaTg", 1e-4)
        channel.init_state(voltage, *ions)
        for gate_value, ratio in ((1., 1/.15), (0., 1.)):
            channel.h.value = u.math.asarray([gate_value])
            before = original(channel, voltage, *ions)
            with recorder._source_closing(True):
                np.testing.assert_allclose(channel.f_h_tau(voltage, *ions), before*ratio, rtol=1e-14)
            assert cls.f_h_tau is original
        with pytest.raises(RuntimeError):
            with recorder._source_closing(True):
                raise RuntimeError("test cleanup")
        assert cls.f_h_tau is original
