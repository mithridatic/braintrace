"""Active transfer preserves source identity and exposes its assumptions."""

from types import SimpleNamespace

import brainstate
import brainunit as u
import numpy as np
import pytest

from .h01_active import make_active_cell
from . import h01_active
from .h01_anatomy_test import imported


def annotations(tags=("L2", "pyramidal")):
    return SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=tags))


@pytest.mark.parametrize("aligned", [False, True])
def test_source_bound_transfer_runs_with_explicit_assumptions(imported, aligned):
    with brainstate.environ.context(precision=64):
        cell, evidence = make_active_cell(imported, annotations(), active_radius_um=2., current_na=0.,
                                          align_active_boundaries=aligned)
        result = cell.run(dt=.005*u.ms, duration=.05*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
    assert evidence["inferred_active_region"]["anchor_source_id"] == 30
    assert "inferred_geodesic" in evidence["inferred_active_region"]["policy"]
    assert "pending" in evidence["qualification"]
    assert evidence["measured_anatomy"] == imported.provenance
    assert evidence["parameters"]["channel_voltage_shift_mv"] == -10.
    assert evidence["numerics"]["solver"] == "staggered"
    assert evidence["numerics"]["align_active_boundaries"] == aligned


@pytest.mark.parametrize("tags", [("L2", "interneuron"), ("L5", "pyramidal"), ()])
def test_reject_wrong_cell_class(imported, tags):
    with pytest.raises(ValueError, match="L2/L3"):
        make_active_cell(imported, annotations(tags), active_radius_um=2., current_na=0.)


@pytest.mark.parametrize("kwargs", [{"active_radius_um": 0.}, {"current_na": np.nan},
    {"max_cv_length_um": -1.}, {"sodium_ms_cm2": -1.}, {"duration_ms": 0.}, {"delay_ms": -1.}])
def test_reject_invalid_parameters(kwargs):
    args = {"active_radius_um": 2., "current_na": 0., **kwargs}
    with pytest.raises(ValueError, match="parameters"):
        make_active_cell(None, None, **args)


@pytest.mark.parametrize("kwargs", [{"pulse_count": 0}, {"pulse_count": 1.5},
    {"pulse_count": True}, {"period_ms": 0.}, {"pulse_count": 2, "period_ms": 3.},
    {"pulse_count": 2, "period_ms": 2.}])
def test_reject_invalid_pulse_trains(kwargs):
    with pytest.raises(ValueError, match="Pulse count"):
        make_active_cell(None, None, active_radius_um=2., current_na=0., **kwargs)


def test_repeated_current_has_separate_onsets_and_stops(imported):
    with brainstate.environ.context(precision=64):
        cell, evidence = make_active_cell(imported, annotations(), active_radius_um=2.,
            current_na=.001, sodium_ms_cm2=0., potassium_ms_cm2=0.,
            delay_ms=.5, duration_ms=.2, pulse_count=3, period_ms=1.)
        result = cell.run(dt=.005*u.ms, duration=4.*u.ms)
    voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV)).ravel()
    times = (np.arange(len(voltage))+1)*.005
    slope = np.gradient(voltage, .005)
    # Passive cells must depolarize during each pulse and relax between them.
    on = ((times > .55) & (times < .65) | (times > 1.55) & (times < 1.65)
          | (times > 2.55) & (times < 2.65))
    off = ((times > .9) & (times < 1.4) | (times > 1.9) & (times < 2.4)
           | (times > 2.9) & (times < 3.9))
    assert np.all(slope[on] > 0.)
    assert np.all(slope[off] < 0.)
    np.testing.assert_allclose(voltage[times < .45], -70., atol=1e-8)
    assert evidence["parameters"]["pulse_count"] == 3
    assert evidence["parameters"]["period_ms"] == 1.


def test_cli_records_source_and_finite_trace(imported, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(h01_active, "H01Archive", lambda _: SimpleNamespace(load=lambda *a, **k: imported))
    monkeypatch.setattr(h01_active, "H01Annotations", lambda _: annotations())
    output = tmp_path / "run.json"
    report = h01_active.main(["--archive", "fixture", "--annotations", "fixture",
        "--active-radius-um", "2", "--current-na", "0", "--duration-ms", ".05", "--output", str(output)])
    assert report["result"]["precision_bits"] == 64
    assert len(report["result"]["voltage_mv"]) == 10
    assert output.is_file()
    assert "pending" in report["qualification"]
    assert "finite" in capsys.readouterr().out


def test_cli_rejects_nonfinite_result(imported, tmp_path, monkeypatch):
    monkeypatch.setattr(h01_active, "H01Archive", lambda _: SimpleNamespace(load=lambda *a, **k: imported))
    monkeypatch.setattr(h01_active, "H01Annotations", lambda _: annotations())
    fake = SimpleNamespace(run=lambda **k: SimpleNamespace(traces={"voltage": np.array([np.nan])*u.mV}))
    monkeypatch.setattr(h01_active, "make_active_cell", lambda *a, **k: (fake, {}))
    with pytest.raises(RuntimeError, match="Non-finite"):
        h01_active.main(["--archive", "fixture", "--annotations", "fixture", "--active-radius-um", "2",
                         "--current-na", "0", "--output", str(tmp_path/"bad.json")])
    assert not (tmp_path/"bad.json").exists()


@pytest.mark.parametrize("dt", ["0", "nan", "-1"])
def test_cli_rejects_bad_time_step_before_loading(dt):
    with pytest.raises(SystemExit):
        h01_active.main(["--archive", "missing", "--annotations", "missing", "--active-radius-um", "2",
                         "--current-na", "0", "--dt-ms", dt, "--output", "unused.json"])
