"""Tests for the scoring parts of the anatomy-transfer runner and its ramp drive."""

from types import SimpleNamespace

import numpy as np
import pytest

from braintrace.datasets.h01_anatomy_test import imported  # noqa: F401 - fixture
from h01_anatomy_transfer_run import (annotations_for, count_verdict, crossings, parse_args, ramp_clamp,
                                      ramp_current_na, ramp_readings, summarize, usable_features, windows)


def _trace(spike_times_ms, duration_ms=100., dt=.01):
    time = np.arange(dt, duration_ms+dt/2, dt)
    voltage = np.full_like(time, -70.)
    for centre in spike_times_ms:
        voltage[(time >= centre) & (time < centre+.5)] = 20.
    return time, voltage


def test_crossings_counts_only_inside_the_pulse():
    time, voltage = _trace([5., 25., 45., 80.])
    observed = crossings(time, voltage, (20., 60.))
    assert observed["count"] == 2
    assert observed["finite"] is True
    assert observed["first_spike_ms"] == pytest.approx(5., abs=.02)
    assert len(observed["crossing_times_ms"]) == 2


def test_crossings_reports_nonfinite_without_raising():
    time, voltage = _trace([30.])
    voltage[-10:] = np.nan
    observed = crossings(time, voltage, (20., 60.))
    assert observed["finite"] is False
    assert observed["count"] == 1


def test_crossings_empty_pulse_window():
    time, voltage = _trace([30.])
    observed = crossings(time, voltage, (200., 300.))
    assert observed == dict(count=0, first_spike_ms=None, finite=True, crossing_times_ms=[])


@pytest.mark.parametrize("model, registered, repeats, exact, band", [
    (10, 10, None, "held", None),
    (11, 10, None, "missed_within_one", None),
    (13, 10, None, "rejected", None),
    (13, 14, (14, 14, 13, 12), "missed_within_one", "held"),
    (16, 14, (14, 14, 13, 12), "rejected", "missed"),
])
def test_count_verdict(model, registered, repeats, exact, band):
    assert count_verdict(model, registered, repeats) == dict(exact=exact, repeat_band=band)


def test_windows_pre_pulse_spike_is_counted_and_excluded_from_the_pulse():
    time, voltage = _trace([5., 30.])
    assert windows(time, voltage, (20., 60.))["pre_pulse_count"] == 1
    assert crossings(time, voltage, (20., 60.))["count"] == 1


def test_windows_rest_mean_equals_the_flat_baseline():
    time, voltage = _trace([30.])
    assert windows(time, voltage, (20., 60.))["rest_mean_mv"] == pytest.approx(-70.)


def test_windows_return_unavailable_when_the_trace_ends_early():
    time, voltage = _trace([30.], duration_ms=100.)
    observed = windows(time, voltage, (20., 60.))
    assert observed["return_mv"] is None and observed["return_available"] is False


def test_windows_return_reads_the_baseline_200ms_after_the_pulse():
    time, voltage = _trace([30.], duration_ms=300.)
    observed = windows(time, voltage, (20., 60.))
    assert observed["return_available"] is True
    assert observed["return_mv"] == pytest.approx(-70.)


def test_windows_post_pulse_spike_is_counted():
    time, voltage = _trace([30., 80.])
    observed = windows(time, voltage, (20., 60.))
    assert observed["post_pulse_count"] == 1 and observed["pre_pulse_count"] == 0


def test_windows_nonfinite_voltage_does_not_raise():
    time, voltage = _trace([30.])
    voltage[-10:] = np.nan
    observed = windows(time, voltage, (20., 60.))
    assert observed["pre_pulse_count"] == 0 and observed["rest_mean_mv"] == pytest.approx(-70.)


def test_windows_reports_the_rest_sd_of_the_pre_pulse_window():
    time, voltage = _trace([30.])
    voltage[(time >= 10.) & (time < 20.)] += np.where(np.arange(int(((time >= 10.) & (time < 20.)).sum())) % 2, 1., -1.)
    observed = windows(time, voltage, (20., 60.), rest_ms=10.)
    assert observed["rest_mean_mv"] == pytest.approx(-70.) and observed["rest_sd_mv"] == pytest.approx(1.)
    assert windows(time, voltage, (0., 60.))["rest_sd_mv"] is None


def test_ramp_current_is_linear_inside_the_window_and_zero_outside():
    current = ramp_current_na([0., 10., 15., 20., 25.], (10., 20.), 2.)
    np.testing.assert_allclose(current, [0., 0., 1., 0., 0.])
    assert ramp_current_na(19.999, (10., 20.), 2.) == pytest.approx(2., abs=1e-3)


def _ramp_trace(spike_times_ms, hold_from_ms=None, duration_ms=120., dt=.01):
    time, voltage = _trace(spike_times_ms, duration_ms=duration_ms, dt=dt)
    if hold_from_ms is not None:
        voltage[time >= hold_from_ms] = -10.
    return time, voltage


def test_ramp_readings_rheobase_is_the_current_at_the_first_crossing():
    time, voltage = _ramp_trace([40., 60., 80., 95.])
    readings = ramp_readings(time, voltage, (20., 100.), 1.6)
    assert readings["rheobase_na"] == pytest.approx(.4, abs=1e-3) and readings["ramp_count"] == 4
    assert readings["block_na"] is None and readings["finite"] and readings["ramp_max_na"] == 1.6
    assert readings["ramp_spike_times_ms"][0] == pytest.approx(40., abs=.02)


def test_ramp_readings_block_is_the_last_crossing_held_above_threshold_to_the_end():
    time, voltage = _ramp_trace([40., 60.], hold_from_ms=70.)
    readings = ramp_readings(time, voltage, (20., 100.), 1.6)
    assert readings["ramp_count"] == 3 and readings["block_na"] == pytest.approx(1., abs=1e-3)
    assert readings["rheobase_na"] == pytest.approx(.4, abs=1e-3)


def test_ramp_readings_short_hold_or_recovery_is_not_a_block():
    time, voltage = _ramp_trace([40., 60.], hold_from_ms=90.)      # 10 ms above threshold: under the hold
    assert ramp_readings(time, voltage, (20., 100.), 1.6)["block_na"] is None
    time, voltage = _ramp_trace([40., 60.], hold_from_ms=70.)
    voltage[(time >= 85.) & (time < 86.)] = -70.                   # a dip below -20 mV before the end
    assert ramp_readings(time, voltage, (20., 100.), 1.6)["block_na"] is None


def test_ramp_readings_without_crossings_and_with_nonfinite_samples():
    time, voltage = _ramp_trace([])
    readings = ramp_readings(time, voltage, (20., 100.), 1.6)
    assert readings["rheobase_na"] is None and readings["block_na"] is None and readings["ramp_count"] == 0
    time, voltage = _ramp_trace([40.])
    voltage[-5:] = np.nan
    readings = ramp_readings(time, voltage, (20., 100.), 1.6)
    assert readings["finite"] is False and readings["ramp_count"] == 1


def test_ramp_clamp_function_is_zero_outside_and_linear_inside_the_window():
    import brainunit as u
    clamp = ramp_clamp(10., 20., .8)
    values = [clamp.fn(t*u.ms).to_decimal(u.nA) for t in (0., 10., 15., 29.99, 30., 40.)]
    np.testing.assert_allclose(values, [0., 0., .2, .8*19.99/20., 0., 0.], atol=1e-9)
    for bad in ((-1., 20., .8), (10., 0., .8), (10., 20., 0.), (10., np.nan, .8)):
        with pytest.raises(ValueError, match="Ramp"):
            ramp_clamp(*bad)


def _soma_trace(imported, clamp):
    import brainstate
    import brainunit as u
    import braincell
    from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
    from braintrace.datasets.h01_network import _regions, _register_cell
    from braintrace.datasets.h01_network_init import init_h01_network_states
    annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=("L2", "pyramidal")))
    with brainstate.environ.context(precision=64, dt=.005*u.ms):
        cell, record = make_h01_ei_cell(imported, annotations, polarity="E", regions=_regions(imported),
            region_basis="test", current_na=0., delay_ms=1., duration_ms=2., pop_size=(1,),
            solver="h01_staggered_calcium_implicit")
        if clamp is not None:
            cell.place(imported.anatomy().soma_location(), clamp)
        network = braincell.Network(name="ramp")
        _register_cell(network, "12", cell, record, imported, {}, 0., lambda _: None)
        init_h01_network_states(network, progress=lambda _: None)
        result = network.run(dt=.005*u.ms, duration=3.*u.ms, spike_recording="population")
        time = np.asarray(result.time.to_decimal(u.ms))+.005
        return time, np.asarray(result.traces["cell_12"]["voltage"].to_decimal(u.mV))[:, 0]


def test_ramp_clamp_drives_a_real_cell_inside_the_compiled_step(imported):
    time, driven = _soma_trace(imported, ramp_clamp(1., 2., 1.))
    _, control = _soma_trace(imported, None)
    assert np.isfinite(driven).all()
    np.testing.assert_allclose(driven[time < 1.], control[time < 1.], atol=1e-9)   # identical before the onset
    late = (time > 2.5) & (time < 3.)
    assert driven[late].mean() > control[late].mean()+1.
    difference = (driven-control)[(time > 1.05) & (time < 1.3)]   # before the small test cell spikes
    assert np.all(np.diff(difference) > 0.) and difference[0] < 2.   # rising from zero, not a step


def _args(**overrides):
    base = dict(cell="c", donor="d", polarity="E", current_na=.3, ramp_na=None, pulse_on_ms=20., pulse_ms=40.,
                duration_ms=100., dt_ms=.01, solver="s", max_cv_um=10., registered_count=2, repeat_counts=None,
                donor_model_count=None, tags=None, cell_table=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def _arrays(spike_times, duration_ms=100.):
    time, voltage = _trace(spike_times, duration_ms=duration_ms)
    return dict(time_ms=time, output_voltage=voltage, voltage=voltage)


def test_summarize_step_run_keeps_the_former_readings_and_adds_rest_sd_and_usable_tier():
    evidence = dict(component=1, source_sha256="x"*64, component_nodes=5, n_compartments=7)
    summary = summarize(_args(), _arrays([30., 45.]), evidence)
    assert summary["drive"] == "step" and summary["ramp"] is None and summary["current_na"] == .3
    assert summary["output_site"]["count"] == 2 and summary["verdict_vs_human"]["exact"] == "held"
    assert summary["rest_sd_mv"] == pytest.approx(0.) and summary["rest_mean_mv"] == pytest.approx(-70.)
    assert summary["usable_tier"]["available"] is True and summary["usable_tier"]["view"]["count"] == 2
    assert summary["peak_mv"] == 20. and summary["pre_pulse_count"] == 0
    assert summary["tags"] == [] and summary["tag_source"] == "proofread annotation cache"


def test_summarize_records_the_tags_the_builder_used_and_their_source(tmp_path):
    evidence = dict(component=1, source_sha256="x"*64, component_nodes=5, n_compartments=7,
                    source_tags=["L2", "interneuron", "neuron"])
    from_table = summarize(_args(cell_table=tmp_path/"c3-segment-properties.json"), _arrays([30.]), evidence)
    assert from_table["tags"] == ["L2", "interneuron", "neuron"]
    assert from_table["tag_source"] == "cell table c3-segment-properties.json"
    explicit = summarize(_args(tags=["L3", "pyramidal"]), _arrays([30.]), evidence)
    assert explicit["tags"] == ["L3", "pyramidal"] and explicit["tag_source"] == "--tags"


def test_summarize_ramp_run_reports_rheobase_block_and_no_human_verdict():
    evidence = dict(component=1, source_sha256="x"*64, component_nodes=5, n_compartments=7)
    time, voltage = _trace([30., 45.], duration_ms=100.)
    voltage[(time >= 50.) & (time < 60.)] = -10.
    voltage[(time >= 60.) & (time < 100.)] = -70.
    arrays = dict(time_ms=time, output_voltage=voltage, voltage=voltage)
    summary = summarize(_args(current_na=None, ramp_na=.8, registered_count=None), arrays, evidence)
    assert summary["drive"] == "ramp" and summary["current_na"] is None and summary["ramp_max_na"] == .8
    assert summary["verdict_vs_human"] is None and summary["verdict_vs_donor_model"] is None
    assert summary["ramp"]["rheobase_na"] == pytest.approx(.2, abs=1e-3)
    assert summary["ramp"]["block_na"] is None and summary["ramp"]["ramp_count"] == 3
    assert summary["ramp"]["soma_site"]["ramp_count"] == 3


def test_usable_features_records_the_import_failure_instead_of_raising(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def failing(name, *args, **kwargs):
        if name == "h01_usable_tier":
            raise ImportError("no h5py")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", failing)
    result = usable_features([0., 1.], [-70., -70.], (0., 1.), "x")
    assert result == dict(available=False, reason="ImportError: no h5py")


@pytest.mark.parametrize("extra, message", [
    (["--current-na", ".3"], "registered-count"),
    (["--ramp-na", "0"], "ramp-na"),
    (["--current-na", ".3", "--ramp-na", "1", "--registered-count", "1"], "not allowed"),
    ([], "required"),
])
def test_parse_args_rejects_incomplete_or_conflicting_drives(extra, message, capsys):
    base = ["--cell", "1", "--donor", "d", "--polarity", "E", "--pulse-on-ms", "20", "--pulse-ms", "40",
            "--duration-ms", "100", "--output", "out.json"]
    with pytest.raises(SystemExit):
        parse_args(base+extra)
    assert message in capsys.readouterr().err


def test_parse_args_ramp_defaults_the_archive_pin_and_accepts_tags():
    from braintrace.datasets.h01 import ARCHIVE_SHA256
    args = parse_args(["--cell", "1", "--donor", "d", "--polarity", "I", "--pulse-on-ms", "270", "--pulse-ms", "1000",
                       "--duration-ms", "1500", "--output", "out.json", "--ramp-na", ".57", "--tags", "interneuron", "L3"])
    assert args.ramp_na == .57 and args.current_na is None and args.registered_count is None
    assert args.archive_sha256 == ARCHIVE_SHA256 and str(args.archive).endswith("proofread104.zip")
    assert args.tags == ["interneuron", "L3"]


def test_parse_args_accepts_a_cell_table_for_c3_ids(tmp_path):
    table = tmp_path/"c3-segment-properties.json"
    args = parse_args(["--cell", "693197378", "--donor", "d", "--polarity", "E", "--pulse-on-ms", "1020",
                       "--pulse-ms", "1000", "--duration-ms", "2300", "--output", "out.json", "--ramp-na", ".93",
                       "--cell-table", str(table), "--cell-table-sha256", "ab"*32])
    assert args.cell_table == table and args.cell_table_sha256 == "ab"*32 and args.tags is None
    assert parse_args(["--cell", "1", "--donor", "d", "--polarity", "E", "--pulse-on-ms", "1", "--pulse-ms", "1",
                       "--duration-ms", "3", "--output", "o", "--ramp-na", "1"]).cell_table is None


def test_annotations_for_prefers_tags_then_the_cell_table_then_the_proofread_cache(tmp_path):
    import hashlib
    import json

    from braintrace.datasets.h01_annotations_test import METADATA
    table = tmp_path/"c3-segment-properties.json"
    payload = json.dumps(METADATA).encode()
    table.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    explicit = annotations_for(SimpleNamespace(tags=["L4", "pyramidal"], cell_table=table, cache=tmp_path,
                                               cell_table_sha256=None))
    assert explicit.metadata("anything").tags == ("L4", "pyramidal")
    from_table = annotations_for(SimpleNamespace(tags=None, cell_table=table, cache=tmp_path, cell_table_sha256=digest))
    assert from_table.metadata("13").tags == ("L3", "interneuron")
    with pytest.raises(KeyError):
        from_table.metadata("99")
    with pytest.raises(ValueError, match="SHA-256"):
        annotations_for(SimpleNamespace(tags=None, cell_table=table, cache=tmp_path, cell_table_sha256="0"*64))
    with pytest.raises(FileNotFoundError):   # the proofread cache is the fallback and this folder has none
        annotations_for(SimpleNamespace(tags=None, cell_table=None, cache=tmp_path/"empty", cell_table_sha256=None))
