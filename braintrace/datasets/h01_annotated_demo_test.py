"""Executable annotation-driven placement and compiled electrical response."""

from dataclasses import replace
from types import SimpleNamespace

import brainunit as u
import numpy as np
import pytest

from .h01_anatomy_test import imported  # noqa: F401 -- shared source geometry fixture
from .h01_annotations_test import _assets
from .h01_annotations import H01Annotations
from . import h01_annotated_demo as demo


def test_compiled_soma_response_and_region_painting(imported, tmp_path, monkeypatch):
    _assets(tmp_path, monkeypatch)
    store = H01Annotations(tmp_path)
    cell, evidence = demo.make_annotated_cell(imported, store, max_distance_um=.1)
    result = cell.run(dt=.025 * u.ms, duration=1 * u.ms)
    voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    assert voltage.shape == (40,) and np.isfinite(voltage).all()
    assert voltage[-1] > -65
    assert evidence["synapses"]["stimulated_source_row"] == 2
    assert evidence["synapses"]["projection_counts"] == {"projected": 1}
    assert evidence["metadata"]["measurements"]["NSI"] == 388
    assert evidence["synapses"]["post_csv_rows"] == 1
    assert evidence["dendrite_intervals"] > 0
    conductances = [float(mech.params["g_max"].to_decimal(u.mS / u.cm**2))
                    for cv in cell.cvs for mech in cv.density_mech if mech.class_name == "IL"]
    assert .1 in conductances and .2 in conductances
    control, _ = demo.make_annotated_cell(imported, store, max_distance_um=.1, current_na=0)
    baseline = control.run(dt=.025 * u.ms, duration=1 * u.ms).traces["voltage"].to_decimal(u.mV)
    np.testing.assert_allclose(baseline, -65, atol=.002)


def test_no_soma_no_synapse_and_no_strict_dendrite(imported, tmp_path, monkeypatch):
    _assets(tmp_path, monkeypatch)
    store = H01Annotations(tmp_path)
    rows = imported.source_rows.copy()
    rows[rows[:, 1] == 1, 1] = -1
    _, evidence = demo.make_annotated_cell(replace(imported, source_rows=rows), store, max_distance_um=.1)
    assert evidence["dendrite_intervals"] == 0
    rows[:, 1] = -1
    with pytest.raises(ValueError, match="no soma"):
        demo.make_annotated_cell(replace(imported, source_rows=rows), store, max_distance_um=.1)
    monkeypatch.setattr(store, "synapses", lambda *a, **k: ())
    with pytest.raises(ValueError, match="No postsynaptic"):
        demo.make_annotated_cell(imported, store, max_distance_um=.1)


def test_cli_writes_evidence_and_rejects_nonfinite(imported, tmp_path, monkeypatch, capsys):
    _assets(tmp_path, monkeypatch)
    monkeypatch.setattr(demo, "H01Archive", lambda path: SimpleNamespace(load=lambda *a, **k: imported))
    output = tmp_path / "output" / "run.json"
    args = ["--archive", "unused.zip", "--annotations", str(tmp_path), "--max-distance-um", ".1", "--output", str(output)]
    report = demo.main(args)
    assert report["simulation"]["finite"] and output.exists()
    assert report["simulation"]["precision_bits"] == 64
    assert '"stimulated_source_row": 2' in capsys.readouterr().out
    monkeypatch.setattr(demo, "make_annotated_cell", lambda *a, **k: (SimpleNamespace(
        run=lambda **k: SimpleNamespace(traces={"voltage": np.array([np.nan]) * u.mV})), {}))
    with pytest.raises(RuntimeError, match="non-finite"):
        demo.main(args)
