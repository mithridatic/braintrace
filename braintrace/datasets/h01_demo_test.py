"""Exercise the installed-style demonstration entry point and its limits."""

import json
from types import SimpleNamespace

import brainunit as u
import numpy as np
import pytest

from . import h01_demo, h01
from .h01_test import _archive


def test_passive_initial_voltage_is_floating_point(tmp_path, monkeypatch):
    path, _ = _archive(tmp_path, monkeypatch)
    cell = h01_demo.make_passive_cell(h01.H01Archive(path).load(12, component=0))
    cell.init_state()
    assert np.issubdtype(cell.V.value.dtype, np.floating)


@pytest.mark.parametrize("download", [False, True])
def test_cli_runs_and_writes_reproducible_evidence(tmp_path, monkeypatch, capsys, download):
    path, _ = _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(h01_demo, "fetch_h01", lambda cache: h01.H01Archive(path))
    output = tmp_path / "evidence" / "run.json"
    source = ["--download", str(tmp_path)] if download else ["--archive", str(path)]
    report = h01_demo.main(source + [
        "--neuron", "12", "--component", "0", "--duration-ms", ".1", "--output", str(output),
    ])
    assert report["finite"] and report["steps"] == 4
    assert report["source_points"] == 3
    assert report["voltage_final_mv"] > -65
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)
    assert report["cell_components"] == (0, 2)


@pytest.mark.parametrize("dt", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_timestep_before_download(dt):
    with pytest.raises(SystemExit):
        h01_demo.main(["--download", "unused", "--dt-ms", dt])


@pytest.mark.parametrize("current,duration", [(np.nan, 1), (np.inf, 1), (1, 0), (1, np.nan), (1, -1)])
def test_passive_cell_rejects_invalid_drive(current, duration):
    with pytest.raises(ValueError, match="finite"):
        h01_demo.make_passive_cell(None, current_na=current, duration_ms=duration)


def test_cli_cannot_report_success_for_nonfinite_simulation(tmp_path, monkeypatch):
    path, _ = _archive(tmp_path, monkeypatch)
    fake = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(
        traces={"voltage": np.array([np.nan]) * u.mV},
    ))
    monkeypatch.setattr(h01_demo, "make_passive_cell", lambda *a, **kw: fake)
    with pytest.raises(RuntimeError, match="non-finite"):
        h01_demo.main(["--archive", str(path), "--neuron", "12"])
