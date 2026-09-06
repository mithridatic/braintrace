"""Check driver guards before source files or NEURON setup are accessed."""

import runpy
import sys
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize("flag,value", [
    ("--dt-ms", "0"), ("--dt-ms", "nan"),
    ("--nseg-factor", "0"), ("--nseg-factor", "2"),
    ("--calcium-decay-factor", "0"), ("--calcium-decay-factor", "-1"),
    ("--calcium-decay-factor", "nan"), ("--calcium-decay-factor", "inf"),
    ("--stop-ms", "2019"), ("--stop-ms", "nan"), ("--stop-ms", "inf"),
    ("--cvode-atol", "0"), ("--cvode-atol", "nan"),
    ("--sodium-recovery-factor", "0"), ("--sodium-recovery-factor", "nan"),
    ("--kv3-closing-factor", "0"), ("--kv3-closing-factor", "-1"),
    ("--kv3-closing-factor", "nan"), ("--kv3-closing-factor", "inf"),
    ("--sodium-opening-factor", "0"), ("--sodium-opening-factor", "nan"),
    ("--sodium-opening-factor", "-1"), ("--sodium-opening-factor", "inf"),
    ("--sodium-density-factor", "0"), ("--sodium-density-factor", "nan"),
    ("--sweep", "53"), ("--sweep", "0"),
    ("--ih-density-factor", "0"), ("--ih-density-factor", "nan"),
    ("--leak-factor", "0"), ("--leak-factor", "nan"),
    ("--leak-reversal-shift-mv", "nan"), ("--leak-reversal-shift-mv", "inf"),
    ("--regional-density", "NaTs:axon"), ("--regional-density", "NaTs:apex:1.1"),
    ("--regional-density", "NaTs:axon:-1"), ("--regional-density", "NaTs:all:nan"),
])
def test_invalid_settings_stop_before_model_setup(monkeypatch, tmp_path, capsys, flag, value):
    """Invalid CLI values must fail before the absent cache is read."""
    folder = Path(__file__).parent
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setattr(sys, "argv", ["reference", "--cache", str(tmp_path / "absent"),
                                    "--output", str(tmp_path / "out"), flag, value])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")
    assert error.value.code == 2
    assert "unrecognized arguments" not in capsys.readouterr().err


def test_candidate_json_sets_defaults_and_rejects_unknown_names(monkeypatch, tmp_path, capsys):
    """Flag values from a file are applied before validation; unknown names stop early."""
    folder = Path(__file__).parent
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.syspath_prepend(str(folder))
    good = tmp_path / "good.json"
    good.write_text('{"kv3_closing_factor": 0, "regional_density": ["NaTs:axon:1.3"]}')
    monkeypatch.setattr(sys, "argv", ["reference", "--cache", str(tmp_path / "absent"),
                                    "--output", str(tmp_path / "out"), "--candidate-json", str(good)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")
    assert error.value.code == 2 and "Kv3-closing factor" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text('{"not_a_flag": 1}')
    monkeypatch.setattr(sys, "argv", ["reference", "--cache", str(tmp_path / "absent"),
                                    "--output", str(tmp_path / "out"), "--candidate-json", str(bad)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")
    assert error.value.code == 2 and "not_a_flag" in capsys.readouterr().err
