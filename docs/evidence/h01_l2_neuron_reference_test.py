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
    ("--sweep", "54"), ("--sweep", "0"),
    ("--ih-density-factor", "0"), ("--ih-density-factor", "nan"),
    ("--leak-factor", "0"), ("--leak-factor", "nan"),
    ("--leak-reversal-shift-mv", "nan"), ("--leak-reversal-shift-mv", "inf"),
    ("--regional-density", "NaTs:axon"), ("--regional-density", "NaTs:apex:1.1"),
    ("--regional-density", "NaTs:axon:-1"), ("--regional-density", "NaTs:all:nan"),
    ("--insert-density", "NaTs:axon"), ("--insert-density", "NaTs:all:1"),
    ("--insert-density", "NaTs:axon:-1"), ("--insert-density", "NaTs:axon:nan"),
    ("--axon-stub-diameter-um", "0"), ("--axon-stub-diameter-um", "nan"), ("--axon-stub-diameter-um", "-2"),
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


@pytest.mark.parametrize("sweep", [43, 50, 53, 54, 55, 56, 52])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("from_candidate", [False, True])
def test_driver_respects_holdout_release(monkeypatch, tmp_path, capsys,
                                       sweep, registered, from_candidate):
    """Calibration inputs reach source loading; sealed 54/55 stop before it; 52 is unknown."""
    import json

    folder = Path(__file__).parent
    monkeypatch.syspath_prepend(str(folder))
    import h01_l2_sweep_export as exporter

    monkeypatch.setattr(exporter, "EVIDENCE", tmp_path)
    if registered:
        for prediction in exporter.SEALED.values():
            (tmp_path / prediction).write_text("{}")
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    argv = ["reference", "--cache", str(tmp_path / "absent"),
            "--output", str(tmp_path / "out")]
    if from_candidate:
        candidate = tmp_path / "candidate.json"
        candidate.write_text(json.dumps({"sweep": sweep}))
        argv += ["--candidate-json", str(candidate)]
    else:
        argv += ["--sweep", str(sweep)]
    monkeypatch.setattr(sys, "argv", argv)
    if sweep == 52:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")
        assert error.value.code == 2
    elif sweep in exporter.SEALED and not registered:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")
        assert error.value.code == 2
        assert f"sealed until {exporter.SEALED[sweep]}" in capsys.readouterr().err
    else:
        with pytest.raises(FileNotFoundError, match="541563728_fit.json"):
            runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")


def test_driver_help_does_not_require_nwb_export_dependency(monkeypatch, capsys):
    """The simulation image reads NPZ and need not install h5py for the seal."""
    folder = Path(__file__).parent
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.delitem(sys.modules, "h01_l2_sweep_export", raising=False)
    monkeypatch.setitem(sys.modules, "h5py", None)
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.setattr(sys, "argv", ["reference", "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")
    assert error.value.code == 0
    out = capsys.readouterr().out
    assert "55" in out and "56" in out and "54" in out


def _fake_neuron(monkeypatch):
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)


def _run(monkeypatch, argv):
    folder = Path(__file__).parent
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setattr(sys, "argv", ["reference", *argv])
    runpy.run_path(str(folder / "h01_l2_neuron_reference.py"), run_name="__main__")


MINIMAL_FIT = {
    "passive": [{"ra": 15., "e_pas": -80., "cm": [{"section": s, "cm": 1.} for s in ("soma", "axon", "dend", "apic")]}],
    "fitting": [{"junction_potential": -14., "sweeps": [69]}],
    "conditions": [{"celsius": 34., "v_init": -80., "erev": [{"section": "soma", "ena": 53., "ek": -107.}]}],
    "genome": [{"section": "soma", "name": "gbar_NaTs", "value": 2., "mechanism": "NaTs"},
               {"section": "soma", "name": "gbar_Ih", "value": 1e-3, "mechanism": "Ih"},
               {"section": "soma", "name": "decay_CaDynamics", "value": 175., "mechanism": "CaDynamics"},
               {"section": "soma", "name": "g_pas", "value": 4e-4, "mechanism": ""},
               {"section": "axon", "name": "g_pas", "value": 2e-4, "mechanism": ""},
               {"section": "dend", "name": "g_pas", "value": 1e-5, "mechanism": ""},
               {"section": "apic", "name": "g_pas", "value": 1e-7, "mechanism": ""}],
}


def _donor(tmp_path, sweeps=(69, 39), fit_text=None, fit_sha256=None):
    import hashlib
    import json

    fit = tmp_path / "527952884_fit.json"
    fit.write_text(json.dumps(MINIMAL_FIT) if fit_text is None else fit_text)
    donor = {"donor_key": "l4-pyramidal-allen-527952884", "model_id": 626170709, "specimen_id": 527952884,
             "fit": fit.name, "fit_sha256": fit_sha256 or hashlib.sha256(fit.read_bytes()).hexdigest(),
             "morphology": "morphology.swc", "sweeps": list(sweeps)}
    path = tmp_path / "donor.json"
    path.write_text(json.dumps(donor))
    return path


def test_donor_json_restricts_sweeps_to_the_donor_registration(monkeypatch, tmp_path, capsys):
    """A sweep outside the donor's list stops before any file is read; the L2 seal does not apply."""
    _fake_neuron(monkeypatch)
    donor = _donor(tmp_path)
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["--cache", str(tmp_path / "absent"), "--output", str(tmp_path / "out"),
                           "--donor-json", str(donor), "--sweep", "50"])
    assert error.value.code == 2 and "not among the donor's registered sweeps" in capsys.readouterr().err
    # Sweep 54 is sealed for the L2 donor; a donor that registers it is not sealed.
    donor = _donor(tmp_path, sweeps=(54,))
    with pytest.raises(FileNotFoundError, match="sweep-54.npz"):
        _run(monkeypatch, ["--cache", str(tmp_path / "absent"), "--output", str(tmp_path / "out"),
                           "--donor-json", str(donor), "--sweep", "54"])


def test_donor_json_rejects_missing_file_and_wrong_fit_hash(monkeypatch, tmp_path, capsys):
    _fake_neuron(monkeypatch)
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["--cache", str(tmp_path), "--output", str(tmp_path / "out"),
                           "--donor-json", str(tmp_path / "absent.json"), "--sweep", "69"])
    assert error.value.code == 2 and "does not exist" in capsys.readouterr().err
    donor = _donor(tmp_path, fit_sha256="0"*64)
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["--cache", str(tmp_path), "--output", str(tmp_path / "out"),
                           "--donor-json", str(donor), "--sweep", "69"])
    assert error.value.code == 2 and "registered sha256" in capsys.readouterr().err


def test_donor_json_reads_fit_and_waveform_from_its_own_directory(monkeypatch, tmp_path):
    """With a matching hash the donor's waveform is the next file the driver opens."""
    _fake_neuron(monkeypatch)
    donor = _donor(tmp_path)
    with pytest.raises(FileNotFoundError) as error:
        _run(monkeypatch, ["--cache", str(tmp_path / "absent"), "--output", str(tmp_path / "out"),
                           "--donor-json", str(donor), "--sweep", "69"])
    assert Path(error.value.filename) == tmp_path / "sweep-69.npz"


def test_default_l2_path_rejects_a_changed_fit_hash(monkeypatch, tmp_path, capsys):
    _fake_neuron(monkeypatch)
    (tmp_path / "541563728_fit.json").write_text("{}")
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["--cache", str(tmp_path), "--output", str(tmp_path / "out"), "--sweep", "50"])
    assert error.value.code == 2 and "2ceca2317ccbd586" in capsys.readouterr().err


def test_help_lists_the_donor_option(monkeypatch, capsys):
    _fake_neuron(monkeypatch)
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, ["--help"])
    assert error.value.code == 0 and "--donor-json" in capsys.readouterr().out


def test_inserted_probes_record_only_soma_inserts_with_registered_fields(monkeypatch):
    folder = Path(__file__).parent
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.syspath_prepend(str(folder))
    driver = types.SimpleNamespace(**runpy.run_path(str(folder / "h01_l2_neuron_reference.py")))

    class Mechanism:
        _ref_ik, _ref_z = "ik-pointer", "z-pointer"

    class Soma:
        sec = "soma[0]"
        KsAHP = Mechanism()

    inserted = [{"mechanism": "KsAHP", "region": "soma", "value": 1e-4},
                {"mechanism": "KsAHP", "region": "axon", "value": 1e-4},
                {"mechanism": "NaTs", "region": "soma", "value": 3.8}]
    probes = driver.inserted_probes(Soma(), inserted, lambda name, sec: name == "KsAHP")
    assert [(p[0], p[1]) for p in probes] == [("soma_KsAHP_ma_cm2", "ik-pointer"), ("soma_KsAHP_z", "z-pointer")]
    assert driver.inserted_probes(Soma(), inserted, lambda name, sec: False) == []


def test_mechanism_parameter_is_parsed_and_applied_to_every_segment(monkeypatch):
    folder = Path(__file__).parent
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.syspath_prepend(str(folder))
    driver = types.SimpleNamespace(**runpy.run_path(str(folder / "h01_l2_neuron_reference.py")))
    assert driver.parse_mechanism_parameter("KsAHP:soma:tau_off:5000") == {
        "mechanism": "KsAHP", "region": "soma", "name": "tau_off", "value": 5000.}
    for bad in ("KsAHP:soma:tau_off", "KsAHP:all:tau_off:1", "KsAHP:soma:tau_off:nan"):
        with pytest.raises(ValueError):
            driver.parse_mechanism_parameter(bad)

    class Mechanism:
        tau_off = 1000.

    class Segment:
        def __init__(self):
            self.KsAHP = Mechanism()

    class Section:
        def __init__(self, name, n=2):
            self._name, self.segments = name, [Segment() for _ in range(n)]

        def name(self):
            return self._name

        def __iter__(self):
            return iter(self.segments)

    sections = [Section("soma[0]"), Section("dend[0]")]
    item = driver.parse_mechanism_parameter("KsAHP:soma:tau_off:5000")
    driver.apply_mechanism_parameters(sections, [item], lambda name, sec: sec.name().startswith("soma"))
    assert [s.KsAHP.tau_off for s in sections[0]] == [5000., 5000.]
    assert [s.KsAHP.tau_off for s in sections[1]] == [1000., 1000.]
    with pytest.raises(ValueError):
        driver.apply_mechanism_parameters(sections, [dict(item, region="dend")], lambda name, sec: sec.name().startswith("soma"))


def test_membrane_area_factor_scales_cm_leak_and_distributed_ih(monkeypatch):
    folder = Path(__file__).parent
    fake_neuron = types.ModuleType("neuron")
    fake_neuron.h = object()
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.syspath_prepend(str(folder))
    driver = types.SimpleNamespace(**runpy.run_path(str(folder / "h01_l2_neuron_reference.py")))

    class Ih:
        gbar = 1e-4

    class Segment:
        def __init__(self):
            self.g_pas, self.Ih = 4e-5, Ih()

    class Section:
        def __init__(self, name, n=2):
            self._name, self.cm, self.segments = name, 2.3, [Segment() for _ in range(n)]

        def name(self):
            return self._name

        def __iter__(self):
            return iter(self.segments)

    sections = [Section("dend[0]"), Section("apic[0]"), Section("soma[0]")]
    factors = [driver.parse_capacitance_factor("dend:2"), driver.parse_capacitance_factor("apic:2")]
    applied = driver.apply_membrane_area_factors(sections, factors, lambda name, sec: sec.name().startswith(("dend", "apic")))
    assert [a["section_count"] for a in applied] == [1, 1]
    for sec in sections[:2]:
        assert sec.cm == pytest.approx(4.6)
        assert all(seg.g_pas == pytest.approx(8e-5) and seg.Ih.gbar == pytest.approx(2e-4) for seg in sec)
    assert sections[2].cm == 2.3 and all(seg.g_pas == 4e-5 and seg.Ih.gbar == 1e-4 for seg in sections[2])
    with pytest.raises(ValueError):
        driver.apply_membrane_area_factors(sections, [driver.parse_capacitance_factor("axon:2")], lambda name, sec: True)
