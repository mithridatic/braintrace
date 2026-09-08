import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_ie_functional_inhibition as inh  # noqa: E402

DT = .005


def spiking(e_spikes, i_spikes=(), duration=300., ipsp=0., site_gain=100.):
    """Synthetic traces: -60 mV baseline, spikes as one-sample flags, an IPSP after every I spike."""
    n = int(round(duration/DT))
    time = np.arange(n)*DT+DT
    e_events, i_events = np.zeros(n, bool), np.zeros(n, bool)
    e_events[[int(round(t/DT))-1 for t in e_spikes]] = True
    i_events[[int(round(t/DT))-1 for t in i_spikes]] = True
    soma = np.full(n, -60.)
    for t in i_spikes:
        soma += np.where(time >= t+.5, ipsp*np.exp(-(time-t-.5)/4.18), 0.)
    return {"time_ms": time, "E_voltage": soma, "E_incoming_voltage": -60.+(soma+60.)*site_gain,
            "E_events": e_events, "I_events": i_events}


CONTROL = (30., 80., 130., 180., 230., 280.)


def loader_factory(shift=0., ipsp=-.01, halved_jitter=.005, soma_shift=2., soma_ipsp=-.4, drop_last=False):
    control = spiking(CONTROL)
    onsets = inh.between_spike_onsets(CONTROL)
    i_spikes = [o+inh.I_SPIKE_LATENCY_MS for o in onsets]
    measured = spiking([t+shift*(k > 0) for k, t in enumerate(CONTROL)], i_spikes, ipsp=ipsp)
    halved = spiking([t+shift*(k > 0)+halved_jitter*(k > 0) for k, t in enumerate(CONTROL)], i_spikes, ipsp=ipsp)
    soma_spikes = [t+soma_shift*(k > 0) for k, t in enumerate(CONTROL)]
    soma = spiking(soma_spikes[:-1] if drop_last else soma_spikes, i_spikes, ipsp=soma_ipsp)
    arms = {"disconnected": control, "measured": measured, "soma": soma, "measured-halved": halved}
    return lambda arm: arms[arm]


def test_provisional_onsets_are_periodic_and_inside_the_window():
    onsets = inh.provisional_onsets(100.)
    assert onsets[:3] == [20., 45., 70.] and onsets[-1]+inh.I_PULSE_MS < 100.


def test_between_spike_onsets_aim_at_cycle_midpoints():
    onsets = inh.between_spike_onsets([30., 80., 130.])
    assert onsets == pytest.approx([55.-6.78, 105.-6.78])
    assert inh.between_spike_onsets([2., 4.]) == []  # midpoint minus latency is negative
    assert inh.between_spike_onsets([10., 12., 14., 40.]) == pytest.approx([4.22, 20.22])  # 6.22 overlaps 4.22


def test_arm_args_and_command_carry_registered_drive_site_dt_and_train(tmp_path):
    args = inh.arm_args("soma", [20., 45.], 300., cache=tmp_path)
    assert args[args.index("--receptor-site")+1] == "soma" and args[args.index("--dt-ms")+1] == "0.005"
    assert args[args.index("--e-current-na")+1] == "0.6" and args[args.index("--e-pulse-ms")+1] == "300.0"
    assert args[args.index("--inhibitory-weight-us")+1] == "0.0031"
    assert args[-3:] == ["--i-pulse-onsets-ms", "20.0", "45.0"] and args[args.index("--cache")+1] == str(tmp_path/"h01")
    halved = inh.arm_args("measured-halved", [20.], 300., cache=tmp_path)
    assert halved[halved.index("--dt-ms")+1] == "0.0025" and halved[halved.index("--control")+1] == "ei"
    command = inh.command_for("disconnected", [20.], 100., cache=tmp_path, python="py")
    assert command[0] == "py" and command[-1] == str(tmp_path/"h01"/"inh-disconnected-100ms")
    assert "." not in Path(command[-1]).name


def test_sibling_cache_prefers_a_local_archive(tmp_path):
    assert inh.sibling_cache(tmp_path) == tmp_path.parent/"h01-braincell"/".cache"
    (tmp_path/".cache"/"h01").mkdir(parents=True)
    (tmp_path/".cache"/"h01"/"proofread104.zip").write_bytes(b"")
    assert inh.sibling_cache(tmp_path) == tmp_path/".cache"


def test_run_arm_logs_time_command_and_events(tmp_path):
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        stem = Path(command[-1])
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stem.with_suffix(".npz"), **spiking([30.], [25.], duration=40.))

    record = inh.run_arm("measured", [18.22], 40., out=tmp_path/"out", cache=tmp_path, python="py", runner=runner)
    assert calls[0][0] == "py" and record["e_events_ms"] == pytest.approx([30.]) and record["i_events_ms"] == pytest.approx([25.])
    assert record["seconds_per_simulated_ms"] is not None
    inh.run_arm("measured", [18.22], 40., out=tmp_path/"out", cache=tmp_path, python="py", runner=lambda c, **k: None)
    assert len(json.loads((tmp_path/"out"/"run-log.json").read_text())) == 2


def test_cycle_rows_retain_length_shift_i_spikes_and_ipsps():
    loader = loader_factory(shift=.5, ipsp=-.01)
    rows = inh.cycle_rows(loader("measured"), loader("disconnected"))
    assert len(rows) == 5 and rows[0]["cycle_ms"] == pytest.approx(50.)
    assert rows[0]["shift_ms"] == pytest.approx(0.) and rows[1]["shift_ms"] == pytest.approx(.5)
    assert rows[0]["connected_cycle_ms"] == pytest.approx(50.5) and rows[1]["connected_cycle_ms"] == pytest.approx(50.)
    assert len(rows[0]["i_spikes_ms"]) == 1 and 30. < rows[0]["i_spikes_ms"][0] < 80.
    assert rows[0]["soma_ipsp_mv"] == pytest.approx(-.01, abs=1e-6)
    assert rows[0]["site_ipsp_mv"] == pytest.approx(-1., abs=1e-4)


def test_cycle_rows_without_i_spike_or_with_missing_connected_spike():
    control = spiking(CONTROL)
    rows = inh.cycle_rows(spiking(CONTROL[:3]), control)
    assert rows[0]["i_spikes_ms"] == [] and rows[0]["soma_ipsp_mv"] is None
    assert rows[2]["connected_cycle_ms"] is None and rows[3]["shift_ms"] is None


def test_halving_limits_use_mean_range_times_dlf_with_a_floor():
    loader = loader_factory(shift=.5, halved_jitter=.01)
    control = loader("disconnected")
    rows, halved = (inh.cycle_rows(loader(a), control) for a in ("measured", "measured-halved"))
    limits = inh.halving_limits(rows, halved)
    assert limits["resolved"] and limits["shift_ms"] == pytest.approx(.01*4/5*2.95, rel=1e-6)
    assert limits["cycle_ms"] == pytest.approx(.01*1/5*2.95, rel=1e-6)
    same = inh.halving_limits(rows, rows)
    assert same["shift_ms"] == same["cycle_ms"] == DT
    broken = inh.halving_limits(rows, inh.cycle_rows(spiking(CONTROL[:4]), control))
    assert broken["resolved"] is False


def test_score_null_measured_arm_and_functional_soma_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(inh, "EVIDENCE", tmp_path)
    decision = inh.score(loader_factory(shift=0., ipsp=-.008, halved_jitter=.005, soma_shift=2., soma_ipsp=-.4), out=tmp_path/"d")
    assert decision["functional_inhibition"] == {"measured": False, "soma": True, "measured-halved": False}
    predictions = decision["predictions"]
    assert predictions == {"measured_soma_ipsp_below_0.05_mv": True, "measured_no_spike_or_cycle_beyond_limit": True,
                           "soma_ipsp_at_least_0.3_mv": True, "soma_spike_delayed_beyond_limit": True}
    assert decision["placement_hypothesis"].startswith("not rejected")
    assert decision["human_tiers"]["contract_1mv"].startswith("not applicable")
    assert decision["receptor_placement"]["soma"].startswith("inferred")
    assert json.loads((tmp_path/"d"/"decision.json").read_text())["e_drive_na"] == .6
    text = (tmp_path/"h01-ie-inhibition-result.md").read_text()
    assert "| soma |" in text and "inferred perisomatic hypothesis" in text and "not applicable" in text


def test_score_rejects_placement_and_reports_count_change(tmp_path, monkeypatch):
    monkeypatch.setattr(inh, "EVIDENCE", tmp_path)
    decision = inh.score(loader_factory(soma_shift=0., soma_ipsp=-.01, drop_last=True), out=tmp_path/"d")
    assert decision["placement_hypothesis"].startswith("rejected")
    assert decision["verdicts"]["soma"]["count_change"] and decision["functional_inhibition"]["soma"] is True
    assert decision["predictions"]["soma_spike_delayed_beyond_limit"] is False


def test_score_with_a_broken_halving_pair_is_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(inh, "EVIDENCE", tmp_path)
    loader = loader_factory()
    arms = {a: loader(a) for a in inh.ARMS}
    arms["measured-halved"] = spiking(CONTROL[:4], [55.])
    decision = inh.score(lambda a: arms[a], out=tmp_path/"d")
    assert decision["limits"]["resolved"] is False
    assert decision["functional_inhibition"]["measured"] is None
    assert decision["predictions"]["measured_no_spike_or_cycle_beyond_limit"] is None
    assert "none" in inh.render(decision)


def test_main_modes_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(inh, "OUT", tmp_path/"out")
    monkeypatch.setattr(inh, "sibling_cache", lambda root=None: tmp_path)
    monkeypatch.setattr(inh, "EVIDENCE", tmp_path)
    (tmp_path/"h01").mkdir()
    np.savez(tmp_path/"h01"/"inh-disconnected-300ms.npz", **spiking(CONTROL))
    inh.main(["schedule"])
    schedule = json.loads((tmp_path/"out"/"schedule.json").read_text())
    assert schedule["i_pulse_onsets_ms"] == inh.between_spike_onsets(CONTROL)
    ran = []
    monkeypatch.setattr(inh, "run_arm", lambda arm, onsets, duration, **kw: ran.append((arm, onsets, duration, kw["drive_na"])) or {})
    inh.main(["run", "--arm", "soma"])
    inh.main(["run", "--arm", "disconnected", "--drive-na", "0.8"])
    inh.main(["benchmark"])
    assert ran[0] == ("soma", schedule["i_pulse_onsets_ms"], 300., .6)
    assert ran[1] == ("disconnected", inh.provisional_onsets(300.), 300., .8)
    assert ran[2] == ("disconnected", inh.provisional_onsets(100.), 100., .6)
    loader = loader_factory()
    for arm in ("measured", "soma", "measured-halved"):
        np.savez(tmp_path/"h01"/f"inh-{arm}-300ms.npz", **loader(arm))
    inh.main(["score"])
    assert (tmp_path/"out"/"decision.json").exists()


def test_manifest_commands_parse_and_match_the_driver():
    from examples.h01_ei_circuit import build_parser
    manifest = json.loads((Path(inh.EVIDENCE)/"h01-ie-inhibition-manifest.json").read_text(encoding="utf-8"))
    assert manifest["cap"] == 6 and manifest["halving_cap"] == 2 and manifest["prior_evaluations"] == 1  # the completed 100 ms benchmark spent run 1 of 6
    assert manifest["registered_stimulus"]["e_drive_na"] == inh.E_DRIVE_NA == .6
    placeholder = "<i_pulse_onsets_ms from docs/evidence/h01-ie-inhibition/schedule.json>"
    for name, arm, onsets, ms in (("benchmark", "disconnected", inh.provisional_onsets(100.), 100.),
                                  ("disconnected", "disconnected", inh.provisional_onsets(300.), 300.),
                                  ("measured", "measured", [placeholder], 300.), ("soma", "soma", [placeholder], 300.),
                                  ("measured-halved", "measured-halved", [placeholder], 300.)):
        command = manifest["runs"][name]["wrapper_command"]
        assert command == inh.command_for(arm, onsets, ms)
        argv = [("50" if a == placeholder else a) for a in command[3:]]
        parsed = build_parser().parse_args(argv)
        assert parsed.e_current_na == .6 and parsed.inhibitory_weight_us == .0031 and parsed.i_pulse_onsets_ms
        assert parsed.receptor_site == ("soma" if name == "soma" else "measured")
    assert all(v.startswith("not applicable") for v in manifest["human_tiers"].values())
    assert "inferred" in manifest["runs"]["soma"]["placement"]
