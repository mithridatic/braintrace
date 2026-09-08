"""Trough audit on synthetic traces: sampling, flags, the pre-registered rule, the files."""

import json

import numpy as np
import pytest

import h01_i_trough_audit as audit

GEOMETRY = {"area_um2": 150., "cm_uf_cm2": 2., "location": "soma",
            "neighbours": [{"location": "left", "resistance_mohm": 1., "voltage_key": "axial_neighbour_0_mv"},
                           {"location": "right", "resistance_mohm": 2., "voltage_key": "axial_neighbour_1_mv"}]}


def _trace(spikes_ms, stop_ms=400.):
    """Spikes as narrow bumps to +20 mV, each followed by a 2 ms trough to -80 mV."""
    time = np.arange(0., stop_ms, .01)
    voltage = np.full_like(time, -70.)
    for spike in spikes_ms:
        voltage += 90.*np.exp(-((time-spike)/.2)**2)-10.*np.exp(-((time-spike-3.)/2.)**2)
    return time, voltage


def _data(spikes_ms):
    time, voltage = _trace(spikes_ms)
    return {"time_ms": time, "voltage_mv": voltage, "axon_voltage_mv": voltage+5.,
            "NaTg_h": time/1000., "NaTg_m": 1.-time/1000., "Kv3_1_m": np.full_like(time, .25),
            "total_current_na": np.full_like(time, .27),
            "axial_neighbour_0_mv": np.full_like(time, -60.), "axial_neighbour_1_mv": np.full_like(time, -80.)}


def test_rows_sample_every_probe_at_the_three_offsets():
    data = _data([300., 330.])
    rows = audit.trough_rows(data, GEOMETRY, pulse_ms=(270., 400.))
    assert [(r["cycle"], r["offset_ms"]) for r in rows] == [(1, 0.), (1, 6.), (1, 10.), (2, 0.), (2, 6.), (2, 10.)]
    first = rows[0]
    assert first["trough_ms"] == pytest.approx(303., abs=.05)
    assert first["soma_mv"] == pytest.approx(-80., abs=.05)
    assert first["axon_mv"] == pytest.approx(first["soma_mv"]+5.)
    assert first["soma_h"] == pytest.approx(.303, abs=1e-4)
    assert first["soma_m"] == pytest.approx(.697, abs=1e-4)
    assert first["kv3_m"] == .25
    # Inward-positive axial current: (-60+80)/1 + (-80+80)/2 = 20 nA at the trough.
    assert first["axial_na"] == pytest.approx(20., abs=.1)
    plus_six = rows[1]
    assert plus_six["at_ms"] == pytest.approx(first["trough_ms"]+6.)
    assert plus_six["soma_h"] == pytest.approx(.309, abs=1e-4)
    assert not any(r["after_next_spike"] for r in rows)


def test_samples_after_the_next_spike_are_flagged_not_dropped():
    data = _data([300., 308.])
    rows = audit.trough_rows(data, GEOMETRY, pulse_ms=(270., 400.))
    flagged = {(r["cycle"], r["offset_ms"]): r["after_next_spike"] for r in rows}
    assert flagged[(1, 0.)] is False and flagged[(1, 6.)] is True and flagged[(1, 10.)] is True
    assert flagged[(2, 6.)] is False


def test_availability_is_the_median_h_over_unflagged_plus_six_rows():
    rows = [{"cycle": 1, "offset_ms": 6., "after_next_spike": False, "soma_h": .8},
            {"cycle": 2, "offset_ms": 6., "after_next_spike": False, "soma_h": .6},
            {"cycle": 3, "offset_ms": 6., "after_next_spike": False, "soma_h": .7},
            {"cycle": 3, "offset_ms": 6., "after_next_spike": True, "soma_h": .1},
            {"cycle": 3, "offset_ms": 10., "after_next_spike": False, "soma_h": .9}]
    assert audit.availability(rows) == pytest.approx(.7)
    assert audit.availability([rows[3]]) is None


@pytest.mark.parametrize("h_by_input,arm,reading", [
    ({"019": .75, "027": .72}, "drive", "available"),
    ({"019": .75, "027": .70}, "drive", "available"),
    ({"019": .69, "027": .95}, "drive", "ambiguous"),
    ({"019": .50, "027": .95}, "drive", "ambiguous"),
    ({"019": .49, "027": .95}, "recovery", "limited"),
    ({"019": .80, "027": .10}, "recovery", "limited"),
])
def test_decision_rule_uses_the_smaller_input_median(h_by_input, arm, reading):
    decision = audit.decide(h_by_input)
    assert decision["arm"] == arm and decision["reading"] == reading
    assert decision["h_used"] == min(h_by_input.values())
    assert decision["arm_flags"] == (audit.DRIVE_FLAGS if arm == "drive" else audit.RECOVERY_FLAGS)
    assert decision["decision"]


def test_decision_refuses_a_missing_input():
    with pytest.raises(ValueError, match="no unflagged"):
        audit.decide({"019": .8, "027": None})


def test_audit_writes_rows_decision_and_pages(tmp_path):
    folder = tmp_path/"runs"
    folder.mkdir()
    for stem, spikes in (("fin-019", [300., 330.]), ("fin-027", [300., 320.]), ("src-019", [300.]), ("src-027", [300.])):
        np.savez(folder/f"{stem}.npz", **_data(spikes))
        (folder/f"{stem}.json").write_text(json.dumps({"charge_balance_geometry": GEOMETRY, "spike_times_ms": spikes}))
    (folder/"fin.candidate.json").write_text(json.dumps({"sodium_h_tau_factor": .15, "nseg_factor": 9}))
    report = audit.audit(folder, stems={"finalist": "fin", "source": "src"}, pulse_ms=(270., 400.))
    assert set(report["rows"]) == {"finalist", "source"} and set(report["rows"]["finalist"]) == {"019", "027"}
    assert report["decision"]["arm"] == "recovery"  # h = t/1000 sits near 0.31 at every trough + 6 ms
    assert report["decision"]["finalist_flags"] == {"sodium_h_tau_factor": .15, "nseg_factor": 9}
    assert report["source_control"]["027"] == pytest.approx(.309, abs=1e-3)
    page = audit.render(report)
    assert "| cycle | offset_ms |" in page and "recovery" in page
    out = tmp_path/"out"
    audit.main(["--folder", str(folder), "--finalist", "fin", "--source", "src", "--out", str(out/"audit"),
                "--decision", str(out/"stage-0-decision.json"), "--pulse-ms", "270", "400"])
    written = json.loads((out/"audit.json").read_text())
    assert written["decision"]["arm"] == "recovery"
    decision = json.loads((out/"stage-0-decision.json").read_text())
    assert decision["stage"] == "0" and decision["arm"] == "recovery" and decision["decision"]
    assert decision["candidate_flags"]["sodium_h_recovery_factor"] == .5
    assert (out/"audit.md").exists()


def test_candidate_flags_merge_the_arm_onto_the_finalist():
    drive = audit.candidate_flags({"a": 1, "sodium_h_recovery_factor": 1.}, audit.DRIVE_FLAGS)
    assert drive == {"a": 1, "sodium_h_recovery_factor": 1., "scale": ["NaTg:axon:1.5"]}
    recovery = audit.candidate_flags({"a": 1, "sodium_h_recovery_factor": 1.}, audit.RECOVERY_FLAGS)
    assert recovery == {"a": 1, "sodium_h_recovery_factor": .5}
