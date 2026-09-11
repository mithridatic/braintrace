"""Failure evidence and resource-limit validation for the diagnostic runner."""

import json
import sys

import pytest

from examples import h01_arc_probe


def test_failed_construction_keeps_phase_evidence(tmp_path, monkeypatch):
    output = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--cells", "4", "--cache", str(tmp_path),
                                     "--output", str(output)])
    monkeypatch.setattr(h01_arc_probe, "H01Archive", lambda _: object())
    monkeypatch.setattr(h01_arc_probe, "H01Annotations", lambda _: object())
    def fail(*args, **kwargs):
        raise ValueError("invalid pinned population")
    monkeypatch.setattr(h01_arc_probe, "make_h01_network", fail)
    h01_arc_probe.main()
    result = json.loads(output.read_text())
    assert result["status"] == "blocked"
    assert result["error"] == "invalid pinned population"
    assert result["phases"]["construction"]["status"] == "failed"
    assert result["phases"]["construction"]["seconds"] >= 0
    assert result["limits"] == {"wall_seconds": 900., "rss_gib": 16.}


@pytest.mark.parametrize("flag,value", [("--rss-limit-gib", "0"),
    ("--rss-limit-gib", "nan"), ("--wall-limit-seconds", "-1")])
def test_invalid_limits_rejected_before_construction(tmp_path, monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["probe", "--cells", "4", "--cache", str(tmp_path),
        "--output", str(tmp_path/"unused.json"), flag, value])
    with pytest.raises(SystemExit) as exc:
        h01_arc_probe.main()
    assert exc.value.code == 2
    assert not (tmp_path/"unused.json").exists()


def test_sparse_probe_uses_grouped_muon_on_actual_cable(tmp_path):
    from braintrace.datasets.h01_network_step_test import _contact_network
    from examples.pp_prop.h01_arc_model import H01ArcModel
    model = H01ArcModel(_contact_network(tmp_path), ['5805562981', '5965472721'])
    report = {}
    phases = []
    def phase(name, call):
        value = call()
        phases.append(name)
        return value
    h01_arc_probe._sparse_learning_probe(model, report, phase)
    assert phases == ['sparse_pp_prop_compile', 'sparse_muon_compile_and_update', 'sparse_muon_warm_update']
    groups = report['optimizer']['groups']
    assert groups['input']['algorithm'] == groups['recurrent']['algorithm'] == 'masked_muon'
    assert groups['readout_weight']['algorithm'] == 'muon'
    assert groups['readout_bias']['algorithm'] == 'adamw'
    assert all(group['weight_decay'] == .1 for group in groups.values())
    assert report['learning_probe']['finite']
    assert report['learning_probe']['updates'] == 2
    assert report['learning_probe']['cold_gradient_norm'] > 0
    assert set(report['learning_probe']['changed_parameters']) == {
        'input', 'recurrent', 'readout_weight', 'readout_bias'}
