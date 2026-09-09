"""Population evidence must qualify before creating an initial manifest."""

import json
import sys
from types import SimpleNamespace

import pytest

from examples import h01_arc_manifest as subject


@pytest.mark.parametrize('status,finite,count', [('running', True, 104),
    ('forward_pass', False, 104), ('forward_pass', True, 40)])
def test_incomplete_population_evidence_is_rejected(tmp_path, monkeypatch, status, finite, count):
    evidence = tmp_path/'evidence.json'
    evidence.write_text(json.dumps(dict(status=status, evidence={'simulated_cell_ids': list(map(str, range(count)))},
        nonfinite_states=[] if finite else ['voltage'])))
    monkeypatch.setattr(sys, 'argv', ['manifest', '--evidence', str(evidence),
        '--archive', str(tmp_path/'absent.zip'), '--output', str(tmp_path/'manifest.json')])
    with pytest.raises(ValueError, match='successful 104-cell'):
        subject.main()
    assert not (tmp_path/'manifest.json').exists()


def test_manifest_retains_pinned_assets_and_no_synthetic_initial_contacts(tmp_path, monkeypatch, capsys):
    evidence, contacts = tmp_path/'evidence.json', tmp_path/'contacts.json'
    evidence.write_text(json.dumps(dict(status='forward_pass', nonfinite_states=[],
        evidence={'simulated_cell_ids': list(map(str, range(104)))})))
    contacts.write_text('{}')
    output = tmp_path/'manifest.json'
    monkeypatch.setattr(sys, 'argv', ['manifest', '--evidence', str(evidence), '--contacts', str(contacts),
        '--archive', str(tmp_path/'source.zip'), '--output', str(output)])
    monkeypatch.setattr(subject, 'store_asset', lambda *args: 'a'*64)
    monkeypatch.setattr(subject, 'H01Archive', lambda path: path)
    monkeypatch.setattr(subject, 'topology_from_evidence', lambda *args:
        SimpleNamespace(to_dict=lambda: {'active_contacts': ['one', 'two']}))
    monkeypatch.setattr(subject, 'numerical_settings', lambda: {'dt_ms': .005})
    subject.main()
    document = json.loads(output.read_text())
    assert document['assets'] == ['a'*64] and document['settings']['dt_ms'] == .005
    assert document['asset_root'] == 'assets'
    assert json.loads(capsys.readouterr().out)['synthetic_contacts'] == 0
