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


def _kept_evidence(tmp_path, kept_ids):
    ids = list(map(str, range(104)))
    contacts = [dict(annotation_id='a', enabled=True, pre_cell='1', post_cell='2'),
                dict(annotation_id='b', enabled=True, pre_cell='2', post_cell='50')]
    evidence, decision = tmp_path/'evidence.json', tmp_path/'decision.json'
    evidence.write_text(json.dumps(dict(status='forward_pass', nonfinite_states=[],
        evidence={'simulated_cell_ids': ids, 'cells': {i: {'id': i} for i in ids}, 'contacts': contacts,
                  'blocked_contacts': []})))
    decision.write_text(json.dumps(dict(kept=kept_ids)))
    return evidence, decision


def test_kept_subset_filters_cells_and_contacts_between_kept_cells():
    evidence = dict(simulated_cell_ids=['1', '2', '3', '50'], cells={i: {} for i in ('1', '2', '3', '50')},
                    contacts=[dict(pre_cell='1', post_cell='2'), dict(pre_cell='2', post_cell='50')], other='kept')
    filtered = subject.kept_subset(evidence, ['3', '1', '2'])
    assert filtered['simulated_cell_ids'] == ['1', '2', '3'] and set(filtered['cells']) == {'1', '2', '3'}
    assert filtered['contacts'] == [dict(pre_cell='1', post_cell='2')] and filtered['other'] == 'kept'
    assert evidence['simulated_cell_ids'] == ['1', '2', '3', '50']   # input untouched
    with pytest.raises(ValueError, match='not simulated cells: 7, 9'):
        subject.kept_subset(evidence, ['1', '9', '7'])
    with pytest.raises(ValueError, match='at least one'):
        subject.kept_subset(evidence, [])


def test_manifest_from_kept_cells_records_count_and_provenance(tmp_path, monkeypatch, capsys):
    evidence, decision = _kept_evidence(tmp_path, ['1', '2', '3'])
    seen = {}
    monkeypatch.setattr(sys, 'argv', ['manifest', '--evidence', str(evidence), '--contacts', str(tmp_path/'c.json'),
        '--archive', str(tmp_path/'source.zip'), '--output', str(tmp_path/'manifest.json'), '--cells', str(decision)])
    (tmp_path/'c.json').write_text('{}')
    monkeypatch.setattr(subject, 'store_asset', lambda *args: 'a'*64)
    monkeypatch.setattr(subject, 'H01Archive', lambda path: path)
    def topology(filtered, *args):
        seen.update(filtered)
        return SimpleNamespace(to_dict=lambda: {'active_contacts': [e['annotation_id'] for e in filtered['contacts']]})
    monkeypatch.setattr(subject, 'topology_from_evidence', topology)
    monkeypatch.setattr(subject, 'numerical_settings', lambda: {'dt_ms': .005})
    subject.main()
    document = json.loads((tmp_path/'manifest.json').read_text())
    assert seen['simulated_cell_ids'] == ['1', '2', '3'] and set(seen['cells']) == {'1', '2', '3'}
    assert [e['annotation_id'] for e in seen['contacts']] == ['a']
    assert document['original_cells'] == 3
    assert document['kept_from']['path'] == str(decision) and document['kept_from']['source_cells'] == 104
    assert len(document['kept_from']['sha256']) == 64
    printed = json.loads(capsys.readouterr().out)
    assert printed['original_cells'] == 3 and printed['anatomical_contacts'] == 1


def test_manifest_from_kept_cells_rejects_unknown_cell(tmp_path, monkeypatch):
    evidence, decision = _kept_evidence(tmp_path, ['1', '999'])
    monkeypatch.setattr(sys, 'argv', ['manifest', '--evidence', str(evidence),
        '--archive', str(tmp_path/'source.zip'), '--output', str(tmp_path/'manifest.json'), '--cells', str(decision)])
    with pytest.raises(ValueError, match='999'):
        subject.main()
    assert not (tmp_path/'manifest.json').exists()
