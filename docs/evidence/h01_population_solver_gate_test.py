"""Reject hidden model changes while checking a numerical solver transition."""
from copy import deepcopy
import json
import sys
import pytest
from docs.evidence import h01_population_solver_gate as gate


def records():
    ids=[str(i) for i in range(104)]
    old=dict(simulated_cell_ids=ids,donors={i:'donor' for i in ids},
        compartments_by_cell={i:1 for i in ids},n_compartments=104,control='ei',
        contacts=[{'annotation_id':'one','weight':.01}],enabled_contacts=['one'],removed_contacts=[],
        cells={i:dict(solver='h01_staggered_scan',geometry=[1,2],current=1.) for i in ids})
    new=deepcopy(old)
    for cell in new['cells'].values(): cell['solver']='h01_staggered_calcium_implicit'
    return old,new


def test_accepts_only_solver_change():
    old,new=records()
    new['construction_seconds']=5
    result=gate.audit_solver_transition(old,new)
    assert result['status']=='passed'
    assert not result['runtime_qualified']


@pytest.mark.parametrize('field',['geometry','current','extra'])
def test_rejects_each_other_cell_change(field):
    old,new=records();new['cells']['0'][field]='changed'
    assert gate.audit_solver_transition(old,new)['status']=='failed'


@pytest.mark.parametrize('field',['simulated_cell_ids','donors','compartments_by_cell',
    'n_compartments','control','contacts','enabled_contacts','removed_contacts','cells'])
def test_rejects_missing_model_fields(field):
    old,new=records();del new[field]
    assert gate.audit_solver_transition(old,new)['status']=='failed'


@pytest.mark.parametrize('kind',['duplicate','short','old_solver','new_solver','missing_old_cell'])
def test_rejects_invalid_identity_or_solver(kind):
    old,new=records()
    if kind=='duplicate': old['simulated_cell_ids'][-1]='0'
    if kind=='short': old['simulated_cell_ids'].pop()
    if kind=='old_solver': old['cells']['0']['solver']='unknown'
    if kind=='new_solver': new['cells']['0']['solver']='h01_staggered_scan'
    if kind=='missing_old_cell': del old['cells']['0']
    assert gate.audit_solver_transition(old,new)['status']=='failed'


@pytest.mark.parametrize('fail',[False,True])
def test_cli_retains_both_hashes_and_failure(tmp_path,monkeypatch,fail):
    old,new=records()
    if fail: new['cells']['0']['current']=2.
    for name,r in [('previous',old),('current',new)]:
        (tmp_path/(name+'.json')).write_text(json.dumps(r))
    output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['gate','--previous',str(tmp_path/'previous.json'),
        '--current',str(tmp_path/'current.json'),'--output',str(output)])
    if fail:
        with pytest.raises(SystemExit): gate.main()
    else: gate.main()
    result=json.loads(output.read_text())
    assert result['status']==('failed' if fail else 'passed')
    assert all(len(v['sha256'])==64 for v in result['inputs'].values())

