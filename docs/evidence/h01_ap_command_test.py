"""Test source-indexed command and human response selection."""

import numpy as np
import pytest

from h01_ap_command import command_window, human_hwide_records


def test_inclusive_command_window_preserves_sample_clock():
    source = np.arange(1000, dtype=float)
    window = command_window(source, 1, 3)
    assert window['first_index'] == 125 and window['last_index'] == 375
    np.testing.assert_array_equal(window['voltage_mv'], source[125:376])
    np.testing.assert_array_equal(window['time_ms'], np.arange(125,376)/125)


@pytest.mark.parametrize('source,start,end', [
    ([],0,1), ([[1,2]],0,1), ([1,np.nan],0,.008),
    ([1,2],-1,0), ([1,2],.008,0), ([1,2],0,.016),
    ([1,2],0,.001), ([1,2],0,np.inf),
])
def test_invalid_command_windows(source,start,end):
    with pytest.raises(ValueError):
        command_window(source,start,end)


def record(name='H20.29.184.11.41.01.nwb'):
    return dict(Filename=name,excel_row=2,leakSubt=True,Rn=5,distance_pia=500,
                **{f'relative_amp_Hwide_{i:3d}':None if i==200 else 1/i for i in range(1,201)})


def test_correct_waveform_human_selection_and_missing_values():
    r=record();r['relative_amp_Hnarrow_  1']=900
    selected,excluded=human_hwide_records([r,record('M20.29.184.11.41.01.nwb')])
    assert selected[0]['ratios'][0]==1
    assert selected[0]['ratios'][-1] is None
    assert selected[0]['metadata']['leakSubt'] is True
    assert selected[0]['source_excel_row']==2
    assert excluded[0]['reason']=='nonhuman'


@pytest.mark.parametrize('kind',['unknown','missing_header','duplicate_header','nonfinite','duplicate_identity','all_missing'])
def test_invalid_response_inputs(kind):
    r=record();rows=[r]
    if kind=='unknown':r['Filename']='unknown'
    if kind=='missing_header':del r['relative_amp_Hwide_  1']
    if kind=='duplicate_header':r['relative_amp_Hwide_1']=1
    if kind=='nonfinite':r['relative_amp_Hwide_  1']=float('inf')
    if kind=='duplicate_identity':rows.append(dict(r))
    if kind=='all_missing':
        for key in r:
            if key.startswith('relative_amp_'):r[key]=None
    with pytest.raises(ValueError):human_hwide_records(rows)


def test_all_missing_row_excluded_when_other_record_available():
    absent=record('H20.29.184.11.41.02.nwb')
    for key in absent:
        if key.startswith('relative_amp_'):absent[key]=None
    selected,excluded=human_hwide_records([record(),absent])
    assert len(selected)==1 and excluded[0]['reason']=='all Hwide responses missing'
