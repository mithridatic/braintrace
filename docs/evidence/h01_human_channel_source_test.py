"""Reject species ambiguity before deriving human channel calibration inputs."""

import pytest
from h01_human_channel_source import select_human_records


def test_human_rows_preserve_qc_and_missing_values_without_mouse_inputs():
    source=[{'Filename':'H20.29.1.nwb','Species':'Human','exclude':True,'tau':None},
            {'Filename':'M20.29.1.nwb','Species':'Mouse','tau':2.}]
    rows,receipt=select_human_records(source,require_species_column=True)
    assert rows == source[:1] and rows[0] is not source[0]
    assert receipt['human_records']==1 and receipt['excluded_nonhuman_records']==1
    assert rows[0]['exclude'] and rows[0]['tau'] is None


def test_documented_filename_rule_for_sodium():
    rows,receipt=select_human_records([{'Filename':'H20.29.1.nwb'},{'Filename':'M20.29.2.nwb'}])
    assert len(rows)==1 and receipt['species_rule']=='documented H/M filename prefix'


@pytest.mark.parametrize('records,explicit',[
    ([{}],False),([{'Filename':'X20.1.nwb'}],False),
    ([{'Filename':'H20.1.nwb','Species':'Mouse'}],True),
    ([{'Filename':'H20.1.nwb'}],True),
    ([{'Filename':'H20.1.nwb','Species':'Unknown'}],True),
    ([{'Filename':'H20.1.nwb'},{'Filename':'H20.1.nwb'}],False),
    ([],False),([{'Filename':'M20.1.nwb'}],False),
])
def test_invalid_source_identity(records,explicit):
    with pytest.raises(ValueError):select_human_records(records,require_species_column=explicit)
