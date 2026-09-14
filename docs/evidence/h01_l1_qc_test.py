"""Protect missing QC values and reject unpinned inputs before processing."""

import json

import numpy as np
import pytest

from h01_l1_qc import json_values, main


def test_missing_readings_are_null_not_passing_zero():
    data = {'readings': np.array([np.nan, np.inf, 2.]),
            'nested': (np.float32(3), np.bool_(False), 'unavailable')}
    converted = json_values(data)
    assert converted == {'readings': [None, None, 2.], 'nested': [3., False, 'unavailable']}
    json.dumps(converted, allow_nan=False)


def test_wrong_source_and_existing_output_stop_before_ipfx(tmp_path):
    source, output = tmp_path/'source.nwb', tmp_path/'result.json'
    source.write_bytes(b'wrong specimen')
    args = ['--nwb', str(source), '--output', str(output), '--cache', str(tmp_path/'cache')]
    with pytest.raises(ValueError, match='Wrong source'): main(args)
    output.write_text('{}')
    with pytest.raises(FileExistsError): main(args)
