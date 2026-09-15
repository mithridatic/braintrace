"""Independent checks for source identity, notebook scope and held-out responses."""

import hashlib
import json

import h5py
import numpy as np
import pytest

import h01_l1_recording as recording


KEYS = ['SweepNum', 'I-Clamp Holding Enable', 'I-Clamp Holding Level', 'Test QC']


def notebook():
    data = np.full((3, 4, 9), np.nan)
    data[:, 0, :] = 4
    data[0, 1, 0], data[0, 2, 0] = 1, -6.25
    data[1, 2, 0] = -7.5
    data[0, 3, 8] = 1
    return data


def test_global_scope_and_last_finite_value():
    data = notebook()
    assert recording.notebook_value(KEYS, data, 'Test QC', 4, scope=0) is None
    assert recording.notebook_value(KEYS, data, 'Test QC', 4, scope=8) == 1
    assert recording.holding_current_pa(KEYS, data, 4) == -7.5
    assert recording.notebook_value(KEYS, data, 'absent', 4, scope=0) is None
    assert recording.notebook_value(KEYS, data, 'Test QC', 5, scope=8) is None


@pytest.mark.parametrize('enable', [None, 2, 1])
def test_missing_enabled_current_rejected(enable):
    data = notebook()
    data[:, 1:3, :] = np.nan
    if enable is not None:
        data[0, 1, 0] = enable
    with pytest.raises(ValueError, match='unavailable'):
        recording.holding_current_pa(KEYS, data, 4)


def test_explicitly_disabled_current_is_zero():
    data = notebook()
    data[2, 1, 0] = 0
    assert recording.holding_current_pa(KEYS, data, 4) == 0


@pytest.mark.parametrize('keys,data,scope', [(KEYS, np.zeros((1, 4)), 0),
                                           (KEYS, notebook(), 9),
                                           (['bad', *KEYS[1:]], notebook(), 0),
                                           (['SweepNum', 'x', 'x', 'z'], notebook(), 0)])
def test_ambiguous_notebook_rejected(keys, data, scope):
    with pytest.raises(ValueError):
        recording.notebook_value(keys, data, 'Test QC', 4, scope=scope)


@pytest.fixture
def nwb(tmp_path):
    path = tmp_path/'source.nwb'
    with h5py.File(path, 'w') as f:
        for prefix, suffix, unit, conversion, kind in (
                ('acquisition', 'AD0', 'volts', 1e-3, 'CurrentClampSeries'),
                ('stimulus/presentation', 'DA0', 'amperes', 1e-12, 'CurrentClampStimulusSeries')):
            g = f.create_group(f'{prefix}/data_00004_{suffix}')
            g.attrs.update(neurodata_type=kind, sweep_number=4, stimulus_description='synthetic')
            d = g.create_dataset('data', data=np.array([0., 1., 2.]))
            d.attrs.update(unit=unit, conversion=conversion, offset=conversion*2)
            t = g.create_dataset('starting_time', data=[7.])
            t.attrs['rate'] = 1000.
        n = f.create_group('general/labnotebook/ITC18USB_Dev_0')
        n.create_dataset('numericalKeys', data=np.array([KEYS, ['', 'On/Off', 'pA', 'On/Off']],
                                                       dtype=h5py.string_dtype()))
        n.create_dataset('numericalValues', data=notebook())
    return path


def pin(monkeypatch, nwb):
    monkeypatch.setattr(recording, 'SOURCE_SHA256', hashlib.sha256(nwb.read_bytes()).hexdigest())


def test_conversion_offset_clock_and_current(nwb, monkeypatch):
    pin(monkeypatch, nwb)
    arrays, meta = recording.read_calibration(nwb, 4)
    np.testing.assert_allclose(arrays['recorded_voltage_mv'], [2, 3, 4])
    np.testing.assert_allclose(arrays['command_current_pa'], [2, 3, 4])
    np.testing.assert_array_equal(arrays['time_ms'], [0, 1, 2])
    assert meta['absolute_start_s'] == 7
    assert meta['holding_current_pa'] == -7.5
    assert meta['online_global_qc']['Test QC'] == 1
    assert not meta['physiological_qualification']


@pytest.mark.parametrize('fault', ['unit', 'rate', 'start', 'nan', 'mode', 'identity',
                                  'holding_unit', 'negative_rate', 'length'])
def test_invalid_recordings_fail(nwb, monkeypatch, fault):
    with h5py.File(nwb, 'a') as f:
        g = f['stimulus/presentation/data_00004_DA0']
        if fault == 'unit': g['data'].attrs['unit'] = 'volts'
        elif fault == 'rate': g['starting_time'].attrs['rate'] = 500
        elif fault == 'start': g['starting_time'][0] = 8
        elif fault == 'nan': g['data'][0] = np.nan
        elif fault == 'mode': g.attrs['neurodata_type'] = 'VoltageClampSeries'
        elif fault == 'identity': g.attrs['sweep_number'] = 5
        elif fault == 'holding_unit': f['general/labnotebook/ITC18USB_Dev_0/numericalKeys'][1, 2] = 'nA'
        elif fault == 'negative_rate': g['starting_time'].attrs['rate'] = -1
        else:
            del g['data']
            d = g.create_dataset('data', data=[0., 1.])
            d.attrs.update(unit='amperes', conversion=1e-12)
    pin(monkeypatch, nwb)
    with pytest.raises(ValueError): recording.read_calibration(nwb, 4)


def test_source_mismatch_and_holdout_rejected_before_access(nwb):
    with pytest.raises(ValueError, match='SHA256'): recording.read_calibration(nwb, 4)
    for sweep in (*recording.HOLDOUT, 0, 99):
        with pytest.raises(ValueError, match='calibration'):
            recording.read_calibration(nwb.parent/'missing.nwb', sweep)


def test_zero_padding_is_not_recorded_voltage(nwb, monkeypatch):
    with h5py.File(nwb, 'a') as f:
        f['acquisition/data_00004_AD0/data'][...] = [0., 1., 0.]
    pin(monkeypatch, nwb)
    arrays, meta = recording.read_calibration(nwb, 4)
    np.testing.assert_array_equal(arrays['recorded_sample'], [True, True, False])
    assert meta['last_recorded_index'] == 1
    assert meta['trailing_padding_samples'] == 1
    assert np.isnan(arrays['recorded_voltage_mv'][-1])
    assert arrays['source_voltage_mv'][-1] == pytest.approx(2.)


def test_all_zero_response_is_unavailable(nwb, monkeypatch):
    with h5py.File(nwb, 'a') as f:
        f['acquisition/data_00004_AD0/data'][...] = 0.
    pin(monkeypatch, nwb)
    with pytest.raises(ValueError, match='recorded response'):
        recording.read_calibration(nwb, 4)


def test_cli_exports_and_preserves_existing_evidence(nwb, monkeypatch, tmp_path):
    pin(monkeypatch, nwb)
    out = tmp_path/'new'/'sweep4'
    args = ['--nwb', str(nwb), '--sweep', '4', '--output', str(out)]
    recording.main(args)
    meta = json.loads(out.with_suffix('.json').read_text())
    assert meta['arrays_sha256'] == hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest()
    with pytest.raises(FileExistsError): recording.main(args)
