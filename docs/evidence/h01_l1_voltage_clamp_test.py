"""Check source binding, electrode command reconstruction and missing evidence."""

import hashlib
import json

import h5py
import numpy as np
import pytest

import h01_l1_voltage_clamp as vc


@pytest.fixture
def nwb(tmp_path):
    path = tmp_path / 'source.nwb'
    with h5py.File(path, 'w') as f:
        for root, suffix, kind, unit, factor, samples in (
                ('acquisition', 'AD0', 'VoltageClampSeries', 'amperes', 1e-12, [0., 1., 0., 3.]),
                ('stimulus/presentation', 'DA0', 'VoltageClampStimulusSeries', 'volts', 1e-3,
                 [0., 40., 40., 0.])):
            g = f.create_group(f'{root}/data_00098_{suffix}')
            g.attrs.update(neurodata_type=kind, sweep_number=98,
                           stimulus_description=vc.FAMILIES[98])
            d = g.create_dataset('data', data=samples)
            d.attrs.update(unit=unit, conversion=factor, offset=2 * factor)
            t = g.create_dataset('starting_time', data=[7.])
            t.attrs['rate'] = 1000.
        n = f.create_group('general/labnotebook/ITC18USB_Dev_0')
        n.create_dataset('numericalKeys', data=np.array([
            ['SweepNum', 'V-Clamp Holding Enable', 'V-Clamp Holding Level', 'RsComp Enable'],
            ['', 'On/Off', 'mV', 'On/Off']], dtype=h5py.string_dtype()))
        values = np.full((2, 4, 9), np.nan)
        values[:, 0, :] = 98
        values[0, 1:, 0] = [1., -89., 0.]
        values[1, 2, 0] = -90.
        values[1, 2, 8] = 200.  # Global scope is deliberately incompatible.
        n.create_dataset('numericalValues', data=values)
    return path


def pin(monkeypatch, path):
    monkeypatch.setattr(vc, 'SOURCE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())


def test_holding_is_added_once_after_si_conversion(nwb, monkeypatch):
    pin(monkeypatch, nwb)
    a, m = vc.read_sweep(nwb, 98)
    np.testing.assert_allclose(a['total_current_pa'], [2, 3, 2, 5])
    np.testing.assert_allclose(a['dac_voltage_mv'], [2, 42, 42, 2])
    np.testing.assert_allclose(a['command_voltage_mv'], [-88, -48, -48, -88])
    np.testing.assert_array_equal(a['time_ms'], [0, 1, 2, 3])
    assert a['holding_voltage_mv'] == -90
    assert m['absolute_start_s'] == 7
    assert m['instrument']['RsComp Enable']['value'] == 0
    assert not m['channel_kinetics_qualified'] and not m['actual_patch_voltage_measured']
    assert not m['junction_correction_applied'] and not m['leak_subtraction_applied']
    assert m['command_segments'][1] == dict(start_index=1, stop_index=3,
                                           start_ms=1., stop_ms=3., command_mv=-48.)


@pytest.mark.parametrize('fault', ['missing_enable', 'unknown_enable', 'missing_holding',
                                  'holding_unit', 'terminal_zero', 'mode', 'identity',
                                  'family', 'unit', 'length', 'rate', 'start', 'nan',
                                  'bad_rate', 'bad_clock'])
def test_invalid_or_ambiguous_inputs_rejected(nwb, monkeypatch, fault):
    with h5py.File(nwb, 'a') as f:
        g = f['stimulus/presentation/data_00098_DA0']
        nb = f['general/labnotebook/ITC18USB_Dev_0']
        if fault == 'missing_enable': nb['numericalValues'][:, 1, 0] = np.nan
        elif fault == 'unknown_enable': nb['numericalValues'][:, 1, 0] = 2
        elif fault == 'missing_holding': nb['numericalValues'][:, 2, 0] = np.nan
        elif fault == 'holding_unit': nb['numericalKeys'][1, 2] = 'V'
        elif fault == 'terminal_zero': f['acquisition/data_00098_AD0/data'][-1] = 0
        elif fault == 'mode': g.attrs['neurodata_type'] = 'CurrentClampStimulusSeries'
        elif fault == 'identity': g.attrs['sweep_number'] = 99
        elif fault == 'family': g.attrs['stimulus_description'] = 'setup'
        elif fault == 'unit': g['data'].attrs['unit'] = 'amperes'
        elif fault == 'rate': g['starting_time'].attrs['rate'] = 500
        elif fault == 'start': g['starting_time'][0] = 8
        elif fault == 'nan': g['data'][0] = np.nan
        elif fault == 'bad_rate': g['starting_time'].attrs['rate'] = 0
        elif fault == 'bad_clock': g['starting_time'][0] = np.inf
        else:
            del g['data']
            d = g.create_dataset('data', data=[0., 40.])
            d.attrs.update(unit='volts', conversion=1e-3)
    pin(monkeypatch, nwb)
    with pytest.raises(ValueError): vc.read_sweep(nwb, 98)


def test_disabled_holding_and_interior_zero_current_preserved(nwb, monkeypatch):
    with h5py.File(nwb, 'a') as f:
        n = f['general/labnotebook/ITC18USB_Dev_0']
        n['numericalValues'][:, 1, 0] = 0
        n['numericalValues'][:, 2, 0] = np.nan
        f['acquisition/data_00098_AD0/data'].attrs['offset'] = 0
    pin(monkeypatch, nwb)
    a, _ = vc.read_sweep(nwb, 98)
    assert a['holding_voltage_mv'] == 0
    np.testing.assert_array_equal(a['command_voltage_mv'], a['dac_voltage_mv'])
    np.testing.assert_array_equal(a['total_current_pa'], [0, 1, 0, 3])


def test_wrong_hash_and_nonchannel_sweeps_fail_before_access(nwb):
    with pytest.raises(ValueError, match='SHA256'): vc.read_sweep(nwb, 98)
    for sweep in [0, 3, 4, 97, 170]:
        with pytest.raises(ValueError, match='registered'):
            vc.read_sweep(nwb.parent / 'missing.nwb', sweep)


@pytest.mark.parametrize('values,rate', [([], 1000), ([[1]], 1000), ([1, np.nan], 1000),
                                       ([1], 0), ([1], np.inf)])
def test_bad_segment_inputs(values, rate):
    with pytest.raises(ValueError): vc.command_segments(values, rate)


def test_segments_preserve_small_edges_and_final_sample():
    segments = vc.command_segments(np.array([0, 1e-9, 1e-9, 0]), 2000)
    assert [s['stop_index'] - s['start_index'] for s in segments] == [1, 2, 1]
    assert segments[-1]['stop_ms'] == 2.
    assert segments[1]['command_mv'] == 1e-9
    assert len(vc.command_segments(np.array([3, 3]), 1000)) == 1


def test_export_hash_and_no_overwrite(nwb, monkeypatch, tmp_path):
    pin(monkeypatch, nwb)
    out = tmp_path / 'new' / 'sweep98'
    args = ['--nwb', str(nwb), '--sweep', '98', '--output', str(out)]
    vc.main(args)
    m = json.loads(out.with_suffix('.json').read_bytes())
    assert m['arrays_sha256'] == hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest()
    with np.load(out.with_suffix('.npz')) as a:
        np.testing.assert_allclose(a['command_voltage_mv'], [-88, -48, -48, -88])
    with pytest.raises(FileExistsError): vc.main(args)


def test_pax6_source_is_separate_and_identity_preserved(nwb, monkeypatch, tmp_path):
    with h5py.File(nwb, 'a') as f:
        for root, suffix in [('acquisition', 'AD0'), ('stimulus/presentation', 'DA0')]:
            f.move(f'{root}/data_00098_{suffix}', f'{root}/data_00070_{suffix}')
            f[f'{root}/data_00070_{suffix}'].attrs['sweep_number'] = 70
        f['general/labnotebook/ITC18USB_Dev_0/numericalValues'][:, 0, :] = 70
    monkeypatch.setattr(vc, 'PAX6_SHA256', hashlib.sha256(nwb.read_bytes()).hexdigest())
    a, m = vc.read_sweep(nwb, 70, session=vc.PAX6_SESSION)
    assert m['session_id'] == '840043481' and m['specimen_id'] == '840043506'
    assert m['family'] == 'NucVCLS0_DA_0'
    np.testing.assert_allclose(a['command_voltage_mv'], [-88, -48, -48, -88])
    with pytest.raises(ValueError, match='registered'): vc.read_sweep(nwb, 70)
    with pytest.raises(ValueError, match='SHA256'): vc.read_sweep(nwb, 98)
    out = tmp_path / 'pax6'
    vc.main(['--nwb', str(nwb), '--sweep', '70', '--session', vc.PAX6_SESSION,
             '--output', str(out)])
    assert json.loads(out.with_suffix('.json').read_bytes())['specimen_id'] == '840043506'


@pytest.mark.parametrize('session', ['835648738', 'unknown'])
def test_external_validation_and_unknown_session_rejected_before_open(tmp_path, session):
    with pytest.raises(ValueError, match='Session'):
        vc.read_sweep(tmp_path / 'unopened.nwb', 70, session=session)
