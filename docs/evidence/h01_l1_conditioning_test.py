"""Matched test commands, conditioning history and descriptive decay oracles."""

import numpy as np
import pytest

from h01_l1_conditioning import paired_conditioning, match_commands, fit_difference


def sources(test_voltage=10.):
    t = np.arange(0., 3100., .5)
    pulse = (t >= 1100) & (t < 2100)
    condition = (t >= 1000) & (t < 1100)
    va = np.full_like(t, -90.)
    va[pulse] = test_voltage
    vb = va.copy()
    vb[condition] = -20.
    ia = np.full_like(t, 4.)
    ib = np.full_like(t, 7.)
    ia[pulse] += 20 + 30*np.exp(-(t[pulse]-1110)/25) + 100*np.exp(-(t[pulse]-1110)/400)
    ib[pulse] += 20
    return [dict(time_ms=t.copy(), total_current_pa=i, command_voltage_mv=v)
            for i, v in ((ia, va), (ib, vb))]


@pytest.mark.parametrize('voltage', [-50., 70.])
def test_exact_pair_retains_baseline_and_conditioning_above_or_below_test(voltage):
    a, b = sources(voltage)
    arrays, meta = paired_conditioning(a, b)
    phase = arrays['phase_ms']
    np.testing.assert_allclose(arrays['difference_pa'], -3 + 30*np.exp(-(phase-10)/25) + 100*np.exp(-(phase-10)/400))
    np.testing.assert_array_equal(arrays['total_time_ms'], arrays['conditioned_time_ms'])
    np.testing.assert_array_equal(arrays['total_time_ms'], phase+1100)
    assert meta['baseline_difference_pa'] == -3
    assert meta['test_command_mv'] == voltage
    assert meta['conditioning_command_mv'] == -20
    assert meta['ionic_isolation_qualified'] is False


@pytest.mark.parametrize('fault', ['clock', 'shifted_clock', 'nan', 'short', 'test_voltage', 'holding', 'condition', 'extra_total', 'extra_test', 'return', 'rank', 'off_grid'])
def test_invalid_protocols(fault):
    a, b = sources()
    if fault == 'clock': b['time_ms'][100] += .1
    if fault == 'shifted_clock': b['time_ms'] += .5
    if fault == 'nan': b['total_current_pa'][100] = np.nan
    if fault == 'short':
        for x in (a,b):
            for key in x: x[key] = x[key][:4000]
    if fault == 'test_voltage': b['command_voltage_mv'][2200:4200] = 20
    if fault == 'holding': b['command_voltage_mv'][:2000] = -80
    if fault == 'condition': b['command_voltage_mv'][2000:2200] = -90
    if fault == 'extra_total': a['command_voltage_mv'][2050:2060] = -10
    if fault == 'extra_test': b['command_voltage_mv'][2500] = 20
    if fault == 'return': b['command_voltage_mv'][4200:] = -80
    if fault == 'rank': b['total_current_pa'] = b['total_current_pa'][:,None]
    if fault == 'off_grid':
        for x in (a,b): x['time_ms'] += .1
    with pytest.raises(ValueError): paired_conditioning(a,b)


def test_matching_uses_commands_and_rejects_missing_or_ambiguous_partner():
    assert match_commands({70:-50.,71:-35.},{90:-35.,89:-50.}) == [(70,89),(71,90)]
    with pytest.raises(ValueError): match_commands({70:-50.},{89:-35.})
    with pytest.raises(ValueError): match_commands({70:-50.},{89:-50.,90:-50.})
    with pytest.raises(ValueError): match_commands({70:np.nan},{89:-50.})
    with pytest.raises(ValueError): match_commands({70:-50.,71:-50.},{89:-50.})


def test_decay_recovers_known_curve_and_preserves_samples():
    phase=np.arange(0.,1000.,.5)
    difference=30*np.exp(-(phase-10)/25)+100*np.exp(-(phase-10)/400)
    arrays, meta=fit_difference(phase,difference)
    np.testing.assert_allclose(meta['parameters'],[30.,100.,25.,400.],rtol=1e-5)
    np.testing.assert_array_equal(arrays['phase_ms'],phase[20:1960])
    np.testing.assert_array_equal(arrays['difference_pa'],difference[20:1960])
    np.testing.assert_allclose(arrays['predicted_pa'],arrays['difference_pa'],atol=1e-6)
    assert meta['optimizer_success'] and not meta['boundary_limited']
    assert not meta['physiological_qualification']


@pytest.mark.parametrize('fault',['shape','nan','order','coverage','sparse'])
def test_decay_rejects_invalid_data(fault):
    phase=np.arange(0.,1000.,.5); y=np.ones_like(phase)
    if fault=='shape': y=y[:-1]
    if fault=='nan': y[12]=np.nan
    if fault=='order': phase[15]=phase[14]
    if fault=='coverage': phase=phase[30:]; y=y[30:]
    if fault=='sparse': phase=np.r_[np.arange(20.)/10,10.,980.,np.arange(990.,1000.)]; y=np.ones_like(phase)
    with pytest.raises(ValueError): fit_difference(phase,y)
