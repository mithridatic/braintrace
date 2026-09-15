"""Tests for human-only recovery observations and constrained curve estimation."""

import numpy as np
import pytest

from h01_k_recovery import extract_curves, fit_recovery, recovery


def row():
    r = dict(Filename='H21.29.190.11.41.01.nwb', Species='Human',
             rec_protocol_name='K_IKrecv2_1_h0_DA_0')
    for i in range(1, 11):
        r[f'recdelay_{i}'] = i * 100.
        r[f'amp1torec_{i}'] = i / 11
    for i in (8, 9, 10):
        r[f'amp1torec_{i:2d}.1'] = r[f'amp1torec_{i}']
    return r


def test_selection_preserves_values_groups_and_duplicate_receipt():
    r = row()
    absent = dict(r, Filename='H21.29.191.11.41.02.nwb', rec_protocol_name=None)
    curves, exclusions = extract_curves([r, absent])
    assert curves[0]['group'] == 'H21.29.190'
    assert curves[0]['ratios'] == [r[f'amp1torec_{i}'] for i in range(1, 11)]
    assert len(curves[0]['duplicate_columns_verified']) == 3
    assert exclusions[0]['reason'] == 'missing recovery protocol'


@pytest.mark.parametrize('change', [
    {'Species': 'Mouse'}, {'Filename': 'M21.29.190.11.41.01.nwb'},
    {'Filename': 'Human'}, {'rec_protocol_name': 'unknown'},
    {'recdelay_1': None}, {'amp1torec_1': float('nan')},
    {'recdelay_1': -1}, {'recdelay_2': 100}, {'amp1torec_ 8.1': 99},
])
def test_invalid_source_rejected(change):
    with pytest.raises(ValueError):
        extract_curves([dict(row(), **change)])


def test_missing_pairs_and_duplicate_identity():
    r = row()
    r['recdelay_1'] = r['amp1torec_1'] = None
    curves, _ = extract_curves([r])
    assert curves[0]['missing_indices'] == [1]
    with pytest.raises(ValueError):
        extract_curves([r, r])
    for i in range(2, 7):
        r[f'recdelay_{i}'] = r[f'amp1torec_{i}'] = None
    with pytest.raises(ValueError):
        extract_curves([r])


def test_no_curves_rejected():
    with pytest.raises(ValueError):
        extract_curves([])


def test_analytic_limits_and_monotonicity():
    p = [.2, .75, .4, np.log(100), np.log(3000)]
    values = recovery(np.r_[0, np.geomspace(.1, 1e7, 100)], p, 'double')
    assert values[0] == .2
    assert values[-1] == pytest.approx(.8)
    assert np.all(np.diff(values) >= 0)
    assert np.all((values >= .2) & (values <= .8))
    p[2] = 1
    np.testing.assert_allclose(recovery([0, 100, 1000], p, 'double'),
                               recovery([0, 100, 1000], [.2, .75, np.log(100)], 'single'))


@pytest.mark.parametrize('family,params', [
    ('double', [.15, .7, .5, np.log(200), np.log(4000)]),
    ('single', [.15, .7, np.log(1000)]),
])
def test_known_synthetic_recovery(family, params):
    t = np.geomspace(10, 20000, 30)
    y = recovery(t, params, family)
    result = fit_recovery([dict(times_ms=t, ratios=y)], family)
    np.testing.assert_allclose(result['parameters'], params, atol=1e-5)
    assert all(s['success'] for s in result['starts'])
    assert result['cost'] < 1e-15


@pytest.mark.parametrize('curves,family', [
    ([], 'single'), ([dict(times_ms=[1, 2], ratios=[.1])], 'single'),
    ([dict(times_ms=[1, float('inf')], ratios=[.1, .2])], 'single'),
    ([dict(times_ms=[2, 1], ratios=[.1, .2])], 'single'),
    ([dict(times_ms=[-1, 2], ratios=[.1, .2])], 'single'),
    ([dict(times_ms=[1, 2], ratios=[.1, .2])], 'unknown'),
])
def test_invalid_fit_inputs(curves, family):
    with pytest.raises(ValueError):
        fit_recovery(curves, family)


def test_optimizer_failure_is_not_candidate(monkeypatch):
    from types import SimpleNamespace
    import h01_k_recovery
    monkeypatch.setattr(h01_k_recovery, 'least_squares',
                        lambda *a, **k: SimpleNamespace(success=False, nfev=2000,
                            cost=1., message='cap', x=np.zeros(3)))
    with pytest.raises(RuntimeError):
        fit_recovery([dict(times_ms=[1, 2], ratios=[.1, .2])], 'single')


def test_unequal_record_lengths_have_equal_total_weight(monkeypatch):
    import h01_k_recovery
    original = h01_k_recovery.least_squares
    t = np.geomspace(10, 10000, 15)
    dense = np.geomspace(10, 10000, 100)
    # Check the fitting objective at a fixed zero prediction: changing the
    # time grid can change the optimum even with constant group targets.
    curves = [dict(times_ms=t, ratios=np.full(t.size, .2)),
              dict(times_ms=dense, ratios=np.full(dense.size, .4))]
    def inspect_objective(fun, *args, **kwargs):
        residual = fun([0, 0, np.log(1000)])
        assert np.sum(residual**2) == pytest.approx((.2**2+.4**2)/2)
        return original(fun, *args, **kwargs)
    monkeypatch.setattr(h01_k_recovery, 'least_squares', inspect_objective)
    fit_recovery(curves, 'single')


def test_unknown_curve_family():
    with pytest.raises(ValueError):
        recovery([1], [0, 1, 2], 'unknown')
