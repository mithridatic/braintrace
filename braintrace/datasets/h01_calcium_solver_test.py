"""Compiled cell checks for the opt-in calcium integration."""
import braincell
import brainstate
import brainunit as u
import numpy as np
import pytest
from braincell.filter import AllRegion, RootLocation
from braincell.mech import Channel, Ion, MechanismProbe, CurrentProbe, StateProbe
from . import h01_pv_channels, h01_calcium_solver


def make_cell(solver, calcium=True):
    soma = braincell.Soma(lengths=np.array([10.])*u.um,
        radii_proximal=np.array([5.])*u.um, radii_distal=np.array([5.])*u.um)
    cell = braincell.Cell(braincell.Morphology(root_name='soma',root_branch=soma),
                          V_init=-40.*u.mV, solver=solver)
    cell.paint(AllRegion(), Channel('IL',name='leak',g_max=.1*u.mS/u.cm**2,E=-60.*u.mV))
    cell.place(RootLocation(.5),StateProbe(field='v',name='v'))
    if calcium:
        cell.paint(AllRegion(),Ion('H01PV_Calcium',name='calcium'))
        cell.paint(AllRegion(),Ion('PotassiumFixed',name='potassium',E=-85.*u.mV))
        cell.paint(AllRegion(),Channel('H01PV_Ca_LVA',name='ca1',g_max=1.*u.mS/u.cm**2))
        cell.paint(AllRegion(),Channel('H01PV_Ca_HVA',name='ca2',g_max=1.*u.mS/u.cm**2))
        cell.paint(AllRegion(),Channel('H01PV_SK',name='sk',g_max=1.*u.mS/u.cm**2))
        cell.place(RootLocation(.5),MechanismProbe(mechanism='calcium',field='Ci',name='ci'))
        cell.place(RootLocation(.5),MechanismProbe(mechanism='sk',field='z',name='z'))
        for name,probe in [('ca',CurrentProbe(ion='calcium',name='ca')),
             ('ca1',CurrentProbe(mechanism='ca1',name='ca1')),('ca2',CurrentProbe(mechanism='ca2',name='ca2'))]:
            cell.place(RootLocation(.5),probe)
    return cell


def test_compiled_calcium_and_sk_have_finite_traces_and_correct_current_ownership():
    with brainstate.environ.context(precision=64):
        result = make_cell('h01_staggered_calcium_implicit').run(dt=.005*u.ms,duration=.2*u.ms)
        ci = np.asarray(result.traces['ci'].to_decimal(u.mM))
        assert np.isfinite(ci).all() and (ci > 0).all()
        assert ci[-1] > ci[0]
        assert np.isfinite(result.traces['z']).all()
        total,one,two = [np.asarray(result.traces[k].to_decimal(u.mA/u.cm**2)) for k in ['ca','ca1','ca2']]
        np.testing.assert_allclose(total,one+two,rtol=1e-12,atol=1e-14)


def test_no_calcium_matches_existing_scan_solver():
    with brainstate.environ.context(precision=64):
        outputs = [np.asarray(make_cell(solver,False).run(dt=.005*u.ms,duration=.1*u.ms).traces['v'].to_decimal(u.mV))
                   for solver in ['h01_staggered_scan','h01_staggered_calcium_implicit']]
    np.testing.assert_array_equal(*outputs)


def test_unsupported_order_rejected():
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='family ordering'):
        h01_calcium_solver._implicit_step(SimpleNamespace(ion_channel_update_order='integration'))


def test_non_ohmic_current_is_rejected():
    from types import SimpleNamespace
    node = SimpleNamespace(E=100.*u.mV,
        current=lambda v, **kwargs: ((100.-v.to_decimal(u.mV))**2)*u.uA/u.cm**2)
    with brainstate.environ.context(precision=64):
        _, g = h01_calcium_solver._calcium_snapshot(node, -40.*u.mV)
    assert np.isnan(np.asarray(g)).all()


def test_ordinary_voltage_trajectory_refines_toward_existing_solver():
    with brainstate.environ.context(precision=64):
        errors = []
        for dt in (.01,.005):
            traces = [np.asarray(make_cell(solver).run(dt=dt*u.ms,duration=.2*u.ms).traces['v'].to_decimal(u.mV))
                      for solver in ['h01_staggered_scan','h01_staggered_calcium_implicit']]
            errors.append(np.max(np.abs(traces[0]-traces[1])))
    assert errors[1] < .7*errors[0]
    assert errors[0] < .01
