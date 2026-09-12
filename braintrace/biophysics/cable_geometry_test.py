"""Analytic tapered integrals, cylinder limit and massless fork flux."""

from types import SimpleNamespace
from dataclasses import replace

import braincell
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .cable_geometry import CableChemicalGeometry, _integrals, SHELL_FRACTIONS
from .astrocyte_network import shell_transport, AstrocyteCalcium
from .astrocyte import CalciumParameters


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64):
        yield


def branch(lengths=(2.,), proximal=(1.,), distal=(2.,)):
    return braincell.Branch(lengths=np.asarray(lengths)*u.um,
        radii_proximal=np.asarray(proximal)*u.um,radii_distal=np.asarray(distal)*u.um)


def cable(*, cylinder=False, fork=False, subdivisions=1):
    root=branch(distal=(1.,)) if cylinder else branch()
    morphology=braincell.Morphology(root_name='trunk',root_branch=root)
    if fork:
        for name,length,radius in [('a',3.,2.),('b',4.,2.)]:
            morphology.attach(parent='trunk',child_name=name,
                child_branch=branch((length,),(radius,),(radius,)),parent_x=1.)
    cell=braincell.Cell(morphology,V_init=-100.*u.mV,pop_size=(1,),cv_policy=braincell.CVPerBranch(subdivisions))
    cell.init_state()
    return cell


def test_exact_frustum_integrals_and_subdivision_invariance():
    cell=cable()
    geometry=CableChemicalGeometry(cell)
    np.testing.assert_allclose(geometry.volume_um3,[14*np.pi/3],rtol=1e-12)
    np.testing.assert_allclose(geometry.area_um2,[3*np.pi*np.sqrt(5)],rtol=1e-12)
    expected=[2/(3*np.pi),1/(3*np.pi)]
    np.testing.assert_allclose(geometry.to_dict()['half_resistance_per_um'][0],expected,rtol=1e-12)
    fine=CableChemicalGeometry(cable(subdivisions=4))
    np.testing.assert_allclose(fine.volume_um3.sum(),geometry.volume_um3.sum(),rtol=1e-12)
    np.testing.assert_allclose(fine.area_um2.sum(),geometry.area_um2.sum(),rtol=1e-12)
    assert fine.sha256 != geometry.sha256
    changed=geometry.to_dict()
    changed['volume_um3'][0]=0
    assert geometry.volume_um3[0] > 0


def test_piecewise_radius_changes_are_not_replaced_by_endpoint_taper():
    source=branch((1.,1.),(1.,3.),(3.,2.))
    actual=_integrals(source,0.,1.)
    np.testing.assert_allclose(actual[0],np.pi*(13+19)/3)
    np.testing.assert_allclose(actual[3],1/(3*np.pi)+1/(6*np.pi))
    np.testing.assert_allclose(_integrals(source,0.,.5)+_integrals(source,.5,1.),actual)


@pytest.mark.parametrize('radial',[True,False])
def test_cylinder_limit_matches_existing_source_discretization(radial):
    geometry=CableChemicalGeometry(cable(cylinder=True,subdivisions=2))
    actual=geometry.transport(diffusion=.3,dt_ms=.005,radial=radial)
    expected=shell_transport([1.,1.],[2.,2.],[[0,1]],diffusion=.3,dt_ms=.005,radial=radial)
    np.testing.assert_allclose(actual.volumes,expected.volumes,rtol=1e-12)
    np.testing.assert_array_equal(actual.edges,expected.edges)
    np.testing.assert_allclose(actual.conductance,expected.conductance,rtol=1e-12)


def test_fork_elimination_matches_shared_boundary_flux_and_conserves():
    geometry=CableChemicalGeometry(cable(fork=True))
    record=geometry.to_dict()
    assert sorted(record['axial_edges'])==[[0,1],[0,2],[1,2]]
    resistance=np.asarray(record['half_resistance_per_um'])
    weights=1/np.array([resistance[0,1],resistance[1,0],resistance[2,0]])
    concentration=np.array([1.,.2,.7])
    boundary=np.sum(weights*concentration)/np.sum(weights)
    wanted=weights*(boundary-concentration)
    actual=np.zeros(3)
    for (i,j),g in zip(record['axial_edges'],record['axial_conductance_um']):
        flux=g*(concentration[j]-concentration[i])
        actual[i]+=flux
        actual[j]-=flux
    np.testing.assert_allclose(actual,wanted,atol=1e-14)
    graph=geometry.transport(diffusion=.3,dt_ms=.005)
    start=jnp.zeros(12).at[0].set(.001)
    result=jax.jit(graph.step)(start)
    assert result.valid and result.concentration[4] > 0 and result.concentration[8] > 0
    np.testing.assert_allclose(jnp.sum(result.concentration*graph.volumes),jnp.sum(start*graph.volumes),atol=1e-13)
    derivative=jax.jit(jax.grad(lambda pulse: graph.step(start.at[0].set(pulse)).concentration[4]))(.001)
    fd=(graph.step(start.at[0].set(.001001)).concentration[4]-graph.step(start.at[0].set(.000999)).concentration[4])/.000002
    np.testing.assert_allclose(derivative,fd,rtol=1e-7)


def test_tapered_calcium_buffer_mass_and_exact_k_pool_volumes():
    geometry=CableChemicalGeometry(cable(fork=True))
    p=CalciumParameters(alpha=0.,pump_velocity_um_ms=0.)
    model=AstrocyteCalcium(geometry=geometry,parameters=p)
    assert model.geometry_sha256==geometry.sha256
    volumes=model.calcium_transport.volumes.reshape(-1,4)
    np.testing.assert_allclose(volumes.sum(axis=1),geometry.volume_um3,rtol=1e-12)
    np.testing.assert_allclose(model.surface_to_volume,geometry.area_um2/(geometry.volume_um3*SHELL_FRACTIONS[0]))
    model.state.value=model.state.value.at[0,0].set(.001)
    def amount():
        state=model.state.value
        return jnp.sum(volumes*(state[:,:4]+state[:,8:12]+state[:,16:20]))
    before=amount()
    brainstate.transform.jit(model.update)(jnp.zeros(3))
    assert model.valid.value and model.state.value[1,0] > p.resting_mm
    np.testing.assert_allclose(amount(),before,atol=1e-12)
    model.reset_state()
    np.testing.assert_array_equal(model.state.value,model.initial)


def test_invalid_geometry_and_mixed_constructor_arguments():
    with pytest.raises(ValueError,match='initialized'):
        CableChemicalGeometry(SimpleNamespace(_runtime=None))
    with pytest.raises(ValueError,match='positive'):
        _integrals(SimpleNamespace(lengths=np.array([2.])*u.um,
            radii_proximal=np.array([0.])*u.um,radii_distal=np.array([2.])*u.um),0.,1.)
    with pytest.raises(ValueError,match='interval'):
        _integrals(branch(),.5,.5)
    geometry=CableChemicalGeometry(cable())
    with pytest.raises(ValueError,match='Diffusion'):
        geometry.transport(diffusion=-1.,dt_ms=.005)
    with pytest.raises(ValueError,match='cannot mix'):
        AstrocyteCalcium([1.],[2.],[],geometry=geometry)
    with pytest.raises(ValueError,match='Complete'):
        AstrocyteCalcium()
    cell=cable()
    changed=replace(cell.cvs[0],area=1.*u.um**2)
    with pytest.raises(ValueError,match='differs'):
        CableChemicalGeometry(SimpleNamespace(_runtime=True,cvs=(changed,),morpho=cell.morpho))


@pytest.mark.parametrize('fault',['order','partition','missing_end','duplicate_end','mid_attachment','collapsed'])
def test_invalid_cv_or_boundary_provenance(fault):
    cell=cable(cylinder=True,subdivisions=2)
    cvs=cell.cvs
    nodes=list(cell.node_tree.nodes)
    edges=list(cell.node_tree.edges)
    if fault=='order':
        cvs=(replace(cvs[0],id=2),*cvs[1:])
    elif fault=='partition':
        cvs=cvs[:1]
    elif fault=='missing_end':
        nodes[-1]=replace(nodes[-1],roles=())
    elif fault=='duplicate_end':
        nodes.append(nodes[0])
    elif fault=='mid_attachment':
        nodes[1]=replace(nodes[1],roles=(*nodes[1].roles,nodes[0].roles[0]))
    else:
        index=next(i for i,e in enumerate(edges) if len(e.roles)==2)
        edges[index]=replace(edges[index],roles=(*edges[index].roles,edges[index].roles[0]))
    malformed=SimpleNamespace(_runtime=True,cvs=cvs,morpho=cell.morpho,
                              node_tree=SimpleNamespace(nodes=nodes,edges=edges))
    with pytest.raises(ValueError):
        CableChemicalGeometry(malformed)
