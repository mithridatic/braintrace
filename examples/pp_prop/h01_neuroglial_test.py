"""Complete explicit neuroglial session construction and physical replay."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import copy

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_anatomy_test import imported
from braintrace.datasets.h01_biology import topology_digest
from braintrace.datasets.h01_glia_test import source
from braintrace.datasets.h01_glia import H01GlialSelection, load_glial_fragment
from braintrace.datasets.h01_glial_cable import build_glial_cable
from braintrace.datasets.h01_neuroglia import H01NeuroglialManifest
from braintrace.biophysics.cable_geometry import CableChemicalGeometry
from braintrace.biophysics.astrocyte import CalciumParameters
from .h01_runtime_test import _manifest
from .h01_runtime import build_network
from .h01_session import H01Session, numerical_settings
from .example21_arc_adapter import Example21ArcAdapter
from .h01_physical_wait import PhysicalWait
from .h01_physical_checkpoint import _pack


@pytest.fixture(autouse=True)
def precision():
    with brainstate.environ.context(precision=64,dt=.005*u.ms):
        yield


def configured(imported,tmp_path):
    topology=_manifest(imported)
    doc=topology.to_dict()
    archive=SimpleNamespace(load=lambda identity,component:imported)
    network,_=build_network(topology,archive,environment_potassium=True)
    network.init_state()
    neuronal=next(iter(network.populations.values())).cell
    neuron_geometry=CableChemicalGeometry(neuronal)
    path,selection,_=source(tmp_path,vertices_nm=np.array([[0.,0.,0.],[1000.,0.,0.]]),
                            radius_nm=np.array([1000.,1000.]),edges=np.array([[0,1]]))
    electrical=dict(initial_mv=-100.,inside_k_mm=140.,outside_k_mm=10.,temperature_c=34.,
                    gkir_ms_cm2=.4,cm_uf_cm2=1.,ra_ohm_cm=100.,max_cv_length_um=10.,basis='Synthetic diagnostic glial cable')
    glia,_=build_glial_cable(load_glial_fragment(path,H01GlialSelection(selection)),electrical)
    glia.init_state()
    glial_geometry=CableChemicalGeometry(glia)
    spines=dict(schema='h01-biology-spines-v1',topology_sha256=topology_digest(doc),coordinate_frame='h01-swc-um',
        cells={'12':dict(source_sha256=imported.source_sha256,spines=[])},contact_heads={},release_probability={})
    ex=dict(centers_um=[[0.,0.,0.],[1.,0.,0.]],volumes_um3=[100.,100.],edges=[[0,1]],origin='synthetic',
            basis='Explicit two-volume diagnostic domain; centers do not infer contacts')
    for species in ('k','gaba','glutamate'):
        ex[species]=dict(conductance_um3_ms=[1.],boundary_um3_ms=[0.,0.],uptake_per_ms=[0.,0.] if species=='k' else [.01,.01])
    biology=dict(schema='h01-biology-neuroglia-v1',spines=spines,glia=dict(selection=selection,electrical=electrical),
        extracellular=ex,membranes={'12':dict(geometry_sha256=neuron_geometry.sha256,
            outside_indices=[0]*len(neuron_geometry.volume_um3)),
            '@glia':dict(geometry_sha256=glial_geometry.sha256,outside_indices=[1])},
        releases=dict(gaba=None,glutamate=dict(sources=['12'],volume_indices=[[0,1]],active_sites=1,
            molecules_per_site=3000.,seed=22,basis='Explicit synthetic diagnostic glutamate dose')),
        potassium=dict(inside_mm=140.,outside_mm=10.,temperature_c=34.),
        tonic_gaba=dict(g_max_ms_cm2=.1,reversal_mv=-70.,ec50_mm=1e-4,hill=1.),
        calcium=dict(parameters=asdict(CalciumParameters()),substeps=4),dt_ms=.005,basis='Synthetic coupled-session fixture')
    return topology,archive,biology,{selection['source_sha256']:path}


def test_full_session_compiles_and_rebuilds_physical_state(imported,tmp_path):
    topology,archive,biology,assets=configured(imported,tmp_path)
    trainer=Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
    settings=numerical_settings()
    settings['biology']=biology
    session=H01Session.build(topology,archive,trainer,settings=settings,biology_assets=assets)
    assert session.model.neuron_count==1 and len(session.model.stepper.chemistry.astrocytes)==1
    session.model.reset_episode(session.learner)
    brainstate.transform.jit(session.learner)(jnp.full(441,.01))
    assert any(np.any(np.asarray(value)!=0) for value in session.learner.factors.value)
    wait=PhysicalWait(session.model,.0002,learner=session.learner)
    advance=brainstate.transform.jit(lambda:wait.update(max_events=1))
    assert not advance()
    assert session.model.stepper.chemistry.valid.value
    snapshot=tmp_path/'coupled-session.npz'
    digest=session.save_physical(snapshot,wait=wait)
    assert advance()
    expected,_,_= _pack(dict(model=session.model,learner=session.learner,wait=wait),{})
    assert all(np.isfinite(value).all() for value in expected.values())
    fresh=H01Session.build(topology,archive,trainer,settings=settings,biology_assets=assets)
    resumed=PhysicalWait(fresh.model,.0002,learner=fresh.learner)
    fresh.restore_physical(snapshot,wait=resumed,expected_sha256=digest)
    assert brainstate.transform.jit(lambda:resumed.update(max_events=1))()
    actual,_,_=_pack(dict(model=fresh.model,learner=fresh.learner,wait=resumed),{})
    assert expected.keys()==actual.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
    with pytest.raises(ValueError,match='differs'):
        session.save_physical(snapshot,wait=wait,biology_manifest=dict(biology,basis='changed'))


def test_missing_asset_and_stale_geometry_fail_before_learning(imported,tmp_path):
    topology,archive,biology,assets=configured(imported,tmp_path)
    settings=numerical_settings()
    settings['biology']=biology
    with pytest.raises(ValueError,match='Missing pinned'):
        H01Session.build(topology,archive,None,settings=settings)
    biology['membranes']['12']['geometry_sha256']='0'*64
    with pytest.raises(ValueError,match='Stale chemical'):
        H01Session.build(topology,archive,None,settings=settings,biology_assets=assets)


def test_snapshot_rejects_postconstruction_biology_mutation(imported,tmp_path):
    topology,_,biology,_=configured(imported,tmp_path)
    identity=H01NeuroglialManifest(biology,topology.to_dict()).sha256
    model=SimpleNamespace(stepper=SimpleNamespace(chemistry=SimpleNamespace(biology_sha256=identity)))
    trainer=SimpleNamespace(parameters={},muon_groups={},updates=0)
    session=H01Session(topology,model,None,trainer,dict(biology=biology),())
    wait=SimpleNamespace(model=model,learner=None,total_events=1)
    session.settings['biology']['basis']='Changed after physical assembly'
    with pytest.raises(ValueError,match='constructed'):
        session._physical_context(wait,None)
