"""Manifest-driven neuroglial construction before H01 learner compilation."""

import numpy as np
import brainunit as u
from braincell.filter import AllRegion
from braincell.mech import Channel

from braintrace.datasets.h01_glia import H01GlialSelection, load_glial_fragment
from braintrace.datasets.h01_glial_cable import build_glial_cable
from braintrace.datasets.h01_neuroglia import species_graph
from braintrace.biophysics.cable_geometry import CableChemicalGeometry
from braintrace.biophysics.cable_gaba import EnvironmentGABA
from braintrace.biophysics.cable_potassium import CablePotassiumBinding
from braintrace.biophysics.environment import ChemicalEnvironment, MembraneMap
from braintrace.biophysics.astrocyte import CalciumParameters
from braintrace.biophysics.astrocyte_network import AstrocyteCalcium
from braintrace.biophysics.gaba import GabaReleaseSites
from braintrace.biophysics.transmitter import GlutamateReleaseSites, TransmitterField
from braintrace.biophysics.neuroglial import NeuroGlialCoupling


def prepare_neuroglia(network, manifest, assets):
    """Resolve pinned glia and paint neuronal receptors before initialization.

    Parameters
    ----------
    network : braincell.Network
        Declared neuronal cables with environment potassium enabled.
    manifest : H01NeuroglialManifest
        Canonical source, electrical and chemical declarations.
    assets : mapping
        Glial source SHA256 to local path; paths are never source identity.

    Returns
    -------
    braincell.Cell
        Initialized independent glial cable, outside the neuronal ARC domain.
    """
    doc=manifest.to_dict()
    selection=H01GlialSelection(doc['glia']['selection'])
    digest=selection.to_dict()['source_sha256']
    if assets is None or digest not in assets:
        raise ValueError('Missing pinned glial source asset: '+digest)
    electrical=doc['glia']['electrical']
    k=doc['potassium']
    if any(electrical.get(key)!=k[target] for key,target in
           [('inside_k_mm','inside_mm'),('outside_k_mm','outside_mm'),('temperature_c','temperature_c')]):
        raise ValueError('Glial initial K and temperature must match shared environment')
    fragment=load_glial_fragment(assets[digest],selection)
    glia,_=build_glial_cable(fragment,electrical)
    glia.init_state()
    tonic=doc['tonic_gaba']
    for population in network.populations.values():
        population.cell.paint(AllRegion(),Channel('EnvironmentGABA',name='biology_tonic_gaba',
            g_max=tonic['g_max_ms_cm2']*u.mS/u.cm**2,E=tonic['reversal_mv']*u.mV,
            ec50_mm=tonic['ec50_mm'],hill=tonic['hill']))
    return glia


def attach_neuroglia(model, topology, manifest, glia):
    """Validate CV identities and bind complete chemistry to an untraced model.

    Parameters
    ----------
    model : H01ArcModel
        Initialized neuronal model, before learner graph compilation.
    topology : dict
        Ordered active neuronal identities and E/I source profiles.
    manifest : H01NeuroglialManifest
        Fully explicit chemical geometry and release settings.
    glia : braincell.Cell
        Independently initialized selected glial fragment.

    Returns
    -------
    NeuroGlialCoupling
        All chemical/glial state is reachable through the model stepper.
    """
    doc=manifest.to_dict()
    cells=(*model.stepper.cells,glia)
    names=[*topology['active_cells'],'@glia']
    geometries=[CableChemicalGeometry(cell) for cell in cells]
    for name,geometry in zip(names,geometries):
        row=doc['membranes'][name]
        if row['geometry_sha256']!=geometry.sha256 or len(row['outside_indices'])!=len(geometry.volume_um3):
            raise ValueError('Stale chemical geometry or CV mapping for '+name)
    ex,k=doc['extracellular'],doc['potassium']
    volumes=np.concatenate([geometry.volume_um3 for geometry in geometries])
    indices=np.concatenate([doc['membranes'][name]['outside_indices'] for name in names])
    environment=ChemicalEnvironment(species_graph(ex,'k',doc['dt_ms']),species_graph(ex,'gaba',doc['dt_ms']),
        MembraneMap(indices,volumes,len(ex['volumes_um3'])),np.full(len(volumes),k['inside_mm']),
        outside_potassium_mm=k['outside_mm'])
    bindings=[]
    offset=0
    for cell,geometry in zip(cells,geometries):
        bindings.append(CablePotassiumBinding(cell,environment,offset=offset,temperature_c=k['temperature_c']))
        offset+=len(geometry.volume_um3)
    for cell,binding in zip(model.stepper.cells,bindings[:-1]):
        channels=[node for node in cell.runtime.runtime_nodes.values() if isinstance(node,EnvironmentGABA)]
        if len(channels)!=1:
            raise ValueError('Exactly one declared tonic GABA channel required per neuron')
        channels[0].bind(binding)
    def release(species,owner):
        row=doc['releases'][species]
        if row is None:
            return None,[]
        return owner(row['volume_indices'],len(ex['volumes_um3']),active_sites=row['active_sites'],
            molecules_per_site=row['molecules_per_site'],seed=row['seed']),[names.index(name) for name in row['sources']]
    gaba,gaba_sources=release('gaba',GabaReleaseSites)
    glutamate,glutamate_sources=release('glutamate',GlutamateReleaseSites)
    calcium=AstrocyteCalcium(geometry=geometries[-1],dt_ms=doc['dt_ms'],
        parameters=CalciumParameters(**doc['calcium']['parameters']),substeps=doc['calcium']['substeps'])
    coupling=NeuroGlialCoupling(environment,bindings[:-1],bindings[-1:],
        [topology['sources'][topology['instances'][name]['source_id']]['polarity'] for name in names[:-1]],
        TransmitterField(species_graph(ex,'glutamate',doc['dt_ms'])),calcium,doc['membranes']['@glia']['outside_indices'],
        gaba_release=gaba,glutamate_release=glutamate,gaba_sources=gaba_sources,
        glutamate_sources=glutamate_sources,basis=doc['basis'])
    coupling.biology_sha256=manifest.sha256
    model.stepper.bind_chemistry(coupling)
    return coupling
