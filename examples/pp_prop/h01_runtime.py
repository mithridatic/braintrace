"""Construct active H01 instances from immutable source and contact manifests."""

import json
from types import SimpleNamespace

import braincell
from braincell.filter import AtLocation
from braincell.mech import MechanismProbe, StateProbe, Synapse
from braincell.network import pairs
import brainunit as u

from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
from braintrace.datasets.h01_network import _regions, _register_cell
from braintrace.datasets.h01_biology import H01SpatialManifest
from braintrace.datasets.h01_spine_assembly import assembled_location
from .h01_topology import H01Topology


def topology_from_evidence(evidence, contact_audit, archive):
    """Pin selected source components and anatomical placements for evolution.

    Parameters
    ----------
    evidence : dict
        Successful source network construction evidence.
    contact_audit : dict
        Verified anatomical contact audit, including blocked fragments.
    archive : H01Archive
        Checksum-verified source morphology archive.

    Returns
    -------
    H01Topology
        Initial original instances and constructible anatomical contacts.
    """
    sources, instances, contacts = {}, {}, {}
    active = list(evidence['simulated_cell_ids'])
    for identity in active:
        record = evidence['cells'][identity]
        anatomy = record['measured_anatomy']
        component = int(anatomy['member'].split('.')[-2])
        imported = archive.load(identity, component=component)
        if imported.source_sha256 != anatomy['source_sha256']:
            raise ValueError('Source morphology differs from construction evidence')
        soma = imported.anatomy().soma_location().evaluate(imported.morphology).points[0]
        sources[identity] = dict(component=component, source_sha256=imported.source_sha256,
            polarity=record['modeled_polarity'], donor=record['donor'],
            profile=record['borrowed_dynamics'], source_tags=record['source_tags'],
            region_basis=record['inferred_region_basis'], soma_site=list(soma),
            output_site=record['output_site']['source_point'],
            archive_sha256=anatomy['archive_sha256'],
            compartment_policy=anatomy['compartment_policy'])
        instances[identity] = dict(source_id=identity, parent_id=None, synthetic=False, created_stage='source')
    audited = {edge['annotation_id']: edge for edge in contact_audit['contacts']}
    for edge in evidence['contacts']:
        if not edge['enabled']:
            continue
        identity = edge['annotation_id']
        source = audited[identity]
        if not source['construction_ready']:
            raise ValueError('Cannot activate a blocked anatomical contact')
        contacts[identity] = dict(pre=edge['pre_cell'], post=edge['post_cell'],
            pre_site=source['pre_placement']['cable_location'],
            post_site=source['post_placement']['cable_location'], synthetic=False,
            parent_id=None, annotation_id=identity, created_stage='source',
            initial_weight_us=edge['weight_us'], reversal_mv=edge['reversal_mv'],
            tau_ms=edge['tau_ms'], delay_ms=edge['delay_ms'])
    return H01Topology.from_dict(dict(schema='h01-topology-v1', sources=sources,
        instances=instances, contacts=contacts, active_cells=active, active_contacts=list(contacts),
        next_id=0, blocked_contacts=evidence['blocked_contacts'], mutations=[]))


def build_network(topology, archive, *, solver='h01_staggered_calcium_implicit',
                  max_cv_length_um=10., progress=None, environment_potassium=False, biology=None,
                  fused=False):
    """Build real selected cables and placed conductance contacts for a topology.

    Parameters
    ----------
    topology : H01Topology
        Immutable source profiles and active instance/contact identities.
    archive : H01Archive
        Verified immutable morphology source.
    solver : str, optional
        Pinned integration algorithm.
    max_cv_length_um : float, optional
        Frozen spatial discretization length.
    progress : callable, optional
        Construction progress callback.
    environment_potassium : bool, optional
        Declare external K pools; caller must bind chemistry after initialization.
    biology : H01SpatialManifest or dict or None, optional
        Explicit topology-pinned spine additions and contact-head assignments.
    fused : bool, optional
        Register the cells as one forest population
        (``braintrace.datasets.h01_forest_cell.H01ForestCell``): every cell is
        built and initialized as usual, then fused so that each mechanism runs
        as one kernel over all compartments. Spec:
        docs/specs/2026-09-16-h01-fused-population.md. Contacts are not yet
        delivered on the fused path.

    Returns
    -------
    tuple
        BrainCell network and per-instance construction records. Initialization
        is separate so its time and memory can be measured independently; on the
        fused path the source cells are initialized here and the forest in
        ``init_state``. Records carry ``forest`` offsets when fused.
    """
    doc = topology.to_dict()
    manifest = None if biology is None else H01SpatialManifest(
        biology.to_dict() if isinstance(biology, H01SpatialManifest) else biology, doc)
    biological = None if manifest is None else manifest.to_dict()
    emit = progress or (lambda message: None)
    loaded, cells, records = {}, {}, {}
    network = braincell.Network(name='h01_evolved')
    for identity in doc['active_cells']:
        source_id = doc['instances'][identity]['source_id']
        source = doc['sources'][source_id]
        if source_id not in loaded:
            loaded[source_id] = archive.load(source_id, component=source['component'])
        imported = loaded[source_id]
        if imported.source_sha256 != source['source_sha256']:
            raise ValueError('Immutable H01 source digest changed')
        tags = tuple(source['source_tags'])
        annotations = SimpleNamespace(metadata=lambda _, tags=tags: SimpleNamespace(tags=tags))
        emit('Building H01 instance '+identity)
        cell, record = make_h01_ei_cell(imported, annotations, polarity=source['polarity'],
            donor=source['donor'], mode=source['profile']['mode'], regions=_regions(imported),
            region_basis=source['region_basis'], current_na=0., delay_ms=2., duration_ms=3.,
            max_cv_length_um=max_cv_length_um, solver=solver, pop_size=(1,),
            environment_potassium=environment_potassium,
            spines=() if manifest is None else manifest.spines(identity))
        if json.dumps(record['borrowed_dynamics'], sort_keys=True) != json.dumps(source['profile'], sort_keys=True):
            raise ValueError('Frozen H01 donor profile differs from this runtime')
        cells[identity], records[identity] = cell, record
        if manifest is not None:
            record['biology_manifest_sha256'] = manifest.sha256
            record['declared_spine_origins'] = {row['identity']: row['origin']
                for row in biological['cells'][identity]['spines']}
    for identity in doc['active_contacts']:
        edge = doc['contacts'][identity]
        record = records[edge['post']]
        mapped_site = assembled_location(record, edge['post_site'])
        if biological is not None and identity in biological['contact_heads']:
            head = biological['contact_heads'][identity]
            mapped_site = (record['spine_assembly']['heads'][head], .5)
        if manifest is not None:
            record.setdefault('contact_sites', {})[identity] = dict(source=list(edge['post_site']),
                                                                   assembled=list(mapped_site))
        cell, site, name = cells[edge['post']], AtLocation(*mapped_site), 'syn_'+identity
        cell.place(site, Synapse('ExpSyn', name=name, e=edge['reversal_mv']*u.mV,
                                tau=edge['tau_ms']*u.ms, weight=1.*u.uS))
        cell.place(site, MechanismProbe(mechanism=name, field='g', name=name+'_g'))
        cell.place(site, StateProbe(field='v', name=name+'_voltage'))
    for identity in doc['active_cells']:
        source_id = doc['instances'][identity]['source_id']
        source = doc['sources'][source_id]
        site = source['output_site'] or source['soma_site']
        for contact in doc['active_contacts']:
            edge = doc['contacts'][contact]
            if edge['pre'] == identity and not (edge['pre_site'] == site or (
                len(edge['pre_site']) == 2 and len(site) == 2 and edge['pre_site'][0] == site[0] and abs(edge['pre_site'][1] - site[1]) < 1e-4
            )):
                raise ValueError('A cell cannot have conflicting designated output sites')
        _register_cell(network, identity, cells[identity], records[identity], loaded[source_id],
                       {identity: assembled_location(records[identity], site)}, 0., emit)
        if manifest is not None:
            records[identity]['original_output_site'] = list(site)
    if fused:
        network = _fuse_populations(network, doc, records, solver, emit)
    for identity in doc['active_contacts']:
        edge, name = doc['contacts'][identity], 'syn_'+identity
        network.add_edges(name=name, pre='cell_'+edge['pre'], post='cell_'+edge['post'], method=pairs([(0, 0)]))
        network.add_projection(name=name, edges=name, synapse=name,
            weight=edge['initial_weight_us']*u.uS, delay=edge['delay_ms']*u.ms)
    release_shared_construction_data()
    return network, records


def _fuse_populations(network, doc, records, solver, emit):
    """Initialize every registered cell and return a network with one forest population."""
    from braintrace.datasets.h01_forest_cell import H01ForestCell
    if doc['active_contacts']:
        raise ValueError('The fused population does not deliver contacts yet (spec section 3)')
    cells = []
    for identity in doc['active_cells']:
        cell = network.populations['cell_'+identity].cell
        emit('Initializing source cell '+identity)
        cell.init_state()
        cells.append(cell)
    forest = H01ForestCell(cells, solver=solver)
    for index, identity in enumerate(doc['active_cells']):
        records[identity]['forest'] = dict(index=index, cv_offset=int(forest.forest_offsets.cv[index]),
            point_offset=int(forest.forest_offsets.point[index]), soma_cv=int(forest.soma_cv_ids[index]),
            output_cv=int(forest.output_cv_ids[index]))
    fused = braincell.Network(name='h01_evolved_forest')
    fused.add_population('forest', forest)
    return fused


def release_shared_construction_data():
    """Match the source builder's bounded-memory handoff before allocating channel states.

    Immutable component data is shared across clones while building; the loaded-component
    table is released here. The CV geometry cache is owner-scoped on each morphology since
    commit 43bd52b5 (no module-level table remains), so nothing else is cleared.
    """
    from braintrace.datasets.h01 import _GLOBAL_LOADED_COMPONENTS
    _GLOBAL_LOADED_COMPONENTS.clear()
