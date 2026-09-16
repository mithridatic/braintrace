"""Construct active H01 instances from immutable source and contact manifests."""

import json
from types import SimpleNamespace

import braincell
from braincell.filter import AtLocation
from braincell.mech import MechanismProbe, StateProbe, Synapse
from braincell.network import pairs
import brainunit as u

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_archive_set import H01ArchiveSet
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
from braintrace.datasets.h01_network import _regions, _register_cell
from braintrace.datasets.h01_biology import H01SpatialManifest
from braintrace.datasets.h01_spine_assembly import assembled_location
from .h01_topology import H01Topology


def is_archive_set(archive):
    """Tell a multi-archive resolver from a single archive.

    Parameters
    ----------
    archive : object
        ``H01Archive`` (loads by cell and component) or ``H01ArchiveSet`` (also
        needs the source's ``archive_sha256``; exposes ``archives()``).

    Returns
    -------
    bool
        True when ``archive`` resolves sources by archive digest.
    """
    return callable(getattr(archive, 'archives', None))


def load_source(archive, identity, component, archive_sha256):
    """Load one source component from a single archive or an archive set.

    Parameters
    ----------
    archive : H01Archive or H01ArchiveSet
        Single archive (the digest is not consulted) or a set keyed by digest.
    identity : str
        Source cell identifier.
    component : int
        Explicit component suffix.
    archive_sha256 : str
        Digest of the archive the source was imported from.

    Returns
    -------
    H01Component
        Loaded component.

    Raises
    ------
    KeyError
        If an archive set holds no archive with ``archive_sha256``.
    """
    if is_archive_set(archive):
        return archive.load(identity, component, archive_sha256)
    return archive.load(identity, component=component)


def open_archive(path, digest):
    """Open one archive pinned to an expected digest.

    Parameters
    ----------
    path : path-like
        Archive file.
    digest : str
        Expected SHA256 of the file.

    Returns
    -------
    H01Archive
        Verified archive.
    """
    return H01Archive(path, expected_sha256=digest)


def open_archives(asset_root, digests):
    """Open every content-addressed archive asset; one gives an archive, more a set.

    Parameters
    ----------
    asset_root : path-like
        Content-addressed asset directory (files named by SHA256).
    digests : sequence of str
        Archive digests listed in a manifest or checkpoint.

    Returns
    -------
    H01Archive or H01ArchiveSet
        Single archive for one digest, otherwise a set resolving by digest.
    """
    from pathlib import Path
    digests = list(dict.fromkeys(digests))
    if not digests:
        raise ValueError('A manifest must list at least one morphology archive asset')
    archives = [open_archive(Path(asset_root)/digest, digest) for digest in digests]
    if len(archives) == 1:
        return archives[0]
    return H01ArchiveSet(archives)


def open_archive_paths(paths):
    """Open archives named by path, pinning each to the digest of its file.

    Parameters
    ----------
    paths : sequence of path-like
        Archive files; one gives a single archive, more give a set.

    Returns
    -------
    tuple
        ``(archive, digests)`` where ``digests`` maps each path string to its
        SHA256 and ``archive`` is an ``H01Archive`` or ``H01ArchiveSet``.
    """
    import hashlib
    from pathlib import Path
    digests = {}
    for path in paths:
        path = Path(path)
        digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digests:
        raise ValueError('Name at least one morphology archive')
    archives = [open_archive(Path(path), digest) for path, digest in digests.items()]
    if len(archives) == 1:
        return archives[0], digests
    return H01ArchiveSet(archives), digests


class H01CellArchives:
    """Present an archive set through the single-archive ``load`` interface.

    Builders written for one archive call ``load(cell, component=...)``; this
    view chooses each cell's archive from a per-cell digest map, so a network
    whose nodes come from several archives constructs unchanged.

    Parameters
    ----------
    archives : H01Archive or H01ArchiveSet
        Source archives.
    digest_by_cell : mapping
        Cell identifier to the ``archive_sha256`` of the archive holding it.
    """

    def __init__(self, archives, digest_by_cell):
        self.archives = archives
        self.digest_by_cell = {str(k): v for k, v in digest_by_cell.items()}

    @property
    def neuron_ids(self):
        """Cells the view can resolve, in numeric order."""
        return tuple(sorted(self.digest_by_cell, key=int))

    def load(self, neuron_id, *, component):
        """Load one component of a cell from the archive its digest names."""
        return load_source(self.archives, str(neuron_id), component, self.digest_by_cell[str(neuron_id)])


def topology_from_evidence(evidence, contact_audit, archive):
    """Pin selected source components and anatomical placements for evolution.

    Parameters
    ----------
    evidence : dict
        Successful source network construction evidence.
    contact_audit : dict
        Verified anatomical contact audit, including blocked fragments.
    archive : H01Archive or H01ArchiveSet
        Checksum-verified source morphology archive, or a set resolved per
        source by the evidence's ``measured_anatomy.archive_sha256``.

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
        imported = load_source(archive, identity, component, anatomy['archive_sha256'])
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
                  max_cv_length_um=10., progress=None, environment_potassium=False, biology=None):
    """Build real selected cables and placed conductance contacts for a topology.

    Parameters
    ----------
    topology : H01Topology
        Immutable source profiles and active instance/contact identities.
    archive : H01Archive or H01ArchiveSet
        Verified immutable morphology source, or a set resolved per source by
        its ``archive_sha256``.
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

    Returns
    -------
    tuple
        BrainCell network and per-instance construction records. Initialization
        is separate so its time and memory can be measured independently.
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
            loaded[source_id] = load_source(archive, source_id, source['component'],
                                            source.get('archive_sha256'))
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
    for identity in doc['active_contacts']:
        edge, name = doc['contacts'][identity], 'syn_'+identity
        network.add_edges(name=name, pre='cell_'+edge['pre'], post='cell_'+edge['post'], method=pairs([(0, 0)]))
        network.add_projection(name=name, edges=name, synapse=name,
            weight=edge['initial_weight_us']*u.uS, delay=edge['delay_ms']*u.ms)
    release_shared_construction_data()
    return network, records


def release_shared_construction_data():
    """Match the source builder's bounded-memory handoff before allocating channel states.

    Immutable component data is shared across clones while building; the loaded-component
    table is released here. The CV geometry cache is owner-scoped on each morphology since
    commit 43bd52b5 (no module-level table remains), so nothing else is cleared.
    """
    from braintrace.datasets.h01 import _GLOBAL_LOADED_COMPONENTS
    _GLOBAL_LOADED_COMPONENTS.clear()
