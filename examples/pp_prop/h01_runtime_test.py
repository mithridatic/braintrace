"""Rebuild and execute real donor channels on original and evolved cables."""

from types import SimpleNamespace

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01_anatomy_test import imported
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
from braintrace.datasets.h01_network import _regions
from braintrace.datasets.h01_network_init import init_h01_network_states
from .h01_topology import H01Topology
from .h01_runtime import build_network, release_shared_construction_data, topology_from_evidence
from .h01_arc_model import H01ArcModel


def _spatial_fixture(imported):
    from dataclasses import asdict
    from braintrace.datasets.h01_biology import topology_digest
    from braintrace.datasets.h01_spine_assembly_test import explicit_spine
    topology = _manifest(imported).clone('12', stage='clone')
    child = topology.to_dict()['active_cells'][-1]
    # Topology permits one active contact per directed pair. Use distinct
    # sources to exercise head-targeted and ordinary contacts on the same cell.
    topology = topology.clone('12', stage='second-source')
    other = topology.to_dict()['active_cells'][-1]
    topology = topology.add_contact('12', child, stage='contact')
    topology = topology.add_contact(other, child, stage='ordinary-contact')
    document = topology.to_dict()
    spine = explicit_spine(imported)
    for contact in document['active_contacts']:
        document['contacts'][contact]['post_site'] = [spine.parent_branch, spine.parent_x]
    topology = H01Topology.from_dict(document)
    record = dict(asdict(spine), origin='synthetic')
    biology = dict(schema='h01-biology-spines-v1', topology_sha256=topology_digest(document),
        coordinate_frame='h01-swc-um', cells={identity: dict(source_sha256=imported.source_sha256,
            spines=[record] if identity == child else []) for identity in document['active_cells']},
        contact_heads={document['active_contacts'][0]: spine.identity},
        release_probability={identity: .36 for identity in document['active_contacts']})
    return topology, biology, child


def test_spine_builder_remaps_actual_contacts_probes_and_compiled_cable(imported):
    from braintrace.datasets.h01_biology import H01SpatialManifest
    from braintrace.datasets.h01_spine_assembly import assembled_location
    with brainstate.environ.context(precision=64, dt=.005*u.ms):
        topology, biology, child = _spatial_fixture(imported)
        archive = SimpleNamespace(load=lambda *a, **k: imported)
        network, records = build_network(topology, archive, biology=biology)
        record = records[child]
        contact, ordinary = topology.to_dict()['active_contacts']
        assert record['contact_sites'][contact]['assembled'] == [record['spine_assembly']['heads']['synthetic_0'], .5]
        site = topology.to_dict()['contacts'][ordinary]['post_site']
        assert record['contact_sites'][ordinary]['assembled'] == list(assembled_location(record, site))
        assert record['biology_manifest_sha256'] == H01SpatialManifest(biology, topology.to_dict()).sha256
        assert record['declared_spine_origins'] == {'synthetic_0': 'synthetic'}
        assert record['measured_anatomy']['source_sha256'] == imported.source_sha256
        init_h01_network_states(network, progress=lambda _: None)
        model = H01ArcModel(network, topology.to_dict()['active_cells'], release_probability=[.36, .36])
        result = brainstate.transform.jit(model.update)(jnp.zeros(441))
        assert np.isfinite(result).all() and int(model.release.tick.value) == 20
        assert len(model.recurrent_weight.value) == 2
        # Deliver a real queued conductance into the head-targeted receptor.
        # The ordinary contact must remain silent on the same postsynaptic cell.
        queue = model.stepper.ring_buffers[0]
        queue.value = .001*u.math.ones_like(queue.value)
        assert np.isfinite(brainstate.transform.jit(model.update)(jnp.zeros(441))).all()
        probes = network.populations['cell_'+child].cell.sample_probes()
        assert float(probes['syn_'+contact+'_g'].to_decimal(u.uS).reshape(())) > 0
        assert float(probes['syn_'+ordinary+'_g'].to_decimal(u.uS).reshape(())) == 0


def test_invalid_spatial_topology_is_rejected_before_archive_loading(imported):
    with brainstate.environ.context(precision=64):
        topology, biology, _ = _spatial_fixture(imported)
        biology['topology_sha256'] = 'f'*64
        with pytest.raises(ValueError, match='topology digest'):
            build_network(topology, None, biology=biology)


def _manifest(imported):
    annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=('L2', 'pyramidal')))
    _, record = make_h01_ei_cell(imported, annotations, polarity='E', regions=_regions(imported),
        region_basis='Source-labelled regions', solver='h01_staggered_calcium_implicit', pop_size=(1,))
    soma = list(imported.anatomy().soma_location().evaluate(imported.morphology).points[0])
    source = dict(component=0, source_sha256=imported.source_sha256, polarity='E', donor=record['donor'],
        profile=record['borrowed_dynamics'], source_tags=record['source_tags'],
        region_basis='Source-labelled regions', soma_site=soma, output_site=soma)
    doc = dict(schema='h01-topology-v1', sources={'12': source},
        instances={'12': dict(source_id='12', parent_id=None, synthetic=False, created_stage='source')},
        contacts={}, active_cells=['12'], active_contacts=[], next_id=0, blocked_contacts=[], mutations=[])
    return H01Topology.from_dict(doc)


def test_real_clone_contact_and_pruning_rebuild_finite_cells(imported):
    archive = SimpleNamespace(load=lambda identity, component: imported)
    with brainstate.environ.context(precision=64, dt=.005*u.ms):
        original = _manifest(imported)
        cloned = original.clone('12', stage='clone')
        child = cloned.to_dict()['active_cells'][-1]
        connected = cloned.add_contact('12', child, stage='contact')
        network, records = build_network(connected, archive)
        init_h01_network_states(network, progress=lambda _: None)
        model = H01ArcModel(network, connected.to_dict()['active_cells'])
        assert len(network.projections) == 1
        assert records['12']['borrowed_dynamics'] == records[child]['borrowed_dynamics']
        assert np.isfinite(brainstate.transform.jit(model.update)(jnp.zeros(441))).all()
        contact = connected.to_dict()['active_contacts'][0]
        removed = connected.prune_contact(contact, stage='prune-contact').prune_cell('12', stage='prune-cell')
        surviving, _ = build_network(removed, archive)
        init_h01_network_states(surviving, progress=lambda _: None)
        model = H01ArcModel(surviving, removed.to_dict()['active_cells'])
        assert len(surviving.projections) == 0
        assert np.isfinite(brainstate.transform.jit(model.update)(jnp.zeros(441))).all()


@pytest.mark.parametrize('change', ['digest', 'profile', 'output'])
def test_changed_source_profile_or_output_placement_rejected(imported, change):
    with brainstate.environ.context(precision=64):
        topology = _manifest(imported).clone('12', stage='clone')
        child = topology.to_dict()['active_cells'][-1]
        document = topology.add_contact('12', child, stage='contact').to_dict()
        if change == 'digest':
            document['sources']['12']['source_sha256'] = 'a'*64
        elif change == 'profile':
            document['sources']['12']['profile']['initial_mv'] += 1
        else:
            contact = document['active_contacts'][0]
            document['contacts'][contact]['pre_site'] = [0, .123]
        with pytest.raises(ValueError, match='digest|profile|output'):
            build_network(H01Topology.from_dict(document), SimpleNamespace(load=lambda *a, **k: imported))


def test_exported_manifest_keeps_source_and_anatomical_contact_evidence(imported):
    archive = SimpleNamespace(load=lambda *a, **k: imported)
    with brainstate.environ.context(precision=64):
        _, records = build_network(_manifest(imported), archive)
        site = records['12']['output_site']['source_point']
        contact = dict(annotation_id='observed', enabled=True, pre_cell='12', post_cell='12',
            weight_us=.01, reversal_mv=0., tau_ms=2., delay_ms=.5)
        evidence = dict(simulated_cell_ids=['12'], cells=records, contacts=[contact],
                        blocked_contacts=[{'reason': 'disconnected fragment'}])
        audited = dict(annotation_id='observed', construction_ready=True,
            pre_placement={'cable_location': site}, post_placement={'cable_location': site})
        topology = topology_from_evidence(evidence, {'contacts': [audited]}, archive).to_dict()
        assert topology['sources']['12']['source_sha256'] == imported.source_sha256
        assert topology['contacts']['observed']['synthetic'] is False
        assert topology['blocked_contacts'] == evidence['blocked_contacts']
        audited['construction_ready'] = False
        with pytest.raises(ValueError, match='blocked'):
            topology_from_evidence(evidence, {'contacts': [audited]}, archive)
        records['12']['measured_anatomy']['source_sha256'] = 'a'*64
        with pytest.raises(ValueError, match='differs'):
            topology_from_evidence(evidence, {'contacts': [audited]}, archive)


def test_release_handoff_clears_loaded_components_without_a_global_geometry_cache():
    from braintrace.datasets import h01, h01_construction
    assert not hasattr(h01_construction, '_GEOMETRY_CACHE')
    h01._GLOBAL_LOADED_COMPONENTS['x'] = object()
    release_shared_construction_data()
    assert h01._GLOBAL_LOADED_COMPONENTS == {}


def _stub_archive(digest, component, calls):
    """A verified-archive stand-in: digest, source label and a recording ``load``."""
    def load(identity, component=None, _component=component):
        calls.append((digest, str(identity), component))
        return _component
    return SimpleNamespace(archive_sha256=digest, source='stub-'+digest[:4], load=load)


def _archive_set(*archives):
    from braintrace.datasets.h01_archive_set import H01ArchiveSet
    return H01ArchiveSet(archives)


def _two_archive_topology(imported):
    from dataclasses import replace
    doc = _manifest(imported).to_dict()
    doc['sources']['12']['archive_sha256'] = 'a'*64
    other = replace(imported, neuron_id='77')
    doc['sources']['77'] = dict(doc['sources']['12'], archive_sha256='b'*64)
    doc['instances']['77'] = dict(source_id='77', parent_id=None, synthetic=False, created_stage='source')
    doc['active_cells'] = ['12', '77']
    return H01Topology.from_dict(doc), other


def test_two_archive_topology_resolves_each_source_from_its_own_archive(imported):
    from .h01_runtime import is_archive_set, load_source
    topology, other = _two_archive_topology(imported)
    calls = []
    first, second = _stub_archive('a'*64, imported, calls), _stub_archive('b'*64, other, calls)
    archives = _archive_set(first, second)
    assert is_archive_set(archives) and not is_archive_set(first)
    with brainstate.environ.context(precision=64):
        _, records = build_network(topology, archives)
    assert sorted(records) == ['12', '77']
    assert calls == [('a'*64, '12', 0), ('b'*64, '77', 0)]
    assert load_source(first, '12', 0, 'ignored') is imported and calls[-1] == ('a'*64, '12', 0)


def test_source_digest_absent_from_the_archive_set_raises(imported):
    topology, _ = _two_archive_topology(imported)
    archives = _archive_set(_stub_archive('a'*64, imported, []))
    with brainstate.environ.context(precision=64), pytest.raises(KeyError, match='b'*8):
        build_network(topology, archives)


def test_evidence_topology_loads_each_cell_through_its_recorded_archive_digest(imported):
    from dataclasses import replace
    other = replace(imported, neuron_id='77')
    calls = []
    archives = _archive_set(_stub_archive('a'*64, imported, calls), _stub_archive('b'*64, other, calls))
    with brainstate.environ.context(precision=64):
        _, records = build_network(_manifest(imported), SimpleNamespace(load=lambda *a, **k: imported))
    record = records['12']
    record['measured_anatomy'] = dict(record['measured_anatomy'], archive_sha256='a'*64)
    second = dict(record, measured_anatomy=dict(record['measured_anatomy'], archive_sha256='b'*64,
                                                member='77.0.swc'))
    evidence = dict(simulated_cell_ids=['12', '77'], cells={'12': record, '77': second}, contacts=[],
                    blocked_contacts=[])
    topology = topology_from_evidence(evidence, {'contacts': []}, archives).to_dict()
    assert calls == [('a'*64, '12', 0), ('b'*64, '77', 0)]
    assert topology['sources']['12']['archive_sha256'] == 'a'*64
    assert topology['sources']['77']['archive_sha256'] == 'b'*64
    evidence['cells']['77']['measured_anatomy']['archive_sha256'] = 'c'*64
    with pytest.raises(KeyError, match='c'*8):
        topology_from_evidence(evidence, {'contacts': []}, archives)


def test_open_archives_gives_one_archive_or_a_digest_keyed_set(monkeypatch, tmp_path):
    from . import h01_runtime
    opened = []
    monkeypatch.setattr(h01_runtime, 'H01Archive', lambda path, expected_sha256: opened.append(
        (path.name, expected_sha256)) or SimpleNamespace(path=path, archive_sha256=expected_sha256, source='stub'))
    single = h01_runtime.open_archives(tmp_path, ['a'*64])
    assert single.archive_sha256 == 'a'*64 and not h01_runtime.is_archive_set(single)
    both = h01_runtime.open_archives(tmp_path, ['a'*64, 'b'*64, 'a'*64])
    assert h01_runtime.is_archive_set(both) and sorted(both.archives()) == ['a'*64, 'b'*64]
    assert both.archive('b'*64).path == tmp_path/('b'*64)
    assert opened == [('a'*64, 'a'*64), ('a'*64, 'a'*64), ('b'*64, 'b'*64)]
    with pytest.raises(ValueError, match='at least one'):
        h01_runtime.open_archives(tmp_path, [])


def test_open_archive_paths_pins_each_file_to_its_own_digest(monkeypatch, tmp_path):
    import hashlib
    from . import h01_runtime
    first, second = tmp_path/'first.zip', tmp_path/'second.zip'
    first.write_bytes(b'one')
    second.write_bytes(b'two')
    monkeypatch.setattr(h01_runtime, 'H01Archive', lambda path, expected_sha256: SimpleNamespace(
        path=path, archive_sha256=expected_sha256, source='stub',
        load=lambda identity, component: (path.name, identity)))
    archive, digests = h01_runtime.open_archive_paths([first])
    assert digests == {str(first): hashlib.sha256(b'one').hexdigest()} and archive.archive_sha256 == digests[str(first)]
    archives, digests = h01_runtime.open_archive_paths([first, second])
    assert h01_runtime.is_archive_set(archives) and list(digests) == [str(first), str(second)]
    view = h01_runtime.H01CellArchives(archives, {'12': digests[str(second)], '7': digests[str(first)]})
    assert view.neuron_ids == ('7', '12')
    assert view.load(12, component=0) == ('second.zip', '12') and view.load('7', component=1) == ('first.zip', '7')
    with pytest.raises(ValueError, match='at least one'):
        h01_runtime.open_archive_paths([])
