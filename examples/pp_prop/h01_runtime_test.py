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
from .h01_runtime import build_network, topology_from_evidence
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
