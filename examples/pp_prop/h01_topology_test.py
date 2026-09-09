"""Immutable H01 provenance and permitted structural mutations."""

import pytest

from .h01_topology import H01Topology


@pytest.mark.parametrize('damage', ['schema', 'empty', 'missing_active', 'counter', 'source_site',
    'source_digest', 'profile_identity', 'unknown_source', 'missing_parent', 'cycle', 'foreign_parent',
    'unknown_endpoint', 'receptor', 'negative_weight', 'duplicate_pair'])
def test_invalid_provenance_and_topology_are_rejected(damage):
    topology = _topology().add_contact('5805562981', '5965472721', stage='seed')
    document = topology.to_dict()
    first = document['active_cells'][0]
    source, instance = document['sources'][first], document['instances'][first]
    contact = document['contacts'][document['active_contacts'][0]]
    if damage == 'schema':
        document['schema'] = 'other'
    elif damage == 'empty':
        document['active_cells'] = []
    elif damage == 'missing_active':
        document['active_cells'].append('missing')
    elif damage == 'counter':
        document['next_id'] = -1
    elif damage == 'source_site':
        source['soma_site'] = [0, 1.1]
    elif damage == 'source_digest':
        source['source_sha256'] = 'bad'
    elif damage == 'profile_identity':
        source['profile']['polarity'] = 'I'
    elif damage == 'unknown_source':
        instance['source_id'] = 'missing'
    elif damage in ('missing_parent', 'cycle', 'foreign_parent'):
        instance['synthetic'] = True
        instance['parent_id'] = {'missing_parent': 'missing', 'cycle': first, 'foreign_parent': '5965472721'}[damage]
    elif damage == 'unknown_endpoint':
        contact['post'] = 'missing'
    elif damage == 'receptor':
        contact['reversal_mv'] = -80.
    elif damage == 'negative_weight':
        contact['initial_weight_us'] = -.1
    else:
        document['contacts']['duplicate'] = dict(contact)
        document['active_contacts'].append('duplicate')
    with pytest.raises(ValueError):
        H01Topology.from_dict(document)


def test_mutation_stage_and_historical_identity_cannot_be_reused():
    topology = _topology()
    with pytest.raises(ValueError, match='stage'):
        topology.clone('5805562981', stage='')
    document = topology.clone('5805562981', stage='first').to_dict()
    document['next_id'] = 0
    with pytest.raises(ValueError, match='reuse'):
        H01Topology.from_dict(document).clone('5805562981', stage='again')


def _topology():
    ids = ['5805562981', '5965472721', '7196644737']
    sources = {identity: {'component': 0, 'source_sha256': 'a'*64,
        'polarity': 'I' if i == 1 else 'E', 'donor': 'frozen-donor',
        'profile': {'polarity': 'I' if i == 1 else 'E'},
        'soma_site': [0, .5], 'output_site': [1, .25] if i == 0 else None}
        for i, identity in enumerate(ids)}
    return H01Topology.from_dict({'schema': 'h01-topology-v1', 'sources': sources,
        'instances': {identity: {'source_id': identity, 'parent_id': None,
                                'synthetic': False, 'created_stage': 'source'} for identity in ids},
        'contacts': {}, 'active_cells': ids, 'active_contacts': [], 'next_id': 0,
        'blocked_contacts': [{'annotation_id': 'fragment', 'reason': 'disconnected fragment'}],
        'mutations': []})


def test_add_contact_uses_designated_output_soma_and_inherited_identity():
    initial = _topology()
    evolved = initial.add_contact('5805562981', '5965472721', stage='add')
    doc = evolved.to_dict()
    edge = doc['contacts'][doc['active_contacts'][0]]
    assert edge['synthetic'] and edge['pre_site'] == [1, .25] and edge['post_site'] == [0, .5]
    assert (edge['reversal_mv'], edge['tau_ms'], edge['delay_ms']) == (0., 2., .5)
    assert initial.to_dict()['contacts'] == {}
    inhibitory = evolved.add_contact('5965472721', '7196644737', stage='inhibitory').to_dict()
    edge = inhibitory['contacts'][inhibitory['active_contacts'][-1]]
    assert edge['pre_site'] == [0, .5]
    assert (edge['reversal_mv'], edge['tau_ms']) == (-80., 5.)


def test_clone_copies_incident_contacts_and_keeps_source_parent_history():
    initial = _topology().add_contact('5805562981', '5965472721', stage='contact')
    cloned = initial.clone('5805562981', stage='clone')
    doc = cloned.to_dict()
    child = doc['active_cells'][-1]
    assert doc['instances'][child]['source_id'] == '5805562981'
    assert doc['instances'][child]['parent_id'] == '5805562981'
    copy = doc['contacts'][doc['active_contacts'][-1]]
    assert copy['pre'] == child and copy['synthetic']
    assert copy['parent_id'] == initial.to_dict()['active_contacts'][0]
    pruned = cloned.prune_cell('5805562981', stage='prune').to_dict()
    assert '5805562981' in pruned['instances'] and '5805562981' not in pruned['active_cells']
    assert pruned['sources'] == initial.to_dict()['sources']
    assert pruned['blocked_contacts'] == initial.to_dict()['blocked_contacts']


def test_prune_contacts_retains_records_and_rejects_empty_network():
    topology = _topology().add_contact('5805562981', '5965472721', stage='contact')
    identity = topology.to_dict()['active_contacts'][0]
    pruned = topology.prune_contact(identity, stage='prune')
    assert identity in pruned.to_dict()['contacts'] and pruned.to_dict()['active_contacts'] == []
    one = pruned.prune_cell('5805562981', stage='a').prune_cell('5965472721', stage='b')
    with pytest.raises(ValueError, match='empty'):
        one.prune_cell('7196644737', stage='c')


def test_rank_absent_pairs_is_deterministic_and_excludes_existing_contacts():
    topology = _topology().add_contact('5805562981', '5965472721', stage='contact')
    ranked = topology.rank_absent_pairs({'5805562981': 2., '7196644737': 1.},
                                       {('5965472721', '7196644737'): -3.})
    assert ranked[0] == ('5965472721', '7196644737')
    assert ('5805562981', '5965472721') not in ranked
    assert all(a != b for a, b in ranked)


def test_round_trip_and_copy_do_not_expose_mutable_provenance():
    topology = _topology()
    document = topology.to_dict()
    document['sources']['5805562981']['donor'] = 'modified'
    assert topology.to_dict()['sources']['5805562981']['donor'] == 'frozen-donor'
    assert H01Topology.from_dict(topology.to_dict()) == topology


@pytest.mark.parametrize('operation', ['self', 'duplicate', 'inactive_clone', 'inactive_prune'])
def test_invalid_mutations(operation):
    topology = _topology().add_contact('5805562981', '5965472721', stage='contact')
    with pytest.raises(ValueError):
        if operation == 'self':
            topology.add_contact('5805562981', '5805562981', stage='bad')
        elif operation == 'duplicate':
            topology.add_contact('5805562981', '5965472721', stage='bad')
        elif operation == 'inactive_clone':
            topology.clone('missing', stage='bad')
        else:
            topology.prune_contact('missing', stage='bad')
