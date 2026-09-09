"""Mutation transport preserves survivors and initializes only new moments."""

import numpy as np
import pytest

from .h01_topology_test import _topology
from .h01_remap import encoder_keys, remap_mutation, remap_optimizer_group


def _parent():
    topology = _topology().add_contact('5805562981', '5965472721', stage='contact')
    parameters = dict(input=np.array([2., 3.]), recurrent=np.array([.12]),
        readout_weight=np.arange(1080.).reshape(3, 360), readout_bias=np.arange(360.))
    return topology, parameters, np.array([0, 1]), np.array([0, 2, *([2]*440)])


def test_clone_copies_weights_but_not_moments():
    old, parameters, indices, indptr = _parent()
    new = old.clone('5805562981', stage='clone')
    result = remap_mutation(old, new, parameters, indices, indptr)
    np.testing.assert_array_equal(result['parameters']['input'], [2., 3., 2.])
    np.testing.assert_array_equal(result['parameters']['recurrent'], [.12, .12])
    np.testing.assert_array_equal(result['parameters']['readout_weight'][-1], parameters['readout_weight'][0])
    np.testing.assert_array_equal(result['survivor_maps']['input'], [0, 1, -1])
    state = {'moment': np.array([5., 6.]), 'count': np.array(7, dtype=np.int32)}
    template = {'moment': np.zeros(3), 'count': np.array(0, dtype=np.int32)}
    restored = remap_optimizer_group(state, template, result['survivor_maps']['input'], (2,), (3,))
    np.testing.assert_array_equal(restored['moment'], [5., 6., 0.])
    assert restored['count'] == 7


def test_prune_and_new_contact_keep_surviving_identity():
    old, parameters, indices, indptr = _parent()
    new = old.prune_cell('5805562981', stage='prune').add_contact('5965472721', '7196644737', stage='add')
    result = remap_mutation(old, new, parameters, indices, indptr)
    np.testing.assert_array_equal(result['parameters']['input'], [3.])
    np.testing.assert_array_equal(result['survivor_maps']['input'], [1])
    np.testing.assert_array_equal(result['parameters']['recurrent'], [.01])
    np.testing.assert_array_equal(result['survivor_maps']['recurrent'], [-1])
    assert encoder_keys(new.to_dict()['active_cells'], result['input_indices'], result['input_indptr']) == ((0, '5965472721'),)


def test_empty_encoder_survives_mutation():
    old, parameters, _, _ = _parent()
    parameters['input'] = np.zeros(0)
    result = remap_mutation(old, old, parameters, np.zeros(0, dtype=int), np.zeros(442, dtype=int))
    assert result['parameters']['input'].shape == (0,)


@pytest.mark.parametrize('case', ['structure', 'map', 'layout'])
def test_invalid_optimizer_transport_rejected(case):
    old, new, mapping = {'m': np.ones(2)}, {'m': np.zeros(3)}, [0, 1, -1]
    if case == 'structure':
        new = {'other': np.zeros(3)}
    elif case == 'map':
        mapping = [0, 1, 2]
    else:
        old = {'m': np.ones((2, 2))}
    with pytest.raises(ValueError, match='Optimizer|optimizer'):
        remap_optimizer_group(old, new, mapping, (2,), (3,))


def test_invalid_or_duplicate_encoder_identity():
    with pytest.raises(ValueError, match='Invalid'):
        encoder_keys(['a'], np.array([2]), np.array([0, *([1]*441)]))
    with pytest.raises(ValueError, match='Duplicate'):
        encoder_keys(['a'], np.array([0, 0]), np.array([0, *([2]*441)]))


def test_muon_coefficients_are_not_parameter_moments():
    from collections import namedtuple
    State = namedtuple('State', 'mu ns_coeffs')
    old = State(np.ones(3), np.array([3., -4., 2.]))
    new = State(np.zeros(4), old.ns_coeffs.copy())
    result = remap_optimizer_group(old, new, [0, 1, 2, -1], (3,), (4,))
    np.testing.assert_array_equal(result.mu, [1., 1., 1., 0.])
    np.testing.assert_array_equal(result.ns_coeffs, old.ns_coeffs)
    with pytest.raises(ValueError, match='coefficients'):
        remap_optimizer_group(old, new._replace(ns_coeffs=np.ones(3)), [0, 1, 2, -1], (3,), (4,))
