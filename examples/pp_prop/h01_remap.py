"""Stable-identity parameter and optimizer transport for H01 mutations."""

import jax
import numpy as np


def encoder_keys(cells, indices, indptr):
    """Return feature and instance identities for a CSR encoder.

    Parameters
    ----------
    cells : sequence of str
        Active instance order.
    indices, indptr : arrays
        Encoder CSR pattern.

    Returns
    -------
    tuple
        Stable keys in parameter order.
    """
    indices, indptr = np.asarray(indices), np.asarray(indptr)
    if (indptr.shape != (442,) or indptr[0] != 0 or indptr[-1] != len(indices)
            or np.any(np.diff(indptr) < 0) or np.any(indices < 0) or np.any(indices >= len(cells))):
        raise ValueError('Invalid encoder pattern for identity remapping')
    keys = tuple((feature, cells[int(indices[offset])]) for feature in range(441)
                 for offset in range(int(indptr[feature]), int(indptr[feature+1])))
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate encoder identity')
    return keys


def remap_mutation(old, new, parameters, indices, indptr):
    """Copy surviving and cloned parameters while retaining stable identities.

    Parameters
    ----------
    old, new : H01Topology
        Parent and mutated topology.
    parameters : mapping
        Four named parameter arrays.
    indices, indptr : arrays
        Parent encoder CSR pattern.

    Returns
    -------
    dict
        Child parameters, CSR pattern and per-group survivor index maps. New
        identities copy parent parameters but have a moment index of minus one.
    """
    before, after = old.to_dict(), new.to_dict()
    cells, contacts = before['active_cells'], before['active_contacts']
    child_cells, child_contacts = after['active_cells'], after['active_contacts']
    keys = encoder_keys(cells, indices, indptr)
    key_index = {key: index for index, key in enumerate(keys)}
    rows = [[] for _ in range(441)]
    for feature, identity in keys:
        if identity in child_cells:
            rows[feature].append(identity)
    for identity in child_cells:
        if identity not in cells:
            parent = after['instances'][identity]['parent_id']
            if parent not in cells:
                raise ValueError('Mutation transport requires a directly active clone parent')
            for feature in range(441):
                if (feature, parent) in key_index:
                    rows[feature].append(identity)
    order = {identity: index for index, identity in enumerate(child_cells)}
    rows = [sorted(row, key=order.__getitem__) for row in rows]
    new_keys = tuple((feature, identity) for feature, row in enumerate(rows) for identity in row)
    child_indices = np.array([order[identity] for _, identity in new_keys], dtype=np.int32)
    child_indptr = np.array([0, *np.cumsum([len(row) for row in rows])], dtype=np.int32)
    input_copy = [key_index.get(key, key_index.get((key[0], after['instances'][key[1]]['parent_id'])))
                  for key in new_keys]
    cell_copy = [cells.index(identity if identity in cells else after['instances'][identity]['parent_id'])
                 for identity in child_cells]
    edge_index = {identity: index for index, identity in enumerate(contacts)}
    edge_copy = [edge_index.get(identity, edge_index.get(after['contacts'][identity]['parent_id']))
                 for identity in child_contacts]
    result = dict(input=np.asarray(parameters['input'])[input_copy],
        recurrent=np.array([parameters['recurrent'][index] if index is not None else
            after['contacts'][identity]['initial_weight_us'] for identity, index in zip(child_contacts, edge_copy)],
            dtype=np.asarray(parameters['recurrent']).dtype),
        readout_weight=np.asarray(parameters['readout_weight'])[cell_copy],
        readout_bias=np.array(parameters['readout_bias']))
    maps = dict(input=np.array([key_index.get(key, -1) for key in new_keys], dtype=int),
        recurrent=np.array([edge_index.get(identity, -1) for identity in child_contacts], dtype=int),
        readout_weight=np.array([cells.index(identity) if identity in cells else -1 for identity in child_cells]),
        readout_bias=np.arange(360))
    return dict(parameters=result, input_indices=child_indices, input_indptr=child_indptr, survivor_maps=maps)


def remap_optimizer_group(old_state, new_template, survivor_map, old_shape, new_shape):
    """Remap one named optimizer group, zeroing moments of new identities.

    Parameters
    ----------
    old_state, new_template : pytree
        Parent optimizer group and freshly initialized child group.
    survivor_map : array
        Parent first-axis indices, or minus one for new identities.
    old_shape, new_shape : tuple
        Parameter shapes for this named group.

    Returns
    -------
    pytree
        Child optimizer group with survivor moments and original counters.
    """
    if jax.tree.structure(old_state) != jax.tree.structure(new_template):
        raise ValueError('Optimizer group structure changed')
    survivor_map = np.asarray(survivor_map)
    if survivor_map.shape != (new_shape[0],) or np.any(survivor_map < -1) or np.any(survivor_map >= old_shape[0]):
        raise ValueError('Invalid optimizer survivor map')
    def transport(path, old, template):
        old, template = np.asarray(old), np.asarray(template)
        if getattr(path[-1], 'name', None) == 'ns_coeffs':
            if old.dtype != template.dtype or not np.array_equal(old, template):
                raise ValueError('Muon polynomial coefficients changed')
            return old.copy()
        if old.shape == () and template.shape == ():
            return old.copy()
        if old.shape != tuple(old_shape) or template.shape != tuple(new_shape) or old.dtype != template.dtype:
            raise ValueError('Unsupported optimizer moment layout')
        result = np.zeros_like(template)
        valid = survivor_map >= 0
        result[valid] = old[survivor_map[valid]]
        return result
    return jax.tree_util.tree_map_with_path(transport, old_state, new_template)
