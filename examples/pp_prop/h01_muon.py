"""Explicit matrix geometry and edge-shaped state for H01 masked Muon."""

from dataclasses import dataclass
import copy

import jax.numpy as jnp
import numpy as np
import optax


POLICY = dict(version='h01-masked-muon-v1', weight_decay=.1, beta=.95,
              ns_steps=5, eps=1e-8, ns_coeffs=[3.4445, -4.7750, 2.0315],
              nesterov=True, adaptive=False, preconditioning='frobenius',
              shape_scaling='width', learning_rates=dict(input=.001, recurrent=.0003,
              readout=.003), bias=dict(algorithm='adamw', b1=.9, b2=.999,
              eps=1e-8, weight_decay=.1, nesterov=True), workspace_limit_bytes=512*1024**2)


@dataclass(frozen=True)
class EdgeLayout:
    """Map distinct edges into independent matrix planes.

    Parameters
    ----------
    shape : tuple of int
        Plane, source and destination dimensions.
    coordinates : array-like
        One unique integer coordinate per edge, with shape ``(edges, 3)``.
    """

    shape: tuple
    coordinates: object

    def __post_init__(self):
        shape = tuple(self.shape)
        coords = np.asarray(self.coordinates)
        if (len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape)
                or coords.ndim != 2 or coords.shape[1] != 3
                or not np.issubdtype(coords.dtype, np.integer)
                or np.any(coords < 0) or np.any(coords >= np.array(shape))
                or len(set(map(tuple, coords))) != len(coords)):
            raise ValueError('Invalid or duplicate Muon edge coordinates')
        coords = coords.copy()
        coords.flags.writeable = False
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'coordinates', coords)

    def scatter(self, values):
        """Expand edge values into zero-filled matrix planes.

        Parameters
        ----------
        values : array
            Values in edge order.

        Returns
        -------
        array
            Dense temporary matrix planes.
        """
        if values.shape != (len(self.coordinates),):
            raise ValueError('Muon edge value shape differs from layout')
        return jnp.zeros(self.shape, dtype=values.dtype).at[tuple(self.coordinates.T)].set(values)

    def gather(self, matrices):
        """Restrict matrix values to existing edges.

        Parameters
        ----------
        matrices : array
            Dense matrix planes.

        Returns
        -------
        array
            Values in original edge order.
        """
        return matrices[tuple(self.coordinates.T)]


def contact_slots(contacts, inherited=None):
    """Assign stable parallel-contact planes without merging contacts.

    Parameters
    ----------
    contacts : sequence of tuple
        Stable contact identity, source identity and destination identity.
    inherited : mapping, optional
        Previously assigned slots; inactive contacts are discarded.

    Returns
    -------
    dict
        Active contact identities mapped to nonnegative plane indices.
    """
    if len({x[0] for x in contacts}) != len(contacts):
        raise ValueError('Duplicate Muon contact identity')
    inherited = inherited or {}
    slots, occupied = {}, {}
    for identity, pre, post in contacts:
        if identity in inherited:
            slot = inherited[identity]
            used = occupied.setdefault((pre, post), set())
            if type(slot) is not int or slot < 0 or slot in used:
                raise ValueError('Invalid inherited Muon contact slot')
            slots[identity] = slot
            used.add(slot)
    for identity, pre, post in sorted(contacts):
        if identity not in slots:
            used = occupied.setdefault((pre, post), set())
            slot = next(x for x in range(len(used)+1) if x not in used)
            slots[identity] = slot
            used.add(slot)
    return slots


class H01Muon:
    """Explicit H01 optimizer, with Muon on weights and AdamW on bias.

    Parameters
    ----------
    layouts : mapping
        Input and recurrent EdgeLayout objects.
    parameters : mapping
        All four H01 parameter arrays.
    slots : mapping, optional
        Stable recurrent contact plane assignments.
    workspace_limit_bytes : int, optional
        Conservative temporary allocation estimate limit.
    """

    def __init__(self, layouts, parameters, *, slots=None,
                 workspace_limit_bytes=512*1024**2):
        if set(layouts) != {'input', 'recurrent'} or set(parameters) != {
                'input', 'recurrent', 'readout_weight', 'readout_bias'}:
            raise ValueError('H01 Muon requires all four explicit parameter groups')
        if parameters['readout_weight'].ndim != 2 or parameters['readout_bias'].ndim != 1:
            raise ValueError('Invalid H01 readout parameter rank')
        self.layouts, self.slots = dict(layouts), dict(slots or {})
        self.shapes = {name: value.shape for name, value in parameters.items()}
        self.cores, self.rates, self.report = {}, {}, {}
        total = 0
        for name, value in parameters.items():
            layout = layouts.get(name)
            if layout is not None and value.shape != (len(layout.coordinates),):
                raise ValueError('Muon parameter shape differs from edge layout')
            shape = layout.shape if layout else ((1, *value.shape) if value.ndim == 2 else None)
            estimate = 0 if shape is None else value.dtype.itemsize * (
                12 * int(np.prod(shape)) + 6 * shape[0] * min(shape[1:])**2)
            total += estimate
            self.rates[name] = POLICY['learning_rates']['readout' if name.startswith('readout') else name]
            self.report[name] = dict(algorithm='masked_muon' if layout else
                ('muon' if shape else 'adamw'), weight_decay=.1,
                matrix_shape=list(shape) if shape else None, parameters=int(value.size),
                learning_rate=self.rates[name], workspace_estimate_bytes=estimate)
            if name != 'readout_bias':
                self.cores[name] = optax.contrib.scale_by_muon(
                    ns_coeffs=tuple(POLICY['ns_coeffs']), ns_steps=5, beta=.95,
                    eps=1e-8, nesterov=True, adaptive=False, preconditioning='frobenius',
                    weight_dimension_numbers=optax.contrib.MuonDimensionNumbers(1, 2) if layout else None)
        if type(workspace_limit_bytes) is not int or workspace_limit_bytes <= 0:
            raise ValueError('Invalid Muon workspace limit')
        if total > workspace_limit_bytes:
            raise ValueError(f'Muon workspace estimate {total} exceeds limit {workspace_limit_bytes}')
        self.workspace_limit_bytes = workspace_limit_bytes
        self.workspace_estimate_bytes = total
        self.bias = optax.adamw(self.rates['readout_bias'], b1=.9, b2=.999,
                                eps=1e-8, weight_decay=.1, nesterov=True)

    def init(self, parameters):
        """Initialize edge-shaped momentum and explicit bias state.

        Parameters
        ----------
        parameters : mapping
            H01 parameter values.

        Returns
        -------
        dict
            Named optimizer states.
        """
        states = {}
        for name, value in parameters.items():
            if name == 'readout_bias':
                states[name] = self.bias.init(value)
            else:
                # Initialization needs no dense workspace or orthogonalization.
                states[name] = self.cores[name].init(value)
        return states

    def update(self, parameters, gradients, states):
        """Apply one masked matrix update and retain sparse momentum.

        Parameters
        ----------
        parameters, gradients, states : mapping
            Matching parameter, gradient and optimizer groups.

        Returns
        -------
        tuple of dict
            Updated parameters and optimizer state.
        """
        if not parameters.keys() == gradients.keys() == states.keys() == self.shapes.keys():
            raise ValueError('H01 Muon requires a gradient and state for every group')
        updated, result = {}, {}
        for name, value in parameters.items():
            gradient, state = gradients[name], states[name]
            if value.shape != self.shapes[name] or gradient.shape != value.shape:
                raise ValueError('H01 Muon parameter or gradient shape mismatch')
            if name == 'readout_bias':
                delta, result[name] = self.bias.update(gradient, state, value)
            else:
                if state.mu.shape != value.shape:
                    raise ValueError('H01 Muon momentum shape mismatch')
                layout = self.layouts.get(name)
                if layout and not value.size:
                    result[name] = state._replace(count=optax.safe_int32_increment(state.count))
                    updated[name] = value
                    continue
                g = layout.scatter(gradient) if layout else gradient
                dense_state = state._replace(mu=layout.scatter(state.mu)) if layout else state
                direction, next_state = self.cores[name].update(g, dense_state)
                rows, cols = g.shape[-2:]
                scale = np.sqrt(max(1., cols / rows))
                if layout:
                    direction = layout.gather(direction)
                    next_state = next_state._replace(mu=layout.gather(next_state.mu))
                delta = -self.rates[name] * (scale * direction + .1 * value)
                result[name] = next_state
            updated[name] = optax.apply_updates(value, delta)
        return updated, result

    def metadata(self):
        """Return serializable policy and topology identities.

        Returns
        -------
        dict
            Policy, contact slots and edge-coordinate layouts.
        """
        return dict(policy=copy.deepcopy(POLICY), slots=dict(self.slots),
            groups=copy.deepcopy(self.report), workspace_estimate_bytes=self.workspace_estimate_bytes,
            workspace_limit_bytes=self.workspace_limit_bytes,
            layouts={name: dict(shape=list(layout.shape), coordinates=layout.coordinates.tolist())
                     for name, layout in self.layouts.items()})


def model_optimizer(model, parameters, *, inherited_slots=None):
    """Build the optimizer from the physical model's actual delivery order.

    Parameters
    ----------
    model : H01ArcModel
        Model with encoder CSR and physical delivery blocks.
    parameters : mapping
        Four H01 parameter groups.
    inherited_slots : mapping, optional
        Parent or checkpoint contact plane assignments.

    Returns
    -------
    H01Muon
        Validated topology-aware optimizer.
    """
    populations = list(model.stepper.network.populations)
    count = model.neuron_count
    table = getattr(model, 'contact_table', None)
    if table is not None:
        # Static capacity: one plane coordinate per table row; dormant rows sit at (0, 0)
        # on their own planes so the recurrent parameter keeps its fixed length.
        cells = {row: (int(np.asarray(table.pre.value)[row]), int(table.post_cell(model.forest, row)))
                 for row in table.rows}
        contacts = [(f'static-row-{row}', *cells.get(row, (0, 0))) for row in range(table.capacity)]
        slots = contact_slots(contacts, inherited_slots)
        coords = [(slots[key], pre, post) for key, pre, post in contacts]
    else:
        contacts = [(block.source.synapse, block.source.pre_population, block.source.post_population)
                    for block in model.stepper.setup.delivery_blocks]
        slots = contact_slots(contacts, inherited_slots)
        coords = [(slots[key], populations.index(pre), populations.index(post)) for key, pre, post in contacts]
    recurrent = EdgeLayout((max(slots.values(), default=0)+1, count, count),
                           np.asarray(coords, dtype=int).reshape(-1, 3))
    from .h01_remap import encoder_keys
    encoder_keys(model.source_ids, model.input_csr.indices, model.input_csr.indptr)
    rows = np.repeat(np.arange(441), np.diff(np.asarray(model.input_csr.indptr)))
    inputs = EdgeLayout((1, 441, count), np.column_stack((np.zeros(len(rows), dtype=int),
                        rows, np.asarray(model.input_csr.indices))))
    return H01Muon(dict(input=inputs, recurrent=recurrent), parameters, slots=slots)
