"""Static-capacity contact table for a forest population.

Spec: docs/specs/2026-09-16-h01-fused-population.md, section 3. Every forest
cell carries two shared BrainCell synapses at its soma (``contact_exc``:
0 mV / 2 ms, ``contact_inh``: -80 mV / 5 ms, the two kinds
``H01Topology.add_contact`` creates), so the forest runtime holds exactly two
synapse layouts whatever the contact count. Contacts live in a table of fixed
capacity: presynaptic forest cell, target position inside the kind's layout,
kind, delay, weight and an active flag, all ``brainstate`` states, with one
ring buffer ``(depth, capacity)`` of delayed events. Adding a contact writes
one row; a dormant row delivers nothing. No array changes shape, so the
compiled substep is reused across mutations.
"""

import math

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from .h01_forest_delivery import forest_presynaptic_spikes

KINDS = {'exc': dict(name='contact_exc', reversal_mv=0., tau_ms=2.),
         'inh': dict(name='contact_inh', reversal_mv=-80., tau_ms=5.)}


def contact_kind(reversal_mv, tau_ms):
    """Kind index (0 excitatory, 1 inhibitory) of a contact's reversal and time constant.

    Parameters
    ----------
    reversal_mv, tau_ms : float
        Contact parameters from the topology.

    Returns
    -------
    int
        0 or 1.

    Raises
    ------
    ValueError
        If the pair is neither shared kind (the static table holds only those).
    """
    for index, kind in enumerate(KINDS.values()):
        if abs(kind['reversal_mv']-reversal_mv) < 1e-9 and abs(kind['tau_ms']-tau_ms) < 1e-9:
            return index
    raise ValueError(f'Static contacts support only {list(KINDS)} kinds, not ({reversal_mv} mV, {tau_ms} ms)')


class ForestContactTable(brainstate.nn.Module):
    """Fixed-capacity contacts delivered inside one forest.

    Parameters
    ----------
    forest : H01ForestCell
        Initialized forest carrying the ``contact_exc``/``contact_inh`` layouts.
    capacity : int
        Contact rows.
    dt_ms : float
        Cable step; delays are quantized with ``ceil``.
    max_delay_ms : float
        Longest delay a row may hold (sets the ring depth).
    """

    def __init__(self, forest, capacity, *, dt_ms, max_delay_ms=.5):
        super().__init__()
        if int(capacity) < 1:
            raise ValueError('Contact capacity must be at least one row')
        self.capacity, self.dt_ms, self.max_delay_ms = int(capacity), float(dt_ms), float(max_delay_ms)
        self.depth = int(math.ceil(max_delay_ms/dt_ms-1e-9))+1
        self.layout_ids, self.point_index = [], []
        for kind in KINDS.values():
            found = [layout for layout in forest.runtime.layouts
                     if getattr(forest.runtime.layout_mechanisms[layout.id], 'instance_name', None) == kind['name']]
            if len(found) != 1:
                raise ValueError(f'The forest needs exactly one {kind["name"]!r} synapse layout')
            self.layout_ids.append(int(found[0].id))
            self.point_index.append(np.asarray(found[0].point_index, dtype=np.int64))
        self.pre = brainstate.ShortTermState(jnp.zeros(self.capacity, dtype=jnp.int32))
        self.target = brainstate.ShortTermState(jnp.zeros(self.capacity, dtype=jnp.int32))
        self.kind = brainstate.ShortTermState(jnp.zeros(self.capacity, dtype=jnp.int32))
        self.delay = brainstate.ShortTermState(jnp.zeros(self.capacity, dtype=jnp.int32))
        self.active = brainstate.ShortTermState(jnp.zeros(self.capacity))
        self.weight = brainstate.ShortTermState(jnp.zeros(self.capacity))
        self.ring = brainstate.HiddenState(jnp.zeros((self.depth, self.capacity)))
        self.cursor = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))
        self.rows = {}

    def position(self, kind, point):
        """Index of ``point`` inside the kind's synapse layout."""
        where = np.flatnonzero(self.point_index[kind] == int(point))
        if len(where) != 1:
            raise ValueError(f'Point {point} carries no {list(KINDS)[kind]} contact synapse')
        return int(where[0])

    def free_row(self):
        """First dormant row, or ``None`` when the table is full."""
        active = np.asarray(self.active.value)
        free = np.flatnonzero(active == 0.)
        return int(free[0]) if len(free) else None

    def write(self, row, *, pre, post_point, kind, weight_us, delay_ms, identity=None):
        """Activate one contact row in place.

        Parameters
        ----------
        row : int
            Table row.
        pre : int
            Presynaptic forest cell.
        post_point : int
            Postsynaptic point (a soma point carrying the shared synapses).
        kind : int
            0 excitatory, 1 inhibitory.
        weight_us : float
            Initial conductance magnitude.
        delay_ms : float
            Delay; must not exceed ``max_delay_ms``.
        identity : str, optional
            Topology contact identity, recorded in ``rows``.
        """
        if delay_ms > self.max_delay_ms+1e-12 or delay_ms < 0.:
            raise ValueError('Contact delay outside the table capacity')
        steps = int(math.ceil(delay_ms/self.dt_ms-1e-9))
        row = int(row)
        self.pre.value = self.pre.value.at[row].set(int(pre))
        self.target.value = self.target.value.at[row].set(self.position(int(kind), post_point))
        self.kind.value = self.kind.value.at[row].set(int(kind))
        self.delay.value = self.delay.value.at[row].set(steps)
        self.weight.value = self.weight.value.at[row].set(float(weight_us))
        self.active.value = self.active.value.at[row].set(1.)
        self.rows[row] = dict(identity=identity, pre=int(pre), post_point=int(post_point), kind=int(kind),
                              delay_steps=steps)

    def clear(self, row):
        """Return one row to dormancy (its queued events are dropped)."""
        row = int(row)
        self.active.value = self.active.value.at[row].set(0.)
        self.ring.value = self.ring.value.at[:, row].set(0.)
        self.rows.pop(row, None)

    def write_arrivals(self, forest):
        """Deliver the events due now into the two synapse layouts' ``pre_spike`` buffers."""
        cursor = self.cursor.value
        arrivals = self.ring.value[cursor]
        for index, layout_id in enumerate(self.layout_ids):
            buffer = jnp.zeros(len(self.point_index[index])).at[self.target.value].add(
                jnp.where(self.kind.value == index, arrivals, 0.))
            forest.runtime.state_buffers[(layout_id, 'pre_spike')] = u.Quantity(buffer[None, :], u.uS)
        self.ring.value = self.ring.value.at[cursor].set(0.)

    def enqueue(self, forest, magnitude):
        """Queue this substep's presynaptic spikes at each row's delay.

        Parameters
        ----------
        forest : H01ForestCell
            Stepped forest.
        magnitude : array
            ``(capacity,)`` conductance magnitudes (the model's trainable values).
        """
        spikes = forest_presynaptic_spikes(forest)
        event = jnp.where(spikes[self.pre.value], magnitude*self.active.value, 0.)
        slot = (self.cursor.value+self.delay.value) % self.depth
        self.ring.value = self.ring.value.at[slot, jnp.arange(self.capacity)].add(event)
        self.cursor.value = (self.cursor.value+1) % self.depth

    def reset_state(self, **kwargs):
        """Drop every queued event and rewind the cursor."""
        self.ring.value = jnp.zeros_like(self.ring.value)
        self.cursor.value = jnp.zeros_like(self.cursor.value)

    def post_cell(self, forest, row):
        """Forest cell that row ``row`` targets."""
        point = self.point_index[int(np.asarray(self.kind.value)[row])][int(np.asarray(self.target.value)[row])]
        return int(forest.cell_of_point[point])
