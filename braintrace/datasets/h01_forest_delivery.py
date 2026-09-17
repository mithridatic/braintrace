"""Contact delivery inside one forest population.

Spec: docs/specs/2026-09-16-h01-fused-population.md, section 2. BrainCell's
network delivery reads one presynaptic spike per population (``any`` over the
cell's compartments) and routes it through per-projection ring buffers. In a
forest every cell is one population member, so the presynaptic spike of a
contact is the forest's spike at the pre cell's output CV. This module builds
the same ``DeliveryBlock`` table BrainCell would build for the per-cell
network (one block per contact, its target the contact's own placed synapse
layout in the forest) and replaces only the spike read: everything downstream
(ring buffers, arrivals, the ARC model's trainable contact magnitudes) is the
per-cell code path unchanged.
"""

from dataclasses import dataclass
import math

import brainunit as u
import jax.numpy as jnp
import numpy as np
from braincell.network.delivery import delivery_blocks, make_delivery_op
from braincell.network.lowering import ConnectionBlock, resolve_synapse_layout


@dataclass(frozen=True)
class ForestContact:
    """One directed contact between two cells of a forest.

    Attributes
    ----------
    pre : int
        Presynaptic cell index in forest order.
    synapse : str
        Instance name of the placed synapse on the postsynaptic cell.
    weight_us : float
        Initial conductance magnitude in microsiemens.
    delay_ms : float
        Axonal delay in milliseconds.
    """

    pre: int
    synapse: str
    weight_us: float
    delay_ms: float


class _Population:
    def __init__(self, cell, name='forest'):
        self.cell, self.name = cell, name


def forest_delivery(forest, contacts, *, dt_ms, event_backend='scatter'):
    """Build delivery blocks and operators for the contacts of a forest.

    Parameters
    ----------
    forest : H01ForestCell
        Initialized forest cell carrying every contact's synapse layout.
    contacts : sequence of ForestContact
        Contacts in topology order.
    dt_ms : float
        Cable step; delays are quantized with ``ceil`` like the network path.
    event_backend : str, optional
        ``make_delivery_op`` backend; the ARC model replaces the operators anyway.

    Returns
    -------
    tuple
        ``(blocks, ops)``: one ``DeliveryBlock`` and one operator per contact.
    """
    population = _Population(forest)
    blocks = []
    for contact in contacts:
        layout_id, n_active, _ = resolve_synapse_layout(population, contact.synapse)
        delay = int(math.ceil(float(contact.delay_ms)/float(dt_ms)-1e-9))
        blocks.append(ConnectionBlock(pre_population='forest', post_population='forest', synapse=contact.synapse,
            layout_id=layout_id, n_active=n_active, pre_index=np.asarray([contact.pre], dtype=np.int32),
            post_index=np.zeros(1, dtype=np.int32), synapse_index=np.zeros(1, dtype=np.int32),
            weight=u.Quantity(np.asarray([contact.weight_us], dtype=np.float64), u.uS),
            delay_steps=np.asarray([delay], dtype=np.int32), buffer_size=delay+1))
    delivered = delivery_blocks(tuple(blocks))
    ops = tuple(make_delivery_op(block, pre_size=forest.forest_offsets.cells, post_size=1, backend=event_backend)
                for block in delivered)
    return delivered, ops


def forest_presynaptic_spikes(forest):
    """One boolean spike per cell: the forest spike at each cell's output CV.

    Parameters
    ----------
    forest : H01ForestCell
        Stepped forest cell.

    Returns
    -------
    array
        ``(cells,)`` booleans, the same value ``population_spike`` returns for
        each cell on the per-cell path (its output-site spike, every other CV
        masked to zero).
    """
    spike = forest.spike.value
    return jnp.any(spike.reshape(-1, spike.shape[-1])[:, np.asarray(forest.output_cv_ids)] != 0, axis=0)


def enqueue_forest_events(blocks, state, forest):
    """Project the forest's per-cell spikes into every block's ring buffer.

    Parameters
    ----------
    blocks : tuple of DeliveryBlock
        Blocks from :func:`forest_delivery`.
    state : DeliveryState
        Ring buffers, cursors and (possibly replaced) operators.
    forest : H01ForestCell
        Stepped forest cell.
    """
    pre_spike = forest_presynaptic_spikes(forest)
    for index, block in enumerate(blocks):
        event = state.delivery_ops[index](pre_spike)
        buffer = state.ring_buffers[index]
        target = (state.ring_cursors[index].value+int(block.delay_steps)) % buffer.value.shape[0]
        buffer.value = buffer.value.at[target].add(event.reshape((1, block.source.n_active)))
