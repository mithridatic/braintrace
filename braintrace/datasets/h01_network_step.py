"""Reusable BrainCell network step with persistent episode delivery queues."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from braincell.network.delivery import (
    advance_delivery_state, create_delivery_state, enqueue_future_events, write_arrivals,
)


class H01NetworkStep(brainstate.nn.Module):
    """Expose the pinned native network ordering as a reusable compiled step.

    Parameters
    ----------
    network : braincell.Network
        Constructed multicompartment network.
    dt_ms : float, optional
        Positive cable integration interval in milliseconds.

    Notes
    -----
    The step follows BrainCell network.engine.Network.run: arrivals, receptor
    preparation, begin-step, dynamics, probe sampling, enqueue, cursor advance.
    Delivery queues persist until reset, including across ARC event boundaries.
    """

    def __init__(self, network, dt_ms=0.005):
        super().__init__()
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be positive and finite")
        if not network.populations:
            raise ValueError("An empty network cannot step")
        network.init_state()
        self.network = network
        self.dt_ms = float(dt_ms)
        self.cells = tuple(pop.cell for pop in network.populations.values())
        self.setup = network._run_setup(dt=dt_ms*u.ms, delay_quantization="ceil",
                                        event_backend="auto", brainevent_backend="jax_raw")
        self.delivery = create_delivery_state(self.setup.delivery_blocks,
                                              populations=network.populations,
                                              delivery_ops=self.setup.delivery_ops)
        self.ring_buffers = self.delivery.ring_buffers
        self.ring_cursors = self.delivery.ring_cursors
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def update(self):
        """Advance a cable step and return all named population probes.

        Returns
        -------
        dict
            Nested population and probe values after dynamics advance.
        """
        dt = self.dt_ms*u.ms
        with brainstate.environ.context(dt=dt, t=self.tick.value*dt):
            write_arrivals(self.setup.delivery_blocks, self.delivery,
                           populations=self.network.populations)
            for cell in self.cells:
                cell._prepare_next_synapse_inputs()
            for cell in self.cells:
                cell._begin_step()
            for cell in self.cells:
                cell._update_dynamics()
            snapshots = {name: pop.cell.sample_probes()
                         for name, pop in self.network.populations.items()}
            enqueue_future_events(self.setup.delivery_blocks, self.delivery,
                                  populations=self.network.populations)
            advance_delivery_state(self.delivery)
            self.tick.value = self.tick.value + 1
            for cell in self.cells:
                cell._set_current_time(self.tick.value*dt)
            return snapshots

    def reset_state(self):
        """Reset physical states, clocks and all pending event deliveries.

        Returns
        -------
        None
        """
        self.network.reset_state()
        self.tick.value = jnp.zeros_like(self.tick.value)
        for state in self.ring_buffers + self.ring_cursors:
            state.value = u.math.zeros_like(state.value)
