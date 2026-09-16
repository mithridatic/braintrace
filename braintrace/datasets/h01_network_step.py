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

    def __init__(self, network, dt_ms=0.000625):   # qualified step, docs/evidence/h01-timestep-ladder.json
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
        self.forest = self.cells[0] if len(self.cells) == 1 and hasattr(self.cells[0], 'forest_offsets') else None
        if self.forest is not None and getattr(self.forest, 'contacts', ()):
            from dataclasses import replace
            from .h01_forest_delivery import forest_delivery
            blocks, ops = forest_delivery(self.forest, self.forest.contacts, dt_ms=dt_ms)
            self.setup = replace(self.setup, delivery_blocks=blocks, delivery_ops=ops)
        self.delivery = create_delivery_state(self.setup.delivery_blocks,
                                              populations=network.populations,
                                              delivery_ops=self.setup.delivery_ops)
        self.ring_buffers = self.delivery.ring_buffers
        self.ring_cursors = self.delivery.ring_cursors
        self.has_delivery = bool(self.setup.delivery_blocks)
        self.has_synapses = any(any(getattr(l, "kind", "").startswith("synapse") for l in getattr(cell._runtime, "layouts", ())) for cell in self.cells)
        self._dt = self.dt_ms * u.ms
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))
        self.chemistry = None

    def bind_chemistry(self, coupling):
        """Attach physical potassium coupling before compiling any steps.

        Parameters
        ----------
        coupling : PotassiumCoupling
            Bindings for these initialized cells and a matching physical dt.
        """
        if self.chemistry is not None or int(self.tick.value) != 0:
            raise ValueError('Bind chemistry once, before stepping')
        if (coupling.environment.dt_ms != self.dt_ms or
                {id(b.cell) for b in coupling.bindings} != {id(c) for c in self.cells}):
            raise ValueError('Chemistry must cover this network at the same dt')
        self.chemistry = coupling

    def update(self, sample_probes=True):
        """Advance a cable step and return all named population probes.

        Parameters
        ----------
        sample_probes : bool, optional
            Whether to sample and return all placed probes; default True.

        Returns
        -------
        dict or None
            Nested population and probe values after dynamics advance, or None.
        """
        dt = self._dt
        with brainstate.environ.context(dt=dt, t=self.tick.value*dt):
            if self.has_delivery:
                write_arrivals(self.setup.delivery_blocks, self.delivery,
                               populations=self.network.populations)
            if self.has_synapses:
                for cell in self.cells:
                    cell._prepare_next_synapse_inputs()
                for cell in self.cells:
                    cell._begin_step()
            potassium_current = self.chemistry.currents() if self.chemistry is not None else None
            for cell in self.cells:
                cell._update_dynamics()
            if self.chemistry is not None:
                self.chemistry.update(potassium_current)
            snapshots = {name: pop.cell.sample_probes()
                         for name, pop in self.network.populations.items()} if sample_probes else None
            if self.has_delivery and self.forest is not None:
                from .h01_forest_delivery import enqueue_forest_events
                enqueue_forest_events(self.setup.delivery_blocks, self.delivery, self.forest)
                advance_delivery_state(self.delivery)
            elif self.has_delivery:
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
        if self.chemistry is not None:
            self.chemistry.reset_state()
        self.network.reset_state()
        self.tick.value = jnp.zeros_like(self.tick.value)
        for state in self.ring_buffers + self.ring_cursors:
            state.value = u.math.zeros_like(state.value)
