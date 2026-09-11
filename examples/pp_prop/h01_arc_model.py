"""Exploratory ARC encoder and readout around actual H01 cable dynamics."""

from dataclasses import replace

import brainevent
import brainstate
import braintrace
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_network_step import H01NetworkStep


class H01ArcModel(brainstate.nn.Module):
    """Encode ARC events into held soma currents on multicompartment cells.

    Parameters
    ----------
    network : braincell.Network
        Constructed H01 cells with zero baseline soma clamps and voltage probes.
    source_ids : sequence of str
        Stable source identifiers in network population order.
    seed : int, optional
        BrainState encoder/readout initialization seed.
    dt_ms : float, optional
        Cable interval dividing the fixed 0.1 ms ARC event.
    input_pattern : tuple of arrays, optional
        Explicit CSR indices and indptr for checkpoint restore or evolution.
    checkpoint_substeps : bool, optional
        Rematerialize cable substeps during reverse-mode differentiation.

    Notes
    -----
    This computational interface does not qualify donor physiology or pp-prop.
    Compiler and allocation gates must pass before a training claim is made.
    """

    def __init__(self, network, source_ids, *, seed=21, dt_ms=0.005, input_pattern=None,
                 checkpoint_substeps=True):
        super().__init__()
        self.source_ids = tuple(source_ids)
        if not self.source_ids or any(not isinstance(x, str) for x in self.source_ids):
            raise ValueError("Nonempty string source IDs are required")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("Source IDs must be unique")
        if len(self.source_ids) != len(network.populations):
            raise ValueError("Source IDs must match network populations")
        if not np.isfinite(dt_ms) or dt_ms <= 0 or not np.isclose(.1/dt_ms, round(.1/dt_ms)):
            raise ValueError("Cable dt must divide the 0.1 ms event interval")
        self.substeps = int(round(.1/dt_ms))
        self.checkpoint_substeps = bool(checkpoint_substeps)
        self.stepper = H01NetworkStep(network, dt_ms)
        self.neuron_count = len(self.source_ids)
        rng = brainstate.random.RandomState(seed)
        if input_pattern is None:
            fanout = min(32, self.neuron_count)
            ranks = rng.uniform(size=(441, self.neuron_count))
            targets = jnp.sort(jnp.argsort(ranks, axis=1)[:, :fanout], axis=1).reshape(-1)
            indptr = jnp.arange(442, dtype=jnp.int32)*fanout
        else:
            from .h01_remap import encoder_keys
            targets, indptr = map(np.asarray, input_pattern)
            if not np.issubdtype(targets.dtype, np.integer) or not np.issubdtype(indptr.dtype, np.integer):
                raise ValueError('Encoder pattern must use integer indices')
            encoder_keys(self.source_ids, targets, indptr)
            targets, indptr = jnp.asarray(targets, dtype=jnp.int32), jnp.asarray(indptr, dtype=jnp.int32)
        values = rng.normal(size=(len(targets),)) / jnp.sqrt(441.)
        self.input_csr = brainevent.CSR(values, targets, indptr,
            shape=(441, self.neuron_count), backend="jax_raw")
        self.input_weight = brainstate.ParamState(values)
        self.readout_weight = brainstate.ParamState(
            rng.normal(size=(self.neuron_count, 360))/jnp.sqrt(float(self.neuron_count)))
        self.readout_bias = brainstate.ParamState(jnp.zeros(360))
        self.drive = brainstate.ShortTermState(jnp.zeros(self.neuron_count))
        blocks = self.stepper.setup.delivery_blocks
        if any(len(block.pre_index) != 1 for block in blocks):
            raise ValueError("H01 requires one named placed contact per delivery block")
        self.recurrent_weight = brainstate.ParamState(jnp.asarray([
            float(np.asarray(block.weight.to_decimal(u.uS)).reshape(())) for block in blocks]))
        self.contact_magnitude = brainstate.ShortTermState(jnp.abs(self.recurrent_weight.value))
        ops = tuple(self._contact_op(index, block) for index, block in enumerate(blocks))
        self.stepper.ring_buffers = tuple(brainstate.HiddenState(state.value)
                                          for state in self.stepper.ring_buffers)
        self.stepper.delivery = replace(self.stepper.delivery, delivery_ops=ops,
                                        ring_buffers=self.stepper.ring_buffers)
        for index, cell in enumerate(self.stepper.cells):
            table = cell.runtime.clamp_routing_table
            if table is None or len(table.midpoint_ids) + len(table.boundary_ids) != 1:
                raise ValueError("Each H01 cell must have exactly one soma clamp route")
            cell.runtime.evaluate_point_clamps = self._current_op(index, cell, table)
            # A native run may have cached a voltage derivative before encoder attachment.
            if hasattr(cell.runtime, "_voltage_linearizer_cache"):
                cell.runtime._voltage_linearizer_cache = None
        soma_cv_ids = []
        for cell in self.stepper.cells:
            found_cv = None
            for layout in cell.runtime.layouts:
                decl = cell.runtime.get_layout_mechanism(layout.id)
                if getattr(decl, 'name', None) == 'voltage':
                    from braincell._multi_compartment.probes import _representative_cv_id
                    found_cv = _representative_cv_id(cell.runtime, point_id=int(layout.point_index[0]))
                    break
            soma_cv_ids.append(found_cv)
        self._soma_cv_ids = tuple(soma_cv_ids)

    def _contact_op(self, index, block):
        def deliver(spikes):
            magnitude = self.contact_magnitude.value[index]
            event = spikes[block.pre_index[0]] * magnitude
            size = self.stepper.network.populations[block.source.post_population].size*block.source.n_active
            return jnp.zeros(size).at[block.flat_target_index[0]].add(event)*u.uS
        return deliver

    def _current_op(self, index, cell, table):
        original = cell.runtime.evaluate_point_clamps
        point = int(np.concatenate((table.midpoint_ids, table.boundary_ids))[0])
        def current(*, t, point_ids=None):
            value = original(t=t, point_ids=point_ids)
            if point_ids is None or point in point_ids or (isinstance(point_ids, np.ndarray) and point in point_ids):
                value = value.at[..., point].add(self.drive.value[index]*u.nA)
            return value
        return current

    def update(self, event):
        """Advance one 0.1 ms event with its bounded soma current held fixed.

        Parameters
        ----------
        event : array-like
            The 441 ARC input features.

        Returns
        -------
        array
            One soma voltage in millivolts per cell.
        """
        drive = braintrace.sparse_matmul(event, self.input_weight.value, sparse_mat=self.input_csr)
        self.drive.value = .35*jnp.tanh(drive)
        self.contact_magnitude.value = braintrace.element_wise(
            self.recurrent_weight.value, weight_fn=jnp.abs)
        def cable_step(_):
            self.stepper.update(sample_probes=False)
        if self.checkpoint_substeps:
            brainstate.transform.for_loop(brainstate.transform.checkpoint(cable_step, prevent_cse=False),
                                          jnp.arange(self.substeps))
        else:
            brainstate.transform.for_loop(cable_step, jnp.arange(self.substeps))
        return self._soma()

    def _soma(self):
        if hasattr(self, '_soma_cv_ids') and all(cv is not None for cv in self._soma_cv_ids):
            return jnp.stack([cell.V.value[..., cv].to_decimal(u.mV).reshape(())
                              for cell, cv in zip(self.stepper.cells, self._soma_cv_ids)])
        return jnp.stack([cell.sample_probes()["voltage"].to_decimal(u.mV).reshape(())
                          for cell in self.stepper.cells])

    def step(self, event, advance=True, blocked_source=None):
        """Advance an event only when its padding mask permits.

        Parameters
        ----------
        event : array-like
            ARC event features.
        advance : bool, optional
            False preserves all state.
        blocked_source : int, optional
            Unsupported here; causal controls use explicit network manifests.

        Returns
        -------
        array
            Soma voltages, or zero for padding.
        """
        if blocked_source is not None:
            raise ValueError("Use an explicit H01 contact-control manifest")
        return brainstate.transform.cond(advance, lambda: self.update(event),
                                        lambda: jnp.zeros_like(self.drive.value))

    def interval(self, event, advance=True, *, substeps=1):
        """Advance one ARC interval using the manifest's cable clock.

        Parameters
        ----------
        event : array-like
            ARC features.
        advance : bool, optional
            Episode padding mask.
        substeps : int, optional
            Must be one; construct a matched finer-clock model for refinement.

        Returns
        -------
        array
            Soma voltages or padding zeros.
        """
        if substeps != 1:
            raise ValueError("H01 cable refinement requires a matched dt_ms model")
        return self.step(event, advance)

    def readout_features(self):
        """Return normalized soma features.

        Returns
        -------
        array
            tanh((V_mV + 65)/20) per cell.
        """
        return jnp.tanh((self._soma()+65)/20)

    def readout(self):
        """Return the 360 ARC logits.

        Returns
        -------
        array
            Direct trainable linear readout.
        """
        return self.readout_features() @ self.readout_weight.value + self.readout_bias.value

    def reset_episode(self, learner=None):
        """Reset physical, delivery, clock and optional eligibility state.

        Parameters
        ----------
        learner : object, optional
            Compiled learner with a reset_state method.
        """
        self.stepper.reset_state()
        self.drive.value = jnp.zeros_like(self.drive.value)
        self.contact_magnitude.value = jnp.abs(self.recurrent_weight.value)
        if learner is not None:
            learner.reset_state()
