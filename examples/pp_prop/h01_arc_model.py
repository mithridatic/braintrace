"""Exploratory ARC encoder and readout around actual H01 cable dynamics."""

from dataclasses import replace

import brainevent
import brainstate
import braintrace
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_network_step import H01NetworkStep


def _forest_cell(network):
    """The forest cell when the network is one fused population, else ``None``."""
    if len(network.populations) != 1:
        return None
    cell = next(iter(network.populations.values())).cell
    return cell if hasattr(cell, 'forest_offsets') else None


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
    release_probability : array-like or None, optional
        Fixed probability per placed contact. None preserves deterministic
        delivery. Explicit probabilities use an episode-reset random stream.

    Notes
    -----
    This computational interface does not qualify donor physiology or pp-prop.
    Compiler and allocation gates must pass before a training claim is made.
    """

    def __init__(self, network, source_ids, *, seed=21, dt_ms=None,   # qualified step, docs/evidence/h01-timestep-ladder.json
                 input_pattern=None,
                 checkpoint_substeps=True, release_probability=None):
        super().__init__()
        self.source_ids = tuple(source_ids)
        if not self.source_ids or any(not isinstance(x, str) for x in self.source_ids):
            raise ValueError("Nonempty string source IDs are required")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("Source IDs must be unique")
        self.forest = _forest_cell(network)
        if self.forest is not None and self.forest.slots > 1 and len(self.source_ids) == self.forest.source_count:
            # Static capacity: dormant clone slots carry derived identities until a clone names them.
            self.source_ids = self.source_ids+tuple(f'{identity}#slot{slot}' for slot in range(1, self.forest.slots)
                                                    for identity in self.source_ids)
        if self.forest is None and len(self.source_ids) != len(network.populations):
            raise ValueError("Source IDs must match network populations")
        if self.forest is not None and len(self.source_ids) != self.forest.forest_offsets.cells:
            raise ValueError("Source IDs must match the forest's cells")
        if dt_ms is None:
            env_dt = brainstate.environ.get('dt', None)
            if env_dt is not None:
                dt_ms = float(env_dt.to_decimal(u.ms)) if isinstance(env_dt, u.Quantity) else float(env_dt)
            else:
                dt_ms = 0.000625
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
        self.release = None
        self.contact_table = self.stepper.contact_table
        if self.contact_table is not None and release_probability is not None:
            raise ValueError('Static contacts do not carry release probabilities yet')
        if release_probability is not None:
            from braintrace.biophysics.release import ReleaseState
            if np.asarray(release_probability).shape != (len(blocks),):
                raise ValueError('Release probabilities must match placed contacts')
            self.release = ReleaseState(release_probability, seed=seed)
            self.release_mask = brainstate.ShortTermState(jnp.ones(len(blocks)))
        if any(len(block.pre_index) != 1 for block in blocks):
            raise ValueError("H01 requires one named placed contact per delivery block")
        if self.contact_table is not None:
            self.recurrent_weight = brainstate.ParamState(jnp.asarray(self.contact_table.weight.value))
            self.contact_magnitude = brainstate.ShortTermState(jnp.abs(self.recurrent_weight.value))
            self.stepper.contact_magnitude = lambda: self.contact_magnitude.value
        else:
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
            routes = self.neuron_count if self.forest is not None else 1
            if table is None or len(table.midpoint_ids) + len(table.boundary_ids) != routes:
                raise ValueError("Each H01 cell must have exactly one soma clamp route")
            cell.runtime.evaluate_point_clamps = self._current_op(index, cell, table)
            # A native run may have cached a voltage derivative before encoder attachment.
            if hasattr(cell.runtime, "_voltage_linearizer_cache"):
                cell.runtime._voltage_linearizer_cache = None
        soma_cv_ids = []
        for cell in () if self.forest is not None else self.stepper.cells:
            found_cv = None
            for layout in cell.runtime.layouts:
                decl = cell.runtime.get_layout_mechanism(layout.id)
                if getattr(decl, 'name', None) == 'voltage':
                    from braincell._multi_compartment.probes import _representative_cv_id
                    found_cv = _representative_cv_id(cell.runtime, point_id=int(layout.point_index[0]))
                    break
            soma_cv_ids.append(found_cv)
        self._soma_cv_ids = tuple(self.forest.soma_cv_ids) if self.forest is not None else tuple(soma_cv_ids)

    def _contact_op(self, index, block):
        def deliver(spikes):
            magnitude = self.contact_magnitude.value[index]
            event = spikes[block.pre_index[0]] * magnitude
            if self.release is not None:
                event = event * self.release_mask.value[index]
            size = self.stepper.network.populations[block.source.post_population].size*block.source.n_active
            return jnp.zeros(size).at[block.flat_target_index[0]].add(event)*u.uS
        return deliver

    def _current_op(self, index, cell, table):
        original = cell.runtime.evaluate_point_clamps
        points = np.sort(np.concatenate((table.midpoint_ids, table.boundary_ids)).astype(int))
        if self.forest is not None:
            # One route per cell, in cell order (each soma point lies in its cell's point range).
            def current(*, t, point_ids=None):
                value = original(t=t, point_ids=point_ids)
                routed = np.ones(len(points), dtype=bool) if point_ids is None else np.isin(points, np.asarray(point_ids))
                return value.at[..., points[routed]].add(self.drive.value[np.flatnonzero(routed)]*u.nA)
            return current
        point = int(points[0])
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
        self.drive.value = .35*jnp.tanh(drive)*self._active()
        self.contact_magnitude.value = braintrace.element_wise(
            self.recurrent_weight.value, weight_fn=jnp.abs)
        def cable_step(_):
            if self.release is not None:
                self.release_mask.value = self.release.update(jnp.ones_like(self.release_mask.value))
            self.stepper.update(sample_probes=False)
        if self.checkpoint_substeps:
            brainstate.transform.for_loop(brainstate.transform.checkpoint(cable_step, prevent_cse=False),
                                          jnp.arange(self.substeps))
        else:
            brainstate.transform.for_loop(cable_step, jnp.arange(self.substeps))
        return self._soma()

    def _active(self):
        """Per-cell activity mask: dormant clone slots contribute nothing."""
        if self.forest is not None and self.forest.slots > 1:
            return self.forest.active.value
        return 1.

    def clone(self, parent):
        """Activate a dormant slot of ``parent``'s source cell in place (no rebuild).

        Parameters
        ----------
        parent : int
            Forest cell index of the parent.

        Returns
        -------
        int
            Forest cell index of the clone.

        Raises
        ------
        ValueError
            No dormant slot is left for that source cell.
        """
        if self.forest is None or self.forest.slots < 2:
            raise ValueError('Cloning in place needs a forest built with clone slots')
        slot = self.forest.free_slot(parent)
        if slot is None:
            raise ValueError('No dormant slot left for this cell; raise the slot capacity')
        self.forest.activate(slot)
        return slot

    def add_contact(self, pre, post, *, kind, weight_us=.01, delay_ms=.5, identity=None):
        """Write one contact into the static table (no rebuild, no recompile).

        Parameters
        ----------
        pre, post : int
            Forest cell indices.
        kind : int
            0 excitatory (0 mV / 2 ms), 1 inhibitory (-80 mV / 5 ms).
        weight_us, delay_ms : float, optional
            Initial magnitude and delay.
        identity : str, optional
            Topology contact identity.

        Returns
        -------
        int
            Table row.
        """
        if self.contact_table is None:
            raise ValueError('Adding contacts in place needs a forest built with a contact capacity')
        row = self.contact_table.free_row()
        if row is None:
            raise ValueError('The contact table is full; raise the contact capacity')
        self.contact_table.write(row, pre=pre, post_cell=post, forest=self.forest, kind=kind, weight_us=weight_us,
                                 delay_ms=delay_ms, identity=identity)
        from braintrace.datasets.h01_forest_contacts import host_write
        host_write(self.recurrent_weight, row, float(weight_us))
        host_write(self.contact_magnitude, row, abs(float(weight_us)))
        return row

    def _soma(self):
        if self.forest is not None:
            return self.forest.V.value[..., np.asarray(self._soma_cv_ids)].to_decimal(u.mV).reshape(-1)
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

    def sparse_structure(self, paths, shapes, operations):
        """Declare how the forest's state blocks split into per-cell segments.

        Parameters
        ----------
        paths : tuple of tuples
            Model paths of the learner's floating-point state blocks.
        shapes : tuple of tuples
            Their shapes.
        operations : tuple of (start, stop)
            ETP output ranges: the encoder drive (one per cell) then the
            contact magnitudes (one per placed contact).

        Returns
        -------
        tuple or None
            ``(blocks, output_labels)`` for ``SparseInfluence.build_segmented``
            on the fused path; ``None`` on the per-cell path (whole blocks).

        Notes
        -----
        Labels are cell indices and ``('contact', c)``. A cable state
        (last axis ``n_cv`` or ``n_point``) is segmented by the forest offsets;
        contact ``c``'s ring buffer reads the pre cell and feeds the post cell;
        its synapse states read and feed the post cell; every other block is
        unrestricted. The drive output of cell ``k`` is labelled ``k``; contact
        output ``c`` is labelled ``('contact', c)``. The compiler's block-level
        analysis still decides which blocks interact; this only refines where.
        """
        if self.forest is None:
            return None
        forest, offsets = self.forest, self.forest.forest_offsets
        runtime = forest.runtime
        n_cv, n_point, cells = forest.n_cv, runtime.n_point, list(range(offsets.cells))
        contacts, by_layout = [], {}
        table = self.contact_table
        if table is not None:
            contacts = [(int(np.asarray(table.pre.value)[row]), table.post_cell(forest, row)) if row in table.rows else None
                        for row in range(table.capacity)]
            shared = {'layout_%d' % layout_id: np.asarray(index, dtype=np.int64)
                      for layout_id, index in zip(table.layout_ids, table.point_index)}
        for index, block in enumerate(self.stepper.setup.delivery_blocks):
            layout = runtime.layouts[int(block.source.layout_id)]
            post = int(np.searchsorted(offsets.point, int(layout.point_index[0]), side='right')-1)
            contacts.append((int(block.pre_index[0]), post))
            by_layout['layout_%d' % int(block.source.layout_id)] = index
        blocks = []
        for path, shape in zip(paths, shapes):
            layout_key = next((p for p in path if isinstance(p, str) and p in by_layout), None)
            shared_key = next((p for p in path if isinstance(p, str) and table is not None and p in shared), None)
            if table is not None and 'contact_table' in path[:2]:
                if path[-1] == 'ring':
                    owners = [({pre, ('contact', c)}, {post, ('contact', c)}) if contact is not None else
                              ({('contact', c)}, {('contact', c)})
                              for c, contact in enumerate(contacts) for pre, post in [contact or (None, None)]]
                    blocks.append(('segments', np.arange(table.capacity+1), owners))
                else:
                    blocks.append(('block', set(), set()))
            elif shared_key is not None:
                points = shared[shared_key]
                owners = [int(forest.cell_of_point[point]) for point in points]
                blocks.append(('segments', np.arange(len(points)+1), owners))
            elif path[-1] == 'active':
                blocks.append(('block', set(), set()))
            elif path[:2] == ('stepper', 'ring_buffers') and int(path[2]) < len(contacts):
                pre, post = contacts[int(path[2])]
                blocks.append(('block', {pre, ('contact', int(path[2]))}, {post, ('contact', int(path[2]))}))
            elif layout_key is not None:
                pre, post = contacts[by_layout[layout_key]]
                blocks.append(('block', {post, ('contact', by_layout[layout_key])}, {post}))
            elif shape and shape[-1] == n_cv and path[0] == 'forest':
                blocks.append(('segments', offsets.cv, cells))
            elif shape and shape[-1] == n_point and path[0] == 'forest':
                blocks.append(('segments', offsets.point, cells))
            else:
                blocks.append(('block', None, None))
        labels = []
        for index, (start, stop) in enumerate(operations):
            if index == 0:
                if stop-start != len(cells):
                    raise ValueError('The first ETP output must be the per-cell drive')
                labels.extend({k} for k in cells)
            else:
                labels.extend({('contact', c)} for c in range(stop-start))
        if len(labels) != sum(stop-start for start, stop in operations) or (
                len(operations) > 1 and operations[1][1]-operations[1][0] != len(contacts)):
            raise ValueError('ETP outputs do not match the forest drive and contact table')
        return tuple(blocks), tuple(labels)

    def sparse_structure_signature(self):
        """Hashable summary of what ``sparse_structure`` would declare now.

        A learner compiled under one signature must be recompiled
        (``learner.graph = None; learner.compile_graph(event)``) when it changes:
        adding or removing a contact changes which cells reach which.
        """
        if self.contact_table is None:
            return None
        return tuple(sorted((row, v['pre'], self.contact_table.post_cell(self.forest, row), v['kind'])
                            for row, v in self.contact_table.rows.items()))

    def readout_features(self):
        """Return normalized soma features.

        Returns
        -------
        array
            tanh((V_mV + 65)/20) per cell.
        """
        return jnp.tanh((self._soma()+65)/20)*self._active()

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
        if self.release is not None:
            self.release.reset_state()
            self.release_mask.value = jnp.ones_like(self.release_mask.value)
        if learner is not None:
            learner.reset_state()
