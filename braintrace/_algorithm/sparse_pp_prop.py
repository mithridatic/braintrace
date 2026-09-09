"""Sparse IO-factorized online learning with full matrix-free state propagation."""

import brainstate
import jax
import jax.numpy as jnp

from braintrace._compiler.sparse_io_graph import SparseIOGraph
from braintrace._input_data import MultiStepData, SingleStepData
from braintrace._misc import NotSupportedError
from braintrace._op import ETP_RULES_XY_TO_DW, get_pp_df_factors, get_pp_x_repr
from .io_dim_vjp import _format_decays, _f_trace_bias_correction, _reduce_to_param_shape
from .sequence import SequenceDriverMixin
from .sparse_io import advance_factors, contract_factors


class SparsePPProp(SequenceDriverMixin, brainstate.nn.Module):
    """Run pp-prop factors on sparse heterogeneous state supports.

    Parameters
    ----------
    model : brainstate.nn.Module
        Model with registered ETP operations and a single array input.
    decay_or_rank : float, int or pair
        Existing pp-prop input/output smoothing settings.
    max_bytes : int, optional
        Sparse output-factor allocation ceiling, default one GiB.
    vjp_method : str, optional
        Must be ``'multi-step'``. Plain inputs form a one-event window.

    Notes
    -----
    The estimator retains IO factorization but propagates its output factors
    through complete state JVPs, including cable coupling. Within each supplied
    window, ordinary AD is exact; entry-boundary temporal credit uses the saved
    factors. This does not make the factorized estimator generally BPTT-exact.
    """

    def __init__(self, model, decay_or_rank=.99, *, max_bytes=2**30,
                 vjp_method='multi-step'):
        super().__init__()
        if vjp_method != 'multi-step':
            raise NotSupportedError('Sparse pp-prop currently requires multi-step VJP windows')
        self.model, self.max_bytes, self.vjp_method = model, max_bytes, vjp_method
        self.decay_x, self.decay_f = _format_decays(decay_or_rank)
        self.graph = None

    @property
    def _seq_param_states(self):
        return self.model.states(brainstate.ParamState)

    @property
    def _seq_vjp_method(self):
        return self.vjp_method

    @property
    def param_states(self):
        """Return all trainable model states, including direct readout parameters.

        Returns
        -------
        FlattedDict
            Parameter states keyed by stable model paths.
        """
        return self.model.states(brainstate.ParamState)

    @property
    def model4compile(self):
        """Return the physical model used by episode trainers.

        Returns
        -------
        brainstate.nn.Module
            Underlying model.
        """
        return self.model

    @property
    def is_compiled(self):
        """Return whether sparse dependency compilation has completed.

        Returns
        -------
        bool
            True once the graph and custom VJP are ready.
        """
        return self.graph is not None and hasattr(self, '_call')

    def compile_graph(self, event):
        """Compile actual state dependencies and initialize bounded factors.

        Parameters
        ----------
        event : array
            One event with the model's input shape and dtype.
        """
        if self.is_compiled:
            return
        self.graph = SparseIOGraph(self.model, event, max_bytes=self.max_bytes)
        info = self.graph.info
        self._pairs = []
        for state, ins, outs in zip(info.compiled_model_states, info.state_tree_invars, info.state_tree_outvars):
            if not isinstance(state, brainstate.ParamState):
                self._pairs.extend((info.jaxpr.invars.index(a), info.jaxpr.outvars.index(b))
                                   for a, b in zip(jax.tree.leaves(ins), jax.tree.leaves(outs), strict=True))
        for op in self.graph.operations:
            if get_pp_df_factors(op.eqn.primitive) is not None or get_pp_x_repr(op.eqn.primitive) is not None:
                raise NotSupportedError('Sparse pp-prop does not yet support transformed input-factor registries')
        key = info.stateful_model.get_arg_cache_key(event)
        out_shapes = info.stateful_model.get_out_shapes_by_cache(key)[0]
        self._out_shapes = out_shapes
        self._out_tree = jax.tree.structure(out_shapes)
        self._state_tree = jax.tree.structure([state.value for state in info.compiled_model_states])
        self.init_etrace_state()
        self._call = jax.custom_vjp(self._run)
        self._call.defvjp(self._forward_vjp, self._backward_vjp)

    def init_etrace_state(self):
        """Initialize output factors, input factors and the episode event count.

        Returns
        -------
        None
        """
        if self.graph is None:
            raise ValueError('Compile sparse pp-prop before initializing traces')
        dtype = self.graph.operations[0].y.aval.dtype
        self.factors = brainstate.ShortTermState(tuple(
            jnp.zeros(shape + (len(row),), dtype=dtype)
            for shape, row in zip(self.graph.layout.shapes, self.graph.layout.outputs)))
        self.inputs_trace = brainstate.ShortTermState(tuple(
            None if op.x is None else jnp.zeros(op.x.aval.shape, dtype=op.x.aval.dtype)
            for op in self.graph.operations))
        self.running_index = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def reset_state(self, **kwargs):
        """Clear eligibility and clock; the model owns its physical-state reset.

        Parameters
        ----------
        **kwargs : dict
            Accepted for compatibility with the model episode-reset interface.
        """
        self.factors.value = jax.tree.map(jnp.zeros_like, self.factors.value)
        self.inputs_trace.value = jax.tree.map(jnp.zeros_like, self.inputs_trace.value)
        self.running_index.value = jnp.asarray(0, dtype=jnp.int32)

    def _advance(self, raw, full):
        result = list(raw)
        for inv, outv in self._pairs:
            result[inv] = full[outv]
        return tuple(result)

    def _native(self, raw, sequence):
        def step(carry, event):
            values = (event,) + carry[1:]
            full, _, _ = self.graph.forward(values)
            return self._advance(values, full), full[:self.graph.info.num_var_out]
        final, outputs = brainstate.transform.scan(step, raw, sequence)
        return outputs, final

    def _run(self, raw, factors, xs, count, sequence):
        graph = self.graph
        def step(carry, event):
            values, previous, input_history, index = carry
            values = (event,) + values[1:]
            full, current_xs, output = graph.forward(values)
            hidden = tuple(values[i] for i in graph.state_input_indices)
            updated = advance_factors(lambda y, h: graph.transition(y, h, values),
                output, hidden, previous, graph.layout, self.decay_f)
            new_xs = tuple(None if x is None else self.decay_x*old+x
                           for old, x in zip(input_history, current_xs))
            return (self._advance(values, full), updated, new_xs, index+1), full[:graph.info.num_var_out]
        (final, fs, xs, count), outputs = brainstate.transform.scan(step, (raw, factors, xs, count), sequence)
        return outputs, final, fs, xs, count

    def _forward_vjp(self, raw, factors, xs, count, sequence):
        result = self._run(raw, factors, xs, count, sequence)
        return result, (raw, factors, xs, count, sequence)

    def _backward_vjp(self, residual, cotangents):
        raw, factors, xs, count, sequence = residual
        graph = self.graph
        _, pullback = jax.vjp(self._native, raw, sequence)
        raw_grad, sequence_grad = pullback((cotangents[0], cotangents[1]))
        hidden_grad = tuple(raw_grad[i] for i in graph.state_input_indices)
        output_grad = contract_factors(factors, hidden_grad, graph.layout)
        output_grad /= _f_trace_bias_correction(self.decay_f, count)
        result = list(raw_grad)
        for op, x in zip(graph.operations, xs):
            weights = {key: raw[graph.info.jaxpr.invars.index(var)] for key, var in op.weights.items()}
            df = output_grad[op.start:op.stop].reshape(op.y.aval.shape)
            updates = ETP_RULES_XY_TO_DW[op.eqn.primitive](x, df, weights, **op.eqn.params)
            for key, gradient in updates.items():
                inv = graph.info.jaxpr.invars.index(op.weights[key])
                result[inv] = result[inv] + _reduce_to_param_shape(gradient, weights[key])
        # The next window receives temporal credit through factors, not a second
        # reverse-mode path through carried physical state or factor construction.
        for index in graph.state_input_indices:
            result[index] = jnp.zeros_like(raw_grad[index])
        return tuple(result), None, None, None, sequence_grad

    def update(self, event, advance=True):
        """Advance a finite VJP window and carry physical and eligibility state.

        Parameters
        ----------
        event : array, SingleStepData or MultiStepData
            One event or a time-major finite window.
        advance : bool, optional
            False returns padding zeros without advancing any model or trace state.

        Returns
        -------
        array or pytree
            Model output, with a leading time axis for a multi-event window.
        """
        if self.graph is None:
            raise ValueError('Compile sparse pp-prop before updating')
        multi = isinstance(event, MultiStepData)
        data = event.data if isinstance(event, (MultiStepData, SingleStepData)) else event
        sequence = data if multi else data[None]
        if sequence.shape[0] == 0:
            raise ValueError('Sparse pp-prop windows must contain at least one event')
        def padded():
            prefix = (sequence.shape[0],) if multi else ()
            return jax.tree.map(lambda value: jnp.zeros(prefix + value.shape, value.dtype), self._out_shapes)
        return brainstate.transform.cond(advance, lambda: self._update_window(sequence, multi), padded)

    def _update_window(self, sequence, multi):
        raw = self.graph.inputs(sequence[0])
        outputs, final, factors, xs, count = self._call(
            raw, self.factors.value, self.inputs_trace.value, self.running_index.value, sequence)
        states = self._state_tree.unflatten(final[1:])
        for state, value in zip(self.graph.info.compiled_model_states, states):
            if not isinstance(state, brainstate.ParamState):
                state.value = value
        self.factors.value, self.inputs_trace.value, self.running_index.value = factors, xs, count
        result = self._out_tree.unflatten(outputs)
        return result if multi else jax.tree.map(lambda value: value[0], result)
