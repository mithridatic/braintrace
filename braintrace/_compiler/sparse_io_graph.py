"""Explicit ETP-output boundaries for heterogeneous sparse IO factors."""

from dataclasses import dataclass
from math import prod

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace._compatible_imports import Jaxpr, Var
from braintrace._misc import NotSupportedError
from braintrace._op import ETP_RULES_XY_TO_DW, get_trainable_invars
from .hid_param_op import _resolve_eqn_vars, _scan_jaxpr_for_etp_eqns
from .module_info import extract_module_info
from .sparse_support import analyze_transition


@dataclass(frozen=True)
class _Operation:
    eqn: object
    x: object
    y: object
    weights: dict
    start: int
    stop: int


def _program(original, *, invars=None, outvars=None, eqns=None):
    return Jaxpr(constvars=original.constvars,
                 invars=original.invars if invars is None else invars,
                 outvars=original.outvars if outvars is None else outvars,
                 eqns=original.eqns if eqns is None else eqns,
                 effects=original.effects,
                 debug_info=original.debug_info._replace(arg_names=None, result_paths=None))


class SparseIOGraph:
    """Extract sparse factors without position-wise hidden-state grouping.

    Parameters
    ----------
    model : brainstate.nn.Module
        Model whose trainable temporal operations use the ETP registry.
    *args : tuple
        Example model inputs.
    max_bytes : int, optional
        Maximum sparse output-factor storage after temporal closure.

    Notes
    -----
    Every floating-point temporal state is represented, including delayed
    conductance queues. States whose incoming value is syntactically unread
    cannot carry temporal credit and retain only their native forward state.
    Integer clocks and Boolean events have no differentiable tangent.
    """

    def __init__(self, model, *args, max_bytes=2**30):
        self.info = extract_module_info(model, *args)
        original = self.info.jaxpr
        found = _scan_jaxpr_for_etp_eqns(original, policy=self.info.control_flow)
        if any(not any(eqn is outer for outer in original.eqns) for eqn in found):
            raise NotSupportedError('Sparse IO requires ETP operations outside nested control flow')
        self.operations = []
        offset = 0
        for eqn in found:
            if eqn.primitive not in ETP_RULES_XY_TO_DW or len(eqn.outvars) != 1:
                raise NotSupportedError('Sparse IO needs a registered single-output parameter VJP')
            x, y = _resolve_eqn_vars(eqn)
            weights = {}
            for key, index in get_trainable_invars(eqn.primitive, eqn.params).items():
                var = eqn.invars[index]
                if not isinstance(var, Var) or var not in self.info.invar_to_weight_path:
                    raise NotSupportedError('Sparse IO trainable inputs must come directly from ParamState')
                weights[key] = var
            size = prod(y.aval.shape)
            self.operations.append(_Operation(eqn, x, y, weights, offset, offset+size))
            offset += size
        if not self.operations:
            raise NotSupportedError('Sparse IO needs at least one registered ETP operation')
        self.output_size = offset
        self.state_input_indices, self.state_output_indices, self.state_paths = [], [], []
        used = {var for eqn in original.eqns for var in eqn.invars if isinstance(var, Var)}
        used.update(var for var in original.outvars if isinstance(var, Var))
        for state, ins, outs in zip(self.info.compiled_model_states,
                                   self.info.state_tree_invars, self.info.state_tree_outvars):
            if isinstance(state, brainstate.ParamState):
                continue
            for inv, outv in zip(jax.tree.leaves(ins), jax.tree.leaves(outs), strict=True):
                if np.issubdtype(inv.aval.dtype, np.inexact) and inv in used:
                    self.state_input_indices.append(original.invars.index(inv))
                    self.state_output_indices.append(original.outvars.index(outv))
                    self.state_paths.append(self.info.state_id_to_path[id(state)])
        xs = [op.x for op in self.operations if op.x is not None]
        self._forward = _program(original, outvars=list(original.outvars)+xs+[op.y for op in self.operations])
        ids = {id(op.eqn) for op in self.operations}
        self._tail = _program(original, invars=list(original.invars)+[op.y for op in self.operations],
                              eqns=[eqn for eqn in original.eqns if id(eqn) not in ids])
        self._num_out = len(original.outvars)
        raw = self.inputs(*args)
        _, _, output = self.forward(raw)
        hidden = tuple(raw[i] for i in self.state_input_indices)
        self.layout = analyze_transition(lambda y, h: self.transition(y, h, raw),
                                         output, hidden, max_bytes=max_bytes)

    def inputs(self, *args):
        """Flatten current inputs and model state for the extracted program.

        Parameters
        ----------
        *args : tuple
            Current model inputs.

        Returns
        -------
        tuple
            Model inputs followed by the compiled state leaves.
        """
        leaves = jax.tree.leaves((args, [state.value for state in self.info.compiled_model_states]))
        return tuple(jnp.asarray(value, dtype=var.aval.dtype)
                     for value, var in zip(leaves, self.info.jaxpr.invars, strict=True))

    def forward(self, raw):
        """Execute the original program and expose registered ETP operands.

        Parameters
        ----------
        raw : tuple
            Flattened inputs from :meth:`inputs`.

        Returns
        -------
        tuple
            Original flat outputs, per-operation inputs and concatenated outputs.
        """
        result = jax.core.eval_jaxpr(self._forward, self.info.closed_jaxpr.consts, *raw)
        cursor = self._num_out
        xs = []
        for op in self.operations:
            xs.append(None if op.x is None else result[cursor])
            cursor += op.x is not None
        output = jnp.concatenate([value.reshape(-1) for value in result[cursor:]])
        return tuple(result[:self._num_out]), tuple(xs), output

    def transition(self, output, hidden, raw):
        """Evaluate actual state dynamics with explicit ETP output boundaries.

        Parameters
        ----------
        output : array
            Concatenated registered ETP outputs.
        hidden : tuple
            Floating-point state blocks at entry.
        raw : tuple
            Remaining model inputs, clocks, parameters and nondifferentiable state.

        Returns
        -------
        tuple
            Updated floating-point state blocks.
        """
        values = list(raw)
        for index, value in zip(self.state_input_indices, hidden, strict=True):
            values[index] = value
        values.extend(output[op.start:op.stop].reshape(op.y.aval.shape) for op in self.operations)
        result = jax.core.eval_jaxpr(self._tail, self.info.closed_jaxpr.consts, *values)
        return tuple(result[i] for i in self.state_output_indices)
