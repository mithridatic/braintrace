"""Conservative dependency interpretation of a pure recurrent transition."""

from dataclasses import dataclass
from functools import reduce
from math import prod
from operator import or_

import jax
import numpy as np

from braintrace._compatible_imports import Var, scan_num_consts_carry
from braintrace._misc import NotSupportedError
from .sparse_influence import SparseInfluence


@dataclass
class _Support:
    shape: tuple
    bits: object

    @property
    def union(self):
        return int(reduce(or_, self.bits.flat, 0)) if isinstance(self.bits, np.ndarray) else int(self.bits)

    def dense(self):
        return np.broadcast_to(np.asarray(self.bits, dtype=object), self.shape)


def _same(left, right):
    return left.shape == right.shape and np.array_equal(left.bits, right.bits)


def _merge(values, shape, *, positional=False):
    if positional and prod(shape) <= 4096 and any(isinstance(value.bits, np.ndarray) for value in values):
        try:
            bits = np.zeros(shape, dtype=object)
            for value in values:
                bits = np.bitwise_or(bits, np.broadcast_to(value.dense(), shape))
            return _Support(shape, bits)
        except ValueError:
            pass
    return _Support(shape, reduce(or_, (value.union for value in values), 0))


_ELEMENTWISE = frozenset(('add', 'sub', 'mul', 'div', 'neg', 'abs', 'exp', 'expm1',
    'log', 'log1p', 'tanh', 'sin', 'cos', 'sqrt', 'rsqrt', 'integer_pow', 'pow',
    'max', 'min', 'convert_element_type', 'select_n', 'eq', 'ne', 'gt', 'ge',
    'lt', 'le', 'and', 'or', 'not', 'is_finite', 'sign', 'floor', 'ceil'))


def _scan(eqn, args):
    params = eqn.params
    nc, nk = scan_num_consts_carry(eqn)
    constants, carry = args[:nc], args[nc:nc+nk]
    xs = [_Support(value.shape[1:], np.bitwise_or.reduce(value.dense(), axis=0))
          if isinstance(value.bits, np.ndarray) else _Support(value.shape[1:], value.bits)
          for value in args[nc+nk:]]
    ys = [_Support(var.aval.shape[1:], 0) for var in eqn.outvars[nk:]]
    if params['length']:
        # Static abstract fixed point, independent of the numerical loop length.
        # Join the entry state so every reachable iteration is represented.
        for _ in range(128):
            result = _evaluate(params['jaxpr'], constants + carry + xs)
            updated = [_merge([old, new], old.shape, positional=True)
                       for old, new in zip(carry, result[:nk], strict=True)]
            ys = [_merge([old, new], old.shape, positional=True)
                  for old, new in zip(ys, result[nk:], strict=True)]
            if all(_same(old, new) for old, new in zip(carry, updated)):
                carry = updated
                break
            carry = updated
        else:
            raise NotSupportedError('Sparse scan support did not converge within 128 passes')
    return carry + [_Support(var.aval.shape, value.bits) for var, value in zip(eqn.outvars[nk:], ys)]


def _equation(eqn, args):
    name, params = eqn.primitive.name, eqn.params
    shapes = [tuple(var.aval.shape) for var in eqn.outvars]
    if name == 'stop_gradient':
        return [_Support(shape, 0) for shape in shapes]
    if name == 'scan':
        return _scan(eqn, args)
    if name == 'cond':
        branches = [_evaluate(branch, args[1:]) for branch in params['branches']]
        return [_merge([branch[i] for branch in branches], shape, positional=True)
                for i, shape in enumerate(shapes)]
    if name in ('jit', 'pjit', 'remat2') and 'jaxpr' in params:
        return _evaluate(params['jaxpr'], args)
    if name in ('slice', 'squeeze', 'reshape', 'transpose', 'broadcast_in_dim'):
        value = args[0]
        if not isinstance(value.bits, np.ndarray):
            return [_Support(shapes[0], value.bits)]
        bits = value.dense()
        if name == 'slice':
            strides = params.get('strides') or (1,) * len(value.shape)
            bits = bits[tuple(slice(a, b, s) for a, b, s in
                              zip(params['start_indices'], params['limit_indices'], strides))]
        elif name == 'squeeze':
            bits = np.squeeze(bits, axis=params['dimensions'])
        elif name == 'reshape':
            if params.get('dimensions') is not None:
                bits = bits.transpose(params['dimensions'])
            bits = bits.reshape(shapes[0])
        elif name == 'transpose':
            bits = bits.transpose(params['permutation'])
        elif prod(shapes[0]) <= 4096:
            intermediate = [1] * len(shapes[0])
            for axis, size in zip(params['broadcast_dimensions'], value.shape):
                intermediate[axis] = size
            bits = np.broadcast_to(bits.reshape(intermediate), shapes[0])
        else:
            return [_Support(shapes[0], value.union)]
        return [_Support(shapes[0], bits)]
    if name == 'concatenate' and prod(shapes[0]) <= 4096:
        return [_Support(shapes[0], np.concatenate([v.dense() for v in args], axis=params['dimension']))]
    # Unknown primitives can mix every input position, including custom JVPs.
    # Widening is safe; guessing a diagonal or dropping the edge is not.
    return [_merge(args, shape, positional=name in _ELEMENTWISE) for shape in shapes]


def _evaluate(closed, args):
    program = getattr(closed, 'jaxpr', closed)
    if len(args) != len(program.invars):
        raise ValueError('Sparse analysis input count differs from transition program')
    env = {var: value for var, value in zip(program.invars, args)}
    env.update({var: _Support(tuple(var.aval.shape), 0) for var in program.constvars})

    def read(var):
        return env[var] if isinstance(var, Var) else _Support(tuple(var.aval.shape), 0)

    for eqn in program.eqns:
        result = _equation(eqn, [read(var) for var in eqn.invars])
        env.update(zip(eqn.outvars, result, strict=True))
    return [read(var) for var in program.outvars]


def analyze_transition(transition, output, state, *, max_bytes=2**30):
    """Infer conservative sparse factors from the actual transition program.

    Parameters
    ----------
    transition : callable
        Pure ``transition(flat_output, state_tuple) -> next_state_tuple``.
    output : array
        Flattened ETP output sample.
    state : tuple of arrays
        Floating-point recurrent state samples, including cable and delay state.
    max_bytes : int, optional
        Maximum output-factor storage, checked after temporal support closure.

    Returns
    -------
    SparseInfluence
        Layout derived from conservative program dependencies.

    Raises
    ------
    ValueError
        State/output shape or dtype contract is invalid.
    MemoryError
        Closed sparse factors exceed the storage limit.
    NotSupportedError
        Abstract scan dependency closure does not converge within its limit.
    """
    if output.ndim != 1 or any(not np.issubdtype(v.dtype, np.inexact) for v in (output, *state)):
        raise ValueError('Sparse transition needs a flat output and floating-point state arrays')
    program = jax.make_jaxpr(transition)(output, state)
    shapes = tuple(tuple(value.shape) for value in state)
    if tuple(tuple(var.aval.shape) for var in program.jaxpr.outvars) != shapes:
        raise ValueError('Transition must preserve its state block shapes')
    n = output.size
    args = [_Support(output.shape, np.asarray([1 << i for i in range(n)], dtype=object))]
    args += [_Support(shape, 1 << (n+i)) for i, shape in enumerate(shapes)]
    support = _evaluate(program, args)
    seeds = [{i for i in range(n) if value.union & (1 << i)} for value in support]
    parents = [{i for i in range(len(state)) if value.union & (1 << (n+i))} for value in support]
    itemsize = max(np.dtype(value.dtype).itemsize for value in (output, *state))
    return SparseInfluence.build(shapes, seeds, parents, output_size=n,
                                  itemsize=itemsize, max_bytes=max_bytes)
