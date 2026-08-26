# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import brainstate
import jax
import numpy as np

from ._compatible_imports import Var

ArrayLike: TypeAlias = brainstate.typing.ArrayLike
DType: TypeAlias = brainstate.typing.DType
DTypeLike: TypeAlias = brainstate.typing.DTypeLike

# --- Types --- #
PyTree: TypeAlias = Any
StateID: TypeAlias = int
WeightID: TypeAlias = int
Size: TypeAlias = brainstate.typing.Size
Axis: TypeAlias = int

# Elementwise, shape-preserving transform applied to a weight/bias/kernel
# inside an ETP op (``weight_fn`` / ``bias_fn`` / ``kernel_fn`` / ``a_fn`` / ``b_fn``).
WeightFn: TypeAlias = Callable[[ArrayLike], ArrayLike]


def as_size_tuple(size: Size) -> tuple[int, ...]:
    """Normalize an ``in_size``/``out_size`` spec to a tuple of ints.

    ``brainstate``'s size setters accept a scalar ``int`` or a sequence, while
    the matching getters are typed as the broad :data:`Size` union, which static
    type checkers do not treat as indexable. Routing values through this helper
    yields a concrete ``tuple[int, ...]`` so both property assignment and
    trailing-dimension lookups (``size[-1]``) type-check cleanly, reproducing
    ``brainstate``'s own normalization at runtime.

    Parameters
    ----------
    size : Size
        A scalar ``int`` / numpy integer, or a sequence of them.

    Returns
    -------
    tuple of int
        The size expressed as a tuple of Python ints.
    """
    if isinstance(size, (int, np.integer)):
        return (int(size),)
    return tuple(int(s) for s in size)
Axes: TypeAlias = int | Sequence[int]
Path: TypeAlias = tuple[str, ...]

# --- Inputs and outputs --- #
Inputs: TypeAlias = PyTree
Outputs: TypeAlias = PyTree

# --- State values --- #
HiddenVals: TypeAlias = dict[Path, PyTree]
StateVals: TypeAlias = dict[Path, PyTree]
WeightVals: TypeAlias = dict[Path, PyTree]
ETraceVals: TypeAlias = dict[Path, PyTree]

HiddenOutVar: TypeAlias = Var
HiddenInVar: TypeAlias = Var

# --- Gradients --- #
dG_Inputs: TypeAlias = PyTree  # gradients of inputs
dG_Weight: TypeAlias = Sequence[PyTree]  # Gradients of weights
dG_Hidden: TypeAlias = Sequence[PyTree]  # Gradients of hidden states
dG_State: TypeAlias = Sequence[PyTree]  # Gradients of other states

VarID: TypeAlias = int

HiddenGroupName: TypeAlias = str
ETraceRawX_Key: TypeAlias = VarID
ETraceX_Key: TypeAlias = tuple[VarID, int]
ETraceY_Key: TypeAlias = VarID
ETraceDF_Key: TypeAlias = tuple[VarID, HiddenGroupName]

_WeightPath: TypeAlias = Path
_HiddenPath: TypeAlias = Path
# D-RTRL keys weight-gradient traces by (weight y-var id, hidden-group index).
ETraceWG_Key: TypeAlias = tuple[ETraceY_Key, int]
HidHidJac_Key: TypeAlias = tuple[Path, Path]

# --- Data --- #
WeightXVar: TypeAlias = Var
WeightYVar: TypeAlias = Var
WeightXs: TypeAlias = dict[Var, jax.Array]
WeightDfs: TypeAlias = dict[Var, jax.Array]
TempData: TypeAlias = dict[Var, jax.Array]
Current: TypeAlias = ArrayLike  # The synaptic current
Conductance: TypeAlias = ArrayLike  # The synaptic conductance
Spike: TypeAlias = ArrayLike  # The spike signal
# The diagonal Jacobian of the hidden-to-hidden function
Hid2HidDiagJacobian: TypeAlias = dict[
    frozenset[HiddenOutVar],
    dict[HiddenOutVar, list[jax.Array]]
]
Hid2WeightJacobian: TypeAlias = tuple[
    dict[ETraceRawX_Key, jax.Array],
    dict[ETraceDF_Key, jax.Array]
]
Hid2HidJacobian: TypeAlias = dict[
    HidHidJac_Key,
    jax.Array
]
HiddenGroupJacobian: TypeAlias = Sequence[jax.Array]
