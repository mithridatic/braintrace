"""Single-kernel DHS elimination with a portable scan fallback."""

from functools import lru_cache

import jax
import jax.numpy as jnp


@lru_cache(maxsize=1)
def _gpu_modules():
    try:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import triton as plt
    except ImportError:
        return None
    return pl, plt


def _eliminate_gpu(d, s, low, up, children, parents, valid):
    modules = _gpu_modules()
    if modules is None:
        from .h01_dhs_scan import _triang_raw
        return _triang_raw(d, s, low, up, (children, parents, valid))
    pl, plt = modules

    def kernel(di, si, lr, ur, cr, pr, vr, dr, sr):
        batch = pl.program_id(0)
        lanes = jnp.arange(32)

        def step(index, unused):
            c = plt.load(cr.at[index, lanes])
            p = plt.load(pr.at[index, lanes])
            mask = plt.load(vr.at[index, lanes])
            multiplier = plt.load(ur.at[c])/plt.load(dr.at[batch, c])
            delta_d = -plt.load(lr.at[c])*multiplier
            delta_s = -plt.load(sr.at[batch, c])*multiplier
            plt.atomic_add(dr, (batch, p), delta_d, mask=mask)
            plt.atomic_add(sr, (batch, p), delta_s, mask=mask)
            plt.debug_barrier()
            return unused

        jax.lax.fori_loop(0, children.shape[0], step, None)

    return pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct(d.shape, d.dtype), jax.ShapeDtypeStruct(s.shape, s.dtype)),
        grid=(d.shape[0],), input_output_aliases={0: 0, 1: 1},
        compiler_params=plt.CompilerParams(num_warps=1, num_stages=1))(
            d, s, low, up, children, parents, valid)


def eliminate(d, s, low, up, levels):
    """Eliminate a tree in its original order using one CUDA kernel if supported.

    Parameters
    ----------
    d, s : arrays
        Two-dimensional diagonal and right-hand side, including sentinel row.
    low, up : arrays
        One-dimensional lower and upper edge coefficients.
    levels : tuple of arrays
        Child indices, parent indices, and validity masks in elimination order.

    Returns
    -------
    tuple of arrays
        Eliminated diagonal and right-hand side; caller inputs remain unchanged.

    Notes
    -----
    The CUDA path supports 32-wide schedules when the Triton extension is
    installed. Other widths and platforms use the existing scan. Use this
    operation inside implicit linear-solve callbacks;
    differentiation of the raw imperative elimination kernel is not supported.
    """
    from .h01_dhs_scan import _triang_raw
    children, parents, valid = levels
    if children.shape[1] != 32 or d.ndim != 2 or low.ndim != 1:
        return _triang_raw(d, s, low, up, levels)

    def fallback(d, s, low, up, children, parents, valid):
        return _triang_raw(d, s, low, up, (children, parents, valid))

    return jax.lax.platform_dependent(d, s, low, up, children, parents, valid,
        cuda=_eliminate_gpu, default=fallback)
