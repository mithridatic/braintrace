"""Experimental single-program tree contraction; no production activation."""

import jax
import jax.numpy as jnp
import numpy as np


def solve(d, rhs, low, up, stages):
    """Solve a static tree in one GPU program using independent Schur stages.

    Parameters
    ----------
    d, rhs, low, up : arrays
        One-dimensional coefficients including an independent sentinel row.
    stages : sequence
        Ragged removed, parent and child index triples.

    Returns
    -------
    array
        Solution without modifying any input.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    n = d.shape[0]-1
    if not stages:
        return rhs/d
    indices = tuple(jnp.asarray(np.concatenate([s[k] for s in stages])) for k in range(3))
    offsets = jnp.asarray(np.cumsum([0]+[len(s[0]) for s in stages]), dtype=jnp.int32)
    width = 256

    def kernel(di, ri, li, ui, removed, parent, child, offsets, ds, rs, ls, us, xs):
        lanes = jnp.arange(width)

        def initialize(chunk, unused):
            ix = chunk*width+lanes
            valid = ix <= n
            for source, dest in ((di, ds), (ri, rs), (li, ls), (ui, us)):
                plt.store(dest.at[ix], plt.load(source.at[ix], mask=valid, other=0.), mask=valid)
            plt.store(xs.at[ix], jnp.zeros(width, dtype=d.dtype), mask=valid)

        jax.lax.fori_loop(0, (n+width)//width, initialize, None)
        plt.debug_barrier()

        def forward(stage, unused):
            start = plt.load(offsets.at[stage])
            end = plt.load(offsets.at[stage+1])

            def chunk_step(chunk, unused):
                pos = start+chunk*width+lanes
                valid = pos < end
                node = plt.load(removed.at[pos], mask=valid, other=n)
                ancestor = plt.load(parent.at[pos], mask=valid, other=n)
                down = plt.load(child.at[pos], mask=valid, other=n)
                has_child = valid & (down < n)
                diagonal = plt.load(ds.at[node], mask=valid, other=1.)
                value = plt.load(rs.at[node], mask=valid, other=0.)
                lower = plt.load(ls.at[node], mask=valid, other=0.)
                upper = plt.load(us.at[node], mask=valid, other=0.)
                child_low = plt.load(ls.at[down], mask=has_child, other=0.)
                child_up = plt.load(us.at[down], mask=has_child, other=0.)
                plt.atomic_add(ds, (ancestor,), -upper*lower/diagonal, mask=valid)
                plt.atomic_add(rs, (ancestor,), -upper*value/diagonal, mask=valid)
                plt.atomic_add(ds, (down,), -child_low*child_up/diagonal, mask=has_child)
                plt.atomic_add(rs, (down,), -child_low*value/diagonal, mask=has_child)
                plt.store(ls.at[down], -child_low*lower/diagonal, mask=has_child)
                plt.store(us.at[down], -upper*child_up/diagonal, mask=has_child)
                plt.store(us.at[node], child_up, mask=valid)

            jax.lax.fori_loop(0, (end-start+width-1)//width, chunk_step, None)
            plt.debug_barrier()

        jax.lax.fori_loop(0, len(stages), forward, None)
        for index in (0, n):
            plt.store(xs.at[index], plt.load(rs.at[index])/plt.load(ds.at[index]))
        plt.debug_barrier()

        def backward(reverse_stage, unused):
            stage = len(stages)-1-reverse_stage
            start = plt.load(offsets.at[stage])
            end = plt.load(offsets.at[stage+1])

            def chunk_step(chunk, unused):
                pos = start+chunk*width+lanes
                valid = pos < end
                node = plt.load(removed.at[pos], mask=valid, other=n)
                ancestor = plt.load(parent.at[pos], mask=valid, other=n)
                down = plt.load(child.at[pos], mask=valid, other=n)
                value = plt.load(rs.at[node], mask=valid, other=0.)
                lower = plt.load(ls.at[node], mask=valid, other=0.)
                upper = plt.load(us.at[node], mask=valid, other=0.)
                x_parent = plt.load(xs.at[ancestor], mask=valid, other=0.)
                x_child = plt.load(xs.at[down], mask=valid, other=0.)
                diagonal = plt.load(ds.at[node], mask=valid, other=1.)
                plt.store(xs.at[node], (value-lower*x_parent-upper*x_child)/diagonal, mask=valid)

            jax.lax.fori_loop(0, (end-start+width-1)//width, chunk_step, None)
            plt.debug_barrier()

        jax.lax.fori_loop(0, len(stages), backward, None)

    shape = jax.ShapeDtypeStruct(d.shape, d.dtype)
    return pl.pallas_call(kernel, out_shape=(shape,)*5, grid=(1,),
                          compiler_params=plt.CompilerParams(num_warps=4))(
        d, rhs, low, up, *indices, offsets)[-1]
