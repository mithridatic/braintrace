"""Compact numerical scan for the opt-in tree contraction experiment."""

import jax
import jax.numpy as jnp
import numpy as np


def solve(d, rhs, low, up, stages):
    """Solve the same contracted tree while retaining eliminated rows in place.

    Parameters
    ----------
    d, rhs, low, up : arrays
        Diagonal, RHS, and edge coefficients including a neutral sentinel.
    stages : list of tuple
        Independent removed, parent, and child index arrays.

    Returns
    -------
    array
        Tree solution, including the independent sentinel row.
    """
    n = d.shape[0]-1
    if not stages:
        return rhs/d
    largest = max(len(stage[0]) for stage in stages)
    groups = []
    previous = None
    for stage in stages:
        width = len(stage[0])
        bucket = 0 if width > largest/4 else (1 if width > largest/64 else 2)
        if bucket != previous:
            groups.append([])
            previous = bucket
        groups[-1].append(stage)
    packs = []
    for group in groups:
        width = max(len(stage[0]) for stage in group)
        packed = tuple(np.full((len(group), width), n, dtype=np.int32) for _ in range(3))
        for index, stage in enumerate(group):
            for target, values in zip(packed, stage):
                target[index, :len(values)] = values
        packs.append(packed)

    def eliminate(carry, indices):
        d, rhs, low, up = carry
        removed, parent, child = indices
        diagonal, value = d[removed], rhs[removed]
        lower, upper = low[removed], up[removed]
        child_lower = jnp.where(child < n, low[child], 0.)
        child_upper = jnp.where(child < n, up[child], 0.)
        parent_target = jnp.where(removed < n, parent, n+1)
        child_target = jnp.where(child < n, child, n+1)
        removed_target = jnp.where(removed < n, removed, n+1)
        d = d.at[parent_target].add(-upper*lower/diagonal, mode='drop')
        rhs = rhs.at[parent_target].add(-upper*value/diagonal, mode='drop')
        d = d.at[child_target].add(-child_lower*child_upper/diagonal, mode='drop')
        rhs = rhs.at[child_target].add(-child_lower*value/diagonal, mode='drop')
        low = low.at[child_target].set(-child_lower*lower/diagonal, mode='drop')
        up = up.at[child_target].set(-upper*child_upper/diagonal, mode='drop')
        up = up.at[removed_target].set(child_upper, mode='drop')
        return (d, rhs, low, up), None

    for packed in packs:
        (d, rhs, low, up), _ = jax.lax.scan(eliminate, (d, rhs, low, up), packed)
    result = jnp.zeros_like(rhs).at[0].set(rhs[0]/d[0]).at[-1].set(rhs[-1]/d[-1])

    def substitute(result, indices):
        removed, parent, child = indices
        values = (rhs[removed]-low[removed]*result[parent]-up[removed]*result[child])/d[removed]
        target = jnp.where(removed < n, removed, n+1)
        return result.at[target].set(values, mode='drop'), None

    for packed in reversed(packs):
        result, _ = jax.lax.scan(substitute, result, packed, reverse=True)
    return result


if __name__ == '__main__':
    import h01_contraction_profile as profile
    profile.solve = solve
    profile.main()
