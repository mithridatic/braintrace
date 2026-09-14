"""Opt-in physical probe for the experimental contraction solve callbacks."""

import hashlib
from pathlib import Path

import jax
import numpy as np

import h01_event_profile as profile
from h01_contraction_profile import schedule, solve
from braintrace.datasets import h01_dhs_scan


def main():
    """Install experimental callbacks only in this isolated profiling process.

    Returns
    -------
    None
        Writes synchronized phase evidence via the existing profile driver.
    """
    schedules = {}
    original = h01_dhs_scan._solve_raw

    def contracted(d, s, low, up, levels, jumps, edges, *, use_gpu=True):
        if not use_gpu or d.ndim != 2:
            return original(d, s, low, up, levels, jumps, edges, use_gpu=use_gpu)
        key = (d.shape[-1], edges.tobytes())
        if key not in schedules:
            parents = np.zeros(d.shape[-1]-1, dtype=np.int32)
            parents[edges[:, 0]] = edges[:, 1]
            schedules[key] = schedule(parents)
        stages = schedules[key]
        child, parent = edges[:, 0], edges[:, 1]

        def matvec(value):
            result = d*value
            result = result.at[..., child].add(low[child]*value[..., parent])
            return result.at[..., parent].add(up[child]*value[..., child])

        def forward(_, rhs):
            return jax.vmap(lambda a, b: solve(a, b, low, up, stages))(d, rhs)

        def transpose(_, rhs):
            return jax.vmap(lambda a, b: solve(a, b, up, low, stages))(d, rhs)

        return jax.lax.custom_linear_solve(matvec, s, solve=forward, transpose_solve=transpose)

    settings = profile.numerical_settings

    def experimental_settings():
        result = settings()
        result['experimental_solver'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name('h01_contraction_profile.py'))}
        return result

    profile.numerical_settings = experimental_settings
    h01_dhs_scan._solve_raw = contracted
    try:
        profile.main()
    finally:
        h01_dhs_scan._solve_raw = original


if __name__ == '__main__':
    main()
