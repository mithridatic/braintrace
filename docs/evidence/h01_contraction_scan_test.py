"""Run the same numerical stress contract against compact contraction."""

import pytest

import h01_contraction_profile_test as contract
from h01_contraction_scan import solve


def test_independent_sentinel_row_is_solved_without_false_edges():
    import brainstate
    import jax
    import jax.numpy as jnp
    import numpy as np
    from h01_contraction_profile import schedule
    with brainstate.environ.context(precision=64):
        stages = schedule([0, 0])
        args = tuple(jnp.asarray(x) for x in ([3., 4., 2.], [1., 2., .4], [0., -.1, 9.], [0., -.2, 7.]))

        def dense(d, rhs, low, up):
            matrix = jnp.diag(d).at[1, 0].set(low[1]).at[0, 1].set(up[1])
            return jnp.linalg.solve(matrix, rhs)

        candidate = lambda *values: solve(*values, stages)
        np.testing.assert_allclose(candidate(*args), dense(*args), rtol=1e-12, atol=1e-12)
        for mode in (jax.jacfwd, jax.jacrev):
            for got, want in zip(mode(candidate, argnums=(0, 1, 2, 3))(*args),
                                 mode(dense, argnums=(0, 1, 2, 3))(*args)):
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.fixture(autouse=True)
def compact_solver(monkeypatch):
    monkeypatch.setattr(contract, 'solve', solve)


test_contraction_matches_dense_stiff_systems = contract.test_contraction_matches_dense_stiff_systems
test_schedule_eliminates_every_nonroot_exactly_once = contract.test_schedule_eliminates_every_nonroot_exactly_once


def test_compact_physical_learning_state_matches_scan(tmp_path, monkeypatch):
    import h01_contraction_event_profile as driver
    from braintrace.datasets.h01_dhs_gpu_test import test_real_branched_cell_sparse_updates_match_scan
    calls = []

    def counted(*args):
        calls.append(True)
        return solve(*args)

    def check():
        # Undo the oracle's temporary solver choices before the driver restores
        # the production solver, avoiding a leaked experimental callback.
        with monkeypatch.context() as oracle_patch:
            test_real_branched_cell_sparse_updates_match_scan(tmp_path, oracle_patch)

    monkeypatch.setattr(driver, 'solve', counted)
    monkeypatch.setattr(driver.profile, 'main', check)
    monkeypatch.setattr(driver.profile, 'numerical_settings', driver.profile.numerical_settings)
    driver.main()
    assert calls
