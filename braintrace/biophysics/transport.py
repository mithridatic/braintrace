"""Conservative, differentiable finite-volume diffusion without dense matrices."""

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np


class TransportResult(NamedTuple):
    """One transport step and its conservation/solver diagnostics.

    Attributes
    ----------
    concentration : array
        Concentrations in mM.
    boundary_amount : scalar
        Net amount entering through fixed boundaries, in mM um^3.
    removed_amount : scalar
        Amount removed by first-order uptake, in mM um^3.
    balance_error : scalar
        Change in amount minus accounted sources, boundaries and sinks.
    relative_residual : scalar
        Relative residual of the symmetric implicit linear system.
    valid : scalar bool
        Finite, nonnegative concentration and converged solve.
    """

    concentration: object
    boundary_amount: object
    removed_amount: object
    balance_error: object
    relative_residual: object
    valid: object


class DiffusionGraph:
    """Compile a finite-volume graph for backward-Euler diffusion and uptake.

    Parameters
    ----------
    volumes : array-like
        Positive accessible volumes in um^3, including extracellular fraction.
    edges : array-like
        Unique undirected index pairs with shape (n_edges, 2).
    conductance : array-like
        D_eff times interface area divided by centre distance, in um^3/ms.
    dt_ms : float
        Positive integration interval.
    boundary_conductance : array-like, optional
        Conductance to an external fixed reservoir per volume, in um^3/ms.
    uptake_per_ms : array-like, optional
        Nonnegative first-order sink rate per volume, in 1/ms.
    tolerance : float, optional
        Relative linear-solve tolerance; default 1e-10 (use float64).
    max_iterations : int, optional
        Positive CG iteration limit; default 1000.

    Notes
    -----
    The mass-normalized system is symmetric positive definite. JAX CG uses
    implicit differentiation of its solve; no dense Jacobian is materialized.
    Callers must reject results with ``valid == False`` before qualification.
    Sources are amount rates, not concentration rates. Reservoir flux is
    evaluated at the new concentration, consistently with backward Euler.
    """

    def __init__(self, volumes, edges, conductance, *, dt_ms,
                 boundary_conductance=None, uptake_per_ms=None,
                 tolerance=1e-10, max_iterations=1000):
        volume = np.asarray(volumes, dtype=float)
        pairs = np.asarray(edges)
        rates = np.asarray(conductance, dtype=float)
        if volume.ndim != 1 or not volume.size or not np.isfinite(volume).all() or np.any(volume <= 0):
            raise ValueError("Volumes must be a nonempty positive finite vector")
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or
                not np.issubdtype(pairs.dtype, np.integer) or
                np.any(pairs < 0) or np.any(pairs >= volume.size)):
            raise ValueError("Edges must be integer pairs inside the volume domain")
        canonical = np.sort(pairs, axis=1)
        if np.any(canonical[:, 0] == canonical[:, 1]) or len(np.unique(canonical, axis=0)) != len(pairs):
            raise ValueError("Edges must be unique and cannot connect a volume to itself")
        if rates.shape != (len(pairs),) or not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("Conductance must be a nonnegative finite vector matching edges")
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be positive and finite")
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("tolerance must lie strictly between zero and one")
        if type(max_iterations) is not int or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        boundary = self._vector(boundary_conductance, volume.shape, "boundary_conductance")
        uptake = self._vector(uptake_per_ms, volume.shape, "uptake_per_ms")
        self.closed = not bool(np.any(boundary) or np.any(uptake))
        self.volumes = jnp.asarray(volume)
        self.edges = jnp.asarray(pairs, dtype=jnp.int32)
        self.conductance = jnp.asarray(rates)
        self.boundary = jnp.asarray(boundary)
        self.uptake = jnp.asarray(uptake)
        self.dt_ms = float(dt_ms)
        self.tolerance = float(tolerance)
        self.max_iterations = max_iterations

    @staticmethod
    def _vector(value, shape, name):
        result = np.zeros(shape) if value is None else np.asarray(value, dtype=float)
        if result.shape != shape or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError(name + " must be a nonnegative finite vector matching volumes")
        return result

    def step(self, concentration, *, source_amount_per_ms=None, reservoir_mm=None):
        """Advance concentrations and return explicit validity and balance evidence.

        Parameters
        ----------
        concentration : array
            One mM concentration per volume.
        source_amount_per_ms : array, optional
            Signed external amount rate in mM um^3/ms; default zero.
        reservoir_mm : array, optional
            Nonnegative reservoir concentration per volume; default zero.

        Returns
        -------
        TransportResult
            New state and numerical diagnostics; invalid values are not clipped.
        """
        c = jnp.asarray(concentration)
        source = jnp.zeros_like(c) if source_amount_per_ms is None else jnp.asarray(source_amount_per_ms)
        reservoir = jnp.zeros_like(c) if reservoir_mm is None else jnp.asarray(reservoir_mm)
        if c.shape != self.volumes.shape or source.shape != c.shape or reservoir.shape != c.shape:
            raise ValueError("Concentration, source and reservoir must match volumes")
        mass = self.volumes
        root = jnp.sqrt(mass)
        left, right = self.edges[:, 0], self.edges[:, 1]
        diagonal = mass * self.uptake + self.boundary
        dt = self.dt_ms

        def operator(x):
            value = x / root
            flux = self.conductance * (value[left] - value[right])
            laplace = jnp.zeros_like(value).at[left].add(flux).at[right].add(-flux)
            return x + dt * (laplace + diagonal * value) / root

        # Solve the concentration increment, not the large baseline pool.
        # Otherwise a small real membrane flux can fall below relative CG
        # tolerance and disappear while the intracellular pool still debits it.
        old_flux = self.conductance*(c[left]-c[right])
        old_laplace = jnp.zeros_like(c).at[left].add(old_flux).at[right].add(-old_flux)
        rhs = dt*(source+self.boundary*reservoir-old_laplace-diagonal*c)/root
        degree = jnp.zeros_like(c).at[left].add(self.conductance).at[right].add(self.conductance)
        preconditioner = 1 + dt*(diagonal + degree)/mass
        solved, _ = cg(operator, rhs, x0=jnp.zeros_like(c), tol=self.tolerance, atol=0.,
                       maxiter=self.max_iterations, M=lambda x: x/preconditioner)
        updated = c+solved/root
        residual = jnp.linalg.norm(operator(solved)-rhs) / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(c.dtype).tiny)
        boundary_amount = dt*jnp.sum(self.boundary*(reservoir-updated))
        removed = dt*jnp.sum(mass*self.uptake*updated)
        error = jnp.sum(mass*(updated-c)) - dt*jnp.sum(source) - boundary_amount + removed
        valid = (jnp.all(jnp.isfinite(updated)) & jnp.all(updated >= 0) &
                 jnp.all(jnp.isfinite(c)) & jnp.all(c >= 0) &
                 jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(reservoir)) &
                 jnp.all(reservoir >= 0) & (residual <= 10*self.tolerance))
        return TransportResult(updated, boundary_amount, removed, error, residual, valid)
