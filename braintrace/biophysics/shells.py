"""Conservative radial-shell geometry for myelin and intracellular diffusion."""

import numpy as np

from .transport import DiffusionGraph


def radial_shell_graph(length_um, radii_um, diffusion_um2_ms, *, dt_ms,
                       outer_reservoir=False, longitudinal=False):
    """Build a radial finite-volume graph around aligned cylindrical segments.

    Parameters
    ----------
    length_um : array-like
        Positive segment lengths; flattening order is segment then shell.
    radii_um : array-like
        Strictly increasing nonnegative shell boundaries, common to segments.
    diffusion_um2_ms : array-like
        Positive diffusivity per shell. Interfaces use series resistances.
    dt_ms : float
        Physical step in ms.
    outer_reservoir : bool or array-like, optional
        Fixed-bath boundary at the outer radius for selected segments.
    longitudinal : bool, optional
        Also connect adjacent segments. False matches the paper's myelin mode.

    Returns
    -------
    DiffusionGraph
        Implicit conservative shell transport, including optional bath exchange.
    """
    length, radii, diff = map(lambda x: np.asarray(x, dtype=float),
                              (length_um, radii_um, diffusion_um2_ms))
    if length.ndim != 1 or not len(length) or not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError("Segment lengths must be positive and finite")
    if (radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all() or
            radii[0] < 0 or np.any(np.diff(radii) <= 0)):
        raise ValueError("Shell radii must be nonnegative and strictly increasing")
    ns = len(radii)-1
    if diff.shape != (ns,) or not np.isfinite(diff).all() or np.any(diff <= 0):
        raise ValueError("Positive diffusivity required for every shell")
    bath = np.broadcast_to(np.asarray(outer_reservoir, dtype=bool), length.shape)
    centres = (radii[1:]+radii[:-1])/2
    area = np.pi*np.diff(radii**2)
    volumes = (length[:, None]*area[None, :]).reshape(-1)
    edges, conductance = [], []
    # Static mesh construction, not model stepping.
    for segment, size in enumerate(length):
        for shell in range(ns-1):
            boundary = radii[shell+1]
            resistance = np.log(boundary/centres[shell])/diff[shell]
            resistance += np.log(centres[shell+1]/boundary)/diff[shell+1]
            edges.append((segment*ns+shell, segment*ns+shell+1))
            conductance.append(2*np.pi*size/resistance)
        if longitudinal and segment:
            for shell in range(ns):
                edges.append(((segment-1)*ns+shell, segment*ns+shell))
                conductance.append(diff[shell]*area[shell]/((length[segment-1]+size)/2))
    boundary_rates = np.zeros_like(volumes)
    for segment in np.flatnonzero(bath):
        boundary_rates[segment*ns+ns-1] = 2*np.pi*length[segment]*diff[-1]/np.log(radii[-1]/centres[-1])
    return DiffusionGraph(volumes, np.asarray(edges, dtype=int).reshape(-1, 2), conductance,
                          dt_ms=dt_ms, boundary_conductance=boundary_rates)


def cartesian_graph(shape, spacing_um, *, diffusion_um2_ms, dt_ms,
                    volume_fraction=.2, tortuosity=1.6, fixed_boundary=True):
    """Build a uniform three-dimensional extracellular finite-volume grid.

    Parameters
    ----------
    shape : tuple of int
        Positive voxel counts along x, y, z.
    spacing_um : float
        Positive cubic voxel width.
    diffusion_um2_ms, dt_ms : float
        Free diffusion coefficient and physical step.
    volume_fraction, tortuosity : float, optional
        Accessible fraction and path-length factor; D_eff = D/tortuosity^2.
    fixed_boundary : bool, optional
        Couple each outer face to a fixed reservoir at half-voxel distance.

    Returns
    -------
    DiffusionGraph
        C-order grid with explicit boundary exchange.
    """
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Grid shape must contain three positive integers")
    if (not np.isfinite([spacing_um, diffusion_um2_ms, volume_fraction, tortuosity]).all() or
            min(spacing_um, diffusion_um2_ms, volume_fraction) <= 0 or volume_fraction > 1 or tortuosity < 1):
        raise ValueError("Invalid spacing, diffusion, volume fraction or tortuosity")
    ids = np.arange(np.prod(shape)).reshape(shape)
    blocks = []
    boundary = np.zeros(shape)
    g = diffusion_um2_ms/tortuosity**2*volume_fraction*spacing_um
    for axis in range(3):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        blocks.append(np.column_stack((ids[tuple(left)].ravel(), ids[tuple(right)].ravel())))
        if fixed_boundary:
            for side in (0, -1):
                face = [slice(None)]*3
                face[axis] = side
                boundary[tuple(face)] += 2*g
    edges = np.concatenate(blocks)
    return DiffusionGraph(np.full(ids.size, volume_fraction*spacing_um**3), edges,
        np.full(len(edges), g), dt_ms=dt_ms, boundary_conductance=boundary.ravel())
