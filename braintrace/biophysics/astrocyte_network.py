"""Spatial astrocyte calcium with explicit shell, buffer and reservoir state."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from .astrocyte import CalciumParameters, initial_calcium, calcium_derivative
from .transport import DiffusionGraph
from .astrocyte_coupled import coupled_calcium_step


def shell_transport(length_um, diameter_um, segment_edges, *, dt_ms, diffusion, radial=True):
    """Construct the four-annulus spatial discretization in cadifus.mod.

    Parameters
    ----------
    length_um, diameter_um : array-like
        Positive cylindrical segment dimensions.
    segment_edges : array-like
        Undirected neighbouring segment index pairs.
    dt_ms, diffusion : float
        Positive step and nonnegative diffusivity in um^2/ms.
    radial : bool, optional
        Include radial flux; bound mobile buffer diffuses longitudinally only
        in the pinned source and must use False.

    Returns
    -------
    DiffusionGraph
        Segment-major, outside-to-inside four-shell transport.
    """
    lengths, diameters = map(lambda x: np.asarray(x, dtype=float), (length_um, diameter_um))
    neighbours = np.asarray(segment_edges)
    if (lengths.ndim != 1 or not len(lengths) or diameters.shape != lengths.shape or
            not np.isfinite(lengths).all() or not np.isfinite(diameters).all() or
            np.any(lengths <= 0) or np.any(diameters <= 0)):
        raise ValueError('Positive segment lengths and diameters required')
    if (neighbours.ndim != 2 or neighbours.shape[1] != 2 or
            not np.issubdtype(neighbours.dtype, np.integer) or
            np.any(neighbours < 0) or np.any(neighbours >= len(lengths))):
        raise ValueError('Invalid segment adjacency')
    if not np.isfinite(diffusion) or diffusion < 0:
        raise ValueError('Diffusion must be finite and nonnegative')
    # Exact fractions from the pinned factors() routine, with radius/diameter=.5.
    fractions = np.pi*np.array([11/144, 1/9, 1/18, 1/144])
    cross = diameters[:, None]**2*fractions
    volumes = lengths[:, None]*cross
    edges, rates = [], []
    if radial:
        for i, length in enumerate(lengths):
            for shell, factor in enumerate(np.pi*np.array([5., 3., 1.])):
                edges.append((4*i+shell, 4*i+shell+1))
                rates.append(diffusion*factor*length)
    for left, right in neighbours:
        for shell in range(4):
            edges.append((4*left+shell, 4*right+shell))
            rates.append(diffusion/(lengths[left]/(2*cross[left, shell])+lengths[right]/(2*cross[right, shell])))
    return DiffusionGraph(volumes.ravel(), np.asarray(edges, dtype=int).reshape(-1, 2), rates, dt_ms=dt_ms)


class AstrocyteCalcium(brainstate.nn.Module):
    """Advance explicit astrocyte reactions and spatial diffusion in BrainState.

    Parameters
    ----------
    length_um, diameter_um, segment_edges : array-like
        Physical cylindrical compartment geometry and adjacency.
    dt_ms : float, optional
        Cable-synchronized physical interval; default .005 ms.
    parameters : CalciumParameters or None, optional
        Fixed paper-configuration rates; no trainable biological parameters.
    substeps : int, optional
        Internal coupled solves per cable tick; default 4 resolves the strong
        local calcium-pulse refinement fixture at .00125 ms.
    geometry : CableChemicalGeometry or None, optional
        Exact tapered cable CV map. When supplied, omit all three cylindrical
        geometry arguments. The map preserves actual volumes and membrane area.

    Notes
    -----
    Coupled backward Euler uses matrix-free Newton-Krylov and requires time
    refinement. Source
    bound mobile buffer has longitudinal but no radial diffusion. ER and plasma
    exchange are accumulated separately; no whole-cell energy claim is made.
    """

    def __init__(self, length_um=None, diameter_um=None, segment_edges=None, *, dt_ms=.005,
                 parameters=None, substeps=4, geometry=None):
        super().__init__()
        if type(substeps) is not int or substeps < 1:
            raise ValueError('Calcium substeps must be a positive integer')
        self.substeps = substeps
        self.parameters = parameters or CalciumParameters()
        p = self.parameters
        args = (length_um, diameter_um, segment_edges)
        self.geometry_sha256 = None
        if geometry is None:
            if any(value is None for value in args):
                raise ValueError('Complete cylindrical geometry or a cable geometry map required')
            self.calcium_transport = shell_transport(*args, dt_ms=dt_ms, diffusion=p.diffusion_um2_ms)
            self.free_transport = shell_transport(*args, dt_ms=dt_ms, diffusion=p.mobile_diffusion_um2_ms)
            self.bound_transport = shell_transport(*args, dt_ms=dt_ms, diffusion=p.mobile_diffusion_um2_ms, radial=False)
            surface = np.pi*np.asarray(diameter_um)*np.asarray(length_um)
            n = len(length_um)
        else:
            from .cable_geometry import CableChemicalGeometry
            if not isinstance(geometry, CableChemicalGeometry) or any(value is not None for value in args):
                raise ValueError('Cable geometry must be explicit and cannot mix with cylindrical arguments')
            self.calcium_transport = geometry.transport(dt_ms=dt_ms, diffusion=p.diffusion_um2_ms)
            self.free_transport = geometry.transport(dt_ms=dt_ms, diffusion=p.mobile_diffusion_um2_ms)
            self.bound_transport = geometry.transport(dt_ms=dt_ms, diffusion=p.mobile_diffusion_um2_ms, radial=False)
            surface, n = geometry.area_um2, len(geometry.volume_um3)
            self.geometry_sha256 = geometry.sha256
        self.dt_ms = dt_ms
        self.initial = jnp.broadcast_to(initial_calcium(p), (n, 25))
        self.state = brainstate.HiddenState(self.initial)
        volumes = self.calcium_transport.volumes.reshape(n, 4)
        self.surface_to_volume = jnp.asarray(surface)/volumes[:, 0]
        self.valid = brainstate.ShortTermState(jnp.asarray(True))
        self.er_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.pumped_amount = brainstate.ShortTermState(jnp.asarray(0.))
        self.tick = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def update(self, glutamate_mm, *, influx_mm_ms=None, clamped_ip3_mm=None):
        """Advance all chemical states and return free calcium per segment/shell.

        Parameters
        ----------
        glutamate_mm : array
            Concentration per physical segment driving PLC-beta.
        influx_mm_ms : array or None, optional
            Additional calcium source rate per segment/shell; excludes own pump.
        clamped_ip3_mm : array or None, optional
            Experimental IP3 clamp per segment, otherwise the source ODE evolves.

        Returns
        -------
        array
            Free calcium (n_segments, 4), in mM. Check cumulative ``valid``.
        """
        old = self.state.value
        glutamate = jnp.asarray(glutamate_mm)
        influx = jnp.zeros((len(old), 4)) if influx_mm_ms is None else jnp.asarray(influx_mm_ms)
        if glutamate.shape != (len(old),) or influx.shape != (len(old), 4):
            raise ValueError('Chemical inputs must match segment and shell counts')
        if clamped_ip3_mm is not None and jnp.shape(clamped_ip3_mm) != (len(old),):
            raise ValueError('IP3 clamp must match segments')
        clamp_axis = None if clamped_ip3_mm is None else 0
        def exchange(state, surface, glu, source, clamp):
            _, er, pump = calcium_derivative(state, self.parameters, surface_to_volume=surface,
                glutamate_mm=glu, influx_mm_ms=source, clamped_ip3_mm=clamp)
            return er, pump
        volume = self.calcium_transport.volumes.reshape(-1, 4)
        dt = self.dt_ms/self.substeps
        def microstep(carry, _):
            state, valid, er_total, pump_total = carry
            updated, local_valid, _ = coupled_calcium_step(state, self.parameters,
                (self.calcium_transport, self.free_transport, self.bound_transport), dt_ms=dt,
                surface_to_volume=self.surface_to_volume, glutamate_mm=glutamate,
                influx_mm_ms=influx, clamped_ip3_mm=clamped_ip3_mm)
            er, pump = jax.vmap(exchange, in_axes=(0, 0, 0, 0, clamp_axis))(
                updated, self.surface_to_volume, glutamate, influx, clamped_ip3_mm)
            return (updated, valid & local_valid, er_total+dt*jnp.sum(er*volume),
                    pump_total+dt*jnp.sum(pump*volume[:, 0])), None
        (updated, valid, er_amount, pump_amount), _ = brainstate.transform.scan(microstep,
            (old, jnp.asarray(True), jnp.asarray(0.), jnp.asarray(0.)), jnp.arange(self.substeps))
        self.state.value = updated
        self.valid.value = self.valid.value & valid & jnp.all(glutamate >= 0)
        self.er_amount.value = self.er_amount.value + er_amount
        self.pumped_amount.value = self.pumped_amount.value + pump_amount
        self.tick.value = self.tick.value + 1
        return updated[:, :4]

    def reset_state(self):
        """Reset all concentrations, reservoir ledgers, validity and clock."""
        self.state.value = self.initial
        self.valid.value = jnp.asarray(True)
        self.er_amount.value = jnp.zeros_like(self.er_amount.value)
        self.pumped_amount.value = jnp.zeros_like(self.pumped_amount.value)
        self.tick.value = jnp.zeros_like(self.tick.value)
