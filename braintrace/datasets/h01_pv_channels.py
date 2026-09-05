"""Stateful channels for the human-fitted ModelDB 267587 HL5BN1 cell.

These channels preserve the released 34 Celsius kinetics. They include
borrowed nonhuman kinetics and are not measurements from the H01 donor.
Import this module to register the ``H01PV_*`` mechanism names.
"""

import brainstate
import jax.numpy as jnp
import numpy as np
import braintools
import brainunit as u
from braincell._base import HHTypedNeuron
from braincell.channel._base import Gate, HH
from braincell.ion import Calcium, Potassium, Sodium
from braincell.mech import register_channel

from .h01_pv_rates import pv_rates, _pair, _Q
from .h01_wilbers import rate_trap


class _PVChannel(HH):
    def __init__(self, size, g_max, name=None, *, m_open=1., m_close=1.,
                 h_open=1., h_close=1., h_slope=6.):
        factors = (m_open, m_close, h_open, h_close, h_slope)
        if any(not np.isfinite(f).all() or np.any(np.asarray(f) <= 0) for f in factors):
            raise ValueError("Gate factors and slope must be positive and finite.")
        self.phase_factors = {"m": (m_open, m_close), "h": (h_open, h_close)}
        self.h_slope = h_slope
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

    def init_state(self, voltage, *ions, batch_size=None):
        """Allocate gate states at equilibrium.

        Parameters
        ----------
        voltage : Quantity
            Initial voltage, including any batch dimension.
        *ions : IonInfo
            Ion information in root-type order.
        batch_size : int, optional
            Leading batch dimension.
        """
        super().init_state(voltage, *ions, batch_size=batch_size)
        self.reset_state(voltage, *ions, batch_size=batch_size)

    def _rates(self, voltage, ions):
        calcium = ions[1].Ci.to_decimal(u.mM) if self.mechanism == "SK" else 1e-4
        v = voltage.to_decimal(u.mV)
        rates = pv_rates(self.mechanism, v, calcium)
        if self.mechanism == "NaTg":
            rates["h"] = _pair(rate_trap(-v, 56., .015, self.h_slope),
                               rate_trap(v, -56., .015, self.h_slope), _Q)
        return rates

    def _rate_for_gate(self, gate, voltage, ions, component=None):
        calcium = ions[1].Ci.to_decimal(u.mM) if self.mechanism == "SK" else 1e-4
        v = voltage.to_decimal(u.mV)
        if self.mechanism == "NaTg" and gate == "h":
            if component == 0:
                a = rate_trap(-v, 56., .015, self.h_slope)
                b = rate_trap(v, -56., .015, self.h_slope)
                return a / (a + b)
            if component == 1:
                return 1. / (_Q * (rate_trap(-v, 56., .015, self.h_slope) + rate_trap(v, -56., .015, self.h_slope)))
            return _pair(rate_trap(-v, 56., .015, self.h_slope),
                         rate_trap(v, -56., .015, self.h_slope), _Q)
        return pv_rates(self.mechanism, v, calcium, gate=gate, component=component)

    def current(self, voltage, *ions):
        """Return inward-positive current density.

        Parameters
        ----------
        voltage : Quantity
            Membrane voltage.
        *ions : IonInfo
            Ion information in root-type order. SK uses potassium then calcium.

        Returns
        -------
        Quantity
            Conductance density times the inward driving potential.
        """
        reversal = -45.*u.mV if self.mechanism == "Ih" else ions[0].E
        return self.g_max * self.conductance_factor(voltage, *ions) * (reversal-voltage)


def _rate_accessor(gate, component):
    def rate(self, voltage, *ions):
        if component == 0:
            return self._rate_for_gate(gate, voltage, ions, component=0)
        tau = self._rate_for_gate(gate, voltage, ions, component=1)
        opening, closing = self.phase_factors.get(gate, (1., 1.))
        state = getattr(self, gate, None)
        equilibrium = self._rate_for_gate(gate, voltage, ions, component=0)
        value = equilibrium if state is None else state.value
        if self.mechanism == "Kv3_1":
            return tau*jnp.where(equilibrium < value, closing, opening)
        return tau*jnp.where(equilibrium > value, opening, closing)
    return rate


_DEFINITIONS = {
    "NaTg": (Sodium, (("m", 3), ("h", 1))),
    "Nap": (Sodium, (("m", 3), ("h", 1))),
    "K_P": (Potassium, (("m", 2), ("h", 1))),
    "K_T": (Potassium, (("m", 4), ("h", 1))),
    "Kv3_1": (Potassium, (("m", 1),)),
    "Im": (Potassium, (("m", 1),)),
    "Ih": (HHTypedNeuron, (("m", 1),)),
    "Ca_HVA": (Calcium, (("m", 2), ("h", 1))),
    "Ca_LVA": (Calcium, (("m", 2), ("h", 1))),
    "SK": (brainstate.mixin.JointTypes[Potassium, Calcium], (("z", 1),)),
}


def _channel_class(mechanism, root, gates):
    attributes = {"__module__": __name__, "mechanism": mechanism,
                  "root_type": root, "gates": tuple(Gate(*g) for g in gates)}
    if mechanism == "SK":
        # SK senses calcium, but carries potassium. Do not add it to calcium flux.
        attributes["current_owner_type"] = Potassium
    for gate, _ in gates:
        for index, suffix in enumerate(("inf", "tau")):
            attributes[f"f_{gate}_{suffix}"] = _rate_accessor(gate, index)
    cls = type(f"_PV_{mechanism}", (_PVChannel,), attributes)
    return register_channel(f"H01PV_{mechanism}")(cls)


_CHANNELS = {key: _channel_class(key, *definition) for key, definition in _DEFINITIONS.items()}
PV_CHANNEL_NAMES = {key: f"H01PV_{key}" for key in _CHANNELS}


def make_pv_channel(mechanism, size, g_max, name=None):
    """Create a channel with the published fitted kinetics.

    Parameters
    ----------
    mechanism : str
        Source mechanism key listed in ``PV_CHANNEL_NAMES``.
    size : int or tuple
        BrainCell state shape.
    g_max : Quantity
        Maximum conductance density, with explicit units.
    name : str, optional
        BrainState node name.

    Returns
    -------
    HH
        Stateful channel. Initialize it with voltage and the required ions.
    """
    if mechanism not in _CHANNELS:
        raise ValueError(f"Unknown PV mechanism: {mechanism!r}.")
    return _CHANNELS[mechanism](size=size, g_max=g_max, name=name)
