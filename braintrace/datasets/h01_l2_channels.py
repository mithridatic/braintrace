"""Allen 626170538 channel laws at 34 Celsius, with selected phase controls.

The pinned source mechanisms are named in the E profile. These laws include
borrowed nonhuman mechanisms; they are not measurements from the H01 donor.
"""

import jax.numpy as jnp
from jax.nn import sigmoid
import brainunit as u
from braincell.channel._base import Gate
from braincell.mech import register_channel
from .h01_pv_channels import _PVChannel, _DEFINITIONS, _rate_accessor
from .h01_pv_rates import pv_rates, _pair, _Q
from .h01_wilbers import rate_trap


def l2_rates(mechanism, voltage_mv, calcium_mm=1e-4):
    """Return the pinned Allen gate laws before phase scaling.

    Parameters
    ----------
    mechanism : str
        Source mechanism name. Keep static during compilation.
    voltage_mv : array_like
        Membrane voltage in millivolts.
    calcium_mm : array_like, optional
        Inside calcium concentration in millimolar.

    Returns
    -------
    dict
        Gate equilibrium and time constant in milliseconds.
    """
    v, ca = jnp.asarray(voltage_mv), jnp.asarray(calcium_mm)
    if mechanism == "NaTs":
        return {"m": _pair(rate_trap(v, -40., .182, 6.), rate_trap(-v, 40., .124, 6.), 2.3**1.1),
                "h": _pair(rate_trap(-v, 66., .015, 6.), rate_trap(v, -66., .015, 6.), 2.3**1.1)}
    if mechanism == "Nap":
        return {"h": pv_rates("Nap", v)["h"]}
    if mechanism == "Ih":
        return {"m": _pair(rate_trap(-v, 154.9, .00643, 11.9), .193*jnp.exp(v/33.1))}
    if mechanism == "K_P":
        tau = jnp.where(v < -50., 1.25+175.03*jnp.exp(.026*v), 1.25+13.*jnp.exp(-.026*v))/_Q
        return {"m": (sigmoid((v+14.3)/14.6), tau),
                "h": (sigmoid(-(v+54.)/11.), (360.+(1010.+24.*(v+55.))*jnp.exp(-((v+75.)/48.)**2))/_Q)}
    if mechanism == "K_T":
        return {"m": (sigmoid((v+47.)/29.), (.34+.92*jnp.exp(-((v+71.)/59.)**2))/_Q),
                "h": (sigmoid(-(v+66.)/10.), (8.+49.*jnp.exp(-((v+73.)/23.)**2))/_Q)}
    if mechanism == "SK":
        adjusted = jnp.where(ca < 1e-7, ca+1e-7, ca)
        return {"z": (1./(1.+(.00043/adjusted)**4.8), jnp.ones_like(ca))}
    if mechanism not in ("Kv3_1", "Im", "Ca_HVA", "Ca_LVA"):
        raise ValueError(f"Unknown L2 mechanism: {mechanism!r}.")
    return pv_rates(mechanism, v, ca)


class _L2Channel(_PVChannel):
    def _rates(self, voltage, ions):
        ca = ions[1].Ci.to_decimal(u.mM) if self.mechanism == "SK" else 1e-4
        return l2_rates(self.mechanism, voltage.to_decimal(u.mV), ca)

    def conductance_factor(self, voltage, *ions):
        if self.mechanism == "Nap":
            # The source has instantaneous activation, first power, not m cubed.
            return sigmoid((voltage.to_decimal(u.mV)+52.6)/4.6)*self.h.value
        return super().conductance_factor(voltage, *ions)


def _register(mechanism, root, gates):
    attributes = {"__module__": __name__, "mechanism": mechanism,
                  "root_type": root, "gates": tuple(Gate(*g) for g in gates)}
    if mechanism == "SK":
        attributes["current_owner_type"] = _DEFINITIONS["K_P"][0]
    for gate, _ in gates:
        for index, suffix in enumerate(("inf", "tau")):
            attributes[f"f_{gate}_{suffix}"] = _rate_accessor(gate, index)
    return register_channel("H01L2_"+mechanism)(type("_L2_"+mechanism, (_L2Channel,), attributes))


_DEFS = {("NaTs" if k == "NaTg" else k): (r, (("h", 1),) if k == "Nap" else g)
         for k, (r, g) in _DEFINITIONS.items()}
L2_CHANNELS = {k: _register(k, *v) for k, v in _DEFS.items()}
