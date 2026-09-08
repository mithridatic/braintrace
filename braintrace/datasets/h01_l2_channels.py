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


def l2_rates(mechanism, voltage_mv, calcium_mm=1e-4, *, gate=None, component=None):
    """Return the pinned Allen gate laws before phase scaling.

    Parameters
    ----------
    mechanism : str
        Source mechanism name. Keep static during compilation.
    voltage_mv : array_like
        Membrane voltage in millivolts.
    calcium_mm : array_like, optional
        Inside calcium concentration in millimolar.
    gate : str or None, optional
        Specific gate identifier ('m', 'h', 'z') to compute only that gate.
    component : int or None, optional
        0 for equilibrium only, 1 for tau_ms only, None for both.

    Returns
    -------
    dict or tuple or array
        Gate equilibrium and time constant in milliseconds, or single component array.
    """
    v, ca = jnp.asarray(voltage_mv), jnp.asarray(calcium_mm)
    if mechanism == "NaTs":
        if gate == "m":
            if component == 0:
                a = rate_trap(v, -40., .182, 6.)
                b = rate_trap(-v, 40., .124, 6.)
                return a / (a + b)
            if component == 1:
                return 1. / (2.3**1.1 * (rate_trap(v, -40., .182, 6.) + rate_trap(-v, 40., .124, 6.)))
            return _pair(rate_trap(v, -40., .182, 6.), rate_trap(-v, 40., .124, 6.), 2.3**1.1)
        if gate == "h":
            if component == 0:
                a = rate_trap(-v, 66., .015, 6.)
                b = rate_trap(v, -66., .015, 6.)
                return a / (a + b)
            if component == 1:
                return 1. / (2.3**1.1 * (rate_trap(-v, 66., .015, 6.) + rate_trap(v, -66., .015, 6.)))
            return _pair(rate_trap(-v, 66., .015, 6.), rate_trap(v, -66., .015, 6.), 2.3**1.1)
        return {"m": _pair(rate_trap(v, -40., .182, 6.), rate_trap(-v, 40., .124, 6.), 2.3**1.1),
                "h": _pair(rate_trap(-v, 66., .015, 6.), rate_trap(v, -66., .015, 6.), 2.3**1.1)}
    if mechanism == "Nap":
        res = pv_rates("Nap", v, gate="h", component=component)
        return res if gate == "h" or component is not None else {"h": res}
    if mechanism == "Ih":
        if component == 0:
            a = rate_trap(-v, 154.9, .00643, 11.9)
            b = .193 * jnp.exp(v / 33.1)
            return a / (a + b)
        if component == 1:
            return 1. / (rate_trap(-v, 154.9, .00643, 11.9) + .193 * jnp.exp(v / 33.1))
        res = _pair(rate_trap(-v, 154.9, .00643, 11.9), .193 * jnp.exp(v / 33.1))
        return res if gate == "m" else {"m": res}
    if mechanism == "K_P":
        if gate == "m":
            if component == 0:
                return sigmoid((v + 14.3) / 14.6)
            tau = jnp.where(v < -50., 1.25 + 175.03 * jnp.exp(.026 * v), 1.25 + 13. * jnp.exp(-.026 * v)) / _Q
            if component == 1:
                return tau
            return (sigmoid((v + 14.3) / 14.6), tau)
        if gate == "h":
            if component == 0:
                return sigmoid(-(v + 54.) / 11.)
            tau = (360. + (1010. + 24. * (v + 55.)) * jnp.exp(-((v + 75.) / 48.) ** 2)) / _Q
            if component == 1:
                return tau
            return (sigmoid(-(v + 54.) / 11.), tau)
        tau_m = jnp.where(v < -50., 1.25 + 175.03 * jnp.exp(.026 * v), 1.25 + 13. * jnp.exp(-.026 * v)) / _Q
        tau_h = (360. + (1010. + 24. * (v + 55.)) * jnp.exp(-((v + 75.) / 48.) ** 2)) / _Q
        return {"m": (sigmoid((v + 14.3) / 14.6), tau_m),
                "h": (sigmoid(-(v + 54.) / 11.), tau_h)}
    if mechanism == "K_T":
        if gate == "m":
            if component == 0:
                return sigmoid((v + 47.) / 29.)
            tau = (.34 + .92 * jnp.exp(-((v + 71.) / 59.) ** 2)) / _Q
            if component == 1:
                return tau
            return (sigmoid((v + 47.) / 29.), tau)
        if gate == "h":
            if component == 0:
                return sigmoid(-(v + 66.) / 10.)
            tau = (8. + 49. * jnp.exp(-((v + 73.) / 23.) ** 2)) / _Q
            if component == 1:
                return tau
            return (sigmoid(-(v + 66.) / 10.), tau)
        return {"m": (sigmoid((v + 47.) / 29.), (.34 + .92 * jnp.exp(-((v + 71.) / 59.) ** 2)) / _Q),
                "h": (sigmoid(-(v + 66.) / 10.), (8. + 49. * jnp.exp(-((v + 73.) / 23.) ** 2)) / _Q)}
    if mechanism == "SK":
        if component == 1:
            return jnp.ones_like(ca)
        adjusted = jnp.where(ca < 1e-7, ca + 1e-7, ca)
        inf = 1. / (1. + (.00043 / adjusted) ** 4.8)
        if component == 0:
            return inf
        res = (inf, jnp.ones_like(ca))
        return res if gate == "z" else {"z": res}
    if mechanism not in ("Kv3_1", "Im", "Ca_HVA", "Ca_LVA"):
        raise ValueError(f"Unknown L2 mechanism: {mechanism!r}.")
    return pv_rates(mechanism, v, ca, gate=gate, component=component)


class _L2Channel(_PVChannel):
    def _rates(self, voltage, ions):
        ca = ions[1].Ci.to_decimal(u.mM) if self.mechanism == "SK" else 1e-4
        return l2_rates(self.mechanism, voltage.to_decimal(u.mV), ca)

    def _rate_for_gate(self, gate, voltage, ions, component=None):
        ca = ions[1].Ci.to_decimal(u.mM) if self.mechanism == "SK" else 1e-4
        return l2_rates(self.mechanism, voltage.to_decimal(u.mV), ca, gate=gate, component=component)

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
