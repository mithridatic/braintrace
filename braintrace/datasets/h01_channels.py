"""BrainCell channels adapted from the Wilbers 2023 current-clamp mechanisms.

Importing this module registers two explicitly named channels. Parameters
refer to the source's uncorrected current-clamp voltage convention; voltage
shift -10 mV accounts for its differing voltage-clamp junction correction.
"""

import brainunit as u
import jax.numpy as jnp
from braincell.channel._base import Gate, HH
from braincell.ion import Sodium, Potassium
from braincell.mech import register_channel

from .h01_wilbers import sodium_rates, potassium_rates


class _WilbersBase(HH):
    def __init__(self, size, g_max, temperature_c=34., voltage_shift_mv=-10., name=None):
        super().__init__(size=size, name=name)
        self.g_max = g_max
        self.temperature_c = temperature_c
        self.voltage_shift_mv = voltage_shift_mv

    def init_state(self, voltage, *ions, batch_size=None):
        """Allocate gates at their voltage-dependent equilibrium.

        Parameters
        ----------
        voltage : Quantity
            Initial membrane voltage.
        *ions : IonInfo
            Ion information supplied by BrainCell.
        batch_size : int, optional
            Optional leading batch dimension.
        """
        super().init_state(voltage, *ions, batch_size=batch_size)
        self.reset_state(voltage, *ions, batch_size=batch_size)

    def _rates(self, voltage):
        return self.rate_function(voltage.to_decimal(u.mV) + self.voltage_shift_mv,
                                  temperature_c=self.temperature_c)

    def f_m_inf(self, voltage, ion):
        """Return activation steady state for voltage and ion information."""
        return self._rates(voltage)[0]

    def f_h_inf(self, voltage, ion):
        """Return inactivation steady state for voltage and ion information."""
        return self._rates(voltage)[1]

    def f_m_tau(self, voltage, ion):
        """Return activation time constant in milliseconds."""
        return self._rates(voltage)[2]

    def f_h_tau(self, voltage, ion):
        """Return fast inactivation time constant in milliseconds."""
        return self._rates(voltage)[3]


@register_channel("H01Na_Wilbers2023")
class HumanSodium(_WilbersBase):
    """Human sodium current with source temperature-scaled conductance.

    Parameters
    ----------
    size : int or tuple
        BrainCell state shape.
    g_max : Quantity
        Base conductance density; the source additionally multiplies sodium
        conductance by 2.3**((temperature_c-25)/10).
    temperature_c : float, optional
        Celsius temperature, default 34.
    voltage_shift_mv : float, optional
        Shift applied to rate-function voltage, default -10 mV.
    name : str, optional
        BrainState node name.
    """

    root_type = Sodium
    rate_function = staticmethod(sodium_rates)
    gates = (Gate("m", power=3), Gate("h"))

    def current(self, voltage, ion):
        """Return inward-positive sodium current density.

        Parameters
        ----------
        voltage : Quantity
            Membrane voltage.
        ion : IonInfo
            Sodium reversal potential in ``E``.

        Returns
        -------
        Quantity
            Current density with BrainCell's inward-positive convention.
        """
        phi = 2.3 ** ((self.temperature_c - 25.) / 10)
        return self.g_max * phi * self.conductance_factor(voltage, ion) * (ion.E - voltage)


@register_channel("H01K_Wilbers2023")
class HumanPotassium(_WilbersBase):
    """Human aggregate potassium current with two inactivation timescales.

    Parameters
    ----------
    size : int or tuple
        BrainCell state shape.
    g_max : Quantity
        Total conductance density before activation/inactivation factors.
    temperature_c : float, optional
        Celsius temperature, default 34.
    voltage_shift_mv : float, optional
        Rate-function voltage shift, default -10 mV.
    name : str, optional
        BrainState node name.

    Notes
    -----
    Source fractions: 0.51 fast-inactivating, 0.34 slow-inactivating and
    0.15 non-inactivating. Slow time constants are not temperature-scaled
    in this source version. The source inactivation-rate factor is fixed at 1.
    """

    root_type = Potassium
    rate_function = staticmethod(potassium_rates)
    gates = (Gate("m", power=2), Gate("h"), Gate("h2"))

    def f_h2_inf(self, voltage, ion):
        """Return the shared slow-inactivation steady state."""
        return self.f_h_inf(voltage, ion)

    def f_h2_tau(self, voltage, ion):
        """Return slow recovery (2789.79 ms) or inactivation (450 ms)."""
        return jnp.where(self.f_h2_inf(voltage, ion) > self.h2.value, 2789.7929428108187, 450.)

    def conductance_factor(self, voltage, *ions):
        """Return the sum of the source's three potassium populations."""
        return self.m.value**2 * (.51*self.h.value + .34*self.h2.value + .15)

    def current(self, voltage, ion):
        """Return inward-positive potassium current density.

        Parameters
        ----------
        voltage : Quantity
            Membrane voltage.
        ion : IonInfo
            Potassium reversal potential in ``E``.

        Returns
        -------
        Quantity
            Current density with BrainCell's inward-positive convention.
        """
        return self.g_max * self.conductance_factor(voltage, ion) * (ion.E - voltage)
