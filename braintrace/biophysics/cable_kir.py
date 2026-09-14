"""Pinned BRAINCELL Kir4.1 current on a dynamic BrainCell potassium cable."""

import brainunit as u
from braincell.channel import K_Leak
from braincell.mech import register_channel

from .astrocyte import kir41_current_ma_cm2


@register_channel('BraincellKir41')
class BraincellKir41(K_Leak):
    """Apply the source Kir law with BrainCell's inward-positive convention.

    Parameters
    ----------
    size : int or tuple
        Runtime point shape supplied by BrainCell.
    g_max : quantity, optional
        Fixed paper conductance density, default .4 mS/cm^2. This differs from
        the unmodified MOD file's default and is not a fitted parameter.
    name : str or None, optional
        Channel instance name.

    Notes
    -----
    K pools and reversal must be supplied by EnvironmentPotassium for coupled
    simulations. The source's NormK and voltage shifts are retained exactly.
    See BRAINCELL-LICENSE.txt for source attribution.
    """

    def __init__(self, size, g_max=.4*u.mS/u.cm**2, name=None):
        super().__init__(size, g_max=g_max, name=name)

    def current(self, V, K):
        """Return potassium current density into the cable.

        Parameters
        ----------
        V : quantity
            Membrane voltage.
        K : IonInfo
            Local potassium concentrations and reversal.

        Returns
        -------
        quantity
            Inward-positive mA/cm^2, with a voltage-dependent rectification.
        """
        return -kir41_current_ma_cm2(V.to_decimal(u.mV), K.E.to_decimal(u.mV),
            K.Co.to_decimal(u.mM), conductance_ms_cm2=self.g_max.to_decimal(u.mS/u.cm**2))*u.mA/u.cm**2
