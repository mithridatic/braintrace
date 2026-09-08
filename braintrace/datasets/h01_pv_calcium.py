"""Buffered calcium state for the released human-fitted HL5BN1 model."""

import braintools
import brainunit as u
from braincell.ion import Calcium
from braincell.ion._base import DynamicNernstIon
from braincell.mech import register_ion


@register_ion("H01PV_Calcium")
class PVCalcium(Calcium, DynamicNernstIon):
    """Track shell calcium with the source CaDynamics equation.

    Parameters
    ----------
    size : int or tuple
        BrainCell state shape.
    decay : Quantity, optional
        Calcium removal time. Default is the fitted soma value, 206.404347 ms.
        Set the fitted axon value explicitly when painting that region.
    gamma : float, optional
        Unbuffered fraction, default 0.0005 from the fitted cell.
    depth : Quantity, optional
        Shell depth, default 0.1 micrometre.
    name : str, optional
        BrainState node name.
    **channels : Channel
        Calcium-current channels owned by this ion.

    Notes
    -----
    ModelDB 267587, revision 82cdd91bc93942ba19315371330a2412e064baf5.
    The source mechanism is adapted from Destexhe calcium dynamics.
    These parameters are borrowed from a fitted cell, not measured in H01.
    """

    uses_total_current = True

    def __init__(self, size, decay=206.4043474695734*u.ms, gamma=.0005,
                 depth=.1*u.um, name=None, **channels):
        super().__init__(size, name=name, **channels)
        self.rest = 1e-4*u.mM
        self._init_dynamic_nernst_ion(
            Co=2.*u.mM, temp=u.celsius2kelvin(34.), valence=2,
            Ci_initializer=braintools.init.Constant(self.rest))
        # Mixed-ion SK can initialize under potassium before calcium initializes.
        # Provide its resting concentration now; the normal hook allocates State.
        self.Ci = braintools.init.param(self.rest, self.varshape, allow_none=False)
        self.decay = braintools.init.param(decay, self.varshape, allow_none=False)
        self.gamma = braintools.init.param(gamma, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        """Return the calcium concentration derivative.

        Parameters
        ----------
        Ci : Quantity
            Inside calcium concentration.
        V : Quantity
            Membrane voltage; not used directly by this equation.
        total_current : Quantity
            Sum of inward-positive calcium current densities.

        Returns
        -------
        Quantity
            Concentration change per unit time.
        """
        influx = self.gamma*total_current/(2*u.faraday_constant*self.depth)
        return influx+(self.rest-Ci)/self.decay
