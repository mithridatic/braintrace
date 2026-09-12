"""Extracellular GABA drives a fixed extrasynaptic conductance on real cables.

Occupancy follows pinned BRAINCELL GabaUpdateLinear.mod; see bundled
BRAINCELL-LICENSE.txt. Chloride reversal is a declared fixed reservoir.
"""

import brainunit as u
from braincell.channel import IL
from braincell.mech import register_channel

from .gaba import tonic_occupancy


@register_channel('EnvironmentGABA')
class EnvironmentGABA(IL):
    """Source Hill occupancy with BrainCell's inward-positive current convention.

    Parameters
    ----------
    size : int or tuple
        Runtime point shape, supplied by BrainCell.
    g_max, E : quantity, optional
        Fixed maximum conductance density and chloride reversal.
    ec50_mm, hill : float, optional
        Fixed source occupancy parameters.
    name : str or None, optional
        Instance name.

    Notes
    -----
    Bind to a CablePotassiumBinding before stepping. Unbound construction has
    zero ambient GABA. Receptor binding does not remove transmitter molecules.
    """

    def __init__(self, size, g_max=.1*u.mS/u.cm**2, E=-70.*u.mV,
                 ec50_mm=1e-4, hill=1., name=None):
        super().__init__(size, g_max=g_max, E=E, name=name)
        self.ec50_mm, self.hill = ec50_mm, hill
        self._chemical_binding = None

    def bind(self, binding):
        """Use the same CV-to-extracellular map as the cable potassium pool.

        Parameters
        ----------
        binding : CablePotassiumBinding
            Geometry and shared environment for this channel's cell.
        """
        if self._chemical_binding is not None:
            raise ValueError('GABA channel is already bound')
        # Stateless channels may omit a leading singleton population axis;
        # runtime currents broadcast it against the cell's voltage.
        if self.varshape[-1] != binding.ion.varshape[-1] or any(n != 1 for n in self.varshape[:-1]):
            raise ValueError('GABA and chemical point shapes differ')
        self._chemical_binding = binding

    def current(self, V):
        """Return inward-positive receptor current density.

        Parameters
        ----------
        V : quantity
            Runtime point voltage in mV.

        Returns
        -------
        quantity
            Current density determined by local GABA and fixed conductance.
        """
        binding = self._chemical_binding
        if binding is None:
            concentration = 0.
        else:
            env = binding.environment
            indices = env.membranes.indices[binding.indices]
            concentration = env.gaba.value[indices][binding.point_cv].reshape(self.varshape)
        activation = tonic_occupancy(concentration, ec50_mm=self.ec50_mm, hill=self.hill)
        return self.g_max*activation*(self.E-V)
