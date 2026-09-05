"""Assemble the published HL5BN1 reference cell with explicit source parameters."""

import braincell
import brainunit as u
import numpy as np
from braincell.filter import AllRegion, BranchInFilter, RootLocation
from braincell.mech import Channel, Ion, StateProbe, MechanismProbe

from .h01_pv_morphology import make_pv_morphology
from . import h01_pv_channels, h01_pv_calcium  # Register source mechanisms.


# S/cm2, copied from the pinned biophys_HL5BN1.hoc parameter table.
SOMA_CONDUCTANCES = {
    "NaTg": .4916517466528855, "Nap": .00018896712739398385,
    "K_P": .0213706236672448, "K_T": .0021990309048621,
    "Kv3_1": .428848473819995, "Im": .0002662914244435036,
    "SK": 3.566097771013125e-05, "Ca_HVA": .0001056598397436326,
    "Ca_LVA": .0265442718245302,
}
AXON_CONDUCTANCES = {
    "NaTg": .6332420031983784, "Nap": .008718100124522113,
    "K_P": .613436651152536, "K_T": .04420481156230963,
    "Kv3_1": 1.5655472280036258, "Im": .04335940360677642,
    "SK": .39879086425875565, "Ca_HVA": .0009562001984392927,
    "Ca_LVA": .00034438125387452307,
}


def make_pv_cell(reference, current_na=.19, *, max_cv_length_um=10., active=True):
    """Build the published cell for transfer validation.

    Parameters
    ----------
    reference : dict
        Independent NEURON geometry export accepted by ``make_pv_morphology``.
    current_na : float, optional
        Somatic current from 270 to 1270 ms, default 0.19 nA.
    max_cv_length_um : float, optional
        Maximum electrical compartment length, default 10 micrometres.
    active : bool, optional
        Include all source active channels. False keeps passive leak only.

    Returns
    -------
    braincell.Cell
        Uninitialized cell with a somatic voltage probe, using staggered steps.
        Active cells also record soma calcium and SK activation.

    Notes
    -----
    This is the borrowed ModelDB reference cell, not an H01 inhibitory neuron.
    Geometry includes the original template's inferred axon replacement.
    Passing numerical transfer checks does not establish human waveform validity.
    """
    if not np.isfinite([current_na, max_cv_length_um]).all() or max_cv_length_um <= 0:
        raise ValueError("Finite current and positive finite compartment length required.")
    morph = make_pv_morphology(reference)
    cell = braincell.Cell(morph, cv_policy=braincell.MaxCVLen(max_cv_length_um*u.um),
                          V_init=-80.*u.mV, solver="staggered")
    cell.paint(AllRegion(), braincell.CableProperty(membrane_capacitance=2.*u.uF/u.cm**2,
               axial_resistivity=100.*u.ohm*u.cm, resting_potential=-96.97510324827309*u.mV))
    cell.paint(AllRegion(), Channel("IL", name="leak", g_max=.00022279797042703468*1000*u.mS/u.cm**2,
                                   E=-96.97510324827309*u.mV))
    if active:
        cell.paint(AllRegion(), Channel("H01PV_Ih", name="ih", g_max=7.50253747155162e-05*1000*u.mS/u.cm**2))
        for family, conductances, decay in (("soma", SOMA_CONDUCTANCES, 206.4043474695734),
                                           ("axon", AXON_CONDUCTANCES, 370.6410785258457)):
            region = BranchInFilter("type", (family,))
            cell.paint(region, Ion("SodiumFixed", name="sodium", E=50.*u.mV))
            cell.paint(region, Ion("PotassiumFixed", name="potassium", E=-85.*u.mV))
            cell.paint(region, Ion("H01PV_Calcium", name="calcium", decay=decay*u.ms))
            for mechanism, density in conductances.items():
                cell.paint(region, Channel(f"H01PV_{mechanism}", name=f"pv_{mechanism}", g_max=density*1000*u.mS/u.cm**2))
        cell.place(RootLocation(.5), MechanismProbe(mechanism="calcium", field="Ci", name="calcium"))
        cell.place(RootLocation(.5), MechanismProbe(mechanism="pv_SK", field="z", name="sk_gate"))
    cell.place(RootLocation(.5), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(.5), braincell.CurrentClamp(delay=270.*u.ms, durations=1000.*u.ms,
                                                       amplitudes=current_na*u.nA))
    return cell
