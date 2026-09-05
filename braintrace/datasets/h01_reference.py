"""Construct the Wilbers reference compartment before transferring it to H01."""

import braincell
import brainunit as u
import numpy as np
from braincell.filter import AllRegion, RootLocation
from braincell.mech import Channel, Ion, StateProbe

from . import h01_channels  # Register the source-specific mechanisms.


def make_reference_cell(current_na=.1, *, sodium_ms_cm2=9.601446023,
                        potassium_ms_cm2=4.884943554, delay_ms=2., duration_ms=3.):
    """Build the published cylindrical reference geometry with human channels.

    Parameters
    ----------
    current_na : float, optional
        Injected current in nA.
    sodium_ms_cm2, potassium_ms_cm2 : float, optional
        Base channel conductances in mS/cm2. Defaults are the source script's
        starting guesses, not its optimized result or H01 measurements.
    delay_ms, duration_ms : float, optional
        Onset and duration of current injection in ms.

    Returns
    -------
    braincell.Cell
        Uninitialized cell with a ``voltage`` probe and staggered solver.

    Notes
    -----
    Source current-clamp convention: 34 Celsius, reversal potentials +68/-86
    mV, leak -70 mV, channel voltage shift -10 mV. Sodium additionally scales
    conductance by its temperature factor. No silent human/mouse mixing.
    This 10 by 10 micron cylinder is a reference geometry, not an H01 neuron.
    """
    values = (current_na, sodium_ms_cm2, potassium_ms_cm2, delay_ms, duration_ms)
    if not np.isfinite(values).all() or min(sodium_ms_cm2, potassium_ms_cm2, delay_ms) < 0 or duration_ms <= 0:
        raise ValueError("Finite parameters, nonnegative conductances/delay and positive duration required.")
    branch = braincell.Soma(lengths=np.array([10.])*u.um,
                           radii_proximal=np.array([5.])*u.um,
                           radii_distal=np.array([5.])*u.um)
    cell = braincell.Cell(braincell.Morphology(root_name="soma", root_branch=branch),
                         V_init=-70.0*u.mV, solver="staggered")
    cell.paint(AllRegion(), braincell.CableProperty(membrane_capacitance=1*u.uF/u.cm**2,
               axial_resistivity=200*u.ohm*u.cm, resting_potential=-70*u.mV))
    cell.paint(AllRegion(), Ion("SodiumFixed", E=68*u.mV))
    cell.paint(AllRegion(), Ion("PotassiumFixed", E=-86*u.mV))
    cell.paint(AllRegion(), Channel("IL", g_max=.1*u.mS/u.cm**2, E=-70*u.mV))
    cell.paint(AllRegion(), Channel("H01Na_Wilbers2023", g_max=sodium_ms_cm2*u.mS/u.cm**2))
    cell.paint(AllRegion(), Channel("H01K_Wilbers2023", g_max=potassium_ms_cm2*u.mS/u.cm**2))
    cell.place(RootLocation(0.), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(0.), braincell.CurrentClamp(delay=delay_ms*u.ms, durations=duration_ms*u.ms,
                                                     amplitudes=current_na*u.nA))
    return cell
