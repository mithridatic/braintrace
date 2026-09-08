"""Assemble the published HL5BN1 reference cell with explicit source parameters."""

import braincell
import brainunit as u
import numpy as np
from braincell.filter import BranchInFilter, RootLocation
from braincell.mech import StateProbe, MechanismProbe

from .h01_construction import H01Cell
from .h01_pv_morphology import make_pv_morphology
from .h01_ei_cell import _paint_profile
from .h01_ei_profiles import get_ei_profile



def make_pv_cell(reference, current_na=.19, *, max_cv_length_um=10., active=True, mode="candidate",
                 cv_policy=None):
    """Build the selected PV candidate on the published reference anatomy.

    Parameters
    ----------
    reference : dict
        Independent NEURON geometry export accepted by ``make_pv_morphology``.
    current_na : float, optional
        Somatic current from 270 to 1270 ms, default 0.19 nA.
    max_cv_length_um : float, optional
        Maximum electrical compartment length, default 10 micrometres.
    mode : str, optional
        Selected candidate (default), or source for published reproduction.
    active : bool, optional
        Include all source active channels. False keeps passive leak only.
    cv_policy : braincell.CVPolicy, optional
        Explicit electrical mesh policy. When given it replaces the maximum
        compartment length rule, for example to copy NEURON segment counts.

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
    policy = braincell.MaxCVLen(max_cv_length_um*u.um) if cv_policy is None else cv_policy
    cell = H01Cell(morph, cv_policy=policy, V_init=-80.*u.mV, solver="staggered")
    profile = get_ei_profile("I", mode=mode)
    regions = {family: BranchInFilter("type", (kind,)) for family, kind in
               (("soma", "soma"), ("axon", "axon"), ("dend", "basal_dendrite"), ("apic", "apical_dendrite"))}
    _paint_profile(cell, profile, regions, active=active)
    if active:
        cell.place(RootLocation(.5), MechanismProbe(mechanism="calcium", field="Ci", name="calcium"))
        cell.place(RootLocation(.5), MechanismProbe(mechanism="pv_SK", field="z", name="sk_gate"))
    cell.place(RootLocation(.5), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(.5), braincell.CurrentClamp(delay=270.*u.ms, durations=1000.*u.ms,
                                                       amplitudes=current_na*u.nA))
    return cell
