"""Selected E candidate on pinned Allen geometry for transfer comparisons."""
import braincell
import brainunit as u
import numpy as np
from braincell.filter import BranchInFilter, RootLocation
from braincell.mech import StateProbe, MechanismProbe
from .h01_pv_morphology import make_pv_morphology
from .h01_ei_profiles import get_ei_profile
from .h01_ei_cell import _paint_profile


def make_l2_cell(reference, *, mode="candidate", current_na=0., delay_ms=2.,
                 duration_ms=3., max_cv_length_um=10.):
    """Build the E candidate on donor geometry with an explicit current pulse.

    Parameters
    ----------
    reference : dict
        Pinned NEURON cable geometry export, including source axon replacement.
    mode : str, optional
        Candidate defaults or original source settings.
    current_na, delay_ms, duration_ms : float, optional
        Somatic pulse amplitude in nA and timing in milliseconds.
    max_cv_length_um : float, optional
        Maximum electrical compartment length in micrometres.

    Returns
    -------
    braincell.Cell
        Uninitialized donor reference cell, not measured H01 anatomy.
    """
    if (not np.isfinite([current_na, delay_ms, duration_ms, max_cv_length_um]).all()
            or min(duration_ms, max_cv_length_um) <= 0 or delay_ms < 0):
        raise ValueError("Finite current and positive finite durations and CV length required.")
    profile = get_ei_profile("E", mode=mode)
    cell = braincell.Cell(make_pv_morphology(reference),
        cv_policy=braincell.MaxCVLen(max_cv_length_um*u.um),
        V_init=profile.initial_mv*u.mV, solver="staggered")
    regions = {family: BranchInFilter("type", (kind,)) for family, kind in
        (("soma", "soma"), ("axon", "axon"), ("dend", "basal_dendrite"), ("apic", "apical_dendrite"))}
    _paint_profile(cell, profile, regions)
    cell.place(RootLocation(.5), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(.5), MechanismProbe(mechanism="calcium", field="Ci", name="calcium"))
    cell.place(RootLocation(.5), braincell.CurrentClamp(delay=delay_ms*u.ms,
        durations=duration_ms*u.ms, amplitudes=current_na*u.nA))
    return cell
