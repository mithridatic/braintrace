"""Selected E candidate on pinned Allen geometry for transfer comparisons."""
import braincell
import brainunit as u
import numpy as np
from braincell.filter import BranchInFilter, RootLocation
from braincell.mech import StateProbe, MechanismProbe
from .h01_pv_morphology import make_pv_morphology
from .h01_ei_profiles import get_ei_profile
from .h01_ei_cell import _paint_profile


def _stimulus_segments(current_na, duration_ms):
    """Return equal-length finite amplitude and positive duration arrays."""
    amplitudes = np.atleast_1d(np.asarray(current_na, dtype=float))
    durations = np.atleast_1d(np.asarray(duration_ms, dtype=float))
    if (amplitudes.ndim != 1 or amplitudes.shape != durations.shape or not amplitudes.size
            or not np.isfinite(amplitudes).all() or not np.isfinite(durations).all()
            or (durations <= 0).any()):
        raise ValueError("Finite current and positive finite durations and CV length required.")
    return amplitudes, durations


def make_l2_cell(reference, *, mode="candidate", current_na=0., delay_ms=2.,
                 duration_ms=3., max_cv_length_um=10., cv_policy=None):
    """Build the E candidate on donor geometry with an explicit current protocol.

    Parameters
    ----------
    reference : dict
        Pinned NEURON cable geometry export, including source axon replacement.
    mode : str, optional
        Candidate defaults or original source settings.
    current_na, duration_ms : float or sequence of float, optional
        Somatic pulse amplitude in nA and length in milliseconds. Equal-length
        sequences describe consecutive constant segments starting at ``delay_ms``.
    delay_ms : float, optional
        Start of the first segment in milliseconds.
    max_cv_length_um : float, optional
        Maximum electrical compartment length in micrometres.
    cv_policy : braincell.CVPolicy, optional
        Explicit electrical mesh policy. When given it replaces the maximum
        compartment length rule, for example to copy NEURON segment counts.

    Returns
    -------
    braincell.Cell
        Uninitialized donor reference cell, not measured H01 anatomy.
    """
    if (not np.isfinite([delay_ms, max_cv_length_um]).all()
            or max_cv_length_um <= 0 or delay_ms < 0):
        raise ValueError("Finite current and positive finite durations and CV length required.")
    amplitudes, durations = _stimulus_segments(current_na, duration_ms)
    profile = get_ei_profile("E", mode=mode)
    policy = braincell.MaxCVLen(max_cv_length_um*u.um) if cv_policy is None else cv_policy
    cell = braincell.Cell(make_pv_morphology(reference), cv_policy=policy,
        V_init=profile.initial_mv*u.mV, solver="staggered")
    regions = {family: BranchInFilter("type", (kind,)) for family, kind in
        (("soma", "soma"), ("axon", "axon"), ("dend", "basal_dendrite"), ("apic", "apical_dendrite"))}
    _paint_profile(cell, profile, regions)
    cell.place(RootLocation(.5), StateProbe(field="v", name="voltage"))
    cell.place(RootLocation(.5), MechanismProbe(mechanism="calcium", field="Ci", name="calcium"))
    if amplitudes.size == 1:
        clamp = braincell.CurrentClamp(delay=delay_ms*u.ms, durations=float(durations[0])*u.ms,
                                       amplitudes=float(amplitudes[0])*u.nA)
    else:
        clamp = braincell.CurrentClamp(delay=delay_ms*u.ms, durations=durations*u.ms,
                                       amplitudes=amplitudes*u.nA)
    cell.place(RootLocation(.5), clamp)
    return cell
