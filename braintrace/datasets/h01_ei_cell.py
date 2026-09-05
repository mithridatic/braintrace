"""Apply selected E/I dynamics to explicit electrical regions of H01 anatomy."""

from dataclasses import asdict
import numpy as np
import braincell
import brainunit as u
from braincell.mech import Channel, Ion, StateProbe
from . import h01_l2_channels, h01_pv_calcium
from .h01_ei_profiles import get_ei_profile, channel_controls
from .h01_discretization import BoundaryAlignedCV


def _validate_regions(morphology, regions, polarity):
    if not regions or set(regions)-{"soma", "axon", "dend", "apic"}:
        raise ValueError("Use explicit soma, axon, dend, or apic electrical regions.")
    intervals = {name: tuple(region.evaluate(morphology).intervals) for name, region in regions.items()}
    required = ("soma", "axon") if polarity == "I" else ("soma",)
    if any(not intervals.get(name) for name in required):
        raise ValueError("Profile requires nonempty "+" and ".join(required)+" regions.")
    by_branch = [[] for _ in morphology.branches]
    for rows in intervals.values():
        for branch, lo, hi in rows:
            if (not isinstance(branch, (int, np.integer)) or not 0 <= branch < len(by_branch)
                    or not np.isfinite([lo, hi]).all() or not 0 <= lo < hi <= 1):
                raise ValueError("Invalid electrical region interval.")
            by_branch[branch].append((lo, hi))
    for rows in by_branch:
        end = 0.
        for lo, hi in sorted(rows):
            if abs(lo-end) > 1e-9:
                raise ValueError("Electrical regions must cover each branch without overlap or gaps.")
            end = hi
        if abs(end-1.) > 1e-9:
            raise ValueError("Electrical regions must cover each branch without overlap or gaps.")
    return intervals


def _paint_profile(cell, profile, regions, *, active=True):
    prefix = "H01L2" if profile.polarity == "E" else "H01PV"
    for family, cm, leak, channels, calcium in profile.regions:
        if family not in regions:
            continue
        region = regions[family]
        cell.paint(region, braincell.CableProperty(membrane_capacitance=cm*u.uF/u.cm**2,
                   axial_resistivity=profile.axial_ohm_cm*u.ohm*u.cm,
                   resting_potential=profile.reversal_mv*u.mV))
        cell.paint(region, Channel("IL", name="leak", g_max=leak*1000*u.mS/u.cm**2,
                                   E=profile.reversal_mv*u.mV))
        if not active:
            continue
        if calcium is not None:
            cell.paint(region, Ion("SodiumFixed", name="sodium", E=(53. if profile.polarity == "E" else 50.)*u.mV))
            cell.paint(region, Ion("PotassiumFixed", name="potassium", E=(-107. if profile.polarity == "E" else -85.)*u.mV))
            cell.paint(region, Ion("H01PV_Calcium", name="calcium", decay=calcium[0]*u.ms, gamma=calcium[1]))
        for mechanism, density in channels:
            cell.paint(region, Channel(prefix+"_"+mechanism, name="pv_"+mechanism,
                       g_max=density*1000*u.mS/u.cm**2,
                       **channel_controls(profile, family, mechanism)))


def make_h01_ei_cell(imported, annotations, *, polarity, regions, region_basis,
                     mode="candidate", current_na=0., delay_ms=2., duration_ms=3.,
                     max_cv_length_um=10., solver="staggered", pop_size=()):
    """Build an H01 cell with the selected candidate for its explicit E/I role.

    Parameters
    ----------
    imported : H01Component
        Measured connected H01 cable component.
    annotations : H01Annotations
        Original source identity and tags, retained without modification.
    polarity : str
        Explicit modeled E or I role; not a claim of measured cell identity.
    regions : dict of RegionExpr
        Complete disjoint electrical partition named soma, axon, dend, apic.
        I requires a nonempty axon. H01 dendrites do not specify apical identity.
    region_basis : str
        Explanation of measured labels and inferred electrical boundaries.
    mode : str, optional
        Candidate defaults or explicit source reproduction.
    current_na, delay_ms, duration_ms : float, optional
        Somatic current pulse in nA and milliseconds.
    max_cv_length_um : float, optional
        Maximum CV length; all region boundaries are retained.
    solver : str, optional
        BrainCell solver, default staggered.
    pop_size : tuple, optional
        BrainCell population shape. Use (1,) for a distinct cell in Network.

    Returns
    -------
    cell : braincell.Cell
        Uninitialized cell with a somatic voltage probe.
    evidence : dict
        Measured identity, borrowed profile, and inferred electrical mapping.
    """
    profile = get_ei_profile(polarity, mode=mode)
    if solver == "h01_staggered_scan":
        from . import h01_dhs_scan  # Register only on explicit selection.
    if not isinstance(region_basis, str) or not region_basis.strip():
        raise ValueError("Provide the basis for the inferred electrical region map.")
    if (not np.isfinite([current_na, delay_ms, duration_ms, max_cv_length_um]).all()
            or min(duration_ms, max_cv_length_um) <= 0 or delay_ms < 0):
        raise ValueError("Pulse and CV settings must be finite with positive durations and lengths.")
    intervals = _validate_regions(imported.morphology, regions, polarity)
    metadata = annotations.metadata(imported.neuron_id)
    anatomy = imported.anatomy()
    soma = anatomy.soma_location()
    for branch, x in soma.evaluate(imported.morphology).points:
        if not any(b == branch and lo <= x <= hi for b, lo, hi in intervals["soma"]):
            raise ValueError("The measured soma probe must lie in the electrical soma region.")
    policy = braincell.MaxCVLen(max_cv_length_um*u.um)
    for region in regions.values():
        policy = BoundaryAlignedCV(policy, region)
    cell = braincell.Cell(imported.morphology, cv_policy=policy,
                          V_init=profile.initial_mv*u.mV, solver=solver, pop_size=pop_size)
    _paint_profile(cell, profile, regions)
    cell.place(soma, StateProbe(field="v", name="voltage"))
    cell.place(soma, braincell.CurrentClamp(delay=delay_ms*u.ms, durations=duration_ms*u.ms,
                                          amplitudes=current_na*u.nA))
    evidence = {"measured_anatomy": anatomy.provenance, "source_tags": list(metadata.tags),
                "borrowed_dynamics": asdict(profile), "modeled_polarity": polarity,
                "inferred_region_basis": region_basis, "electrical_intervals": intervals,
                "solver": solver, "max_cv_length_um": max_cv_length_um,
                "synaptic_connectivity": "Not supplied by this single-cell builder."}
    return cell, evidence
