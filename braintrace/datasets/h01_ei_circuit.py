"""A two-cell H01 diagnostic circuit with measured default wiring and explicit illustrative controls."""
import numpy as np
import braincell
import brainunit as u
from braincell.filter import AtLocation
from braincell.mech import Synapse, MechanismProbe, StateProbe
from braincell.network import pairs
from .h01_ei_cell import make_h01_ei_cell
from .h01_spike_output import restrict_spike_output
from .h01_measured_contact import measured_ie_contact


def make_h01_ei_circuit(components, annotations, *, regions, region_basis,
                        currents_na=None, control="ei", connectivity="measured", excitatory_weight_us=.01,
                        inhibitory_weight_us=.02, delay_ms=.5, max_cv_length_um=10., solver="staggered",
                        pulse_delays_ms=None, pulse_durations_ms=None):
    """Build the measured I-to-E contact or explicit illustrative wiring.

    Parameters
    ----------
    components : dict
        Distinct imported H01 components keyed E and I.
    annotations : H01Annotations
        Source tags. E must be pyramidal; I must be an interneuron.
    regions, region_basis : dict
        Per-role electrical partitions and explanations for the cell builder.
    currents_na : dict, optional
        Soma pulse amplitudes. Default E=1 nA; I=1 for measured or 0 for illustrative wiring.
    pulse_delays_ms, pulse_durations_ms : dict, optional
        Exactly E/I pulse start times and durations in milliseconds.
        Default starts are 2 ms and durations are 3 ms for both cells.
    control : str, optional
        ei, e_only, i_only, or disconnected. Only projections change.
    connectivity : str, optional
        measured (default) uses annotation 8105899 on its pinned components.
        illustrative explicitly enables the prior reciprocal diagnostic.
    excitatory_weight_us, inhibitory_weight_us : float, optional
        Nonnegative contact conductances in microsiemens.
    delay_ms : float, optional
        Nonnegative transmission delay. Network rounds up to a whole step.
    max_cv_length_um : float, optional
        Maximum spatial compartment length.
    solver : str, optional
        BrainCell integrator. The default is staggered. h01_staggered_scan
        selects the experimental compiled DHS loops.

    Returns
    -------
    network : braincell.Network
        Uninitialized circuit with candidate defaults and local probes.
    evidence : dict
        Anatomy, borrowed dynamics, inferred contacts, and unresolved validity.
    """
    if set(components) != {"E", "I"} or set(regions) != {"E", "I"} or set(region_basis) != {"E", "I"}:
        raise ValueError("Supply exactly E and I components, regions, and region bases.")
    if components["E"].neuron_id == components["I"].neuron_id:
        raise ValueError("E and I must use distinct source neuron identities.")
    if control not in ("ei", "e_only", "i_only", "disconnected"):
        raise ValueError("Unknown E/I projection control.")
    starts = {"E": 2., "I": 2.} if pulse_delays_ms is None else dict(pulse_delays_ms)
    durations = {"E": 3., "I": 3.} if pulse_durations_ms is None else dict(pulse_durations_ms)
    if (set(starts) != {"E", "I"} or set(durations) != {"E", "I"}
            or not np.isfinite(list(starts.values())+list(durations.values())).all()
            or min(starts.values()) < 0 or min(durations.values()) <= 0):
        raise ValueError("Supply finite E/I pulse delays >= 0 and durations > 0.")
    if connectivity not in ("measured", "illustrative"):
        raise ValueError("Unknown connectivity basis.")
    pre_site, post_site, measured = (measured_ie_contact(components)
        if connectivity == "measured" else (None, None, None))
    values = (excitatory_weight_us, inhibitory_weight_us, delay_ms)
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("Weights and delay must be nonnegative and finite.")
    currents = {"E": 1., "I": 1. if measured else 0.} if currents_na is None else dict(currents_na)
    if set(currents) != {"E", "I"}:
        raise ValueError("Supply both E and I current amplitudes.")
    network = braincell.Network(name="h01_ei_candidates")
    evidence = {"cells": {}, "control": control, "inferred_contacts": [],
                "qualification": "Candidate diagnostic circuit; human and E/I behavior validation incomplete.",
                "connectivity_basis": ("Measured I-to-E annotation 8105899; synapse dynamics are borrowed." if measured else
                                       "Illustrative reciprocal wiring; no measured H01 partner mapping."),
                "connectivity": connectivity, "measured_contacts": []}
    for role, source_tag in (("E", "pyramidal"), ("I", "interneuron")):
        imported = components[role]
        if source_tag not in annotations.metadata(imported.neuron_id).tags:
            raise ValueError(f"{role} requires the source {source_tag} tag.")
        cell, record = make_h01_ei_cell(imported, annotations, polarity=role,
            regions=regions[role], region_basis=region_basis[role],
            current_na=currents[role], delay_ms=starts[role], duration_ms=durations[role],
            max_cv_length_um=max_cv_length_um, pop_size=(1,), solver=solver)
        soma = imported.anatomy().soma_location()
        incoming = "inh" if role == "E" else "exc"
        reversal, tau = (-80., 5.) if role == "E" else (0., 2.)
        receptor_site = post_site if measured and role == "E" else soma
        output_site = pre_site if measured and role == "I" else soma
        cell.place(receptor_site, Synapse("ExpSyn", name=incoming, e=reversal*u.mV, tau=tau*u.ms, weight=1.*u.uS))
        cell.place(receptor_site, MechanismProbe(mechanism=incoming, field="g", name="synaptic_conductance"))
        cell.place(receptor_site, StateProbe(field="v", name="incoming_voltage"))
        site = restrict_spike_output(cell, output_site)
        cell.place(AtLocation(*site["output_midpoint"]), StateProbe(field="v", name="output_voltage"))
        record["output_site"] = site
        record["synaptic_connectivity"] = "See circuit measured_contacts and inferred_contacts."
        record["incoming_synapse"] = dict(name=incoming, reversal_mv=reversal, tau_ms=tau)
        record["current_na"] = currents[role]
        record["pulse_delay_ms"] = starts[role]
        record["pulse_duration_ms"] = durations[role]
        evidence["cells"][role] = record
        network.add_population(role, cell)
    for pre, post, synapse, weight, selected in (
        ("E", "I", "exc", excitatory_weight_us, control in ("ei", "e_only")),
        ("I", "E", "inh", inhibitory_weight_us, control in ("ei", "i_only"))):
        if measured and pre == "E":
            continue
        contact = dict(pre=pre, post=post, pre_index=0, post_index=0,
                       synapse=synapse, weight_us=weight, delay_ms=delay_ms, enabled=selected)
        if measured:
            contact["anatomy"] = measured
            contact["dynamics_basis"] = "Borrowed conductance, delay, reversal potential, and decay time."
            evidence["measured_contacts"].append(contact)
        else:
            evidence["inferred_contacts"].append(contact)
        if selected:
            name = pre+"_to_"+post
            network.add_edges(name=name, pre=pre, post=post, method=pairs([(0, 0)]))
            network.add_projection(name=name, edges=name, synapse=synapse,
                                   weight=weight*u.uS, delay=delay_ms*u.ms)
    return network, evidence
