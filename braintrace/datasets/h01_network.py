"""Build a BrainCell network from the verified H01 contact table."""

from copy import deepcopy

import braincell
from braincell.filter import AllRegion, AtLocation
from braincell.mech import MechanismProbe, StateProbe, Synapse
from braincell.network import pairs
import brainunit as u
import numpy as np

from .h01 import ARCHIVE_SHA256
from .h01_cell_types import donor_for_tags
from .h01_connectivity import cell_sign
from .h01_ei_cell import make_h01_ei_cell
from .h01_spike_output import restrict_spike_output


def _regions(imported):
    anatomy = imported.anatomy()
    soma = anatomy.region("soma", policy="sample_neighborhood")
    axon = (anatomy.region("axon", policy="sample_neighborhood") |
            anatomy.region("axon_initial_segment", policy="sample_neighborhood") |
            anatomy.region("myelinated_axon", policy="sample_neighborhood"))-soma
    return dict(soma=soma, axon=axon, dend=AllRegion()-soma-axon)


def _location(imported, site, limit):
    if imported.source_sha256 != site["source_swc_sha256"]:
        raise ValueError("Contact source component hash changed.")
    projection = imported.anatomy().project([site["position_um"]], max_distance_um=limit)[0]
    if projection.status != "projected":
        raise ValueError("Contact no longer has an unambiguous cable location.")
    actual = projection.location.evaluate(imported.morphology).points[0]
    if not np.allclose(actual, site["cable_location"], rtol=0., atol=1e-9):
        raise ValueError("Contact cable location changed.")
    return projection.location


def make_h01_network(topology, archive, annotations, *, disconnected=False,
                     excitatory_weight_us=.01, inhibitory_weight_us=.02,
                     delay_ms=.5, max_cv_length_um=10., currents_na=None, progress=None,
                     solver="h01_staggered_scan"):
    """Construct the cells incident on verified and placed H01 synapses.

    Parameters
    ----------
    topology : dict
        Output of ``prepare_connectivity``. Blocked contacts remain in evidence
        and are omitted from the simulated subset.
    archive : H01Archive
        Pinned local morphology archive.
    annotations : H01Annotations
        Pinned cell-type annotations.
    disconnected : bool, optional
        Remove projections but retain cells, receptor placements, and inputs.
    excitatory_weight_us, inhibitory_weight_us : float, optional
        Nonnegative assumed conductance magnitudes. Signs select receptors.
    delay_ms : float, optional
        Nonnegative assumed transmission delay.
    max_cv_length_um : float, optional
        Positive spatial discretization setting.
    currents_na : dict, optional
        Soma pulse amplitudes keyed by source cell ID; unspecified cells get
        zero input. Pulses start at 2 ms and last 3 ms.
    progress : callable, optional
        Receives a message before each construction stage. It does not alter
        the model or numerical settings.
    solver : str, optional
        BrainCell integrator. Default is ``"h01_staggered_scan"``.

    Returns
    -------
    network : braincell.Network
        Uninitialized network of incident cells, named ``cell_<source ID>``.
    evidence : dict
        Actual cell and contact settings, including the signed adjacency.

    Notes
    -----
    All 104 cells remain in the topology, but isolated cells are not simulated.
    The current engine emits at one site per source cell. Multiple different
    output sites for the same cell are rejected instead of silently combined.
    Conductance weights remain nonnegative even for inhibitory contacts.
    """
    values = [excitatory_weight_us, inhibitory_weight_us, delay_ms, max_cv_length_um]
    emit = progress if progress is not None else lambda message: None
    if not np.isfinite(values).all() or min(values[:3]) < 0 or max_cv_length_um <= 0:
        raise ValueError("Invalid conductance, delay, or compartment length.")
    if topology["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Topology uses a different archive.")
    nodes = {n["cell_id"]: n for n in topology["nodes"]}
    if len(nodes) != len(topology["nodes"]):
        raise ValueError("Duplicate node identity.")
    contacts = [e for e in topology["contacts"] if e["construction_ready"]]
    blocked = [deepcopy(e) for e in topology["contacts"] if not e["construction_ready"]]
    if not contacts:
        raise ValueError("Network requires nonempty contacts with resolved placements.")
    identities = sorted({e[s+"_cell"] for e in contacts for s in ("pre", "post")}, key=int)
    currents = {} if currents_na is None else dict(currents_na)
    if set(currents)-set(identities) or not np.isfinite(list(currents.values())).all():
        raise ValueError("Inputs must be finite and refer to simulated cells.")
    for identity in identities:
        sign = cell_sign(annotations.metadata(identity).tags)
        if nodes[identity]["dale_sign"] != sign:
            raise ValueError("Node Dale sign conflicts with source tags.")
    imported, locations, source_sites, seen = {}, {}, {}, set()
    for edge in contacts:
        annotation = edge["annotation_id"]
        if annotation in seen:
            raise ValueError("Duplicate contact annotation.")
        seen.add(annotation)
        sign = nodes[edge["pre_cell"]]["dale_sign"]
        if (edge["dale_sign"] != sign or edge["type"] != (2 if sign == 1 else 1)
                or edge.get("endpoints_verified") is not True):
            raise ValueError("Contact verification or Dale sign is inconsistent.")
        for side in ("pre", "post"):
            identity, site = edge[side+"_cell"], edge[side+"_placement"]
            if (edge[side+"_proofread_label"] != identity or edge[side+"_nearest_offset"] != [0, 0, 0]
                    or not np.allclose(site["position_um"], np.asarray(edge[side+"_voxel"])*[.008,.008,.033], rtol=0., atol=1e-9)):
                raise ValueError("Contact endpoint evidence changed.")
            if identity not in imported:
                emit(f"Loading cell {identity}, component {site['component']}")
                imported[identity] = archive.load(identity, component=site["component"])
            if imported[identity].component_id != site["component"]:
                raise ValueError("Cannot join disconnected components of a cell.")
            emit(f"Checking {side} cable location for contact {annotation}")
            locations[annotation, side] = _location(imported[identity], site, topology["max_distance_um"])
            if side == "pre":
                point = site["cable_location"]
                if identity in source_sites and source_sites[identity] != point:
                    raise ValueError("Multiple output sites require a contact-specific event engine.")
                source_sites[identity] = point
    network = braincell.Network(name="h01_verified")
    cells, records = {}, {}
    basis = ("Source soma/axon/AIS samples extend halfway along adjacent edges; "
             "remaining cable gets dendrite properties. Myelin is not modeled. "
             "This is the existing inferred electrical partition, not measured channel placement.")
    donors = {}
    for identity in identities:
        emit(f"Building electrical cell {identity}")
        polarity = "E" if nodes[identity]["dale_sign"] == 1 else "I"
        donors[identity] = donor_for_tags(annotations.metadata(identity).tags, polarity)
        cell, record = make_h01_ei_cell(imported[identity], annotations,
            polarity=polarity, donor=donors[identity],
            regions=_regions(imported[identity]), region_basis=basis,
            current_na=currents.get(identity, 0.), delay_ms=2., duration_ms=3.,
            max_cv_length_um=max_cv_length_um, pop_size=(1,), solver=solver)
        cells[identity], records[identity] = cell, record
    edge_records = []
    for edge in contacts:
        name = "syn_"+edge["annotation_id"]
        excitatory = edge["dale_sign"] == 1
        reversal, tau = (0., 2.) if excitatory else (-80., 5.)
        weight = excitatory_weight_us if excitatory else inhibitory_weight_us
        cell = cells[edge["post_cell"]]
        location = locations[edge["annotation_id"], "post"]
        cell.place(location, Synapse("ExpSyn", name=name, e=reversal*u.mV, tau=tau*u.ms, weight=1.*u.uS))
        cell.place(location, MechanismProbe(mechanism=name, field="g", name=name+"_g"))
        cell.place(location, StateProbe(field="v", name=name+"_voltage"))
        edge_records.append(dict(annotation_id=edge["annotation_id"], pre_cell=edge["pre_cell"],
            post_cell=edge["post_cell"], dale_sign=edge["dale_sign"], weight_us=weight,
            signed_weight_us=edge["dale_sign"]*weight, reversal_mv=reversal, tau_ms=tau,
            delay_ms=delay_ms, enabled=not disconnected))
    for identity, cell in cells.items():
        emit(f"Discretizing cell {identity} and selecting its output site")
        site = AtLocation(*source_sites[identity]) if identity in source_sites else imported[identity].anatomy().soma_location()
        record = records[identity]
        points = site.evaluate(cell.morpho).points
        branch_id, x = points[0]
        bounds = cell.cv_policy.resolve_cv_bounds(cell.morpho)[branch_id]
        midpoints = [(lo + hi) / 2.0 for lo, hi in bounds]
        best_idx = min(range(len(midpoints)), key=lambda i: (abs(midpoints[i] - x), i))
        output_midpoint = (branch_id, midpoints[best_idx])
        cell.place(AtLocation(*output_midpoint), StateProbe(field="v", name="output_voltage"))
        record["output_site"] = restrict_spike_output(cell, site)
        record["input_na"] = currents.get(identity, 0.)
        network.add_population("cell_"+identity, cell)
        record["n_compartments"] = cell.n_cv
        emit(f"Registered cell {identity}: {cell.n_cv} compartments")
    if not disconnected:
        for edge in edge_records:
            name = "syn_"+edge["annotation_id"]
            network.add_edges(name=name, pre="cell_"+edge["pre_cell"], post="cell_"+edge["post_cell"], method=pairs([(0, 0)]))
            network.add_projection(name=name, edges=name, synapse=name,
                                   weight=edge["weight_us"]*u.uS, delay=delay_ms*u.ms)
    emit(f"Construction complete: {len(cells)} cells, {len(network.projections)} projections")
    return network, dict(nodes=deepcopy(topology["nodes"]), simulated_cell_ids=identities,
        cells=records, donors=donors, contacts=edge_records, blocked_contacts=blocked, disconnected=disconnected,
        qualification="Verified anatomical subset with assumed synapse dynamics and unqualified candidate cells.",
        topology_audit_sha256=topology.get("endpoint_audit_sha256"))
