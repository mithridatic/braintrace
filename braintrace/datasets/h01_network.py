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

CONTROLS = ("ei", "e_only", "i_only", "disconnected")
_ENABLED_SIGNS = {"ei": (1, -1), "e_only": (1,), "i_only": (-1,), "disconnected": ()}


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


def _resolve_control(control, disconnected):
    if control not in CONTROLS:
        raise ValueError("Unknown projection control; expected one of "+", ".join(CONTROLS)+".")
    if disconnected and control != "ei" and control != "disconnected":
        raise ValueError("disconnected=True conflicts with control="+control+".")
    return "disconnected" if disconnected else control


def _component_rows(components):
    if components is None:
        return None
    rows = components.get("cells") if isinstance(components, dict) and "cells" in components else components
    if isinstance(rows, dict):
        return {str(k): v for k, v in rows.items()}
    return {str(row["cell_id"]): row for row in rows}


def _isolated_record(identity, rows):
    row = None if rows is None else rows.get(identity)
    if row is None:
        raise ValueError(f"Isolated cell {identity} has no component inventory row.")
    if "selected_component" in row:
        selected = row["selected_component"]
        if not isinstance(selected, dict):
            raise ValueError("Explicit component selection must be an evidence record.")
        component, count = selected.get("component"), selected.get("nodes")
        digest, evidence = selected.get("source_sha256"), selected.get("evidence")
        if (type(component) is not int or component < 0 or type(count) is not int
                or not 0 < count <= row["nodes"] or selected.get("has_soma") is not True
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Explicit component selection requires valid source and soma evidence.")
        fraction = count / row["nodes"]
        return dict(component=component, component_nodes=count, cell_nodes=int(row["nodes"]),
                    fraction=fraction, omitted_fraction=1-fraction, components=int(row["components"]),
                    source_sha256=digest, selection_evidence=evidence,
                    summary=f"isolated, explicitly selected component {component} of "
                            f"{count} nodes, fraction {fraction:.3f}")
    if not row.get("largest_has_soma"):
        raise ValueError(f"Isolated cell {identity}: largest component lacks a soma; nothing substituted.")
    component_nodes = int(round(row["largest_share"]*row["nodes"]))
    fraction = float(row["largest_share"])
    return dict(component=int(row["largest_component"]), component_nodes=component_nodes,
                cell_nodes=int(row["nodes"]), fraction=fraction, components=int(row["components"]),
                summary=f"isolated, largest component {int(row['largest_component'])} of "
                        f"{component_nodes} nodes, fraction {fraction:.3f}")


def _cell_order(nodes, incident, include_isolated, components):
    isolated = {}
    if include_isolated:
        rows = _component_rows(components)
        for identity in nodes:
            if identity not in incident:
                isolated[identity] = _isolated_record(identity, rows)
    order = list(incident)+sorted(isolated, key=lambda k: (isolated[k]["component_nodes"], int(k)))
    return order, isolated


def _select_cells(order, incident, cells):
    if cells is None:
        return list(order)
    if cells < len(incident) or cells > len(order):
        raise ValueError(f"cells must lie between {len(incident)} incident cells and {len(order)} candidates.")
    return list(order[:cells])


def plan_h01_cells(topology, *, include_isolated=False, cells=None, components=None, control="ei"):
    """Resolve the build order and built prefix without loading any morphology.

    Parameters
    ----------
    topology : dict
        Output of ``prepare_connectivity``.
    include_isolated, cells, components : optional
        As for :func:`make_h01_network`.
    control : str, optional
        Accepted for keyword compatibility with :func:`make_h01_network`; it
        does not affect which cells are built.

    Returns
    -------
    dict
        ``incident_cell_ids``, ``cell_order``, ``simulated_cell_ids`` (the
        built prefix) and ``isolated_cells`` (records for the built isolated
        cells), computed by the same rules the builder applies.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01_network import plan_h01_cells
        >>> topology = {"nodes": [{"cell_id": "7", "dale_sign": 1}], "contacts": []}
        >>> plan_h01_cells(topology, include_isolated=True, components={"7": dict(
        ...     components=1, nodes=10, largest_component=0, largest_share=1., largest_has_soma=True)})["cell_order"]
        ['7']
    """
    del control
    nodes = [n["cell_id"] for n in topology["nodes"]]
    contacts = [e for e in topology["contacts"] if e["construction_ready"]]
    incident = sorted({e[s+"_cell"] for e in contacts for s in ("pre", "post")}, key=int)
    order, isolated = _cell_order(nodes, incident, include_isolated, components)
    identities = _select_cells(order, incident, cells)
    return dict(incident_cell_ids=incident, cell_order=order, simulated_cell_ids=identities,
                isolated_cells={k: v for k, v in isolated.items() if k in identities})


def _check_contact(edge, nodes):
    sign = nodes[edge["pre_cell"]]["dale_sign"]
    if (edge["dale_sign"] != sign or edge["type"] != (2 if sign == 1 else 1)
            or edge.get("endpoints_verified") is not True):
        raise ValueError("Contact verification or Dale sign is inconsistent.")
    for side in ("pre", "post"):
        identity, site = edge[side+"_cell"], edge[side+"_placement"]
        if (edge[side+"_proofread_label"] != identity or edge[side+"_nearest_offset"] != [0, 0, 0]
                or not np.allclose(site["position_um"], np.asarray(edge[side+"_voxel"])*[.008, .008, .033], rtol=0., atol=1e-9)):
            raise ValueError("Contact endpoint evidence changed.")


def _load_contact_cells(contacts, nodes, topology, archive, emit):
    imported, locations, source_sites, seen = {}, {}, {}, set()
    for edge in contacts:
        annotation = edge["annotation_id"]
        if annotation in seen:
            raise ValueError("Duplicate contact annotation.")
        seen.add(annotation)
        _check_contact(edge, nodes)
        for side in ("pre", "post"):
            identity, site = edge[side+"_cell"], edge[side+"_placement"]
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
    return imported, locations, source_sites


def _place_contact(cells, locations, edge, weights, delay_ms, enabled):
    name = "syn_"+edge["annotation_id"]
    excitatory = edge["dale_sign"] == 1
    reversal, tau = (0., 2.) if excitatory else (-80., 5.)
    weight = weights[0] if excitatory else weights[1]
    cell = cells[edge["post_cell"]]
    location = locations[edge["annotation_id"], "post"]
    cell.place(location, Synapse("ExpSyn", name=name, e=reversal*u.mV, tau=tau*u.ms, weight=1.*u.uS))
    cell.place(location, MechanismProbe(mechanism=name, field="g", name=name+"_g"))
    cell.place(location, StateProbe(field="v", name=name+"_voltage"))
    return dict(annotation_id=edge["annotation_id"], pre_cell=edge["pre_cell"],
                post_cell=edge["post_cell"], dale_sign=edge["dale_sign"], weight_us=weight,
                signed_weight_us=edge["dale_sign"]*weight, reversal_mv=reversal, tau_ms=tau,
                delay_ms=delay_ms, enabled=enabled)


def _register_cell(network, identity, cell, record, imported, source_sites, current, emit):
    emit(f"Discretizing cell {identity} and selecting its output site")
    site = AtLocation(*source_sites[identity]) if identity in source_sites else imported.anatomy().soma_location()
    points = site.evaluate(cell.morpho).points
    branch_id, x = points[0]
    bounds = cell.cv_policy.resolve_cv_bounds(cell.morpho)[branch_id]
    midpoints = [(lo + hi) / 2.0 for lo, hi in bounds]
    best_idx = min(range(len(midpoints)), key=lambda i: (abs(midpoints[i] - x), i))
    cell.place(AtLocation(branch_id, midpoints[best_idx]), StateProbe(field="v", name="output_voltage"))
    record["output_site"] = restrict_spike_output(cell, site)
    record["input_na"] = current
    network.add_population("cell_"+identity, cell)
    record["n_compartments"] = cell.n_cv
    emit(f"Registered cell {identity}: {cell.n_cv} compartments")


def make_h01_network(topology, archive, annotations, *, disconnected=False, control="ei",
                     include_isolated=False, cells=None, components=None,
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
        Deprecated alias of ``control="disconnected"``: remove projections but
        retain cells, receptor placements, and inputs. Combining it with any
        other ``control`` raises ``ValueError``.
    control : str, optional
        ``"ei"`` keeps every projection; ``"e_only"`` keeps projections whose
        presynaptic Dale sign is +1; ``"i_only"`` those with -1;
        ``"disconnected"`` keeps none. Only projections change.
    include_isolated : bool, optional
        Also build every topology node without a constructible contact from its
        explicitly selected component, or its largest component when no override
        is supplied. The selection must carry a soma per ``components``. Isolated
        cells receive no receptors or projections; their output site is the
        soma. Fragments are never joined.
    cells : int, optional
        Build only the first ``cells`` entries of the evidence ``cell_order``
        (incident cells first, then isolated cells by ascending selected-component
        node count). Must be between the incident count and the candidate count.
    components : dict or list, optional
        Parsed ``h01-population-components.json`` (the object, its ``cells``
        list, or a ``cell_id -> row`` mapping). Required with ``include_isolated``.
        An optional ``selected_component`` record supplies ``component``, ``nodes``,
        ``has_soma``, ``source_sha256`` and an ``evidence`` reference. The loaded
        source must match its component, hash and node count before construction.
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
        Uninitialized network of built cells, named ``cell_<source ID>``.
    evidence : dict
        Actual cell and contact settings, including ``control``,
        ``enabled_contacts``, ``removed_contacts``, ``isolated_cells`` and
        ``cell_order``.

    Notes
    -----
    All 104 cells remain in the topology; isolated cells are simulated only
    with ``include_isolated``. The current engine emits at one site per source
    cell. Multiple different output sites for the same cell are rejected
    instead of silently combined. Conductance weights remain nonnegative even
    for inhibitory contacts.
    """
    values = [excitatory_weight_us, inhibitory_weight_us, delay_ms, max_cv_length_um]
    emit = progress if progress is not None else lambda message: None
    control = _resolve_control(control, disconnected)
    if not np.isfinite(values).all() or min(values[:3]) < 0 or max_cv_length_um <= 0:
        raise ValueError("Invalid conductance, delay, or compartment length.")
    if topology["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Topology uses a different archive.")
    nodes = {n["cell_id"]: n for n in topology["nodes"]}
    if len(nodes) != len(topology["nodes"]):
        raise ValueError("Duplicate node identity.")
    contacts = [e for e in topology["contacts"] if e["construction_ready"]]
    blocked = [deepcopy(e) for e in topology["contacts"] if not e["construction_ready"]]
    if not contacts and not include_isolated:
        raise ValueError("Network requires nonempty contacts with resolved placements.")
    plan = plan_h01_cells(topology, include_isolated=include_isolated, cells=cells, components=components)
    incident, order, identities, isolated = (plan["incident_cell_ids"], plan["cell_order"],
                                             plan["simulated_cell_ids"], plan["isolated_cells"])
    currents = {} if currents_na is None else dict(currents_na)
    if set(currents)-set(identities) or not np.isfinite(list(currents.values())).all():
        raise ValueError("Inputs must be finite and refer to simulated cells.")
    for identity in identities:
        sign = cell_sign(annotations.metadata(identity).tags)
        if nodes[identity]["dale_sign"] != sign:
            raise ValueError("Node Dale sign conflicts with source tags.")
    imported, locations, source_sites = _load_contact_cells(contacts, nodes, topology, archive, emit)
    for identity, record in isolated.items():
        emit(f"Loading cell {identity}, component {record['component']} ({record['summary']})")
        imported[identity] = archive.load(identity, component=record["component"])
        if "source_sha256" in record:
            source = imported[identity]
            if (source.component_id != record["component"]
                    or source.source_sha256 != record["source_sha256"]
                    or len(source.source_rows) != record["component_nodes"]):
                raise ValueError("Explicit component selection does not match loaded source.")
    network = braincell.Network(name="h01_verified")
    cells_built, records = {}, {}
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
        cells_built[identity], records[identity] = cell, record
    weights = (excitatory_weight_us, inhibitory_weight_us)
    edge_records = [_place_contact(cells_built, locations, edge, weights, delay_ms,
                                   edge["dale_sign"] in _ENABLED_SIGNS[control]) for edge in contacts]
    for identity, cell in cells_built.items():
        _register_cell(network, identity, cell, records[identity], imported[identity], source_sites,
                       currents.get(identity, 0.), emit)
    for edge in edge_records:
        if edge["enabled"]:
            name = "syn_"+edge["annotation_id"]
            network.add_edges(name=name, pre="cell_"+edge["pre_cell"], post="cell_"+edge["post_cell"], method=pairs([(0, 0)]))
            network.add_projection(name=name, edges=name, synapse=name,
                                   weight=edge["weight_us"]*u.uS, delay=delay_ms*u.ms)
    emit(f"Construction complete: {len(cells_built)} cells, {len(network.projections)} projections")
    return network, dict(nodes=deepcopy(topology["nodes"]), simulated_cell_ids=identities, cell_order=order,
        cells=records, donors=donors, contacts=edge_records, blocked_contacts=blocked,
        control=control, disconnected=control == "disconnected",
        enabled_contacts=[e["annotation_id"] for e in edge_records if e["enabled"]],
        removed_contacts=[e["annotation_id"] for e in edge_records if not e["enabled"]],
        isolated_cells=isolated,
        qualification="Verified anatomical subset with assumed synapse dynamics and unqualified candidate cells.",
        topology_audit_sha256=topology.get("endpoint_audit_sha256"))
