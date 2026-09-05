"""Transfer the published PV template's cable geometry to BrainCell."""

import braincell
import brainunit as u
import numpy as np


def make_pv_morphology(reference):
    """Build branches from an independent NEURON geometry export.

    Parameters
    ----------
    reference : dict
        Export containing ``sections`` with arc lengths and diameters in um,
        parent names, normalized attachment points, and child orientations.

    Returns
    -------
    braincell.Morphology
        Published reference geometry. Source names such as ``soma[0]`` become
        ``soma_0``. The source template replaces its axon.

    Notes
    -----
    This is not H01 anatomy. Electrical cable geometry is retained without
    inventing spatial coordinates for the template's synthetic axon.
    """
    sections = reference["sections"]
    names = [s["name"] for s in sections]
    roots = [s for s in sections if s["parent"] is None]
    if len(names) != len(set(names)) or len(roots) != 1:
        raise ValueError("Geometry requires unique section names and exactly one root.")
    if any(s["parent"] not in names for s in sections if s["parent"] is not None):
        raise ValueError("Geometry has a missing parent.")
    branches = {s["name"]: _branch(s) for s in sections}
    public_names = {name: name.replace("[", "_").replace("]", "") for name in names}
    if len(set(public_names.values())) != len(names):
        raise ValueError("Source names do not map to unique branch names.")
    root = roots[0]
    morph = braincell.Morphology(root_name=public_names[root["name"]], root_branch=branches[root["name"]])
    attached = {root["name"]}
    pending = [s for s in sections if s is not root]
    # This traverses static geometry; it does not advance a model.
    while pending:
        ready = [s for s in pending if s["parent"] in attached]
        if not ready:
            raise ValueError("Geometry contains a disconnected cycle.")
        for section in ready:
            morph.attach(parent=public_names[section["parent"]], child_name=public_names[section["name"]],
                         child_branch=branches[section["name"]],
                         parent_x=section["parent_x"], child_x=section["child_x"])
            attached.add(section["name"])
        pending = [s for s in pending if s["name"] not in attached]
    return morph


def _branch(section):
    family = section["name"].split("[", 1)[0]
    types = {"soma": "soma", "dend": "basal_dendrite", "apic": "apical_dendrite", "axon": "axon"}
    if family not in types:
        raise ValueError(f"Unsupported source section type: {family}.")
    arcs = np.asarray(section["arc_um"], dtype=float)
    diameters = np.asarray(section["diameter3d_um"], dtype=float)
    if arcs.size == 0 and diameters.size == 0:
        arcs = np.array([0., section["length_um"]])
        diameters = np.repeat(section["diameter_um"], 2)
    if (arcs.ndim != 1 or arcs.size < 2 or diameters.shape != arcs.shape
            or not np.isfinite(arcs).all() or not np.isfinite(diameters).all()
            or np.any(np.diff(arcs) < 0) or np.any(diameters <= 0)
            or arcs[0] != 0 or not np.isclose(arcs[-1], section["length_um"], rtol=1e-10)):
        raise ValueError("Invalid source arc lengths or diameters.")
    return braincell.Branch(lengths=np.diff(arcs)*u.um,
                            radii_proximal=diameters[:-1]/2*u.um,
                            radii_distal=diameters[1:]/2*u.um,
                            type=types[family])
