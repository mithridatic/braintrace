"""Filter the H01 population topology and components inventory down to the kept cells.

Drop-in replacements for ``docs/evidence/h01-verified-network.json`` and
``docs/evidence/h01-population-components.json`` restricted to the cells listed under
``kept`` in the keep/drop decision (spec ``docs/specs/2026-09-16-h01-keep-drop.md``).

Fields matched to what ``braintrace.datasets.h01_network`` and
``examples/h01_verified_network.py`` read, without importing them (they pull in
braincell/jax): topology ``nodes[*].cell_id`` / ``index`` / ``polarity`` / ``dale_sign``,
``contacts[*].pre_cell`` / ``post_cell`` / ``pre_index`` / ``post_index`` /
``construction_ready``, ``archive_sha256``, ``max_distance_um``, ``counts``; components
``cells[*].cell_id`` (with ``components``, ``nodes``, ``largest_component``,
``largest_share``, ``largest_has_soma``) and ``contacts[*].cell_id``. Kept nodes are
re-indexed 0..n-1 in their original order and contact indexes are remapped, because
the example builds a ``len(nodes)`` square contact matrix from ``pre_index``/``post_index``.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _kept_contacts(contacts, kept):
    """Split contacts into (kept, dropped-endpoint pairs); a contact survives only with both ends kept."""
    survivors, dropped = [], []
    for contact in contacts:
        ends = (str(contact["pre_cell"]), str(contact["post_cell"]))
        (survivors if ends[0] in kept and ends[1] in kept else dropped).append((contact, ends))
    return [c for c, _ in survivors], [ends for _, ends in dropped]


def _recount(nodes, contacts, counts):
    """Recompute the topology's ``counts`` block for the filtered nodes and contacts."""
    new = dict(counts)
    new.update(nodes=len(nodes), excitatory=sum(1 for n in nodes if n.get("polarity") == "E"),
               inhibitory=sum(1 for n in nodes if n.get("polarity") == "I"),
               verified_contacts=len(contacts),
               construction_ready=sum(1 for c in contacts if c.get("construction_ready")))
    return new


def filter_network(topology, components, kept_ids):
    """Return (filtered_topology, filtered_components, report) restricted to ``kept_ids``.

    Every top-level key of both inputs is preserved; ``nodes``, ``contacts`` and
    ``counts`` (topology) and ``cells``, ``contacts`` and ``summary['cells']``
    (components) are filtered. Raises ``ValueError`` for a kept id absent from the
    topology nodes or the components inventory.
    """
    kept = {str(k) for k in kept_ids}
    node_ids = {str(n["cell_id"]) for n in topology["nodes"]}
    row_ids = {str(r["cell_id"]) for r in components["cells"]}
    unknown = sorted(kept-node_ids | kept-row_ids, key=int)
    if unknown:
        raise ValueError("kept ids missing from topology or components: "+", ".join(unknown))
    nodes = [deepcopy(n) for n in topology["nodes"] if str(n["cell_id"]) in kept]
    index = {str(n["cell_id"]): i for i, n in enumerate(nodes)}
    for i, node in enumerate(nodes):
        node["index"] = i
    contacts, dropped = _kept_contacts(topology["contacts"], kept)
    contacts = [deepcopy(c) for c in contacts]
    for contact in contacts:
        contact["pre_index"], contact["post_index"] = index[str(contact["pre_cell"])], index[str(contact["post_cell"])]
    new_topology = dict(topology, nodes=nodes, contacts=contacts,
                        counts=_recount(nodes, contacts, topology.get("counts", {})))
    rows = [deepcopy(r) for r in components["cells"] if str(r["cell_id"]) in kept]
    comp_contacts = [deepcopy(c) for c in components.get("contacts", []) if str(c["cell_id"]) in kept]
    new_components = dict(components, cells=rows, contacts=comp_contacts,
                          summary=dict(components.get("summary", {}), cells=len(rows)))
    report = dict(kept=len(kept), nodes_before=len(topology["nodes"]), nodes_after=len(nodes),
                  contacts_before=len(topology["contacts"]), contacts_after=len(contacts),
                  dropped_contacts=dropped, components_before=len(components["cells"]), components_after=len(rows))
    return new_topology, new_components, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=Path("docs/evidence/h01-verified-network.json"))
    parser.add_argument("--components", type=Path, default=Path("docs/evidence/h01-population-components.json"))
    parser.add_argument("--decision", type=Path, default=Path("docs/evidence/h01-keep-drop/decision.json"))
    parser.add_argument("--out-topology", type=Path, default=Path("docs/evidence/h01-kept-network.json"))
    parser.add_argument("--out-components", type=Path, default=Path("docs/evidence/h01-kept-components.json"))
    args = parser.parse_args()
    kept = json.loads(args.decision.read_text())["kept"]
    topology, components, report = filter_network(json.loads(args.topology.read_text()),
                                                  json.loads(args.components.read_text()), kept)
    provenance = dict(topology=str(args.topology), topology_sha256=sha256(args.topology),
                      components=str(args.components), components_sha256=sha256(args.components),
                      decision=str(args.decision), decision_sha256=sha256(args.decision))
    topology["filtered_from"] = components["filtered_from"] = provenance
    for path, payload in ((args.out_topology, topology), (args.out_components, components)):
        path.write_text(json.dumps(payload, indent=2)+"\n", newline="\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
