"""Audit exact repeated synapse coordinates for candidate H01 partner links."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

folder = Path(__file__).parent
path = folder.parents[1]/".cache/h01/synapse_locations.csv"
sha = hashlib.sha256(path.read_bytes()).hexdigest()
assert sha == "640ccb12c75b930f96373c7273bbf9688aa111b97b425ee22d8ef6db19f496be"
groups = defaultdict(list)
with path.open(newline="") as stream:
    reader = csv.DictReader(stream)
    for line, row in enumerate(reader, 2):
        key = tuple(int(row[name]) for name in ("prex", "prey", "prez", "postx", "posty", "postz"))
        groups[key].append({"line": line, "cell_id": row["104"], "role": row["prepost"],
                            "center": [int(row[name]) for name in ("x", "y", "z")]})
candidates = []
stats = Counter()
for key, rows in groups.items():
    roles = {role: {r["cell_id"] for r in rows if r["role"] == role} for role in ("pre", "post")}
    if not roles["pre"] or not roles["post"]:
        stats["one_role_only"] += 1
        continue
    if len(roles["pre"]) != 1 or len(roles["post"]) != 1:
        stats["ambiguous_cell_ids"] += 1
        continue
    if len({tuple(r["center"]) for r in rows}) != 1:
        stats["conflicting_centers"] += 1
        continue
    pre, post = next(iter(roles["pre"])), next(iter(roles["post"]))
    candidates.append({"pre_cell_id": pre, "post_cell_id": post,
                       "pre_voxel": list(key[:3]), "post_voxel": list(key[3:]),
                       "rows": rows, "self_link": pre == post})
    stats["candidate_synapses"] += 1
stats["source_rows"] = sum(map(len, groups.values()))
stats.update({"role_"+role: count for role, count in Counter(r["role"] for rows in groups.values() for r in rows).items()})
stats["unique_endpoint_pairs"] = len(groups)
stats["candidate_cell_pairs"] = len({(c["pre_cell_id"], c["post_cell_id"]) for c in candidates})
report = {"source_sha256": sha, "matching_rule": "exact six endpoint coordinates, matching center, one cell ID per role",
          "status": "coordinate-derived candidates; not explicit source partner IDs or manual synapse verification",
          "coordinate_um_per_voxel": [.008, .008, .033], "statistics": dict(stats),
          "candidates": candidates}
(folder/"h01-partner-coordinate-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["statistics"]))
