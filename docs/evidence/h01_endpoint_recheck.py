"""Re-check candidate contact endpoints against the proofread volume with a wider box.

Run with the isolated h01-inspect Python. One edge at a time, each under its own
time budget, so a network stall costs one edge, not the batch. Reads the
candidate records from the resolved edge list and writes one JSON with the
nearest offset of each endpoint's expected label inside the wider box.
"""

import argparse
import json
import time
from pathlib import Path

from docs.evidence.h01_c3_edge_list import VOLUME, neighbourhood

DEFAULT_IDS = ("54906016", "95907584")


def box_offsets(labels, voxel, expected, radius_xy, radius_z):
    """Offset of the nearest voxel carrying ``expected`` inside the box, or None."""
    for point in neighbourhood(voxel, radius_xy, radius_z):
        if str(int(labels[point])) == expected:
            return [point[i]-voxel[i] for i in range(3)]
    return None


def recheck_edge(volume, edge, radius_xy, radius_z):
    """Nearest offsets for both endpoints of one edge with the wider box."""
    pre_v = tuple(int(v) for v in edge["pre_voxel"])
    post_v = tuple(int(v) for v in edge["post_voxel"])
    labels = volume.scattered_points(neighbourhood(pre_v, radius_xy, radius_z)+neighbourhood(post_v, radius_xy, radius_z))
    pre = box_offsets(labels, pre_v, edge["pre_cell"], radius_xy, radius_z)
    post = box_offsets(labels, post_v, edge["post_cell"], radius_xy, radius_z)
    return {"annotation_id": edge["annotation_id"], "pre_cell": edge["pre_cell"], "post_cell": edge["post_cell"],
            "type": edge["type"], "radius_xy": radius_xy, "radius_z": radius_z,
            "pre_nearest_offset": pre, "post_nearest_offset": post,
            "both_within_box": pre is not None and post is not None}


def select_edges(edge_list, ids):
    """Requested annotations first, then every edge already within the 2-voxel box."""
    wanted = [e for e in edge_list["edges"] if e["annotation_id"] in ids]
    in_box = [e for e in edge_list["edges"] if e["endpoints_within_box"] and e["annotation_id"] not in ids]
    return wanted+in_box


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-list", type=Path, default=Path("docs/evidence/h01-resolved-edge-list.json"))
    parser.add_argument("--ids", nargs="*", default=list(DEFAULT_IDS))
    parser.add_argument("--radius-xy", type=int, default=5)
    parser.add_argument("--radius-z", type=int, default=2)
    parser.add_argument("--edge-seconds", type=float, default=300.)
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/h01-endpoint-recheck-5voxel.json"))
    args = parser.parse_args(argv)
    from cloudvolume import CloudVolume
    volume = CloudVolume("precomputed://"+VOLUME, mip=0, progress=False, fill_missing=True, cache=False)
    edges = select_edges(json.loads(args.edge_list.read_text()), set(args.ids))
    results, budget_exceeded = [], []
    for edge in edges:
        t0 = time.time()
        result = recheck_edge(volume, edge, args.radius_xy, args.radius_z)
        result["seconds"] = round(time.time()-t0, 1)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps({"results": results, "budget_exceeded": budget_exceeded}, indent=1))
        if result["seconds"] > args.edge_seconds:
            budget_exceeded.append(edge["annotation_id"])
            print("stopping: edge exceeded its time budget", flush=True)
            break
    args.output.write_text(json.dumps({"results": results, "budget_exceeded": budget_exceeded,
                                       "requested": len(edges), "completed": len(results)}, indent=1))


if __name__ == "__main__":
    main()
