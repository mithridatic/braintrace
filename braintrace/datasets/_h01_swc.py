"""Translate the H01 archive's voxel coordinates into simulation geometry."""

from collections import deque
import io

import numpy as np

# The H01 analysis scripts use 32 x 32 x 33 nm skeleton voxels. Radii
# are already physical nanometers, unlike the position columns.
POSITION_UM = np.array([0.032, 0.032, 0.033])
RADIUS_UM = 0.001


def _parse_swc_bytes(source: bytes):
    lines = [line for line in source.splitlines() if line and not line.startswith(b'#')]
    if not lines:
        return np.empty((0, 7), dtype=np.float64)
    text = b' '.join(lines)
    data = np.fromstring(text.decode('ascii'), sep=' ', dtype=np.float64)
    if data.size % 7 != 0:
        raise ValueError("H01 component must contain seven finite columns.")
    return data.reshape(-1, 7)


def normalize(source: bytes):
    """Return raw rows and an SWC with physical geometry and neutral labels."""
    rows = _parse_swc_bytes(source)
    if rows.shape[1] != 7 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("H01 component must contain seven finite columns.")
    integer_columns = rows[:, [0, 1, 6]]
    if not np.equal(integer_columns, np.floor(integer_columns)).all():
        raise ValueError("Node IDs, annotation codes and parents must be integers.")
    ids = rows[:, 0].astype(np.int64)
    parents = rows[:, 6].astype(np.int64)
    if (ids < 0).any() or len(set(ids)) != len(ids):
        raise ValueError("Node IDs must be unique and nonnegative.")
    if (rows[:, 5] <= 0).any():
        raise ValueError("H01 radii must be positive; no radius is invented.")
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1:
        raise ValueError("Each component must have exactly one root.")
    index = {int(node): i for i, node in enumerate(ids)}
    children = {int(node): [] for node in ids}
    for i, parent in enumerate(parents):
        if parent == -1:
            continue
        if parent not in index:
            raise ValueError(f"Missing parent {parent}.")
        children[int(parent)].append(i)
    order, pending = [], deque([int(roots[0])])
    while pending:
        i = pending.popleft()
        order.append(i)
        pending.extend(children[int(ids[i])])
    if len(order) != len(rows):
        raise ValueError("Component contains a cycle disconnected from its root.")
    if len(rows) < 2:
        raise ValueError("A singleton component has no cable geometry to simulate.")
    new_ids = {int(ids[i]): j + 1 for j, i in enumerate(order)}
    lines = [
        "# H01 proofread_104; positions and radii converted to micrometers.",
        "# All SWC types are custom (0); original H01 annotations remain separate.",
    ]
    for i in order:
        x, y, z = rows[i, 2:5] * POSITION_UM
        radius = rows[i, 5] * RADIUS_UM
        parent = -1 if parents[i] == -1 else new_ids[int(parents[i])]
        lines.append(f"{new_ids[int(ids[i])]} 0 {x:.12g} {y:.12g} {z:.12g} {radius:.12g} {parent}")
    rows.flags.writeable = False
    return rows, "\n".join(lines) + "\n"
