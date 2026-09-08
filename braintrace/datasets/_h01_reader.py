"""Preserve voxel-scale H01 attachments at large absolute coordinates."""

from dataclasses import replace

import numpy as np
from braincell.io.swc.reader import SwcReader
from braincell.io.swc.soma import row_point, row_radius


class _H01Reader(SwcReader):
    """Keep distinct custom-cable attachment points in the upstream pipeline."""

    def _make_branch(self, branch, nodes, *, parent_branch_type=None):
        if branch.attach is not None and branch.override_points is None:
            points = [row_point(nodes[node]) for node in branch.point_ids]
            attach_point, attach_radius = self._attach_geometry(branch.attach, nodes)
            # H01 normalizes all types to custom. Retain the upstream semantics
            # for any other type and for exact duplicate attachment coordinates.
            if (branch.branch_type == "custom"
                    and not np.array_equal(points[0], attach_point)
                    and np.allclose(points[0], attach_point)):
                radii = [row_radius(nodes[node]) for node in branch.point_ids]
                branch = replace(
                    branch,
                    override_points=tuple(tuple(p) for p in [attach_point, *points]),
                    override_radii=tuple([float(attach_radius), *radii]),
                )
        return super()._make_branch(branch, nodes, parent_branch_type=parent_branch_type)
