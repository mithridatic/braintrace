"""Local SWC compatibility preserving distinct H01 branch attachment geometry."""

from dataclasses import replace

import numpy as np
from braincell.io.swc import SwcReader
from braincell.io.swc.soma import row_point, row_radius


class H01SwcReader(SwcReader):
    """Read physical H01 coordinates without relative-tolerance node collapse.

    Notes
    -----
    H01 uses neutral branch types. Only attachment geometry is overridden;
    parsing, branch identities, topology and reports retain upstream behavior.
    No global BrainCell or NumPy functions are patched.
    """

    def _make_branch(self, branch, nodes, *, parent_branch_type=None):
        if branch.attach is None or branch.override_points is not None:
            return super()._make_branch(branch, nodes, parent_branch_type=parent_branch_type)
        points = [row_point(nodes[node]) for node in branch.point_ids]
        radii = [row_radius(nodes[node]) for node in branch.point_ids]
        point, radius = self._attach_geometry(branch.attach, nodes)
        if parent_branch_type == "soma":
            radius = radii[0]
        copy = self._should_copy_attach_geometry(
            parent_branch_type=parent_branch_type, attach=branch.attach,
            point_ids=list(branch.point_ids), points=points, attach_point=point,
        )
        if copy and (not np.array_equal(points[0], point) or radii[0] != radius):
            points.insert(0, point)
            radii.insert(0, radius)
        exact = replace(branch, override_points=tuple(map(tuple, points)),
                        override_radii=tuple(radii))
        return super()._make_branch(exact, nodes, parent_branch_type=parent_branch_type)
