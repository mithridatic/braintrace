"""Restrict a multicompartment cell to one declared circuit emission site."""
from dataclasses import dataclass
import numpy as np
import brainunit as u


@dataclass(frozen=True)
class _SiteSpike:
    base: object
    cv_id: int
    n_cv: int
    mask: object = None

    def __init__(self, base: object, cv_id: int, n_cv: int):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "cv_id", cv_id)
        object.__setattr__(self, "n_cv", n_cv)
        mask = np.zeros(n_cv, dtype=bool)
        mask[cv_id] = True
        object.__setattr__(self, "mask", mask)

    def __call__(self, voltage):
        if voltage.shape[-1] != self.n_cv:
            raise ValueError("Output-site mask no longer matches the cell mesh.")
        return self.base(voltage)*self.mask


def restrict_spike_output(cell, location):
    """Emit only from the nearest membrane CV on the selected source branch.

    Parameters
    ----------
    cell : braincell.Cell
        Uninitialized cell after its final geometry, paint, and placement edits.
        Reapply this function if the mesh changes before initialization.
    location : LocsetExpr
        Exactly one source location, usually the measured soma sample.

    Returns
    -------
    dict
        Source point and selected CV midpoint. This is a modeled output site,
        not a measured axon terminal or an interpolated source-node voltage.

    Notes
    -----
    The installed network reduces compartment events with logical any. Masking
    all other CVs prevents propagation through them from emitting extra events.
    This function retains the cell's existing threshold and surrogate rule.
    """
    points = location.evaluate(cell.morpho).points
    if len(points) != 1:
        raise ValueError("A single source location is required for circuit output.")
    branch, x = points[0]
    cached_disc = cell.__dict__.get("_discretization_cache")
    if cached_disc is not None:
        cvs = cached_disc.cvs
        candidates = [cv for cv in cvs if cv.branch_id == branch and
                      (cv.area.mantissa if isinstance(cv.area, u.Quantity) else float(cv.area)) > 0.]
        if not candidates:
            raise ValueError("The source branch has no membrane CV for output.")
        cv = min(candidates, key=lambda item: (abs((item.prox+item.dist)/2-x), item.id))
        cv_id = cv.id
        midpoint = (cv.branch_id, (cv.prox+cv.dist)/2)
        n_cv = len(cvs)
    else:
        bounds_by_branch = cell.cv_policy.resolve_cv_bounds(cell.morpho)
        if branch >= len(bounds_by_branch) or not bounds_by_branch[branch]:
            raise ValueError("The source branch has no membrane CV for output.")
        branch_bounds = bounds_by_branch[branch]
        branch_offset = sum(len(b) for b in bounds_by_branch[:branch])
        best_k = min(range(len(branch_bounds)), key=lambda k: (abs((branch_bounds[k][0] + branch_bounds[k][1])/2.0 - x), k))
        cv_id = branch_offset + best_k
        lo, hi = branch_bounds[best_k]
        midpoint = (int(branch), (lo + hi) / 2.0)
        n_cv = sum(len(b) for b in bounds_by_branch)

    base = cell.spk_fun.base if isinstance(cell.spk_fun, _SiteSpike) else cell.spk_fun
    cell.spk_fun = _SiteSpike(base, cv_id, n_cv)
    return dict(source_point=(int(branch), float(x)), output_cv_id=cv_id,
                output_midpoint=midpoint,
                policy="nearest membrane CV midpoint on source branch; one output site")
