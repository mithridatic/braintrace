"""Restrict a multicompartment cell to one declared circuit emission site."""
from dataclasses import dataclass
import numpy as np
import brainunit as u


@dataclass(frozen=True)
class _SiteSpike:
    base: object
    cv_id: int
    n_cv: int

    def __call__(self, voltage):
        if voltage.shape[-1] != self.n_cv:
            raise ValueError("Output-site mask no longer matches the cell mesh.")
        return self.base(voltage)*(np.arange(self.n_cv) == self.cv_id)


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
    candidates = [cv for cv in cell.cvs if cv.branch_id == branch and
                  float(cv.area.to_decimal(u.um**2)) > 0.]
    if not candidates:
        raise ValueError("The source branch has no membrane CV for output.")
    cv = min(candidates, key=lambda item: (abs((item.prox+item.dist)/2-x), item.id))
    base = cell.spk_fun.base if isinstance(cell.spk_fun, _SiteSpike) else cell.spk_fun
    cell.spk_fun = _SiteSpike(base, cv.id, cell.n_cv)
    return dict(source_point=(int(branch), float(x)), output_cv_id=cv.id,
                output_midpoint=(cv.branch_id, (cv.prox+cv.dist)/2),
                policy="nearest membrane CV midpoint on source branch; one output site")
