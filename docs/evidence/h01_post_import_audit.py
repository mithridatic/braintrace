"""Measure source-edge preservation through the installed BrainCell geometry stage."""

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np


def compare_edges(expected, actual):
    """Compare every directed edge with its endpoint coordinates and radii.

    Parameters
    ----------
    expected, actual : numpy.ndarray
        Rows contain proximal xyz, distal xyz, proximal radius, distal radius,
        all in micrometers. Row order is irrelevant.

    Returns
    -------
    float
        Maximum absolute discrepancy; incomplete or changed geometry raises.
    """
    if expected.shape != actual.shape or expected.ndim != 2 or expected.shape[1] != 8:
        raise ValueError('Source and imported edge inventories differ.')
    if not expected.size or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Edge geometry is empty or nonfinite.')
    left = expected[np.lexsort(expected.T[::-1])]
    right = actual[np.lexsort(actual.T[::-1])]
    error = float(np.abs(left - right).max())
    if error > 1e-8:
        raise ValueError('Imported endpoint coordinates or radii changed.')
    return error


def main(argv=None):
    """Retain actual source, branch and control-volume observations.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments; requires the source archive and a new output prefix.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    assert not args.output.with_suffix('.json').exists()
    assert not args.output.with_suffix('.npz').exists()
    import brainunit as u
    from braincell import CVPerBranch
    from braincell._discretization import geometry
    from braintrace.datasets import h01
    from braintrace.datasets import _h01_swc

    started = time.monotonic()
    imported = h01.H01Archive(args.archive).load('955432427', component=0)
    rows = imported.source_rows
    # Unique coordinates make endpoint correspondence unambiguous for this cell.
    assert len(np.unique(rows[:, 2:5], axis=0)) == len(rows)
    indices = {int(row[0]): i for i, row in enumerate(rows)}
    children = np.flatnonzero(rows[:, 6] != -1)
    parents = [indices[int(rows[i, 6])] for i in children]
    expected = np.column_stack((rows[parents, 2:5] * (.032, .032, .033),
                                rows[children, 2:5] * (.032, .032, .033),
                                rows[parents, 5] * .001, rows[children, 5] * .001))
    actual = np.concatenate([np.column_stack([
        np.asarray(value.to_decimal(u.um)) for value in (
            view.branch.points_proximal, view.branch.points_distal,
            view.branch.radii_proximal, view.branch.radii_distal)])
        for view in imported.morphology.branches])
    error = compare_edges(expected, actual)
    lengths = np.linalg.norm(expected[:, 3:6] - expected[:, :3], axis=1)
    source_area = float(np.sum(np.pi * (expected[:, 6] + expected[:, 7]) *
                              np.hypot(lengths, expected[:, 7] - expected[:, 6])))
    policy = CVPerBranch()
    cvs = geometry.build_cv_geometry(imported.morphology,
                                    policy.resolve_cv_bounds(imported.morphology)).geos
    areas = np.array([cv.lateral_area_um2 for cv in cvs])
    cv_lengths = np.array([cv.length_um for cv in cvs])
    assert np.isfinite(areas).all() and (areas > 0).all()
    assert np.isclose(areas.sum(), source_area, rtol=1e-8, atol=0)
    assert np.isclose(cv_lengths.sum(), lengths.sum(), rtol=1e-8, atol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), source_edges_um=expected,
                        imported_edges_um=actual, cv_areas_um2=areas, cv_lengths_um=cv_lengths)
    sources = [Path(h01.__file__), Path(_h01_swc.__file__), Path(geometry.__file__),
               Path(inspect.getfile(CVPerBranch)), Path(__file__)]
    report = dict(status='passed', cell_id='955432427', component=0,
                  source_sha256=imported.source_sha256, archive_sha256=h01.ARCHIVE_SHA256,
                  edges_checked=len(expected), max_endpoint_error_um=error,
                  source_area_um2=source_area, cv_area_um2=float(areas.sum()),
                  source_length_um=float(lengths.sum()), cv_length_um=float(cv_lengths.sum()),
                  policy='CVPerBranch', cv_count=len(cvs), cvs=[asdict(cv) for cv in cvs],
                  scope='static geometry only; no mechanisms, simulation or human physiology',
                  host='local CPU', elapsed_seconds=time.monotonic()-started,
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  arrays_sha256=hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest())
    with args.output.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cvs', 'sources')}))


if __name__ == '__main__':
    main()
