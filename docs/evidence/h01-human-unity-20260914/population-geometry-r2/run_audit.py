"""Audit the exact recorded population mesh without biological mechanisms."""

from pathlib import Path
import argparse
import gc
import hashlib
import inspect
import json
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
repository = root.parents[3]
sys.path.insert(0, str(repository))
sys.path.insert(0, str(root.parents[1]))
from h01_population_geometry import validate_cv_partition
from h01_post_import_audit import compare_edges

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
args = parser.parse_args()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, obj: p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
if (root/'launch.json').exists():raise FileExistsError('Evidence prefix already launched.')
decision = json.loads((repository/'docs/evidence/h01-ready-104-implicit-build-decision.json').read_bytes())
assert sha(args.reference) == decision['inputs']['build']['sha256']
reference = json.loads(args.reference.read_bytes())
assert len(reference['cells']) == 104 and reference['n_compartments'] == 808495
assert set(reference['cells']) == set(reference['simulated_cell_ids'])
write(root/'launch.json',dict(reference_sha256=sha(args.reference), reference_path=str(args.reference),
       archive_path=str(args.archive), expected_cells=104, expected_compartments=808495,
       wall_cap_seconds=600, human_anatomy_only=True, biological_mechanisms_executed=False))

import braincell
import brainunit as u
from braincell._discretization import geometry
from braintrace.datasets import h01, _h01_swc, h01_construction, h01_discretization, h01_network

archive = h01.H01Archive(args.archive)
assert set(archive.neuron_ids) == set(reference['cells'])
modules = [h01, _h01_swc, h01_construction, h01_discretization, h01_network, geometry]
sources = [Path(m.__file__) for m in modules]+[Path(inspect.getfile(braincell.MaxCVLen)),
           Path(__file__), root.parents[1]/'h01_population_geometry.py', root.parents[1]/'h01_post_import_audit.py']
write(root/'executed-sources.json', {str(p):sha(p) for p in sources})
started = time.monotonic()
reports = []
for identity in sorted(reference['cells'], key=int):
    cell_started = time.monotonic()
    saved = reference['cells'][identity]
    component = int(saved['measured_anatomy']['member'].split('.')[-2])
    imported = archive.load(identity, component=component)
    assert imported.source_sha256 == saved['measured_anatomy']['source_sha256']
    rows = imported.source_rows
    indices = {int(row[0]): i for i,row in enumerate(rows)}
    children = np.flatnonzero(rows[:,6] != -1)
    parents = [indices[int(rows[i,6])] for i in children]
    expected = np.column_stack((rows[parents,2:5]*(.032,.032,.033), rows[children,2:5]*(.032,.032,.033),
                                rows[parents,5]*.001, rows[children,5]*.001))
    branches, branch_sources = [], []
    for view in imported.morphology.branches:
        edge = np.column_stack([np.asarray(x.to_decimal(u.um)) for x in
             (view.branch.points_proximal, view.branch.points_distal, view.branch.radii_proximal, view.branch.radii_distal)])
        lengths = np.linalg.norm(edge[:,3:6]-edge[:,:3],axis=1)
        areas = np.pi*(edge[:,6]+edge[:,7])*np.hypot(lengths,edge[:,7]-edge[:,6])
        branches.append(edge)
        branch_sources.append([areas.sum(), lengths.sum()])
    actual = np.concatenate(branches)
    error = compare_edges(expected, actual)
    regions = h01_network._regions(imported)
    region_comparisons, region_arrays = {}, {}
    policy = braincell.MaxCVLen(saved['max_cv_length_um']*u.um)
    assert saved['max_cv_length_um'] == 10.
    for label, region in regions.items():
        intervals = np.asarray(region.evaluate(imported.morphology).intervals).reshape(-1,3)
        previous = np.asarray(saved['electrical_intervals'][label]).reshape(-1,3)
        same_structure = intervals.shape == previous.shape and np.array_equal(intervals[:,0],previous[:,0])
        difference = np.abs(intervals[:,1:]-previous[:,1:]) if same_structure else None
        region_comparisons[label] = dict(same_structure=bool(same_structure),
            passed=bool(same_structure and np.all(difference <= 1e-12)),
            absolute_parameter_tolerance=1e-12,
            maximum_parameter_difference=float(difference.max(initial=0)) if same_structure else None,
            boundaries_exceeding_tolerance=int(np.sum(difference > 1e-12)) if same_structure else None)
        region_arrays['current_'+label+'_intervals'] = intervals
        region_arrays['reference_'+label+'_intervals'] = previous
        policy = h01_discretization.BoundaryAlignedCV(policy, region)
    cvs = geometry.build_cv_geometry(imported.morphology,policy.resolve_cv_bounds(imported.morphology)).geos
    cv_data = np.array([[cv.branch_id,cv.prox,cv.dist,cv.lateral_area_um2,cv.length_um] for cv in cvs])
    report = validate_cv_partition(branch_sources,cv_data)
    assert len(cvs) == saved['n_compartments']
    path = root/(identity+'.npz')
    np.savez_compressed(path, source_rows=rows,source_edges_um=expected, imported_edges_um=actual,
          imported_branch_edge_counts=np.array([len(b) for b in branches]),
          source_branch_area_um2_length_um=np.asarray(branch_sources), cv_observations=cv_data, **region_arrays)
    region_passed = all(r['passed'] for r in region_comparisons.values())
    report.update(status='passed' if region_passed else 'region_comparison_failed',
          source_edge_and_cv_geometry_passed=True, region_comparison=region_comparisons,
          cell_id=identity, component=component,source_sha256=imported.source_sha256,
          directed_edges=len(expected),maximum_endpoint_error_um=error,
          duplicate_source_positions=len(rows)-len(np.unique(rows[:,2:5],axis=0)),
          lateral_area_um2=float(cv_data[:,3].sum()),length_um=float(cv_data[:,4].sum()),
          arrays_sha256=sha(path),seconds=time.monotonic()-cell_started)
    write(path.with_suffix('.json'), report)
    reports.append(report)
    print(json.dumps({k:report[k] for k in ('cell_id','directed_edges','cv_count','seconds')}),flush=True)
    del imported, rows, branches, actual, expected, cvs, cv_data, regions, policy
    gc.collect()
assert sum(r['cv_count'] for r in reports) == 808495
write(root/'result.json',dict(status='passed' if all(r['status']=='passed' for r in reports) else 'region_comparison_failed',
    cells=reports,cell_count=len(reports), source_edge_and_cv_geometry_passed=all(r['source_edge_and_cv_geometry_passed'] for r in reports),
    total_compartments=sum(r['cv_count'] for r in reports),total_directed_edges=sum(r['directed_edges'] for r in reports),
    elapsed_seconds=time.monotonic()-started,scope='source-to-production compartment geometry only',
    source_mesh_completeness_qualified=False,physiology_qualified=False,scores_promoted=False))
