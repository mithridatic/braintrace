"""Verify every retained population record against the original archive bytes."""

from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

root = Path(__file__).resolve().parent
repository = root.parents[3]
primary = repository.parents[1]
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result = json.loads((root/'result.json').read_bytes())
launch = json.loads((root/'launch.json').read_bytes())
terminal = json.loads((root/'run-receipt.json').read_bytes())
assert terminal['returncode'] == 0 and not terminal['timed_out']
assert terminal['log_sha256'] == sha(root/'audit.log')
assert result['cell_count'] == 104 and result['total_compartments'] == 808495
assert len({r['cell_id'] for r in result['cells']}) == 104
sources = json.loads((root/'executed-sources.json').read_bytes())
for path,digest in sources.items():assert sha(Path(path)) == digest,path
reference_path = Path(launch['reference_path'])
assert sha(reference_path) == launch['reference_sha256']
reference = json.loads(reference_path.read_bytes())
assert set(reference['cells']) == {r['cell_id'] for r in result['cells']}
region_failed, duplicated, displacement = [], [], []
with zipfile.ZipFile(primary/'.cache/h01/proofread104.zip') as archive:
    for record in result['cells']:
        identity = record['cell_id']
        saved = reference['cells'][identity]
        source = archive.read(saved['measured_anatomy']['member'])
        assert hashlib.sha256(source).hexdigest() == record['source_sha256']
        raw = np.fromstring(b' '.join(line for line in source.splitlines() if line and not line.startswith(b'#')).decode('ascii'),sep=' ').reshape(-1,7)
        path = root/(identity+'.npz')
        assert sha(path) == record['arrays_sha256']
        assert json.loads(path.with_suffix('.json').read_bytes()) == record
        with np.load(path) as a:
            np.testing.assert_array_equal(a['source_rows'],raw)
            edges = a['source_edges_um']
            lengths = np.linalg.norm(edges[:,3:6]-edges[:,:3],axis=1)
            area = np.sum(np.pi*(edges[:,6]+edges[:,7])*np.sqrt(lengths**2+(edges[:,7]-edges[:,6])**2))
            np.testing.assert_allclose(area,record['lateral_area_um2'],rtol=1e-8,atol=0)
            counts = a['imported_branch_edge_counts']
            branch_lengths = a['source_branch_area_um2_length_um'][:,1]
            assert counts.sum() == len(edges) == record['directed_edges']
            assert len(a['cv_observations']) == record['cv_count'] == saved['n_compartments']
            if record['duplicate_source_positions']:duplicated.append(identity)
            for label,comparison in record['region_comparison'].items():
                current,old = a['current_'+label+'_intervals'],a['reference_'+label+'_intervals']
                assert current.shape == old.shape and np.array_equal(current[:,0],old[:,0])
                delta = np.abs(current[:,1:]-old[:,1:])
                assert comparison['passed'] == bool(np.all(delta <= 1e-12))
                displacement.append(float(np.max(delta*branch_lengths[current[:,0].astype(int),None],initial=0)))
                if not comparison['passed']:region_failed.append(dict(cell_id=identity,region=label,**comparison))
assert bool(region_failed) == (result['status'] == 'region_comparison_failed')
report = dict(archive_rows_verified_cells=104,arrays_verified_cells=104,
    directed_edges_verified=sum(r['directed_edges'] for r in result['cells']),
    compartments_verified=sum(r['cv_count'] for r in result['cells']),
    cells_with_duplicate_source_positions=duplicated,
    historical_region_comparison_failed_cells=len({r['cell_id'] for r in region_failed}),
    failed_region_comparisons=region_failed,
    maximum_region_boundary_displacement_um=max(displacement),
    geometry_passed=result['source_edge_and_cv_geometry_passed'],
    full_comparison_passed=result['status']=='passed',physiology_qualified=False,scores_promoted=False)
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('failed_region_comparisons','cells_with_duplicate_source_positions')},indent=2))
