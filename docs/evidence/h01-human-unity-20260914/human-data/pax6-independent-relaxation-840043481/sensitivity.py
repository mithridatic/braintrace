"""Post-fit local sensitivity diagnosis; no refit, new observations or gate change."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize._numdiff import approx_derivative

out = Path(__file__).resolve().parent
repo = out.parents[4]
sys.path.insert(0, str(repo))
from docs.evidence.h01_pax6_relaxation_current import prepare_commands, predict_active, LOWER, UPPER, ORDER

began = time.perf_counter()
inputs = json.loads((out / 'training-inputs.json').read_bytes())
candidate = json.loads((out / 'frozen-candidate.json').read_bytes())
with np.load(out / inputs['source_arrays_path']) as source:
    raw = dict(source)
ids, times, weights = [], [], []
for row, record in enumerate(inputs['records']):
    s = record['sweep']
    index = raw[f's{s}_fit_indices']
    ids.append(np.full(len(index), row, int))
    times.append(raw[f's{s}_time_ms'][index])
    weights.append(np.sqrt(raw[f's{s}_fit_weights']))
ids, times, weight = (np.concatenate(x) for x in (ids, times, weights))
ctx = prepare_commands(raw['command_starts'], raw['command_stops'], raw['command_volts'], ids, times,
                       inputs['observation_parameters'])

# SciPy differentiates the static batched closed-form response. No time-step loop.
def response(parameters):
    return predict_active(parameters, ctx)*weight/np.sqrt(np.sum(weight**2))


jacobian = approx_derivative(response, np.array(candidate['parameters']), method='cs')
scaled = jacobian*(UPPER-LOWER)
_, singular, vectors = np.linalg.svd(scaled, full_matrices=False)
norm = np.linalg.norm(scaled, axis=0)
cosine = (scaled.T @ scaled)/(norm[:, None]*norm[None, :])
with (out / 'local-sensitivity.npz').open('xb') as stream:
    np.savez_compressed(stream, normalized_parameter_jacobian=scaled,
                        singular_values=singular, right_singular_vectors=vectors, derivative_cosines=cosine,
                        protocol_ids=ids, time_ms=times, quadrature_weights=weight**2)
weakest = [dict(parameter=ORDER[j], coefficient=float(vectors[-1,j]))
           for j in np.argsort(-np.abs(vectors[-1]))[:5]]
result = dict(status='post-fit local diagnosis only', parameters_unchanged=True, new_response_access=False,
              candidate_sha256=hashlib.sha256((out / 'frozen-candidate.json').read_bytes()).hexdigest(),
              parameter_order=ORDER, normalized_parameter_derivative_norms=norm.tolist(),
              singular_values=singular.tolist(), smallest_largest_ratio=float(singular[-1]/singular[0]),
              weakest_direction=weakest, availability_half_slope_derivative_cosine=float(cosine[6,7]),
              boundary_active=True, confidence_intervals_estimated=False,
              interpretation='Local response sensitivity under full bound-range parameter scaling; neither a global identifiability proof nor a confidence interval.',
              seconds=time.perf_counter()-began, scores_promoted=False)
with (out / 'local-sensitivity.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
