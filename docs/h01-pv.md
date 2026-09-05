# Published human-fitted PV reference

This model transfers the released HL5BN1 cell from ModelDB 267587 to BrainCell.
It is a reference for inhibitory-cell development. It is not an H01 donor cell.
Its channels include borrowed nonhuman kinetics and published fitted parameters.
Its axon is the source template's inferred replacement.

The model runs, but numerical convergence and human waveform validation are incomplete.
Do not treat a matching spike count as a completed validation.
See the [transfer evidence](evidence/h01-pv-cell-transfer.md) and
[human comparison](evidence/h01-pv-reference-assessment.md).

## Build and run

The installed API is `braintrace.datasets.h01_pv_cell.make_pv_cell`.
Provide the [reference geometry export](evidence/h01-pv-geometry-reference.json).
The builder preserves the exported cable taper, length, area, and attachments.
It converts source conductance densities from S/cm2 to mS/cm2.

```python
import json
from pathlib import Path

import brainstate
import brainunit as u
from braintrace.datasets.h01_pv_cell import make_pv_cell

geometry = json.loads(Path("h01-pv-geometry-reference.json").read_text())
with brainstate.environ.context(precision=64):
    cell = make_pv_cell(geometry, current_na=0.19, max_cv_length_um=5.0)
    result = cell.run(dt=0.0025*u.ms, duration=1500*u.ms)
    voltage_mv = result.traces["voltage"].to_decimal(u.mV)
    calcium_mm = result.traces["calcium"].to_decimal(u.mM)
    sk_gate = result.traces["sk_gate"]
```

The current step runs from 270 to 1270 ms at soma position 0.5.
Initial voltage is -80 mV. Channel parameters use the source's 34 Celsius convention.
The solver is BrainCell's staggered method. `Cell.run` uses a compiled loop.
The recorded values are end-of-step samples. The first sample is at `dt`.
No holding current, background input, or synaptic circuit is added.

Set `active=False` to keep passive leak only. This also removes Ih.
This option supports a passive cable comparison; it is not an inhibitory model.

The 5 micrometre setting above has produced a full response with 11 spikes.
It is a tested configuration, not a claim of spatial convergence.
The 2.5 micrometre mesh also gives 11 spikes. Its late spike timing still differs
from NEURON. Halving the time step from 0.0025 to 0.00125 ms moves the last
spike by about 0.99 ms. Full waveform convergence is not established.

## Installed-package evidence

Wheel SHA256: `12cb31b2bcbbb79875fb23a35449f7c32fa09db33ca0342c8a3fa154cfc311f7`.
The isolated check imported the wheel, verified byte parity for 15 H01 modules,
and ran this active cell for 2 ms. Voltage and calcium were finite; calcium was
positive and SK activation stayed between zero and one.
The [check record](evidence/h01-pv-installed-check.json) records module hashes.
This short installed run checks packaging and initialization, not physiology.

The offline wheel build uses `--no-cache-dir` because the user-level pip cache
is outside this worktree's writable area. Dependencies were already available.
No package index access is needed for this validation.
