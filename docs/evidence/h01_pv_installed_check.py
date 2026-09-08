"""Verify wheel origin, source parity, and a short installed PV simulation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--installed", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
installed, source = args.installed.resolve(), args.source.resolve()
sys.path.insert(0, str(installed))

import braintrace
import brainstate
import brainunit as u
import numpy as np
from braintrace.datasets.h01_pv_cell import make_pv_cell

assert Path(braintrace.__file__).resolve().is_relative_to(installed)
parity = {}
for path in sorted((source/"braintrace/datasets").glob("h01*.py")):
    if path.name.endswith("_test.py"):
        assert not (installed/"braintrace/datasets"/path.name).exists()
        continue
    other = installed/"braintrace/datasets"/path.name
    assert other.read_bytes() == path.read_bytes(), path.name
    parity[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
geometry = json.loads((source/"docs/evidence/h01-pv-geometry-reference.json").read_text())
with brainstate.environ.context(precision=64):
    result = make_pv_cell(geometry).run(dt=.005*u.ms, duration=2.*u.ms)
    voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    calcium = np.asarray(result.traces["calcium"].to_decimal(u.mM))
    gate = np.asarray(result.traces["sk_gate"])
assert np.isfinite(voltage).all() and np.isfinite(calcium).all()
assert np.all(calcium > 0.) and np.all((gate >= 0.) & (gate <= 1.))
report = {"import_path": str(Path(braintrace.__file__).resolve()),
          "isolated_python": bool(sys.flags.isolated), "module_sha256": parity,
          "module_count": len(parity), "duration_ms": 2., "dt_ms": .005,
          "voltage_range_mv": [float(voltage.min()), float(voltage.max())],
          "qualification": "installed source parity and active initialization; not physiological validation"}
(source/"docs/evidence/h01-pv-installed-check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
