"""Probe an installed alternative integrator on the actual PV cell runtime."""

import argparse
import json
from pathlib import Path

import braincell
import brainstate
import brainunit as u
import numpy as np

from braintrace.datasets.h01_pv_cell import make_pv_cell

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", choices=("cn_rk4", "rk4"), required=True)
parser.add_argument("--explicit-time", action="store_true")
args = parser.parse_args()
folder = Path(__file__).parent
reference = json.loads((folder/"h01-pv-geometry-reference.json").read_text())
report = {"solver": args.solver, "explicit_time_wrapper": args.explicit_time,
          "purpose": "one-step API compatibility only; not stability or accuracy qualification"}
try:
    with brainstate.environ.context(precision=64):
        cell = make_pv_cell(reference)
        integrator = braincell.quad.get_integrator(args.solver)
        if args.explicit_time:
            cell.solver = lambda target: integrator(target, 0.*u.ms, brainstate.environ.get_dt())
        else:
            cell.solver = integrator
        result = cell.run(dt=.0001*u.ms, duration=.0001*u.ms)
        voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV))
        assert np.isfinite(voltage).all()
        report.update(status="one step completed", voltage_mv=voltage.ravel().tolist())
except Exception as error:
    report.update(status="failed", exception_type=type(error).__name__, message=str(error))
suffix = "-explicit" if args.explicit_time else ""
(folder/f"h01-pv-solver-{args.solver}{suffix}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
