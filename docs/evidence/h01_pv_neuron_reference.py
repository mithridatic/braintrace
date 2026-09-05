"""Run the pinned published PV circuit-cell template in an isolated container.

The input files stay in the cache. This driver does not modify their equations.
NEURON advances the entire simulation through continuerun, without Python steps.
"""

import argparse
import json
from pathlib import Path

import neuron
from neuron import h
import numpy as np
from scipy.signal import find_peaks


def _time_average(times, values, start, stop):
    interior = times[(times > start) & (times < stop)]
    grid = np.r_[start, interior, stop]
    return float(np.trapezoid(np.interp(grid, times, values), grid)/(stop-start))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--current-na", type=float, required=True)
parser.add_argument("--dt-ms", type=float, default=.025)
parser.add_argument("--initial-mv", type=float, default=-80.)
parser.add_argument("--scale-conductance", choices=("NaTg", "Kv3_1", "SK"))
parser.add_argument("--conductance-factor", type=float, default=1.)
parser.add_argument("--cvode-atol", type=float)
parser.add_argument("--passive", action="store_true")
parser.add_argument("--nseg-factor", type=int, default=1)
parser.add_argument("--refine-region", choices=("all", "soma", "axon", "dendrites"), default="all")
parser.add_argument("--output", required=True)
args = parser.parse_args()
if not np.isfinite(args.conductance_factor) or args.conductance_factor < 0:
    parser.error("Conductance factor must be nonnegative and finite.")
if args.scale_conductance is None and args.conductance_factor != 1.:
    parser.error("A non-unit conductance factor requires a named mechanism.")
if args.nseg_factor < 1 or args.nseg_factor % 2 != 1:
    parser.error("Segment factor must be a positive odd integer.")
if not np.isfinite([args.current_na, args.dt_ms, args.initial_mv]).all() or args.dt_ms <= 0:
    parser.error("Current must be finite and time step must be positive.")
if args.cvode_atol is not None and (not np.isfinite(args.cvode_atol) or args.cvode_atol <= 0):
    parser.error("CVode absolute tolerance must be positive and finite.")

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")
h.load_file("/work/source/NeuronTemplate.hoc")
h.load_file("/work/source/biophys_HL5BN1.hoc")
cell = h.NeuronTemplate("/work/source/HL5BN1.swc")
h.biophys_HL5BN1(cell)
if args.scale_conductance is not None:
    for section in cell.all:
        if args.scale_conductance in section.psection()["density_mechs"]:
            parameter = "gbar_"+args.scale_conductance
            setattr(section, parameter, getattr(section, parameter)*args.conductance_factor)
for section in cell.all:
    family = section.name().split(".", 1)[1].split("[", 1)[0]
    selected = (args.refine_region == "all" or family == args.refine_region
                or (args.refine_region == "dendrites" and family in ("dend", "apic")))
    if selected:
        section.nseg *= args.nseg_factor
if args.passive:
    # Static configuration only. Remove every density mechanism except leak.
    for section in cell.all:
        for mechanism in tuple(section.psection()["density_mechs"]):
            if mechanism != "pas":
                section.uninsert(mechanism)
h.celsius = 34.
h.dt = args.dt_ms
h.steps_per_ms = 1./args.dt_ms
h.CVode().active(0)
if args.cvode_atol is not None:
    h.CVode().atol(args.cvode_atol)
    h.CVode().active(1)
clamp = h.IClamp(cell.soma[0](.5))
clamp.delay = 270.
clamp.dur = 1000.
clamp.amp = args.current_na
t = h.Vector().record(h._ref_t)
v = h.Vector().record(cell.soma[0](.5)._ref_v)
current = h.Vector().record(clamp._ref_i)
axon_v = h.Vector().record(cell.axon[0](.5)._ref_v)
calcium = h.Vector().record(cell.soma[0](.5)._ref_cai) if not args.passive else []
sk = h.Vector().record(cell.soma[0](.5).SK._ref_z) if not args.passive else []
ih = h.Vector().record(cell.soma[0](.5).Ih._ref_m) if not args.passive else []
nap_h = h.Vector().record(cell.soma[0](.5).Nap._ref_h) if not args.passive else []
geometry = [{"name": sec.name(), "length_um": sec.L, "diameter_um": sec.diam,
             "nseg": sec.nseg, "area_um2": sum(seg.area() for seg in sec),
             "ra_ohm_cm": sec.Ra, "cm_uf_cm2": sec.cm,
             "parent": None if sec.parentseg() is None else str(sec.parentseg())}
            for sec in cell.all]
h.finitialize(args.initial_mv)
h.continuerun(1500.)
times, voltage, applied, axon = map(np.asarray, (t, v, current, axon_v))
assert np.isfinite(voltage).all() and np.isfinite(axon).all()
peaks, _ = find_peaks(voltage, height=0., prominence=40.)
peaks = peaks[(times[peaks] >= 270.) & (times[peaks] < 1270.)]
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output.with_suffix(".npz"), time_ms=times, voltage_mv=voltage,
                    current_na=applied, axon_voltage_mv=axon,
                    calcium_mm=np.asarray(calcium), sk_gate=np.asarray(sk),
                    ih_gate=np.asarray(ih), nap_h_gate=np.asarray(nap_h))
report = {"neuron_version": neuron.__version__, "source_commit": "82cdd91bc93942ba19315371330a2412e064baf5",
          "active_channels": not args.passive,
          "conductance_intervention": {"mechanism": args.scale_conductance, "factor": args.conductance_factor},
          "nseg_factor": args.nseg_factor,
          "refine_region": args.refine_region,
          "model": "ModelDB267587 released HL5BN1 circuit cell, original template and mechanisms",
          "current_na": args.current_na, "dt_ms": args.dt_ms, "temperature_c": 34.,
          "integration": {"method": "CVode" if args.cvode_atol is not None else "fixed step",
                          "cvode_atol": args.cvode_atol},
          "initial_voltage_mv": args.initial_mv, "stimulus_on_ms": 270., "stimulus_off_ms": 1270.,
          "duration_ms": 1500., "synaptic_background": "none",
          "sample_convention": "NEURON recorded time, includes initial state at t=0",
          "spike_times_ms": times[peaks].tolist(), "spike_peaks_mv": voltage[peaks].tolist(),
          "baseline_mean_mv": _time_average(times, voltage, 200., 270.),
          "baseline_mean_convention": "time integral of piecewise-linear voltage over 200-270 ms, divided by 70 ms",
          "geometry": geometry,
          "qualification": "independent reference response; comparison with human data and BrainCell transfer pending"}
output.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "geometry"}))
