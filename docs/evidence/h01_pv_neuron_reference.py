"""Run the pinned published PV circuit-cell template in an isolated container.

The input files stay in the cache. This driver does not modify their equations.
NEURON advances the entire simulation through continuerun, without Python steps.
"""

import argparse
import hashlib
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


DEFAULT_TEMPLATE = "/work/source/NeuronTemplate.hoc"
DEFAULT_BIOPHYS = "/work/source/biophys_HL5BN1.hoc"
DEFAULT_MORPHOLOGY = "/work/source/HL5BN1.swc"


def _biophys_procedure(path):
    """Name of the hoc procedure defined by a ``biophys_<CELL>.hoc`` file (its stem)."""
    stem = Path(str(path)).name
    if not stem.startswith("biophys_") or not stem.endswith(".hoc"):
        raise ValueError("Biophysics file must be named biophys_<CELL>.hoc.")
    return stem[:-len(".hoc")]


def _existing_sections(cell, name):
    """Live sections of a template section array such as ``cell.myelin``.

    The template's ``init`` deletes every declared section and recreates only some of
    them (HL5BN1 takes ``delete_axon_BPO`` and keeps no myelin). Reading a deleted
    section from Python raises a hoc error that is not converted and aborts the
    process (exit 139), so each index is checked with ``section_exists`` first.
    """
    sections = []
    while h.section_exists(name, len(sections), cell):
        sections.append(getattr(cell, name)[len(sections)])
    return sections


SCALE_MECHANISMS = ("NaTg", "Kv3_1", "SK")
SCALE_REGIONS = ("soma", "axon", "dend", "apic", "all")


def _parse_scale(text):
    """Parse ``MECHANISM:REGION:FACTOR`` (mirrors ``h01_l2_regional_density.parse_regional_density``)."""
    parts = str(text).split(":")
    if len(parts) != 3:
        raise ValueError("Scale must be MECHANISM:REGION:FACTOR.")
    if parts[0] not in SCALE_MECHANISMS:
        raise ValueError("Scale mechanism must be one of "+", ".join(SCALE_MECHANISMS)+".")
    if parts[1] not in SCALE_REGIONS:
        raise ValueError("Scale region must be one of "+", ".join(SCALE_REGIONS)+".")
    factor = float(parts[2])
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Scale factor must be nonnegative and finite.")
    return {"mechanism": parts[0], "region": parts[1], "factor": factor}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--current-na", type=float, required=True)
parser.add_argument("--bias-na", type=float, default=0.)
parser.add_argument("--axon-calcium-decay-ms", type=float)
parser.add_argument("--axon-calcium-gamma", type=float)
parser.add_argument("--observe-spike-currents", action="store_true")
parser.add_argument("--observe-charge-balance", action="store_true")
parser.add_argument("--sodium-h-tau-factor", type=float, default=1.)
parser.add_argument("--sodium-h-recovery-factor", type=float)
parser.add_argument("--sodium-h-slope-mv", type=float)
parser.add_argument("--somatic-calva-factor", type=float, default=1.)
parser.add_argument("--somatic-kv3-factor", type=float, default=1.)
parser.add_argument("--somatic-kv3-tau-factor", type=float, default=1.)
parser.add_argument("--somatic-kv3-close-factor", type=float)
parser.add_argument("--dt-ms", type=float, default=.025)
parser.add_argument("--initial-mv", type=float, default=-80.)
parser.add_argument("--scale-conductance", choices=("NaTg", "Kv3_1", "SK"))
parser.add_argument("--conductance-factor", type=float, default=1.)
parser.add_argument("--conductance-region", choices=("all", "soma", "axon"), default="all")
parser.add_argument("--scale", action="append", default=[], metavar="MECH:REGION:FACTOR",
                    help="repeatable regional density scale applied after the single --scale-conductance slot; "
                         "mechanism NaTg, Kv3_1 or SK; region soma, axon, dend, apic or all")
parser.add_argument("--cvode-atol", type=float)
parser.add_argument("--passive", action="store_true")
parser.add_argument("--nseg-factor", type=int, default=1)
parser.add_argument("--unselected-nseg-factor", type=int, default=1)
parser.add_argument("--refine-region", choices=("all", "soma", "axon", "dendrites"), default="all")
parser.add_argument("--duration-ms", type=float, default=1500.)
parser.add_argument("--output", required=True)
parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="in-container NeuronTemplate.hoc path")
parser.add_argument("--biophys", default=DEFAULT_BIOPHYS,
                    help="in-container biophys_<CELL>.hoc path; the procedure name is the file stem")
parser.add_argument("--morphology", default=DEFAULT_MORPHOLOGY,
                    help="in-container SWC path; the template reads the cell name after 'morphologies/'")
parser.add_argument("--candidate-json", type=Path,
                    help="JSON object of flag names to values used as defaults; explicit flags override")
preliminary, _ = parser.parse_known_args()
candidate_record = None
if preliminary.candidate_json is not None:
    candidate_values = json.loads(preliminary.candidate_json.read_text())
    known = vars(parser.parse_args(["--current-na", "0", "--output", "x"]))
    unknown = [k for k in candidate_values if k not in known] if isinstance(candidate_values, dict) else ["<not an object>"]
    if unknown:
        parser.error("Candidate JSON must map known flag names to values: " + ", ".join(unknown))
    parser.set_defaults(**candidate_values)
    candidate_record = {"file": preliminary.candidate_json.name, "values": candidate_values,
                        "sha256": hashlib.sha256(preliminary.candidate_json.read_bytes()).hexdigest()}
args = parser.parse_args()
if args.axon_calcium_gamma is not None and (not np.isfinite(args.axon_calcium_gamma) or not 0 <= args.axon_calcium_gamma <= 1):
    parser.error("Axonal calcium gamma must be finite and between zero and one.")
if args.somatic_kv3_close_factor is not None and (not np.isfinite(args.somatic_kv3_close_factor) or args.somatic_kv3_close_factor <= 0):
    parser.error("Somatic Kv3 closing factor must be positive and finite.")
if not np.isfinite(args.somatic_kv3_tau_factor) or args.somatic_kv3_tau_factor <= 0:
    parser.error("Somatic Kv3 time factor must be positive and finite.")
if not np.isfinite(args.somatic_kv3_factor) or args.somatic_kv3_factor < 0:
    parser.error("Somatic Kv3 factor must be nonnegative and finite.")
if not np.isfinite(args.somatic_calva_factor) or args.somatic_calva_factor < 0:
    parser.error("Somatic Ca_LVA factor must be nonnegative and finite.")
if args.sodium_h_slope_mv is not None and (not np.isfinite(args.sodium_h_slope_mv) or args.sodium_h_slope_mv <= 0):
    parser.error("Sodium inactivation slope must be positive and finite.")
if args.axon_calcium_decay_ms is not None and (not np.isfinite(args.axon_calcium_decay_ms) or args.axon_calcium_decay_ms <= 0):
    parser.error("Calcium removal time must be positive and finite.")
if args.sodium_h_recovery_factor is not None and (not np.isfinite(args.sodium_h_recovery_factor) or args.sodium_h_recovery_factor <= 0):
    parser.error("Sodium recovery time factor must be positive and finite.")
if not np.isfinite(args.sodium_h_tau_factor) or args.sodium_h_tau_factor <= 0:
    parser.error("Sodium h time factor must be positive and finite.")
if not np.isfinite(args.conductance_factor) or args.conductance_factor < 0:
    parser.error("Conductance factor must be nonnegative and finite.")
if args.scale_conductance is None and args.conductance_factor != 1.:
    parser.error("A non-unit conductance factor requires a named mechanism.")
if args.nseg_factor < 1 or args.nseg_factor % 2 != 1:
    parser.error("Segment factor must be a positive odd integer.")
if args.unselected_nseg_factor < 1 or args.unselected_nseg_factor % 2 != 1:
    parser.error("Unselected segment factor must be a positive odd integer.")
if not np.isfinite([args.current_na, args.bias_na, args.dt_ms, args.initial_mv]).all() or args.dt_ms <= 0:
    parser.error("Current must be finite and time step must be positive.")
if not np.isfinite(args.duration_ms) or args.duration_ms <= 0:
    parser.error("Duration must be positive and finite.")
if args.cvode_atol is not None and (not np.isfinite(args.cvode_atol) or args.cvode_atol <= 0):
    parser.error("CVode absolute tolerance must be positive and finite.")
try:
    regional_scales = [_parse_scale(text) for text in args.scale]
    biophys_procedure = _biophys_procedure(args.biophys)
except ValueError as error:
    parser.error(str(error))

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")
for required in (args.template, args.biophys, args.morphology):
    if not Path(required).is_file():
        raise SystemExit(f"Donor file not found: {required}")
h.load_file(args.template)
h.load_file(args.biophys)
cell = h.NeuronTemplate(args.morphology)
getattr(h, biophys_procedure)(cell)
if args.axon_calcium_gamma is not None:
    for section in cell.axonal:
        section.gamma_CaDynamics = args.axon_calcium_gamma
if args.somatic_kv3_close_factor is not None:
    for section in cell.somatic:
        if not hasattr(section, "m_close_factor_Kv3_1"):
            raise RuntimeError("Kv3 phase intervention requires the isolated phase mechanisms.")
        section.m_close_factor_Kv3_1 = args.somatic_kv3_close_factor
if args.somatic_kv3_tau_factor != 1.:
    for section in cell.somatic:
        if not hasattr(section, "m_tau_factor_Kv3_1"):
            raise RuntimeError("Kv3 timing intervention requires the isolated kinetics mechanisms.")
        section.m_tau_factor_Kv3_1 = args.somatic_kv3_tau_factor
if args.somatic_kv3_factor != 1.:
    for section in cell.somatic:
        section.gbar_Kv3_1 *= args.somatic_kv3_factor
if args.somatic_calva_factor != 1.:
    for section in cell.somatic:
        section.gbar_Ca_LVA *= args.somatic_calva_factor
if args.sodium_h_slope_mv is not None:
    for section in cell.all:
        if "NaTg" in section.psection()["density_mechs"]:
            section.slopeh_NaTg = args.sodium_h_slope_mv
if args.axon_calcium_decay_ms is not None:
    for section in cell.axonal:
        section.decay_CaDynamics = args.axon_calcium_decay_ms
if args.sodium_h_recovery_factor is not None:
    for section in cell.all:
        if "NaTg" in section.psection()["density_mechs"]:
            if not hasattr(section, "h_recovery_factor_NaTg"):
                raise RuntimeError("This intervention requires the isolated recovery mechanisms.")
            section.h_recovery_factor_NaTg = args.sodium_h_recovery_factor
if args.sodium_h_tau_factor != 1.:
    for section in cell.all:
        if "NaTg" in section.psection()["density_mechs"]:
            if not hasattr(section, "h_tau_factor_NaTg"):
                raise RuntimeError("This intervention requires the isolated inactivation mechanisms.")
            section.h_tau_factor_NaTg = args.sodium_h_tau_factor
if args.scale_conductance is not None:
    for section in cell.all:
        family = section.name().split(".", 1)[1].split("[", 1)[0]
        selected = args.conductance_region == "all" or family == args.conductance_region
        if selected and args.scale_conductance in section.psection()["density_mechs"]:
            parameter = "gbar_"+args.scale_conductance
            setattr(section, parameter, getattr(section, parameter)*args.conductance_factor)
for scale in regional_scales:
    touched = 0
    for section in cell.all:
        family = section.name().split(".", 1)[1].split("[", 1)[0]
        selected = scale["region"] == "all" or family == scale["region"]
        if selected and scale["mechanism"] in section.psection()["density_mechs"]:
            parameter = "gbar_"+scale["mechanism"]
            setattr(section, parameter, getattr(section, parameter)*scale["factor"])
            touched += 1
    if not touched:
        raise RuntimeError(f"Scale {scale} matched no section carrying the mechanism.")
for section in cell.all:
    family = section.name().split(".", 1)[1].split("[", 1)[0]
    selected = (args.refine_region == "all" or family == args.refine_region
                or (args.refine_region == "dendrites" and family in ("dend", "apic")))
    if selected:
        section.nseg *= args.nseg_factor
    else:
        section.nseg *= args.unselected_nseg_factor
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
bias = h.IClamp(cell.soma[0](.5))
bias.delay = 0.
bias.dur = args.duration_ms+1.
bias.amp = args.bias_na
t = h.Vector().record(h._ref_t)
v = h.Vector().record(cell.soma[0](.5)._ref_v)
current = h.Vector().record(clamp._ref_i)
bias_current = h.Vector().record(bias._ref_i)
axon_v = h.Vector().record(cell.axon[0](.5)._ref_v)
calcium = h.Vector().record(cell.soma[0](.5)._ref_cai) if not args.passive else []
sk = h.Vector().record(cell.soma[0](.5).SK._ref_z) if not args.passive else []
ih = h.Vector().record(cell.soma[0](.5).Ih._ref_m) if not args.passive else []
nap_h = (h.Vector().record(cell.soma[0](.5).Nap._ref_h)
         if not args.passive and "Nap" in cell.soma[0].psection()["density_mechs"] else [])
spike_probes = {}
balance_geometry = None
if (args.observe_spike_currents or args.observe_charge_balance) and not args.passive:
    soma = cell.soma[0](.5)
    axon_segment = cell.axon[0](.5)
    spike_probes["axon_calcium_mm"] = h.Vector().record(axon_segment._ref_cai)
    spike_probes["axon_sk_gate"] = h.Vector().record(axon_segment.SK._ref_z)
    spike_probes["axon_sk_current_ma_cm2"] = h.Vector().record(axon_segment.SK._ref_ik)
    for mechanism, field in (("NaTg", "ina"), ("Nap", "ina"),
                             ("K_P", "ik"), ("K_T", "ik"), ("Kv3_1", "ik"),
                             ("Im", "ik"), ("SK", "ik"),
                             ("Ca_HVA", "ica"), ("Ca_LVA", "ica")):
        spike_probes[mechanism+"_current_ma_cm2"] = h.Vector().record(
            getattr(getattr(soma, mechanism), "_ref_"+field))
    for mechanism, field in (("NaTg", "m"), ("NaTg", "h"), ("Kv3_1", "m")):
        spike_probes[mechanism+"_"+field] = h.Vector().record(
            getattr(getattr(soma, mechanism), "_ref_"+field))
    for field in ("ina", "ik", "ica"):
        spike_probes[field+"_ma_cm2"] = h.Vector().record(getattr(soma, "_ref_"+field))
if args.observe_charge_balance:
    section = cell.soma[0]
    soma = section(.5)
    assert section.nseg >= 3 and section.nseg % 2 == 1
    assert section.parentseg() is None
    middle = section.nseg // 2
    left = section((middle-.5)/section.nseg)
    right = section((middle+1.5)/section.nseg)
    neighbours = [(left, soma.ri()), (right, right.ri())]
    for child in section.children():
        parent = child.parentseg()
        if int(parent.x*section.nseg) == middle:
            assert child.orientation() == 0.
            first = child(.5/child.nseg)
            neighbours.append((first, first.ri()))
    balance_geometry = {"area_um2": soma.area(), "cm_uf_cm2": soma.cm,
                        "location": str(soma), "neighbours": []}
    for index, (neighbour, resistance) in enumerate(neighbours):
        assert 0 < resistance < 1e20
        key = f"axial_neighbour_{index}_mv"
        spike_probes[key] = h.Vector().record(neighbour._ref_v)
        balance_geometry["neighbours"].append({"location": str(neighbour),
                                               "resistance_mohm": resistance, "voltage_key": key})
    spike_probes["capacitive_current_ma_cm2"] = h.Vector().record(soma._ref_i_cap)
    spike_probes["leak_current_ma_cm2"] = h.Vector().record(soma.pas._ref_i)
    if not args.passive:
        spike_probes["Ih_current_ma_cm2"] = h.Vector().record(soma.Ih._ref_ihcn)
geometry = [{"name": sec.name(), "length_um": sec.L, "diameter_um": sec.diam,
             "nseg": sec.nseg, "area_um2": sum(seg.area() for seg in sec),
             "ra_ohm_cm": sec.Ra, "cm_uf_cm2": sec.cm,
             "parent": None if sec.parentseg() is None else str(sec.parentseg())}
            for sec in list(cell.all)+[m for m in _existing_sections(cell, "myelin") if m.parentseg() is not None]]
h.finitialize(args.initial_mv)
h.continuerun(args.duration_ms)
times, voltage, applied, axon = map(np.asarray, (t, v, current, axon_v))
assert np.isfinite(voltage).all() and np.isfinite(axon).all()
peaks, _ = find_peaks(voltage, height=0., prominence=40.)
peaks = peaks[(times[peaks] >= 270.) & (times[peaks] < 1270.)]
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output.with_suffix(".npz"), time_ms=times, voltage_mv=voltage,
                    current_na=applied, axon_voltage_mv=axon,
                    bias_current_na=np.asarray(bias_current),
                    total_current_na=applied+np.asarray(bias_current),
                    **{key: np.asarray(value) for key, value in spike_probes.items()},
                    calcium_mm=np.asarray(calcium), sk_gate=np.asarray(sk),
                    ih_gate=np.asarray(ih), nap_h_gate=np.asarray(nap_h))
report = {"neuron_version": neuron.__version__, "source_commit": "82cdd91bc93942ba19315371330a2412e064baf5",
          "active_channels": not args.passive,
          "conductance_intervention": {"mechanism": args.scale_conductance, "factor": args.conductance_factor,
                                       "region": args.conductance_region},
          "regional_scales": regional_scales,
          "nseg_factor": args.nseg_factor,
          "unselected_nseg_factor": args.unselected_nseg_factor,
          "refine_region": args.refine_region,
          "model": ("ModelDB267587 released HL5BN1 circuit cell, original template and mechanisms"
                    if args.biophys == DEFAULT_BIOPHYS and args.morphology == DEFAULT_MORPHOLOGY
                    else f"ModelDB267587 released {biophys_procedure} circuit cell, original template and mechanisms"),
          "donor": {"template": args.template, "biophys": args.biophys, "morphology": args.morphology,
                    "procedure": biophys_procedure,
                    "sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for name, path in (("template", args.template), ("biophys", args.biophys),
                                                  ("morphology", args.morphology))}},
          "current_na": args.current_na, "dt_ms": args.dt_ms, "temperature_c": 34.,
          "bias_na": args.bias_na, "bias_on_ms": 0.,
          "axon_calcium_decay_ms": args.axon_calcium_decay_ms,
          "axon_calcium_gamma": args.axon_calcium_gamma,
          "spike_current_probes": args.observe_spike_currents,
          "charge_balance_geometry": balance_geometry,
          "sodium_h_tau_factor": args.sodium_h_tau_factor,
          "sodium_h_recovery_factor": args.sodium_h_recovery_factor,
          "sodium_h_slope_mv": args.sodium_h_slope_mv,
          "somatic_calva_factor": args.somatic_calva_factor,
          "somatic_kv3_factor": args.somatic_kv3_factor,
          "somatic_kv3_tau_factor": args.somatic_kv3_tau_factor,
          "somatic_kv3_close_factor": args.somatic_kv3_close_factor,
          "channel_current_convention": "NEURON outward positive, mA/cm2",
          "integration": {"method": "CVode" if args.cvode_atol is not None else "fixed step",
                          "cvode_atol": args.cvode_atol},
          "candidate_json": candidate_record,
          "initial_voltage_mv": args.initial_mv, "stimulus_on_ms": 270., "stimulus_off_ms": 1270.,
          "duration_ms": args.duration_ms, "synaptic_background": "none",
          "mechanism_library": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in sorted(Path.cwd().glob("mod/*.mod"))+sorted(Path.cwd().glob("x86_64/libnrnmech.so"))},
          "sample_convention": "NEURON recorded time, includes initial state at t=0",
          "spike_times_ms": times[peaks].tolist(), "spike_peaks_mv": voltage[peaks].tolist(),
          "baseline_mean_mv": _time_average(times, voltage, 200., 270.),
          "baseline_mean_convention": "time integral of piecewise-linear voltage over 200-270 ms, divided by 70 ms",
          "geometry": geometry,
          "qualification": "independent reference response; comparison with human data and BrainCell transfer pending"}
output.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "geometry"}))
