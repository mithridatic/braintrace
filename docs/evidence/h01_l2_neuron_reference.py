"""Reproduce the retrieved Allen layer-2 perisomatic fit in NEURON."""

import argparse
import hashlib
import json
import time as wall_clock
from pathlib import Path

import numpy as np
import neuron
from neuron import h
from h01_l2_input import stimulus_current
from h01_l2_mesh import section_segments
from h01_l2_calcium import calcium_removal_fit
from h01_l2_sodium_density import sodium_density_fit
from h01_l2_ih_density import ih_density_fit
from h01_l2_leak import leak_fit
from h01_l2_ih_location import distribute_ih
from h01_l2_leak_reversal import shift_leak_reversal
from h01_l2_recording import right_limit_recording
from h01_l2_regional_density import REGIONS, parse_regional_density, regional_density_fit
from h01_l2_insert_density import insert_density_fit, parse_insert_density


def parse_capacitance_factor(text):
    """Parse ``REGION:FACTOR`` for a membrane capacitance scaling."""
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("Capacitance factor must be REGION:FACTOR.")
    region, factor = parts[0], float(parts[1])
    if region not in REGIONS+("all",):
        raise ValueError("Region must be one of soma, axon, dend, apic, or all.")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Capacitance factor must be positive and finite.")
    return {"region": region, "factor": factor}


def axial_probes(section, vectors, units):
    """Record every electrical neighbour of the soma midpoint for an offline axial balance.

    Mirrors ``h01_pv_neuron_reference.py --observe-charge-balance``: neighbour
    voltages plus the axial resistance to each, so that the axial current is
    ``(V_neighbour - V_soma) / R`` in nA when R is in megohms.
    """
    assert section.nseg >= 3 and section.nseg % 2 == 1, "soma nseg must be odd and at least 3"
    soma = section(.5)
    middle = section.nseg // 2
    left = section((middle-.5)/section.nseg)
    right = section((middle+1.5)/section.nseg)
    neighbours = [(left, soma.ri()), (right, right.ri())]
    for child in section.children():
        if int(child.parentseg().x*section.nseg) == middle:
            first = child(.5/child.nseg) if child.orientation() == 0. else child(1.-.5/child.nseg)
            neighbours.append((first, first.ri()))
    geometry = {"area_um2": soma.area(), "cm_uf_cm2": soma.cm, "location": str(soma), "neighbours": []}
    for index, (neighbour, resistance) in enumerate(neighbours):
        assert 0 < resistance < 1e20
        key = f"axial_neighbour_{index}_mv"
        vectors[key] = h.Vector().record(neighbour._ref_v)
        units[key] = "mV; electrical neighbour of soma(0.5)"
        geometry["neighbours"].append({"location": str(neighbour), "resistance_mohm": resistance,
                                       "voltage_key": key})
    vectors["soma_pas_i_ma_cm2"] = vectors.get("soma_pas_ma_cm2", h.Vector().record(soma.pas._ref_i))
    return geometry


def main():
    """Load source parameters and save one compiled NEURON rollout.

    Returns
    -------
    None
        Write the voltage, input, and setup evidence to the requested paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep", type=int, choices=(43, 50, 53), default=50)
    parser.add_argument("--dt-ms", type=float, default=.005)
    parser.add_argument("--include-recorded-bias", action="store_true")
    parser.add_argument("--nseg-factor", type=int, default=1)
    parser.add_argument("--calcium-decay-factor", type=float, default=1.)
    parser.add_argument("--stop-ms", type=float)
    parser.add_argument("--cvode-atol", type=float)
    parser.add_argument("--sodium-recovery-factor", type=float)
    parser.add_argument("--sodium-opening-factor", type=float)
    parser.add_argument("--kv3-closing-factor", type=float)
    parser.add_argument("--sodium-density-factor", type=float, default=1.)
    parser.add_argument("--record-soma-currents", action="store_true")
    parser.add_argument("--ih-density-factor", type=float, default=1.)
    parser.add_argument("--leak-factor", type=float, default=1.)
    parser.add_argument("--distribute-ih", action="store_true")
    parser.add_argument("--leak-reversal-shift-mv", type=float, default=0.)
    parser.add_argument("--regional-density", action="append", default=[],
                        help="MECHANISM:REGION:FACTOR, repeatable; REGION is soma, axon, dend, apic, or all")
    parser.add_argument("--capacitance-factor", action="append", default=[],
                        help="REGION:FACTOR, repeatable; scales membrane capacitance of soma, axon, dend, apic, or all")
    parser.add_argument("--insert-density", action="append", default=[],
                        help="MECHANISM:REGION:VALUE, repeatable; adds a mechanism at an absolute density (S/cm2) "
                             "to a region that has none, with the soma's reversal potentials")
    parser.add_argument("--candidate-json", type=Path,
                        help="JSON object of flag names to values used as defaults; explicit flags override")
    preliminary, _ = parser.parse_known_args()
    candidate_record = None
    if preliminary.candidate_json is not None:
        candidate_values = json.loads(preliminary.candidate_json.read_text())
        unknown = [k for k in candidate_values if k not in vars(parser.parse_args([
            "--cache", "x", "--output", "y"]))]
        if not isinstance(candidate_values, dict) or unknown:
            parser.error("Candidate JSON must map known flag names to values: " + ", ".join(unknown))
        parser.set_defaults(**candidate_values)
        candidate_record = {"file": preliminary.candidate_json.name, "values": candidate_values,
                            "sha256": hashlib.sha256(preliminary.candidate_json.read_bytes()).hexdigest()}
    args = parser.parse_args()
    try:
        regional = [parse_regional_density(text) for text in args.regional_density]
        capacitance = [parse_capacitance_factor(text) for text in args.capacitance_factor]
        inserted = [parse_insert_density(text) for text in args.insert_density]
    except ValueError as error:
        parser.error(str(error))
    if args.kv3_closing_factor is not None and (not np.isfinite(args.kv3_closing_factor)
                                               or args.kv3_closing_factor <= 0):
        parser.error("Kv3-closing factor must be positive and finite.")
    if args.sodium_opening_factor is not None and (not np.isfinite(args.sodium_opening_factor)
                                                  or args.sodium_opening_factor <= 0):
        parser.error("Sodium-opening factor must be positive and finite.")
    if not np.isfinite(args.leak_reversal_shift_mv):
        parser.error("Leak reversal shift must be finite.")
    if not np.isfinite(args.leak_factor) or args.leak_factor <= 0:
        parser.error("Leak factor must be positive and finite.")
    if not np.isfinite(args.ih_density_factor) or args.ih_density_factor <= 0:
        parser.error("Ih-density factor must be positive and finite.")
    if not np.isfinite(args.sodium_density_factor) or args.sodium_density_factor <= 0:
        parser.error("Sodium-density factor must be positive and finite.")
    if not np.isfinite(args.dt_ms) or args.dt_ms <= 0:
        parser.error("Time step must be positive and finite.")
    if not np.isfinite(args.calcium_decay_factor) or args.calcium_decay_factor <= 0:
        parser.error("Calcium-removal factor must be positive and finite.")
    if args.stop_ms is not None and (not np.isfinite(args.stop_ms) or args.stop_ms < 2020.):
        parser.error("Observation endpoint must be finite and at least 2020 ms.")
    if args.cvode_atol is not None and (not np.isfinite(args.cvode_atol) or args.cvode_atol <= 0):
        parser.error("CVode tolerance must be positive and finite.")
    if args.sodium_recovery_factor is not None and (not np.isfinite(args.sodium_recovery_factor)
                                                   or args.sodium_recovery_factor <= 0):
        parser.error("Sodium-recovery factor must be positive and finite.")
    try:
        section_segments(30., args.nseg_factor)
    except ValueError as error:
        parser.error(str(error))
    fit_path = args.cache / "541563728_fit.json"
    assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == (
        "2ceca2317ccbd586adde4b1e72507ad4bdf2fc10fc26ad4b281484324dd5f0c3")
    source_fit = json.loads(fit_path.read_text())
    fit = calcium_removal_fit(source_fit, args.calcium_decay_factor)
    fit = sodium_density_fit(fit, args.sodium_density_factor)
    fit = ih_density_fit(fit, args.ih_density_factor)
    fit = leak_fit(fit, args.leak_factor)
    fit = shift_leak_reversal(fit, args.leak_reversal_shift_mv)
    for item in inserted:
        fit = insert_density_fit(fit, item["mechanism"], item["region"], item["value"])
    for item in regional:
        fit = regional_density_fit(fit, item["mechanism"], item["region"], item["factor"])
    source_decay = next(row["value"] for row in source_fit["genome"]
                        if row["section"] == "soma" and row["name"] == "decay_CaDynamics")
    waveform_path = args.cache / f"sweep-{args.sweep}.npz"
    source = np.load(waveform_path)
    time = source["time_ms"]
    command = source["command_current_na"]
    applied_command = stimulus_current(command, source["bias_current_na"],
                                       include_bias=args.include_recorded_bias)
    assert time.shape == command.shape and np.isfinite(command).all()
    assert np.isfinite(time).all() and time[0] == 0
    assert np.allclose(np.diff(time), .02, rtol=0, atol=1e-9)
    assert np.isclose(.02 / args.dt_ms, round(.02 / args.dt_ms))
    stop_ms = float(time[-1]) if args.stop_ms is None else args.stop_ms
    if stop_ms > time[-1]:
        parser.error("Observation endpoint exceeds the source waveform.")

    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    swc = h.Import3d_SWC_read()
    swc.input(str(args.cache / "source-model/morphology.swc"))
    importer = h.Import3d_GUI(swc, 0)
    h("objref this")
    importer.instantiate(h.this)
    h("soma[0] area(0.5)")
    for sec in list(h.allsec()):
        sec.nseg = section_segments(sec.L, args.nseg_factor)
        if sec.name().startswith("axon"):
            h.delete_section(sec=sec)
    h("create axon[2]")
    for sec in h.axon:
        sec.L, sec.diam, sec.nseg = 30., 1., section_segments(30., args.nseg_factor)
    h.axon[0].connect(h.soma[0], .5, 0.)
    h.axon[1].connect(h.axon[0], 1., 0.)
    h.define_shape()

    passive, conditions = fit["passive"][0], fit["conditions"][0]
    sections = list(h.allsec())
    ih_distribution = None
    if args.distribute_ih:
        areas = {region: sum(seg.area() for sec in sections if sec.name().startswith(region + "[")
                            for seg in sec) for region in ("soma", "dend", "apic")}
        fit, ih_distribution = distribute_ih(fit, areas)
    for sec in sections:
        sec.Ra = passive["ra"]
        sec.insert("pas")
        sec.e_pas = passive["e_pas"]
        assert all(seg.e_pas == passive["e_pas"] for seg in sec)
    applied = []
    for row in passive["cm"]:
        selected = [sec for sec in sections if sec.name().split("[")[0] == row["section"]]
        assert selected, row["section"]
        for sec in selected:
            sec.cm = row["cm"]
    for item in capacitance:
        regions = REGIONS if item["region"] == "all" else (item["region"],)
        for sec in sections:
            if sec.name().split("[")[0] in regions:
                sec.cm *= item["factor"]
    for row in fit["genome"]:
        selected = [sec for sec in sections if sec.name().split("[")[0] == row["section"]]
        assert selected, row["section"]
        for sec in selected:
            if row["mechanism"]:
                sec.insert(row["mechanism"])
            setattr(sec, row["name"], row["value"])
            for segment in sec:
                assert np.isclose(getattr(segment, row["name"]), row["value"], rtol=1e-12)
        applied.append({**row, "section_count": len(selected)})
    for row in conditions["erev"]:
        for sec in sections:
            if sec.name().split("[")[0] == row["section"]:
                sec.ena, sec.ek = row["ena"], row["ek"]
    if args.sodium_recovery_factor is not None:
        for sec in sections:
            if sec.name().split("[")[0] == "soma":
                for segment in sec:
                    segment.NaTs.h_recovery_factor = args.sodium_recovery_factor
                    assert segment.NaTs.h_recovery_factor == args.sodium_recovery_factor
    if args.sodium_opening_factor is not None:
        for sec in sections:
            if sec.name().split("[")[0] == "soma":
                for segment in sec:
                    segment.NaTs.m_opening_factor = args.sodium_opening_factor
                    assert segment.NaTs.m_opening_factor == args.sodium_opening_factor
    if args.kv3_closing_factor is not None:
        for sec in sections:
            if sec.name().split("[")[0] == "soma":
                for segment in sec:
                    segment.Kv3_1.m_closing_factor = args.kv3_closing_factor
                    assert segment.Kv3_1.m_closing_factor == args.kv3_closing_factor
    h.celsius = conditions["celsius"]
    h.CVode().active(0)
    if args.cvode_atol is not None:
        h.CVode().atol(args.cvode_atol)
        h.CVode().active(1)
    h.dt = args.dt_ms
    h.steps_per_ms = 1. / args.dt_ms
    stim = h.IClamp(h.soma[0](.5))
    stim.delay, stim.dur, stim.amp = 0., 1e12, 0.
    command_vector = h.Vector(applied_command)
    command_vector.play(stim._ref_amp, .02)
    vectors = {"time_ms": h.Vector().record(h._ref_t),
               "voltage_mv": h.Vector().record(h.soma[0](.5)._ref_v),
               "applied_current_na": h.Vector().record(stim._ref_i)}
    observation_units = {}
    vectors["axon_voltage_mv"] = h.Vector().record(h.axon[1](.5)._ref_v)
    observation_units["axon_voltage_mv"] = "mV; axon[1](0.5), distal half of the 60 um stub"
    balance_geometry = None
    if args.record_soma_currents:
        soma = h.soma[0](.5)
        balance_geometry = axial_probes(h.soma[0], vectors, observation_units)
        for mechanism, field in (("NaTs", "ina"), ("Nap", "ina"), ("K_P", "ik"),
                                 ("K_T", "ik"), ("Kv3_1", "ik"), ("Im", "ik"),
                                 ("SK", "ik"), ("Ca_HVA", "ica"), ("Ca_LVA", "ica"),
                                 ("Ih", "ihcn"), ("pas", "i")):
            name = "soma_" + mechanism + "_ma_cm2"
            vectors[name] = h.Vector().record(getattr(getattr(soma, mechanism), "_ref_" + field))
            observation_units[name] = "mA/cm2; outward positive; local soma(0.5)"
        for name, pointer, unit in (("soma_icap_ma_cm2", soma._ref_i_cap, "mA/cm2"),
                                    ("soma_cai_mm", soma._ref_cai, "mM"),
                                    ("soma_Ih_m", soma.Ih._ref_m, "dimensionless"),
                                    ("soma_Im_m", soma.Im._ref_m, "dimensionless")):
            vectors[name] = h.Vector().record(pointer)
            observation_units[name] = unit + "; local soma(0.5)"
    integration_start = wall_clock.perf_counter()
    h.finitialize(conditions["v_init"])
    if args.cvode_atol is None:
        h.continuerun(stop_ms)
    else:
        h.CVode().solve(stop_ms)
    integration_seconds = wall_clock.perf_counter() - integration_start
    arrays = {key: np.asarray(value) for key, value in vectors.items()}
    if args.cvode_atol is not None:
        raw_path = args.output.parent / (args.output.name + "-raw.npz")
        np.savez_compressed(raw_path, **arrays)
        raw_report = {"solver": "CVode", "cvode_atol": args.cvode_atol,
                      "integration_seconds": integration_seconds,
                      "nonpositive_time_steps": int(np.count_nonzero(np.diff(arrays["time_ms"]) <= 0)),
                      "qualification": "Raw recording before output validation; not a qualified trace."}
        raw_path.with_suffix(".json").write_text(json.dumps(raw_report, indent=2) + "\n")
        derived = right_limit_recording(arrays["time_ms"], arrays["voltage_mv"], arrays["applied_current_na"])
        derived.update({key: arrays[key][derived["raw_sample_index"]] for key in observation_units})
        arrays = derived
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert np.all(np.diff(arrays["time_ms"]) > 0)
    assert abs(arrays["time_ms"][-1] - stop_ms) <= (args.dt_ms if args.cvode_atol is None else 1e-8)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    report = {"model_id": 626170538, "specimen_id": 541563728, "sweep": args.sweep,
              "waveform_sha256": hashlib.sha256(waveform_path.read_bytes()).hexdigest(),
              "soma_observation_units": observation_units,
              "neuron_version": neuron.__version__, "dt_ms": args.dt_ms if args.cvode_atol is None else None,
              "solver": "CVode" if args.cvode_atol is not None else "fixed step",
              "cvode_atol": args.cvode_atol, "integration_seconds": integration_seconds,
              "input": ("command plus recorded bias" if args.include_recorded_bias
                                                   else "source command only; bias not added"),
              "added_bias_na": float(source["bias_current_na"]) if args.include_recorded_bias else 0.,
              "nseg_factor": args.nseg_factor,
              "requested_stop_ms": stop_ms, "observed_stop_ms": float(arrays["time_ms"][-1]),
              "recording_convention": "raw fixed-step samples" if args.cvode_atol is None else
                  "right-limit samples with indices into separately preserved raw recording",
              "calcium_removal_intervention": {"factor": args.calcium_decay_factor,
                  "source_decay_ms": source_decay,
                  "applied_decay_ms": source_decay * args.calcium_decay_factor},
              "sodium_recovery_factor": args.sodium_recovery_factor,
              "sodium_opening_factor": args.sodium_opening_factor,
              "kv3_closing_factor": args.kv3_closing_factor,
              "sodium_density_factor": args.sodium_density_factor,
              "ih_density_factor": args.ih_density_factor,
              "leak_factor": args.leak_factor,
              "ih_distribution": ih_distribution,
              "leak_reversal_shift_mv": args.leak_reversal_shift_mv,
              "regional_density_interventions": regional,
              "insert_density_interventions": inserted,
              "capacitance_factors": capacitance,
              "charge_balance_geometry": balance_geometry,
              "candidate_json": candidate_record,
              "applied_passive_parameters": fit["passive"],
              "mechanism_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(Path.cwd().glob("*.mod"))},
              "mechanism_library_sha256": hashlib.sha256(
                  (Path.cwd() / "x86_64/libnrnmech.so").read_bytes()).hexdigest(),
              "initial_mv": conditions["v_init"], "temperature_c": h.celsius,
              "voltage_convention": "model internal voltage; compare to corrected human voltage",
              "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
              "applied_genome": applied, "sections": [
                  {"name": sec.name(), "length_um": sec.L, "nseg": sec.nseg,
                   "cm_uf_cm2": sec.cm, "ra_ohm_cm": sec.Ra,
                   "parent": sec.parentseg().sec.name() if sec.parentseg() is not None else None,
                   "area_um2": sum(seg.area() for seg in sec)} for sec in sections],
              "samples": len(arrays["time_ms"]),
              "qualification": "Source setup reproduction; physiology and convergence unqualified"}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"samples": report["samples"], "sections": len(sections),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
