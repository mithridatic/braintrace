"""Isolate the PV transfer drift between simulators with a 2x2 half-split.

Elements: A mesh (BrainCell MaxCVLen versus counts copied from NEURON) and
B integration (NEURON CVode versus fixed step at the BrainCell dt). Everything
else is asserted equal before any comparison. See the specification
``docs/specs/2026-09-05-h01-transfer-isolation.md``.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_ei_spike_transfer import compare_spike_transfer

FOLDER = Path(__file__).parent
PREFIX = "h01-pv-transfer-isolation-"
STEMS = {name: PREFIX+name.replace("_", "-") for name in
         ("neuron_cvode", "neuron_fixed", "neuron_fixed_halfdt",
          "braincell_maxcv", "braincell_matched", "braincell_matched_halfdt")}
WINDOW_MS = (270., 330.)
NEURON_HELD = ("source_commit", "neuron_version", "active_channels", "conductance_intervention",
               "nseg_factor", "unselected_nseg_factor", "refine_region", "current_na", "temperature_c",
               "bias_na", "axon_calcium_decay_ms", "axon_calcium_gamma", "sodium_h_tau_factor",
               "sodium_h_recovery_factor", "sodium_h_slope_mv", "somatic_calva_factor", "somatic_kv3_factor",
               "somatic_kv3_tau_factor", "somatic_kv3_close_factor", "initial_voltage_mv", "stimulus_on_ms",
               "duration_ms", "mechanism_library")
BRAINCELL_HELD = ("source_commit", "profile", "duration_ms", "solver", "active_channels", "current_na",
                  "temperature_c", "initial_voltage_mv", "stimulus_on_ms", "sample_convention")
CROSS_HELD = ("current_na", "temperature_c", "initial_voltage_mv", "stimulus_on_ms", "duration_ms")


def neuron_counts_by_branch(geometry):
    """Map ``NeuronTemplate[0].soma[0]`` style names to BrainCell branch names and counts."""
    counts = {}
    for section in geometry:
        name = section["name"].split(".", 1)[-1].replace("[", "_").replace("]", "")
        if name in counts:
            raise ValueError("Duplicate section name: "+section["name"])
        counts[name] = int(section["nseg"])
    return counts


def assert_held_equal(metas, neuron_json_sha256):
    """Raise when any element other than the two swapped ones differs."""
    base = metas["neuron_cvode"]
    for name in ("neuron_fixed", "neuron_fixed_halfdt"):
        for key in NEURON_HELD:
            if metas[name].get(key) != base.get(key):
                raise ValueError(f"NEURON field {key} differs in {name}")
        if [s["nseg"] for s in metas[name]["geometry"]] != [s["nseg"] for s in base["geometry"]]:
            raise ValueError(f"NEURON mesh differs in {name}")
    if base["integration"]["method"] != "CVode" or metas["neuron_fixed"]["integration"]["method"] != "fixed step":
        raise ValueError("neuron_cvode must use CVode and neuron_fixed a fixed step")
    cell = metas["braincell_maxcv"]
    for name in ("braincell_matched", "braincell_matched_halfdt"):
        for key in BRAINCELL_HELD:
            if metas[name].get(key) != cell.get(key):
                raise ValueError(f"BrainCell field {key} differs in {name}")
        mesh = metas[name]["mesh"]
        if mesh["policy"] != "CVPerBranchList" or mesh["source_sha256"] != neuron_json_sha256:
            raise ValueError(f"{name} mesh must be copied from the neuron_cvode run")
    if cell["mesh"]["policy"] != "MaxCVLen" or cell["solver"] != "staggered":
        raise ValueError("braincell_maxcv must use MaxCVLen and the staggered solver")
    if metas["braincell_matched"]["mesh"]["cv_per_branch"] != metas["braincell_matched_halfdt"]["mesh"]["cv_per_branch"]:
        raise ValueError("matched meshes differ between dt levels")
    for key in CROSS_HELD:
        if base.get(key) != cell.get(key):
            raise ValueError(f"cross-simulator field {key} differs")
    dt = metas["neuron_fixed"]["dt_ms"]
    if not (cell["dt_ms"] == metas["braincell_matched"]["dt_ms"] == dt):
        raise ValueError("fixed-step dt must equal the BrainCell dt")
    for name in ("neuron_fixed_halfdt", "braincell_matched_halfdt"):
        if not np.isclose(metas[name]["dt_ms"], dt/2., rtol=1e-12, atol=0.):
            raise ValueError(f"{name} must use half the shared dt")
    return {"dt_ms": dt, "half_dt_ms": dt/2., "neuron_nseg": [s["nseg"] for s in base["geometry"]]}


def assert_matched_mesh(neuron_meta, braincell_meta):
    """Raise unless every BrainCell branch count equals the NEURON section count."""
    counts = neuron_counts_by_branch(neuron_meta["geometry"])
    mesh = braincell_meta["mesh"]
    mismatch = [(b, counts.get(b), c) for b, c in zip(mesh["branches"], mesh["cv_per_branch"]) if counts.get(b) != c]
    if mismatch or len(mesh["branches"]) != len(counts):
        raise ValueError(f"matched mesh differs from NEURON counts: {mismatch}")
    return dict(zip(mesh["branches"], mesh["cv_per_branch"]))


def cell_result(reference, actual):
    """Event gates over the window plus the signed rise errors at events 7 and 8."""
    result = compare_spike_transfer(reference, actual, start_ms=WINDOW_MS[0], stop_ms=WINDOW_MS[1])
    rises = [row["rise_crossing_ms"] for row in result["signed_errors"]]
    result["rise_error_event_7_ms"] = rises[6] if len(rises) > 6 else None
    result["rise_error_event_8_ms"] = rises[7] if len(rises) > 7 else None
    result["max_abs_rise_error_ms"] = max(map(abs, rises)) if rises else None
    return result


def decide(passed):
    """Name the literal prediction pattern from the four cell outcomes."""
    matched = passed["matched_cvode"] and passed["matched_fixed"]
    maxcv = passed["maxcv_cvode"] or passed["maxcv_fixed"]
    fixed = passed["maxcv_fixed"] and passed["matched_fixed"]
    cvode = passed["maxcv_cvode"] or passed["matched_cvode"]
    if matched and not maxcv:
        return "mesh"
    if fixed and not cvode:
        return "integration"
    if not any(passed.values()):
        return "time_level_open"
    if passed["matched_fixed"] and sum(passed.values()) == 1:
        return "interaction"
    return "no_prediction_matched"


def main():
    """Assert the held-equal design, run the six comparisons, and write the audit."""
    metas, traces, hashes = {}, {}, {}
    for name, stem in STEMS.items():
        json_path, npz_path = FOLDER/(stem+".json"), FOLDER/(stem+".npz")
        metas[name] = json.loads(json_path.read_text())
        hashes[stem+".json"] = hashlib.sha256(json_path.read_bytes()).hexdigest()
        hashes[stem+".npz"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        with np.load(npz_path) as data:
            traces[name] = {"time_ms": np.asarray(data["time_ms"]), "voltage_mv": np.asarray(data["voltage_mv"])}
    held = assert_held_equal(metas, hashes[STEMS["neuron_cvode"]+".json"])
    held["matched_counts"] = assert_matched_mesh(metas["neuron_cvode"], metas["braincell_matched"])
    isolation = {"neuron": cell_result(traces["neuron_fixed"], traces["neuron_fixed_halfdt"]),
                 "braincell": cell_result(traces["braincell_matched"], traces["braincell_matched_halfdt"])}
    gate_valid = all(row["passed"] for row in isolation.values())
    cells = {f"{mesh}_{scheme}": cell_result(traces["neuron_"+scheme], traces["braincell_"+mesh])
             for mesh in ("maxcv", "matched") for scheme in ("cvode", "fixed")}
    passed = {name: row["passed"] for name, row in cells.items()}
    report = {"window_ms": list(WINDOW_MS), "inputs_sha256": hashes, "held_equal": held,
              "reversible_isolation": isolation, "gate_valid": gate_valid, "cells": cells,
              "passed": passed, "decision": decide(passed) if gate_valid else "gate_invalid",
              "qualification": "Numerical transfer only; identifies no channel cause and no human waveform validity."}
    (FOLDER/(PREFIX+"audit.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps({"decision": report["decision"], "gate_valid": gate_valid, "passed": passed,
                      "event_8": {k: v["rise_error_event_8_ms"] for k, v in cells.items()}}))


if __name__ == "__main__":
    main()
