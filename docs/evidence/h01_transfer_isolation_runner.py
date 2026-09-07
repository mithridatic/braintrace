"""Name, check, and score the SP2 transfer-isolation arms from a manifest.

This module launches nothing. ``--print-commands`` lists the exact command of
every arm so the user launches each as a detached job; ``--score`` reads the
finished traces, compares every event over the registered window with
``compare_spike_transfer``, and writes the decision JSON. See
``docs/specs/2026-09-07-h01-transfer-isolation.md``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from docs.evidence.h01_ei_spike_transfer import compare_spike_transfer

ARM_FIELDS = ("name", "simulator", "mesh", "current_na", "dt_ms", "duration_ms", "reference", "window")
CONTROL_MAP = (("somatic_kv3_close_factor", "Kv3_1", "m_close"), ("somatic_kv3_tau_factor", "Kv3_1", "m_open"),
               ("sodium_h_tau_factor", "NaTg", "h_close"), ("sodium_h_recovery_factor", "NaTg", "h_open"),
               ("sodium_h_slope_mv", "NaTg", "h_slope"))


def check_manifest(manifest):
    """Raise when arms break the caps, the dt relations, or the reference and window names."""
    arms = manifest["arms"]
    names = [arm["name"] for arm in arms]
    if len(names) != len(set(names)):
        raise ValueError("Arm names must be unique.")
    by_name = {arm["name"]: arm for arm in arms}
    for arm in arms:
        missing = [field for field in ARM_FIELDS if field not in arm]
        if missing:
            raise ValueError(f"{arm['name']} lacks {', '.join(missing)}")
        if arm["simulator"] not in ("braincell", "neuron"):
            raise ValueError(f"{arm['name']} names an unknown simulator")
        if arm["reference"] not in manifest["references"] or arm["window"] not in manifest["windows_ms"]:
            raise ValueError(f"{arm['name']} names an unknown reference or window")
        if arm["mesh"] == "MaxCVLen" and not arm.get("max_cv_um"):
            raise ValueError(f"{arm['name']} uses MaxCVLen without max_cv_um")
        if arm["simulator"] == "neuron" and not arm.get("candidate_json"):
            raise ValueError(f"{arm['name']} needs a candidate_json")
        pair = arm.get("pair_of")
        if pair is not None and not np.isclose(arm["dt_ms"], by_name[pair]["dt_ms"]/2., rtol=1e-12, atol=0.):
            raise ValueError(f"{arm['name']} must use half the dt of {pair}")
    for simulator, cap in manifest["caps"].items():
        count = sum(arm["simulator"] == simulator for arm in arms)
        if count > cap:
            raise ValueError(f"{count} {simulator} arms exceed the cap {cap}")
    for field in ("prediction", "rejection", "unchanged"):
        if not manifest.get(field):
            raise ValueError(f"Manifest lacks a pre-registered {field}.")
    return names


def braincell_command(root, manifest, arm):
    """Host command for one BrainCell arm, output stem under the manifest output directory."""
    evidence = root/"docs/evidence"
    mesh = (["--max-cv-um", str(arm["max_cv_um"])] if arm["mesh"] == "MaxCVLen"
            else ["--mesh-from", str(evidence/manifest["mesh_source"])])
    return [str(root/manifest["braincell_python"]), "-m", manifest["braincell_module"],
            "--mode", manifest["braincell_profile_key"], "--current-na", str(arm["current_na"]),
            "--duration-ms", str(arm["duration_ms"]), "--dt-ms", str(arm["dt_ms"]), *mesh,
            "--output", str(evidence/manifest["output_dir"]/arm["name"])]


def neuron_command(root, manifest, arm):
    """Container command for one fixed-step NEURON arm (no ``--cvode-atol``: fixed step)."""
    evidence = (root/"docs/evidence").resolve().as_posix()
    cache = (root/manifest["cache_dir"]).resolve().as_posix()
    out = "/evidence/"+manifest["output_dir"]
    return ["docker", "run", "--rm", "--name", (manifest["output_dir"]+"-"+arm["name"]).replace(".", "-"),
            "-v", f"{cache}:/work", "-v", f"{evidence}:/evidence", "-w", "/work/"+manifest["library"],
            manifest["image"], "python", "/evidence/"+manifest["driver"],
            "--candidate-json", out+"/"+arm["candidate_json"], "--current-na", str(arm["current_na"]),
            "--dt-ms", str(arm["dt_ms"]), "--duration-ms", str(arm["duration_ms"]),
            "--output", out+"/"+arm["name"]]


def arm_command(root, manifest, arm):
    """Dispatch on the simulator named by the arm."""
    if arm["simulator"] == "braincell":
        return braincell_command(root, manifest, arm)
    return neuron_command(root, manifest, arm)


def profile_alignment(neuron_meta, profile, controls):
    """List every phase control or density on which the BrainCell profile differs from the NEURON run."""
    mismatches = []
    for flag, mechanism, key in CONTROL_MAP:
        expected = neuron_meta.get(flag)
        actual = controls(profile, "soma", mechanism).get(key)
        if expected is None or actual is None or not np.isclose(expected, actual, rtol=1e-12, atol=0.):
            mismatches.append({"flag": flag, "neuron": expected, "braincell": actual})
    soma = next(region for region in profile.regions if region[0] == "soma")
    intervention = neuron_meta.get("conductance_intervention") or {}
    if intervention.get("mechanism") != "NaTg" or intervention.get("region") != "soma":
        mismatches.append({"flag": "conductance_intervention", "neuron": intervention, "braincell": "NaTg soma factor"})
    axon = next(region for region in profile.regions if region[0] == "axon")
    for flag, value in (("axon_calcium_decay_ms", axon[4][0]), ("axon_calcium_gamma", axon[4][1])):
        if not np.isclose(neuron_meta.get(flag) or np.nan, value, rtol=1e-9, atol=0.):
            mismatches.append({"flag": flag, "neuron": neuron_meta.get(flag), "braincell": value})
    return {"aligned": not mismatches, "mismatches": mismatches, "soma_channels": dict(soma[3])}


def event_rows(result, gate):
    """One row per compared event with signed errors and in-gate flags."""
    rows = []
    for index, (reference, actual, error) in enumerate(zip(result["reference_events"], result["actual_events"],
                                                          result["signed_errors"]), start=1):
        rows.append({"event": index, "reference_rise_ms": reference["rise_crossing_ms"],
                     "actual_rise_ms": actual["rise_crossing_ms"], "errors": error,
                     "in_gate": {key: abs(error[key]) <= limit for key, limit in gate.items()}})
    return rows


def first_failed_event(rows):
    """Index of the first event outside any gate, or None."""
    return next((row["event"] for row in rows if not all(row["in_gate"].values())), None)


def score_pair(reference, actual, window, gate):
    """Compare two traces over a window under a gate and report per-event rows."""
    result = compare_spike_transfer(reference, actual, start_ms=window[0], stop_ms=window[1])
    rows = event_rows(result, gate)
    same_count = result["same_nonzero_event_count"]
    passed = same_count and all(all(row["in_gate"].values()) for row in rows)
    return {"passed": bool(passed), "same_nonzero_event_count": bool(same_count),
            "reference_count": len(result["reference_events"]), "actual_count": len(result["actual_events"]),
            "first_failed_event": None if same_count and passed else first_failed_event(rows) or "count",
            "max_abs_rise_error_ms": max((abs(r["errors"]["rise_crossing_ms"]) for r in rows), default=None),
            "window_ms": list(window), "gate": gate, "events": rows}


def decide(a1_passed, b1_passed, gate_valid, first_failures):
    """Name the literal decision from the A1 and B1 outcomes and the halving validity."""
    if not gate_valid:
        return "time_level_open"
    if all(a1_passed.values()) and all(b1_passed.values()):
        return "mesh"
    if not all(a1_passed.values()) and all(b1_passed.values()):
        k = min(v for v in first_failures.values() if isinstance(v, int)) if any(
            isinstance(v, int) for v in first_failures.values()) else "count"
        return f"implementation_difference_at_event_{k}"
    if not all(b1_passed.values()):
        return "integration"
    return "no_prediction_matched"


def load_trace(path):
    """Voltage trace arrays from a saved run."""
    with np.load(path) as data:
        return {"time_ms": np.asarray(data["time_ms"]), "voltage_mv": np.asarray(data["voltage_mv"])}


def score_manifest(root, manifest, load=load_trace):
    """Score every arm present on disk against its reference and the halving pairs; missing arms stay untested."""
    evidence = root/"docs/evidence"
    gate, half_gate = manifest["gate"], {k: v*manifest["halving_fraction"] for k, v in manifest["gate"].items()}
    arms = {arm["name"]: arm for arm in manifest["arms"]}
    scores, hashes = {}, {}
    for name, arm in arms.items():
        actual_path = evidence/manifest["output_dir"]/(name+".npz")
        if not actual_path.exists():
            scores[name] = {"status": "untested", "reason": "trace absent"}
            continue
        reference_path = evidence/manifest["reference_dir"]/(manifest["references"][arm["reference"]]+".npz")
        hashes[name] = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        scores[name] = score_pair(load(reference_path), load(actual_path), manifest["windows_ms"][arm["window"]], gate)
    halving = {}
    for label in ("braincell_halving", "neuron_halving"):
        full, half = manifest["decision_arms"][label]
        paths = [evidence/manifest["output_dir"]/(n+".npz") for n in (full, half)]
        if all(p.exists() for p in paths):
            halving[label] = score_pair(load(paths[0]), load(paths[1]), manifest["windows_ms"]["halving"], half_gate)
        else:
            halving[label] = {"status": "untested", "passed": False}
    return {"scores": scores, "halving": halving, "inputs_sha256": hashes}


def decision_report(manifest, scored):
    """Assemble the decision JSON from the per-arm scores."""
    groups = manifest["decision_arms"]
    passed = lambda names: {n: bool(scored["scores"][n].get("passed", False)) for n in names}
    a1, b1 = passed(groups["a1"]), passed(groups["b1"])
    tested = all(scored["scores"][n].get("status") != "untested" for n in groups["a1"]+groups["b1"])
    gate_valid = all(row.get("passed", False) for row in scored["halving"].values())
    failures = {n: scored["scores"][n].get("first_failed_event") for n in groups["a1"]}
    return {"spec": manifest["spec"], "prediction": manifest["prediction"], "rejection": manifest["rejection"],
            "gate_valid": gate_valid, "a1_passed": a1, "b1_passed": b1,
            "a0_passed": passed(groups["a0"]), "first_failed_event": failures,
            "decision": decide(a1, b1, gate_valid, failures) if tested else "untested",
            "qualification": manifest["qualification"], **scored}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--check-alignment", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text())
    check_manifest(manifest)
    if args.print_commands:
        for name in manifest["order"]:
            arm = next(a for a in manifest["arms"] if a["name"] == name)
            print(name+": "+" ".join(arm_command(root, manifest, arm)))
    if args.check_alignment or args.score:
        from braintrace.datasets.h01_ei_profiles import channel_controls, get_ei_profile
        meta = json.loads((root/"docs/evidence"/manifest["mesh_source"]).read_text())
        alignment = profile_alignment(meta, get_ei_profile("I", mode=manifest["braincell_profile_key"]), channel_controls)
        print(json.dumps({"profile_alignment": alignment}))
        if args.score and manifest.get("profile_alignment_required") and not alignment["aligned"]:
            raise SystemExit("BrainCell profile is not the NEURON finalist; scoring refused until the registry re-key lands.")
    if args.score:
        report = decision_report(manifest, score_manifest(root, manifest))
        out = root/"docs/evidence"/manifest["output_dir"]/"sp2-i-decision.json"
        out.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"decision": report["decision"], "gate_valid": report["gate_valid"],
                          "a1_passed": report["a1_passed"], "b1_passed": report["b1_passed"]}))


if __name__ == "__main__":
    main()
