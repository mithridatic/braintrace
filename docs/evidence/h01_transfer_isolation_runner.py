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

from docs.evidence.h01_ei_spike_transfer import PEAK_KEYS, compare_spike_transfer

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
    check_dt_series(manifest, names)
    for name, partner in manifest.get("identity_pairs", {}).items():
        if name not in by_name or partner not in by_name or by_name[name]["simulator"] == by_name[partner]["simulator"]:
            raise ValueError(f"identity pair {name}/{partner} must name two arms of different simulators")
    return names


def check_dt_series(manifest, arm_names):
    """Raise when the NEURON dt ladder breaks its cap, its halving relation, or its prediction fields."""
    series = manifest.get("dt_series")
    if series is None:
        return
    for field in ("cap", "gate", "prediction", "rejection", "rungs"):
        if not series.get(field):
            raise ValueError(f"dt_series lacks {field}")
    new = [rung for rung in series["rungs"] if rung["name"] not in arm_names]
    if len(new) > series["cap"]:
        raise ValueError(f"{len(new)} new dt_series runs exceed the cap {series['cap']}")
    for current in {rung["current_na"] for rung in series["rungs"]}:
        ladder = sorted((rung["dt_ms"] for rung in series["rungs"] if rung["current_na"] == current), reverse=True)
        for coarse, fine in zip(ladder, ladder[1:]):
            if not np.isclose(fine, coarse/2., rtol=1e-12, atol=0.):
                raise ValueError(f"dt_series rungs at {current} nA must halve dt: {coarse} -> {fine}")


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


def gate_for(gate, peak_method):
    """Manifest gate re-keyed so its peak limit names the peak metric that is gated."""
    return {(PEAK_KEYS[peak_method] if key in PEAK_KEYS.values() else key): limit for key, limit in gate.items()}


def richardson_peaks(reference, full, half, window, peak_method="sample"):
    """Per-event first-order Richardson peak, ``2 x peak(dt/2) - peak(dt)``, and its error vs the reference.

    Derived column, never gated. Empty when the two actual traces do not hold the same
    number of events over the window.
    """
    key = PEAK_KEYS[peak_method]
    at_full = compare_spike_transfer(reference, full, start_ms=window[0], stop_ms=window[1], peak_method=peak_method)
    at_half = compare_spike_transfer(reference, half, start_ms=window[0], stop_ms=window[1], peak_method=peak_method)
    if len(at_full["actual_events"]) != len(at_half["actual_events"]):
        return []
    rows = []
    for index, (ref, coarse, fine) in enumerate(zip(at_full["reference_events"], at_full["actual_events"],
                                                    at_half["actual_events"]), start=1):
        extrapolated = 2.*fine[key]-coarse[key]
        rows.append({"event": index, "peak_dt_mv": coarse[key], "peak_half_dt_mv": fine[key],
                     "peak_richardson_mv": extrapolated, "peak_richardson_error_mv": extrapolated-ref[key]})
    return rows


def attach_richardson(score, richardson):
    """Fill the derived Richardson columns on the events that have a partner value; null elsewhere."""
    by_event = {row["event"]: row for row in richardson}
    for row in score["events"]:
        partner = by_event.get(row["event"])
        row["peak_richardson_mv"] = None if partner is None else partner["peak_richardson_mv"]
        row["peak_richardson_error_mv"] = None if partner is None else partner["peak_richardson_error_mv"]
    score["richardson_events"] = len(richardson)
    return score


def score_pair(reference, actual, window, gate, peak_method="sample"):
    """Compare two traces over a window under a gate and report per-event rows (both peaks in every row)."""
    gate = gate_for(gate, peak_method)
    result = compare_spike_transfer(reference, actual, start_ms=window[0], stop_ms=window[1],
                                    peak_method=peak_method)
    rows = event_rows(result, gate)
    same_count = result["same_nonzero_event_count"]
    passed = same_count and all(all(row["in_gate"].values()) for row in rows)
    return {"passed": bool(passed), "same_nonzero_event_count": bool(same_count),
            "reference_count": len(result["reference_events"]), "actual_count": len(result["actual_events"]),
            "first_failed_event": None if same_count and passed else first_failed_event(rows) or "count",
            "max_abs_rise_error_ms": max((abs(r["errors"]["rise_crossing_ms"]) for r in rows), default=None),
            "window_ms": list(window), "gate": gate, "peak_method": peak_method, "events": rows}


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


def score_manifest(root, manifest, load=load_trace, peak_method="sample", reference_root=None):
    """Score every arm present on disk against its reference and the halving pairs; missing arms stay untested.

    ``reference_root`` names the tree holding the reference ``.npz`` traces when they are not
    in ``root`` (they are tracked on another branch); the hash of every reference used is recorded.
    Arms named in ``richardson_pairs`` get the derived Richardson peak column from their dt/2 partner.
    """
    evidence = root/"docs/evidence"
    reference_root = Path(reference_root) if reference_root is not None else root
    reference_dir = reference_root/"docs/evidence"/manifest["reference_dir"]
    reference_hashes = {}
    gate, half_gate = manifest["gate"], {k: v*manifest["halving_fraction"] for k, v in manifest["gate"].items()}
    arms = {arm["name"]: arm for arm in manifest["arms"]}
    scores, hashes = {}, {}
    for name, arm in arms.items():
        actual_path = evidence/manifest["output_dir"]/(name+".npz")
        if not actual_path.exists():
            scores[name] = {"status": "untested", "reason": "trace absent"}
            continue
        reference_path = reference_dir/(manifest["references"][arm["reference"]]+".npz")
        hashes[name] = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        reference_hashes.setdefault(arm["reference"], hashlib.sha256(reference_path.read_bytes()).hexdigest())
        scores[name] = score_pair(load(reference_path), load(actual_path), manifest["windows_ms"][arm["window"]],
                                  gate, peak_method)
    for name, partner in manifest.get("richardson_pairs", {}).items():
        paths = [evidence/manifest["output_dir"]/(n+".npz") for n in (name, partner)]
        if scores[name].get("status") == "untested" or not paths[1].exists():
            continue
        reference_path = reference_dir/(manifest["references"][arms[name]["reference"]]+".npz")
        attach_richardson(scores[name], richardson_peaks(load(reference_path), load(paths[0]), load(paths[1]),
                                                         manifest["windows_ms"][arms[partner]["window"]], peak_method))
    halving = {}
    for label in ("braincell_halving", "neuron_halving"):
        full, half = manifest["decision_arms"][label]
        paths = [evidence/manifest["output_dir"]/(n+".npz") for n in (full, half)]
        if all(p.exists() for p in paths):
            halving[label] = score_pair(load(paths[0]), load(paths[1]), manifest["windows_ms"]["halving"], half_gate,
                                        peak_method)
        else:
            halving[label] = {"status": "untested", "passed": False}
    return {"scores": scores, "halving": halving, "inputs_sha256": hashes, "peak_method": peak_method,
            "reference_root": str(reference_root), "reference_sha256": reference_hashes}


def decision_report(manifest, scored):
    """Assemble the decision JSON from the per-arm scores."""
    groups = manifest["decision_arms"]
    passed = lambda names: {n: bool(scored["scores"][n].get("passed", False)) for n in names}
    a1, b1 = passed(groups["a1"]), passed(groups["b1"])
    tested = all(scored["scores"][n].get("status") != "untested" for n in groups["a1"]+groups["b1"])
    gate_valid = all(row.get("passed", False) for row in scored["halving"].values())
    failures = {n: scored["scores"][n].get("first_failed_event") for n in groups["a1"]}
    return {"spec": manifest["spec"], "prediction": manifest["prediction"], "rejection": manifest["rejection"],
            "gate": manifest["gate"], "gate_amendment": manifest.get("gate_amendment"),
            "gate_valid": gate_valid, "a1_passed": a1, "b1_passed": b1,
            "a0_passed": passed(groups["a0"]), "first_failed_event": failures,
            "decision": decide(a1, b1, gate_valid, failures) if tested else "untested",
            "qualification": manifest["qualification"], **scored}


def rise_summary(score):
    """Late-event and maximal rise error of a scored pair, and whether the 1 ms human crossing tolerance is met."""
    rows = score["events"]
    max_abs = score["max_abs_rise_error_ms"]
    return {"count_equal": score["same_nonzero_event_count"], "reference_count": score["reference_count"],
            "actual_count": score["actual_count"], "paired_events": len(rows),
            "last_event_rise_error_ms": rows[-1]["errors"]["rise_crossing_ms"] if rows else None,
            "max_abs_rise_error_ms": max_abs,
            "met": bool(score["same_nonzero_event_count"] and rows and all(row["in_gate"]["rise_crossing_ms"] for row in rows))}


def adjacent_ratios(rungs):
    """Error ratios of every adjacent (dt, dt/2) pair present at one input; first order predicts 2."""
    present = sorted((r for r in rungs if r.get("status") != "untested"), key=lambda r: -r["dt_ms"])
    ratios = []
    for coarse, fine in zip(present, present[1:]):
        if not np.isclose(fine["dt_ms"], coarse["dt_ms"]/2., rtol=1e-12, atol=0.):
            continue
        pair = {"dt_ms": coarse["dt_ms"], "half_dt_ms": fine["dt_ms"]}
        for key in ("last_event_rise_error_ms", "max_abs_rise_error_ms"):
            a, b = coarse[key], fine[key]
            pair[key+"_ratio"] = None if a is None or not b else a/b
        common = min(coarse["paired_events"], fine["paired_events"])
        pair["common_event"] = common or None
        if common:
            a, b = (r["events"][common-1]["errors"]["rise_crossing_ms"] for r in (coarse, fine))
            pair["common_event_rise_errors_ms"] = [a, b]
            pair["common_event_rise_error_ratio"] = None if not b else a/b
        ratios.append(pair)
    return ratios


def score_dt_series(root, manifest, load=load_trace, peak_method="sample", reference_root=None):
    """Score every present NEURON dt rung against the CVode finalist under the human 1 ms rise tolerance."""
    series = manifest.get("dt_series")
    if series is None:
        return None
    evidence = root/"docs/evidence"
    reference_dir = Path(reference_root or root)/"docs/evidence"/manifest["reference_dir"]
    window = manifest["windows_ms"][series["window"]]
    rungs, hashes = [], {}
    for rung in series["rungs"]:
        path = evidence/manifest["output_dir"]/(rung["name"]+".npz")
        row = {k: rung[k] for k in ("name", "current_na", "dt_ms", "reference")}
        if not path.exists():
            rungs.append({**row, "status": "untested", "reason": "trace absent"})
            continue
        hashes[rung["name"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        reference = load(reference_dir/(manifest["references"][rung["reference"]]+".npz"))
        score = score_pair(reference, load(path), window, series["gate"], peak_method)
        rungs.append({**row, **rise_summary(score), "events": score["events"]})
    by_current = {c: [r for r in rungs if r["current_na"] == c] for c in sorted({r["current_na"] for r in rungs})}
    ratios = {str(c): adjacent_ratios(rows) for c, rows in by_current.items()}
    met = {}
    for dt in sorted({r["dt_ms"] for r in rungs}, reverse=True):
        at_dt = [r for r in rungs if r["dt_ms"] == dt]
        met[str(dt)] = (len(at_dt) == len(by_current) and all(r.get("met") for r in at_dt))
    found = [float(dt) for dt, ok in met.items() if ok]
    by_input = {str(c): max((r["dt_ms"] for r in rows if r.get("met")), default="not reached within cap")
                for c, rows in by_current.items()}
    return {"gate": series["gate"], "window_ms": list(window), "prediction": series["prediction"],
            "rejection": series["rejection"], "rungs": rungs, "adjacent_ratios": ratios,
            "met_at_both_inputs_by_dt": met, "dt_found_ms": max(found) if found else "not reached within cap",
            "dt_found_by_input_ms": by_input, "inputs_sha256": hashes}


def score_identity(root, manifest, load=load_trace, peak_method="sample"):
    """Per-event identity of each BrainCell arm against its NEURON fixed-step partner at the same dt and mesh."""
    evidence = root/"docs/evidence"/manifest["output_dir"]
    arms = {arm["name"]: arm for arm in manifest["arms"]}
    out = {}
    for name, partner in manifest.get("identity_pairs", {}).items():
        paths = [evidence/(n+".npz") for n in (partner, name)]
        if not all(p.exists() for p in paths):
            out[name] = {"status": "untested", "reason": "trace absent", "reference": partner}
            continue
        window = manifest["windows_ms"][arms[name]["window"]]
        reference, actual = load(paths[0]), load(paths[1])
        score = score_pair(reference, actual, window, manifest["identity_gate"], peak_method)
        maxima = {key: max((abs(r["errors"][key]) for r in score["events"]), default=None)
                  for key in ("peak_interpolated_voltage_mv", "peak_sample_voltage_mv", "time_above_threshold_ms")}
        out[name] = {"reference": partner, **score, "max_abs_peak_error_mv": maxima[PEAK_KEYS[peak_method]],
                     "max_abs_peak_sample_error_mv": maxima["peak_sample_voltage_mv"],
                     "max_abs_width_error_ms": maxima["time_above_threshold_ms"],
                     "raw_voltage_max_abs_diff_mv": raw_voltage_difference(reference, actual, window)}
    return out


def raw_voltage_difference(reference, actual, window):
    """Largest |voltage difference| over the window, the reference interpolated onto the actual grid."""
    keep = (actual["time_ms"] >= window[0]) & (actual["time_ms"] <= window[1])
    on_grid = np.interp(actual["time_ms"][keep], reference["time_ms"], reference["voltage_mv"])
    return float(np.max(np.abs(actual["voltage_mv"][keep]-on_grid)))


def reproduces(measured, quoted):
    """Whether each measured maximum, rounded to two significant figures, equals the quoted decision value."""
    return all(measured.get(key) is not None and float(f"{measured[key]:.2g}") == value for key, value in quoted.items())


def wall_clocks(root, manifest):
    """Every ``*.timing.json`` under the output directory, keyed by file stem."""
    folder = root/"docs/evidence"/manifest["output_dir"]
    return {p.name[:-len(".timing.json")]: json.loads(p.read_text()) for p in sorted(folder.glob("*.timing.json"))}


def close_report(manifest, report, series, identity, clocks):
    """Assemble the SP2 close decision: identity gate, dt qualification, BrainCell identity rows, clocks, hashes."""
    basis = identity.get(manifest["close"]["identity_basis"], {})
    confirming = {n: row for n, row in identity.items() if n != manifest["close"]["identity_basis"]}
    measured = {"rise_crossing_ms": basis.get("max_abs_rise_error_ms"),
                "peak_interpolated_voltage_mv": basis.get("max_abs_peak_error_mv"),
                "time_above_threshold_ms": basis.get("max_abs_width_error_ms")}
    quoted = manifest["close"]["identity_values"]
    return {"spec": manifest["spec"], "date": manifest["close"]["date"], "amendment": manifest["close"]["amendment"],
            "identity_gate": {"closed": bool(basis) and reproduces(measured, quoted),
                              "closing_rule": "user decision (a): the basis arm reproduces the quoted identity values",
                              "decision_values": quoted, "basis_measured": measured,
                              "basis_arm": manifest["close"]["identity_basis"], "basis": basis,
                              "confirming_full_train": confirming, "confirming_gate": manifest["identity_gate"],
                              "confirming_prediction_held": bool(confirming) and all(
                                  row.get("passed", False) for row in confirming.values())},
            "dt_qualification": series, "full_train_vs_cvode": {n: report["scores"][n] for n in manifest["decision_arms"]["a1"]
                                                                + manifest["decision_arms"]["b1"]},
            "decision": report["decision"], "gate_valid": report["gate_valid"], "peak_method": report["peak_method"],
            "wall_clocks": clocks, "inputs_sha256": report["inputs_sha256"], "reference_sha256": report["reference_sha256"],
            "qualification": manifest["qualification"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--check-alignment", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--peak-method", choices=tuple(PEAK_KEYS), default="sample",
                        help="peak metric that enters the gate; both peaks are reported either way")
    parser.add_argument("--reference-root", type=Path, help="tree holding the reference .npz traces")
    parser.add_argument("--close", action="store_true",
                        help="with --score: also write sp2-close-decision.json (identity gate, dt series, clocks)")
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
        profile = get_ei_profile(manifest["polarity"], mode=manifest["braincell_profile_key"])
        alignment = profile_alignment(meta, profile, channel_controls)
        print(json.dumps({"profile_alignment": alignment}))
        if args.score and manifest.get("profile_alignment_required") and not alignment["aligned"]:
            raise SystemExit("BrainCell profile is not the NEURON finalist; scoring refused "
                             "(set braincell_profile_key to the aligned registry mode).")
    if args.score:
        report = decision_report(manifest, score_manifest(root, manifest, peak_method=args.peak_method,
                                                          reference_root=args.reference_root))
        out = root/"docs/evidence"/manifest["output_dir"]/"sp2-i-decision.json"
        out.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"decision": report["decision"], "gate_valid": report["gate_valid"], "peak_method": args.peak_method,
                          "a1_passed": report["a1_passed"], "b1_passed": report["b1_passed"]}))
        if args.close:
            series = score_dt_series(root, manifest, peak_method=args.peak_method, reference_root=args.reference_root)
            identity = score_identity(root, manifest, peak_method=args.peak_method)
            close = close_report(manifest, report, series, identity, wall_clocks(root, manifest))
            out.with_name("sp2-close-decision.json").write_text(json.dumps(close, indent=2)+"\n")
            print(json.dumps({"identity_closed": close["identity_gate"]["closed"],
                              "dt_found_ms": None if series is None else series["dt_found_ms"]}))


if __name__ == "__main__":
    main()
