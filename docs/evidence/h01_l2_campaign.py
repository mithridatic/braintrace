"""Run the bounded layer-2 family campaign from a manifest, at most two at a time.

Each candidate runs both calibration sweeps in the pinned NEURON container.
The manifest fixes the cap, the stages, and every flag. Finished candidates
whose flag file hash matches are skipped. See
``docs/specs/2026-09-05-h01-l2-family-campaign.md``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STAGE_GATES = {"A": "stage0-preservation.json", "B": "stage-a-ranking.json"}


def check_manifest(manifest):
    """Raise when the manifest exceeds its cap or repeats a name."""
    names = [c["name"] for c in manifest["candidates"]]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique.")
    prior = manifest.get("prior_evaluations", 0)
    if isinstance(prior, bool) or not isinstance(prior, int) or prior < 0:
        raise ValueError("prior_evaluations must be a nonnegative integer.")
    if len(names) + prior > manifest["cap"]:
        raise ValueError(f"Manifest lists {len(names)} candidates plus {prior} prior evaluations; cap is {manifest['cap']}.")
    if manifest["max_concurrent"] > 2:
        raise ValueError("At most two concurrent simulations are allowed.")
    # These are the latest observations in the campaign's two fixed protocols.
    required_end = max({"43": 2120., "50": 2020.}[str(s)] for s in manifest["sweeps"])
    for candidate in manifest["candidates"]:
        stop = candidate_values(candidate).get("stop_ms")
        if stop is not None and not float(stop) >= required_end:
            raise ValueError(f"{candidate['name']} must cover the final required sample at {required_end:g} ms.")
    return names


def candidate_values(candidate):
    """Driver flag values written to the candidate JSON file."""
    values = dict(candidate["flags"])
    values.update(candidate["settings"])
    return values


def flag_hash(candidate):
    """Stable hash of the exact driver values a candidate uses."""
    text = json.dumps(candidate_values(candidate), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def docker_command(root, manifest, candidate, sweep):
    """The container invocation for one candidate and one sweep."""
    evidence = (root/"docs/evidence").resolve().as_posix()
    cache = (root/manifest["cache_dir"]).resolve().as_posix()
    out = "/evidence/"+manifest["output_dir"]+"/"+candidate["name"]
    return ["docker", "run", "--rm", "-v", f"{cache}:/work", "-v", f"{evidence}:/evidence",
            "-w", "/work/"+manifest["library"], manifest["image"], "python",
            "/evidence/h01_l2_neuron_reference.py", "--cache", "/work",
            "--candidate-json", out+".candidate.json", "--sweep", str(sweep),
            *manifest["sweeps"][str(sweep)], "--output", f"{out}-sweep{sweep}"]


def needs_run(root, manifest, candidate):
    """True unless both sweep reports exist and record this candidate's flag file hash."""
    folder = root/"docs/evidence"/manifest["output_dir"]
    expected = hashlib.sha256(json.dumps(candidate_values(candidate), sort_keys=True, indent=2).encode()).hexdigest()
    for sweep in manifest["sweeps"]:
        path = folder/f"{candidate['name']}-sweep{sweep}.json"
        if not path.exists():
            return True
        record = json.loads(path.read_text()).get("candidate_json") or {}
        if record.get("sha256") != expected:
            return True
    return False


def stage_ready(root, manifest, stage):
    """Stage A needs preserved coarse settings or a verified fine fallback."""
    gate = STAGE_GATES.get(stage)
    if gate is None:
        return True
    path = root/"docs/evidence"/manifest["output_dir"]/gate
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if stage != "A":
        return True
    if record.get("preserved") is True:
        return True
    required = {"nseg_factor": 9, "cvode_atol": 1e-10}
    candidates = [c for c in manifest["candidates"] if str(c["stage"]) == "A"]
    return (record.get("preserved") is False and record.get("selected_mesh") == "fine"
            and record.get("coverage_verified") is True
            and record.get("selected_settings") == required and bool(candidates)
            and all(all(candidate_values(c).get(k) == v for k, v in required.items())
                    for c in candidates))


def run_candidate(root, manifest, candidate, dry_run):
    """Write the flag file, run each sweep in turn, and return timings."""
    folder = root/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder/(candidate["name"]+".candidate.json")).write_text(json.dumps(candidate_values(candidate), sort_keys=True, indent=2))
    rows = []
    for sweep in manifest["sweeps"]:
        command = docker_command(root, manifest, candidate, sweep)
        start = time.perf_counter()
        code = None if dry_run else subprocess.run(command, env={**os.environ, "MSYS_NO_PATHCONV": "1"},
                                                    capture_output=True, text=True).returncode
        rows.append({"name": candidate["name"], "sweep": sweep, "seconds": time.perf_counter()-start,
                     "returncode": code, "command": command})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text())
    check_manifest(manifest)
    if not stage_ready(root, manifest, args.stage):
        raise SystemExit(f"Stage {args.stage} is gated by {STAGE_GATES[args.stage]}; it is missing or not preserved.")
    selected = [c for c in manifest["candidates"] if str(c["stage"]) == args.stage and needs_run(root, manifest, c)]
    with ThreadPoolExecutor(max_workers=manifest["max_concurrent"]) as pool:
        results = list(pool.map(lambda c: run_candidate(root, manifest, c, args.dry_run), selected))
    log_path = root/"docs/evidence"/manifest["output_dir"]/"campaign-log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log += [row for rows in results for row in rows]
    if not args.dry_run:
        log_path.write_text(json.dumps(log, indent=2))
    for row in (r for rows in results for r in rows):
        print(row["name"], row["sweep"], f"{row['seconds']:.0f}s", "rc", row["returncode"])


if __name__ == "__main__":
    main()
