"""Run a bounded, manifest-driven candidate campaign in the pinned NEURON container.

One runner serves every cell. The manifest names the driver script, the inputs
(each with its extra flags), the cap, the stages with their pre-registered
predictions, and a per-run abort. Finished candidates whose flag file hash
matches are skipped. A stage opens only when its gate file records a decision.
See ``docs/specs/2026-09-06-h01-progressive-search-plan.md``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def check_manifest(manifest):
    """Raise when the manifest breaks the cap, concurrency, naming, or stage rules."""
    names = [c["name"] for c in manifest["candidates"]]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique.")
    prior = manifest.get("prior_evaluations", 0)
    if isinstance(prior, bool) or not isinstance(prior, int) or prior < 0:
        raise ValueError("prior_evaluations must be a nonnegative integer.")
    if len(names)+prior > manifest["cap"]:
        raise ValueError(f"Manifest lists {len(names)} candidates plus {prior} prior evaluations; cap is {manifest['cap']}.")
    if manifest["max_concurrent"] > 2:
        raise ValueError("At most two concurrent simulations are allowed.")
    if not isinstance(manifest.get("abort_seconds"), (int, float)) or manifest["abort_seconds"] <= 0:
        raise ValueError("abort_seconds must be a positive number.")
    for candidate in manifest["candidates"]:
        stage = manifest["stages"].get(str(candidate["stage"]))
        if stage is None:
            raise ValueError(f"{candidate['name']} names stage {candidate['stage']} that the manifest does not define.")
        for field in ("prediction", "rejection", "unchanged"):
            if not stage.get(field):
                raise ValueError(f"Stage {candidate['stage']} lacks a pre-registered {field}.")
    return names


def candidate_values(candidate):
    """Driver flag values written to the candidate JSON file."""
    values = dict(candidate["flags"])
    values.update(candidate.get("settings", {}))
    return values


def flag_hash(candidate):
    """Stable hash of the exact candidate file text the driver records."""
    text = json.dumps(candidate_values(candidate), sort_keys=True, indent=2)
    return hashlib.sha256(text.encode()).hexdigest()


def docker_command(root, manifest, candidate, input_name):
    """The container invocation for one candidate and one input."""
    evidence = (root/"docs/evidence").resolve().as_posix()
    cache = (root/manifest["cache_dir"]).resolve().as_posix()
    out = "/evidence/"+manifest["output_dir"]+"/"+candidate["name"]
    return ["docker", "run", "--rm", "--name", container_name(manifest, candidate, input_name),
            "-v", f"{cache}:/work", "-v", f"{evidence}:/evidence",
            "-w", "/work/"+manifest["library"], manifest["image"], "python",
            "/evidence/"+manifest["driver"], "--candidate-json", out+".candidate.json",
            *manifest["inputs"][input_name], "--output", f"{out}-{input_name}"]


def container_name(manifest, candidate, input_name):
    """Deterministic container name so an aborted run can be killed, not only abandoned."""
    return f"{manifest['output_dir']}-{candidate['name']}-{input_name}".replace(".", "-")


def needs_run(root, manifest, candidate):
    """True unless every input report exists and records this candidate's flag hash."""
    folder = root/"docs/evidence"/manifest["output_dir"]
    for input_name in manifest["inputs"]:
        path = folder/f"{candidate['name']}-{input_name}.json"
        if not path.exists():
            return True
        record = json.loads(path.read_text()).get("candidate_json") or {}
        if record.get("sha256") != flag_hash(candidate):
            return True
    return False


def stage_ready(root, manifest, stage):
    """A stage opens when its gate file exists and records ``decision`` as a non-empty value."""
    gate = manifest["stages"][str(stage)].get("gate")
    if not gate:
        return True
    path = root/"docs/evidence"/manifest["output_dir"]/gate
    if not path.exists():
        return False
    return bool(json.loads(path.read_text()).get("decision"))


def _capture_text(value):
    """Decode a captured stream (str, bytes or None) to text."""
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def persist_container_output(folder, stem, stdout, stderr):
    """Write ``<stem>.stderr.log`` (stderr, then the stdout tail) and return the last 20 stderr lines.

    The log is written on every real run so a container that dies before its report
    (for example exit 139) still leaves its diagnostic text behind ``--rm``.
    """
    stderr, stdout = _capture_text(stderr), _capture_text(stdout)
    stdout_tail = stdout.splitlines()[-50:]
    text = stderr+"\n--- stdout tail ---\n"+"\n".join(stdout_tail)+"\n"
    (folder/f"{stem}.stderr.log").write_text(text, encoding="utf-8")
    return stderr.splitlines()[-20:]


def run_candidate(root, manifest, candidate, dry_run):
    """Write the flag file, run each input in turn under the abort limit, and return timings."""
    folder = root/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder/(candidate["name"]+".candidate.json")).write_text(json.dumps(candidate_values(candidate), sort_keys=True, indent=2))
    rows = []
    for input_name in manifest["inputs"]:
        command = docker_command(root, manifest, candidate, input_name)
        stem = f"{candidate['name']}-{input_name}"
        start = time.perf_counter()
        code, aborted, tail, log_name = None, False, [], None
        if not dry_run:
            try:
                completed = subprocess.run(command, env={**os.environ, "MSYS_NO_PATHCONV": "1"}, capture_output=True,
                                           text=True, timeout=manifest["abort_seconds"])
                code = completed.returncode
                stdout, stderr = completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as expired:
                aborted = True
                stdout, stderr = expired.stdout, expired.stderr
                subprocess.run(["docker", "kill", container_name(manifest, candidate, input_name)],
                               capture_output=True, text=True, check=False)
                for suffix in (".json", ".npz"):
                    stale = folder/f"{stem}{suffix}"
                    if stale.exists():
                        stale.unlink()
            tail = persist_container_output(folder, stem, stdout, stderr)
            log_name = f"{manifest['output_dir']}/{stem}.stderr.log"
        rows.append({"name": candidate["name"], "input": input_name, "seconds": time.perf_counter()-start,
                     "returncode": code, "aborted": aborted, "stderr_tail": tail, "stderr_log": log_name,
                     "command": command})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text())
    check_manifest(manifest)
    if not stage_ready(root, manifest, args.stage):
        raise SystemExit(f"Stage {args.stage} is gated by {manifest['stages'][str(args.stage)]['gate']}; no decision is recorded.")
    selected = [c for c in manifest["candidates"] if str(c["stage"]) == str(args.stage) and needs_run(root, manifest, c)]
    with ThreadPoolExecutor(max_workers=manifest["max_concurrent"]) as pool:
        results = list(pool.map(lambda c: run_candidate(root, manifest, c, args.dry_run), selected))
    log_path = root/"docs/evidence"/manifest["output_dir"]/"campaign-log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log += [row for rows in results for row in rows]
    if not args.dry_run:
        log_path.write_text(json.dumps(log, indent=2))
    for row in (r for rows in results for r in rows):
        print(row["name"], row["input"], f"{row['seconds']:.0f}s", "rc", row["returncode"], "aborted" if row["aborted"] else "")


if __name__ == "__main__":
    main()
