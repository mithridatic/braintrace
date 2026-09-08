"""Run a bounded, manifest-driven candidate campaign in the pinned NEURON container.

One runner serves every cell. The manifest names the driver script, the inputs
(each with its extra flags), the cap, the stages with their pre-registered
predictions, and a per-run abort. A (candidate, input) pair whose report records
the candidate's flag hash is skipped. A stage opens only when its gate file
records a decision.

An evaluation is counted once per candidate, when the candidate is first
launched; ``campaign-log.json`` holds one entry per evaluation with an
``evaluation_index`` and a per-input status (``untested``, ``completed``,
``failed``). An input killed at the abort stays ``untested`` and is re-run by
the next invocation of the same stage without a second evaluation being counted.
See ``docs/specs/2026-09-06-h01-progressive-search-plan.md`` and the
2026-09-08 addendum of ``docs/specs/2026-09-07-h01-e-gain-split.md``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STATUS_UNTESTED = "untested"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


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


def candidate_text(candidate):
    """Exact bytes of the candidate JSON file the runner writes and the driver hashes.

    Line endings are fixed to CRLF on every platform: the driver records
    ``sha256(file bytes)``, and every candidate identity recorded so far in the
    E and I campaigns (e.g. B3 ``e4825c83...``) was produced from that byte form,
    so the skip rule and the documented identities stay valid everywhere.
    """
    text = json.dumps(candidate_values(candidate), sort_keys=True, indent=2)
    return text.replace("\n", "\r\n").encode()


def flag_hash(candidate):
    """Stable hash of the exact candidate file bytes the driver records as ``candidate_json.sha256``."""
    return hashlib.sha256(candidate_text(candidate)).hexdigest()


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


def input_finished(root, manifest, candidate, input_name):
    """Whether one (candidate, input) report exists and records this candidate's flag hash.

    Parameters
    ----------
    root : Path
        Repository root holding ``docs/evidence``.
    manifest : dict
        Campaign manifest.
    candidate : dict
        Candidate entry of the manifest.
    input_name : str
        Key of ``manifest["inputs"]``.

    Returns
    -------
    bool
        True when ``<output_dir>/<name>-<input>.json`` exists and its
        ``candidate_json.sha256`` equals ``flag_hash(candidate)``.
    """
    path = root/"docs/evidence"/manifest["output_dir"]/f"{candidate['name']}-{input_name}.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text()).get("candidate_json") or {}
    return record.get("sha256") == flag_hash(candidate)


def needs_run(root, manifest, candidate):
    """True unless every input report exists and records this candidate's flag hash."""
    return not all(input_finished(root, manifest, candidate, name) for name in manifest["inputs"])


def stage_ready(root, manifest, stage):
    """A stage opens when its gate file exists and records ``decision`` as a non-empty value."""
    gate = manifest["stages"][str(stage)].get("gate")
    if not gate:
        return True
    path = root/"docs/evidence"/manifest["output_dir"]/gate
    if not path.exists():
        return False
    return bool(json.loads(path.read_text()).get("decision"))


def row_status(row):
    """Per-input status of one run row: aborted -> untested, rc 0 -> completed, else failed."""
    if row["aborted"] or row["returncode"] is None:
        return STATUS_UNTESTED
    return STATUS_COMPLETED if row["returncode"] == 0 else STATUS_FAILED


def upgrade_log(log, manifest):
    """Group a legacy flat list of run rows into evaluation entries; pass a new-format log through.

    Parameters
    ----------
    log : list
        Either legacy rows (``{"name", "input", "seconds", "returncode", "aborted", "command"}``)
        or evaluation entries (``{"name", "evaluation_index", "inputs", "runs", ...}``).
    manifest : dict
        Supplies ``prior_evaluations``, ``abort_seconds`` and the candidate flag hashes.

    Returns
    -------
    list of dict
        Evaluation entries in first-launch order. Legacy rows of one candidate become one
        evaluation numbered ``prior_evaluations + k``; a candidate absent from the manifest
        gets ``candidate_sha256`` None.
    """
    if not log or "evaluation_index" in log[0]:
        return log
    by_name = {c["name"]: c for c in manifest["candidates"]}
    entries = []
    for row in log:
        entry = next((e for e in entries if e["name"] == row["name"]), None)
        if entry is None:
            candidate = by_name.get(row["name"])
            entry = {"name": row["name"], "candidate_sha256": flag_hash(candidate) if candidate else None,
                     "evaluation_index": manifest.get("prior_evaluations", 0)+len(entries)+1,
                     "abort_seconds": manifest["abort_seconds"], "inputs": {}, "runs": []}
            entries.append(entry)
        record_rows(entry, [row])
    return entries


def load_log(path, manifest):
    """Read ``campaign-log.json`` (legacy or evaluation format) as evaluation entries."""
    return upgrade_log(json.loads(path.read_text()) if path.exists() else [], manifest)


def find_evaluation(log, candidate):
    """The log entry of ``candidate`` whose recorded hash matches its current flags, or None."""
    sha = flag_hash(candidate)
    return next((e for e in log if e["name"] == candidate["name"] and e["candidate_sha256"] == sha), None)


def evaluations_used(log, manifest):
    """Evaluations spent so far: manifest ``prior_evaluations`` plus one per log entry."""
    return manifest.get("prior_evaluations", 0)+len(log)


def register_evaluation(log, manifest, candidate, abort_seconds, reason):
    """Return the candidate's evaluation entry, creating and counting it only on first launch.

    Parameters
    ----------
    log : list of dict
        Evaluation entries; a new entry is appended in place.
    manifest : dict
        Campaign manifest (cap, prior_evaluations, inputs).
    candidate : dict
        Candidate to launch or resume.
    abort_seconds : float
        Abort limit in force for this launch; recorded on the entry.
    reason : str or None
        ``abort_override_reason`` when the limit exceeds the manifest's value.

    Returns
    -------
    dict
        The existing entry (same name and flag hash: a resume, index unchanged) or a new
        entry with ``evaluation_index = evaluations_used + 1`` and every input ``untested``.

    Raises
    ------
    ValueError
        When a new entry would exceed the manifest cap.
    """
    entry = find_evaluation(log, candidate)
    if entry is None:
        if evaluations_used(log, manifest)+1 > manifest["cap"]:
            raise ValueError(f"Launching {candidate['name']} would be evaluation {evaluations_used(log, manifest)+1}; cap is {manifest['cap']}.")
        entry = {"name": candidate["name"], "candidate_sha256": flag_hash(candidate),
                 "evaluation_index": evaluations_used(log, manifest)+1, "abort_seconds": abort_seconds,
                 "inputs": {name: STATUS_UNTESTED for name in manifest["inputs"]}, "runs": []}
        log.append(entry)
    entry["abort_seconds"] = abort_seconds
    if reason:
        entry["abort_override_reason"] = reason
    return entry


def record_rows(entry, rows):
    """Append run rows to an evaluation entry and set each input's status; the index is untouched."""
    for row in rows:
        entry["runs"].append(row)
        entry["inputs"][row["input"]] = row_status(row)


def plan_inputs(root, manifest, candidate, log, only):
    """Decide per input whether the candidate would ``skip``, ``resume`` or ``run`` it.

    Parameters
    ----------
    root : Path
        Repository root.
    manifest : dict
        Campaign manifest.
    candidate : dict
        Candidate entry.
    log : list of dict
        Evaluation entries (see ``load_log``).
    only : list of str or None
        Restrict to these inputs (``--only-input``); None means every manifest input.

    Returns
    -------
    dict
        ``{input: "skip" | "resume" | "run"}``. ``skip``: the report records this flag hash.
        ``resume``: an evaluation of this candidate and hash exists in the log, so the input
        is completed inside it without a new evaluation. ``run``: first launch.

    Raises
    ------
    ValueError
        When ``only`` names something that is not a manifest input.
    """
    names = list(manifest["inputs"]) if only is None else list(only)
    unknown = [n for n in names if n not in manifest["inputs"]]
    if unknown:
        raise ValueError(f"{unknown} is not an input of the manifest; inputs are {list(manifest['inputs'])}.")
    entry = find_evaluation(log, candidate)
    plan = {}
    for name in names:
        if input_finished(root, manifest, candidate, name):
            plan[name] = "skip"
        else:
            plan[name] = "resume" if entry is not None else "run"
    return plan


def abort_limit(manifest, override, reason):
    """Abort seconds in force: the CLI override when given, else the manifest's.

    Parameters
    ----------
    manifest : dict
        Campaign manifest with ``abort_seconds``.
    override : float or None
        ``--abort-seconds`` value.
    reason : str or None
        ``--abort-override-reason``; required (non-blank) when ``override`` exceeds the manifest.

    Returns
    -------
    float
        The limit to pass to ``run_candidate``.

    Raises
    ------
    ValueError
        When the override is above the manifest's value and no reason is registered.
    """
    if override is None:
        return manifest["abort_seconds"]
    if override > manifest["abort_seconds"] and not (reason or "").strip():
        raise ValueError(f"--abort-seconds {override} exceeds the manifest's {manifest['abort_seconds']}; "
                         "an abort_override_reason must be registered (--abort-override-reason).")
    return override


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


def run_candidate(root, manifest, candidate, dry_run, inputs=None, abort_seconds=None):
    """Write the flag file, run the chosen inputs in turn under the abort limit, and return timings.

    Parameters
    ----------
    root : Path
        Repository root.
    manifest : dict
        Campaign manifest.
    candidate : dict
        Candidate entry.
    dry_run : bool
        Build commands without launching containers.
    inputs : list of str, optional
        Inputs to run; default every manifest input.
    abort_seconds : float, optional
        Per-run limit; default ``manifest["abort_seconds"]``.

    Returns
    -------
    list of dict
        One row per input: name, input, seconds, returncode, aborted, stderr_tail,
        stderr_log (``<output_dir>/<stem>.stderr.log``, written on every real run), command.
    """
    folder = root/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder/(candidate["name"]+".candidate.json")).write_bytes(candidate_text(candidate))
    limit = manifest["abort_seconds"] if abort_seconds is None else abort_seconds
    rows = []
    for input_name in (list(manifest["inputs"]) if inputs is None else inputs):
        command = docker_command(root, manifest, candidate, input_name)
        stem = f"{candidate['name']}-{input_name}"
        start = time.perf_counter()
        code, aborted, tail, log_name = None, False, [], None
        if not dry_run:
            try:
                completed = subprocess.run(command, env={**os.environ, "MSYS_NO_PATHCONV": "1"}, capture_output=True,
                                           text=True, timeout=limit)
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


def repo_root():
    """Repository root: two directories above ``docs/evidence``."""
    return Path(__file__).resolve().parents[2]


def print_plan(log, manifest, plans):
    """Print one line per (candidate, input) naming skip, resume or run and the evaluation it belongs to."""
    used = evaluations_used(log, manifest)
    pending = {}
    for candidate, plan in plans:
        entry = find_evaluation(log, candidate)
        index = entry["evaluation_index"] if entry else None
        if index is None and any(v == "run" for v in plan.values()):
            index = pending.setdefault(candidate["name"], used+len(pending)+1)
        for name, verdict in plan.items():
            where = "" if verdict == "skip" else f" evaluation {index}"
            print(candidate["name"], name, verdict+where)
    print(f"evaluations used {used+len(pending)} of {manifest['cap']} after this invocation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true", help="print skip/resume/run per (candidate, input); launch nothing, write nothing")
    parser.add_argument("--only-input", action="append", help="run this manifest input only (repeatable)")
    parser.add_argument("--only-candidate", action="append", help="launch this candidate of the stage only (repeatable); other candidates of the stage are neither planned nor counted")
    parser.add_argument("--abort-seconds", type=float, help="override the manifest abort; above the manifest's value it needs --abort-override-reason")
    parser.add_argument("--abort-override-reason", help="registered reason for an abort above the manifest's value; written to the log")
    args = parser.parse_args(argv)
    root = repo_root()
    manifest = json.loads(args.manifest.read_text())
    check_manifest(manifest)
    try:
        limit = abort_limit(manifest, args.abort_seconds, args.abort_override_reason)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not stage_ready(root, manifest, args.stage):
        raise SystemExit(f"Stage {args.stage} is gated by {manifest['stages'][str(args.stage)]['gate']}; no decision is recorded.")
    log_path = root/"docs/evidence"/manifest["output_dir"]/"campaign-log.json"
    log = load_log(log_path, manifest)
    chosen = [c for c in manifest["candidates"] if str(c["stage"]) == str(args.stage)]
    if args.only_candidate:
        unknown = [n for n in args.only_candidate if n not in {c["name"] for c in chosen}]
        if unknown:
            raise SystemExit(f"{unknown} is not a candidate of stage {args.stage}; candidates are {[c['name'] for c in chosen]}.")
        chosen = [c for c in chosen if c["name"] in args.only_candidate]
    plans = [(c, plan_inputs(root, manifest, c, log, args.only_input)) for c in chosen]
    print_plan(log, manifest, plans)
    if args.dry_run:
        return
    launches = [(c, [n for n, v in plan.items() if v != "skip"]) for c, plan in plans if any(v != "skip" for v in plan.values())]
    try:
        entries = [register_evaluation(log, manifest, c, limit, args.abort_override_reason) for c, _ in launches]
    except ValueError as error:
        raise SystemExit(str(error)) from error
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    with ThreadPoolExecutor(max_workers=manifest["max_concurrent"]) as pool:
        results = list(pool.map(lambda item: run_candidate(root, manifest, item[0], False, item[1], limit), launches))
    for entry, rows in zip(entries, results):
        record_rows(entry, rows)
    log_path.write_text(json.dumps(log, indent=2))
    for row in (r for rows in results for r in rows):
        print(row["name"], row["input"], f"{row['seconds']:.0f}s", "rc", row["returncode"], "aborted" if row["aborted"] else "")


if __name__ == "__main__":
    main()
