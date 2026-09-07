"""Measure the wall-clock cost of the 4-cell verified H01 network per simulated millisecond.

Two subcommands. ``run`` launches ``examples.h01_verified_network`` once for one
duration as a child process, samples the child tree's resident memory, kills the
child when its log is silent for a watchdog period (killed means untested, never
negative), and writes ``run.json`` next to the example outputs. ``report`` reads
every ``run.json`` under ``bench-*`` directories, fits T(steps) = a + b*steps by
least squares over the completed ``h01_staggered_scan`` runs, derives the
two-repeat decision limit, and writes the JSON decision record and its page.

No physiology is asserted; the network is the measurement instrument being costed.
Spec: ``docs/specs/2026-09-07-h01-network-throughput.md``.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

DECISION_LIMIT_FACTOR = 2.95
"""Decision-limit factor for two repeats (Hartshorne 2020, pp. 199-201)."""

RUN_CAP = 5
DEFAULT_DT_MS = .005
STEPS_10MS = 2000
EMPTY_RECORD = {"construction_seconds": None, "initialization_and_run_seconds": None, "dt_ms": None,
                "duration_ms": None, "steps": None}
MAIN_SOLVER = "h01_staggered_scan"


def step_count(duration_ms, dt_ms):
    """Return the number of solver steps for a duration at a fixed step.

    Parameters
    ----------
    duration_ms : float
        Simulated duration in milliseconds.
    dt_ms : float
        Solver step in milliseconds; must be positive.

    Returns
    -------
    int
        ``round(duration_ms / dt_ms)``.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_network_throughput import step_count
        >>> step_count(1., .005)
        200
    """
    if dt_ms <= 0:
        raise ValueError("dt must be positive")
    return int(round(duration_ms/dt_ms))


def parse_build_record(path):
    """Read the timing fields the example writes to ``<output>-build.json``.

    Parameters
    ----------
    path : pathlib.Path
        Build record path; a missing file yields ``EMPTY_RECORD``.

    Returns
    -------
    dict
        ``construction_seconds``, ``initialization_and_run_seconds``, ``dt_ms``,
        ``duration_ms`` and derived ``steps`` (``None`` when the run never started).
    """
    path = Path(path)
    if not path.exists():
        return dict(EMPTY_RECORD)
    record = json.loads(path.read_text())
    parsed = {key: record.get(key) for key in ("construction_seconds", "initialization_and_run_seconds", "dt_ms", "duration_ms")}
    ran = parsed["initialization_and_run_seconds"] is not None and parsed["dt_ms"] is not None
    parsed["steps"] = step_count(parsed["duration_ms"], parsed["dt_ms"]) if ran else None
    return parsed


def fit_linear(steps, times):
    """Fit T(steps) = a + b*steps by ordinary least squares.

    Parameters
    ----------
    steps : sequence of int
        Step counts; at least two distinct values are required.
    times : sequence of float
        Measured ``initialization_and_run_seconds`` for each point.

    Returns
    -------
    dict
        ``a`` (seconds, init + compile), ``b`` (seconds per step), ``residuals``
        (measured minus fitted, seconds), ``points``, ``steps`` and ``times``.
    """
    if len(set(steps)) < 2:
        raise ValueError("a line needs two distinct step counts")
    n = len(steps)
    mean_x, mean_y = sum(steps)/n, sum(times)/n
    sxx = sum((x-mean_x)**2 for x in steps)
    sxy = sum((x-mean_x)*(y-mean_y) for x, y in zip(steps, times))
    b = sxy/sxx
    a = mean_y-b*mean_x
    residuals = [y-(a+b*x) for x, y in zip(steps, times)]
    return {"a": a, "b": b, "residuals": residuals, "points": n, "steps": list(steps), "times": list(times)}


def predict(fit, steps):
    """Return the fitted time at ``steps``."""
    return fit["a"]+fit["b"]*steps


def decision_limit(values):
    """Return range x 2.95 for a repeat pair, or ``None`` with fewer than two values."""
    if len(values) < 2:
        return None
    return (max(values)-min(values))*DECISION_LIMIT_FACTOR


def _fit_runs(runs):
    return [r for r in runs if r["solver"] == MAIN_SOLVER and r["status"] == "completed"
            and r.get("initialization_and_run_seconds") is not None]


def repeat_slopes(runs):
    """Return one slope estimate per 1 ms repeat, each against the shortest completed point.

    Parameters
    ----------
    runs : list of dict
        Run records; only completed ``h01_staggered_scan`` runs are used.

    Returns
    -------
    list of float
        ``(T_repeat - T_anchor) / (steps_repeat - steps_anchor)`` for every point
        whose step count equals the most repeated non-anchor step count.
    """
    usable = sorted(_fit_runs(runs), key=lambda r: r["steps"])
    if len(usable) < 2:
        return []
    anchor, rest = usable[0], usable[1:]
    counts = {}
    for run in rest:
        counts[run["steps"]] = counts.get(run["steps"], 0)+1
    repeated = max(counts, key=lambda s: (counts[s], -s))
    return [(r["initialization_and_run_seconds"]-anchor["initialization_and_run_seconds"])/(r["steps"]-anchor["steps"])
            for r in rest if r["steps"] == repeated]


def check_cap(runs, cap=RUN_CAP):
    """Raise when ``cap`` runs were already launched; otherwise return the next run number."""
    if len(runs) >= cap:
        raise RuntimeError(f"run cap {cap} reached; {len(runs)} runs already launched")
    return len(runs)+1


def is_silent(last_progress, now, silence_seconds):
    """Return whether the watchdog must kill a child whose log last grew at ``last_progress``."""
    return now-last_progress >= silence_seconds


def tree_rss_mb(process):
    """Sum resident memory (MiB) of a process and its descendants, ignoring ones that exit mid-sample."""
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total/2**20


def build_command(python, run_dir, duration_ms, solver, cache, topology, dt_ms):
    """Assemble the example invocation for one benchmark run.

    Parameters
    ----------
    python : str
        Interpreter path.
    run_dir : pathlib.Path
        Directory receiving ``network*.json/npz``.
    duration_ms, dt_ms : float
        Simulated duration and step.
    solver : str
        ``h01_staggered_scan`` or ``staggered``.
    cache, topology : pathlib.Path
        H01 cache directory (archive + annotations) and prepared topology JSON.

    Returns
    -------
    list of str
        Argument vector.
    """
    return [python, "-u", "-m", "examples.h01_verified_network", "--topology", str(topology), "--cache", str(cache),
            "--build", "--dt-ms", repr(float(dt_ms)), "--duration-ms", repr(float(duration_ms)),
            "--solver", solver, "--output", str(Path(run_dir)/"network")]


def load_runs(root):
    """Return every ``bench-*/run.json`` under ``root`` sorted by start time."""
    records = [json.loads(p.read_text()) for p in sorted(Path(root).glob("bench-*/run.json"))]
    return sorted(records, key=lambda r: r.get("started_at", ""))


def summarize(runs, anchor_seconds, cap_seconds):
    """Fit, limit and pre-registered decisions over a list of run records.

    Parameters
    ----------
    runs : list of dict
        Run records from ``run.json``.
    anchor_seconds : float
        The one-step anchor (157 s) the 0.05 ms prediction is tested against.
    cap_seconds : float
        Per-run cap the predicted 10 ms cost must stay under.

    Returns
    -------
    dict
        Fit, ``seconds_per_simulated_ms`` (200*b at dt 0.005), decision limits,
        linearity verdict, prediction verdict, 10 ms gate and peak RSS.
    """
    usable = _fit_runs(runs)
    slopes = repeat_slopes(runs)
    limit_b = decision_limit(slopes)
    summary = {"fit": None, "seconds_per_simulated_ms": None, "decision_limit_b": limit_b, "decision_limit_seconds": None,
               "repeat_slopes": slopes, "linear_within_limit": None, "max_abs_residual_seconds": None,
               "prediction_0p05_within_anchor": None, "predicted_10ms_seconds": None, "run_10ms_allowed": None,
               "anchor_seconds": anchor_seconds, "cap_seconds": cap_seconds,
               "untested": [r["label"] for r in runs if r["status"] != "completed"],
               "controls": [r for r in runs if r["solver"] != MAIN_SOLVER],
               "peak_rss_mb": max((r["peak_rss_mb"] for r in runs if r.get("peak_rss_mb") is not None), default=None)}
    if len({r["steps"] for r in usable}) < 2:
        return summary
    fit = fit_linear([r["steps"] for r in usable], [r["initialization_and_run_seconds"] for r in usable])
    summary["fit"] = fit
    summary["seconds_per_simulated_ms"] = fit["b"]/usable[0]["dt_ms"]
    summary["max_abs_residual_seconds"] = max(abs(r) for r in fit["residuals"])
    summary["predicted_10ms_seconds"] = predict(fit, STEPS_10MS)
    summary["run_10ms_allowed"] = summary["predicted_10ms_seconds"] < cap_seconds
    if limit_b is not None:
        repeated_steps = max(_repeated_steps(usable))
        repeats = [r for r in usable if r["steps"] == repeated_steps]
        summary["decision_limit_seconds"] = decision_limit([r["initialization_and_run_seconds"] for r in repeats])
        shortest = min(usable, key=lambda r: r["steps"])
        summary["prediction_0p05_within_anchor"] = abs(shortest["initialization_and_run_seconds"]-anchor_seconds) <= summary["decision_limit_seconds"]
        if len({r["steps"] for r in usable}) >= 3:
            summary["linear_within_limit"] = summary["max_abs_residual_seconds"] <= summary["decision_limit_seconds"]
    return summary


def _repeated_steps(usable):
    counts = {}
    for run in usable:
        counts[run["steps"]] = counts.get(run["steps"], 0)+1
    return {s for s, c in counts.items() if c >= 2}


def _fmt(value, digits=3):
    return "not tested" if value is None else f"{value:.{digits}f}"


def render_markdown(runs, summary):
    """Render the result page from run records and their summary."""
    lines = ["# H01 verified-network throughput (SP1)", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')}. dt 0.005 ms, 4 cells, 2 projections, "
             "75,605 compartments. Cost qualification only; no physiology claim.", "",
             "| Run | Solver | Duration ms | Steps | Construction s | Init+run s | Wall s | Peak RSS MB | Status |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in runs:
        lines.append(f"| {r['label']} | {r['solver']} | {r['duration_ms']} | {r['steps']} | {_fmt(r.get('construction_seconds'), 1)} | "
                     f"{_fmt(r.get('initialization_and_run_seconds'), 1)} | {_fmt(r.get('wall_seconds'), 1)} | "
                     f"{_fmt(r.get('peak_rss_mb'), 0)} | {r['status']} |")
    fit = summary["fit"]
    lines += ["", "## Fit T(steps) = a + b*steps (h01_staggered_scan, completed runs)", ""]
    if fit is None:
        lines.append("Fewer than two distinct durations completed; no fit.")
    else:
        lines += [f"- a (init + compile): {fit['a']:.2f} s", f"- b (per step): {fit['b']:.5f} s",
                  f"- 200*b, seconds per simulated ms: {summary['seconds_per_simulated_ms']:.3f} s",
                  f"- residuals (s): {', '.join(f'{x:.2f}' for x in fit['residuals'])}",
                  f"- two-repeat decision limit on b: {_fmt(summary['decision_limit_b'], 5)} s/step; in seconds at the repeated point: {_fmt(summary['decision_limit_seconds'])}",
                  f"- prediction (0.05 ms within {summary['anchor_seconds']} s +/- limit): {_fmt_bool(summary['prediction_0p05_within_anchor'])}",
                  f"- three points linear within limit: {_fmt_bool(summary['linear_within_limit'])}",
                  f"- predicted 10 ms cost: {summary['predicted_10ms_seconds']:.1f} s (cap {summary['cap_seconds']} s; allowed: {summary['run_10ms_allowed']})"]
    if summary["untested"]:
        lines += ["", "Killed or incomplete (untested, not negative): "+", ".join(summary["untested"])]
    return "\n".join(lines)+"\n"


def _fmt_bool(value):
    return "not tested" if value is None else ("held" if value else "REJECTED")


def report(root, output_json, anchor_seconds, cap_seconds):
    """Aggregate ``bench-*/run.json`` under ``root`` into the decision JSON and page."""
    runs = load_runs(root)
    summary = summarize(runs, anchor_seconds, cap_seconds)
    record = {"spec": "docs/specs/2026-09-07-h01-network-throughput.md", "runs": runs, "summary": summary}
    output_json = Path(output_json)
    output_json.write_text(json.dumps(record, indent=2)+"\n")
    output_json.with_suffix(".md").write_text(render_markdown(runs, summary))
    return record


def run_once(args):
    """Launch one benchmark child, sample its RSS, enforce the watchdog and write ``run.json``."""
    run_dir = args.root/f"bench-{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    check_cap(load_runs(args.root))
    command = build_command(args.python, run_dir, args.duration_ms, args.solver, args.cache, args.topology, args.dt_ms)
    log = run_dir/"child.log"
    record = {"label": args.label, "solver": args.solver, "duration_ms": args.duration_ms, "dt_ms": args.dt_ms,
              "steps": step_count(args.duration_ms, args.dt_ms), "command": command, "log": str(log),
              "started_at": datetime.now().isoformat(timespec="seconds"), "status": "running", "peak_rss_mb": 0.}
    started = time.monotonic()
    with log.open("wb") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(args.worktree))
    process, last_size, last_progress = psutil.Process(child.pid), -1, time.monotonic()
    while child.poll() is None:
        record["peak_rss_mb"] = max(record["peak_rss_mb"], tree_rss_mb(process))
        size = log.stat().st_size
        if size != last_size:
            last_size, last_progress = size, time.monotonic()
        if is_silent(last_progress, time.monotonic(), args.silence_seconds):
            for member in [*process.children(recursive=True), process]:
                member.kill()
            record["status"] = "killed"
            print(f"watchdog: killed after {args.silence_seconds} s of silence", flush=True)
        time.sleep(1.)
    record["wall_seconds"] = time.monotonic()-started
    record["finished_at"] = datetime.now().isoformat(timespec="seconds")
    record["exit_code"] = child.returncode
    record.update({k: v for k, v in parse_build_record(run_dir/"network-build.json").items() if v is not None})
    if record["status"] == "running":
        record["status"] = "completed" if child.returncode == 0 and record["initialization_and_run_seconds"] is not None else "failed"
    (run_dir/"run.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in ("label", "status", "wall_seconds", "construction_seconds",
                                              "initialization_and_run_seconds", "peak_rss_mb")}), flush=True)
    return record


def main(argv=None):
    """Command-line entry: ``run`` one duration or ``report`` all runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--duration-ms", type=float, required=True)
    run.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS)
    run.add_argument("--solver", default=MAIN_SOLVER, choices=[MAIN_SOLVER, "staggered"])
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--worktree", type=Path, default=Path.cwd())
    run.add_argument("--root", type=Path, default=Path(".cache/h01"))
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--topology", type=Path, default=Path("docs/evidence/h01-verified-network.json"))
    run.add_argument("--silence-seconds", type=float, default=600.)
    rep = sub.add_parser("report")
    rep.add_argument("--root", type=Path, default=Path(".cache/h01"))
    rep.add_argument("--output", type=Path, default=Path("docs/evidence/h01-network-throughput.json"))
    rep.add_argument("--anchor-seconds", type=float, default=156.96669389998715)
    rep.add_argument("--cap-seconds", type=float, default=900.)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_once(args)
    else:
        report(args.root, args.output, args.anchor_seconds, args.cap_seconds)


if __name__ == "__main__":
    main()
