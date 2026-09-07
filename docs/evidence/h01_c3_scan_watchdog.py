"""Resume the 3,000-sample C3 edge-list scan under a stall watchdog.

Launches ``h01_c3_edge_list.py`` so that it resumes from its per-cell checkpoint
(``<output>.cells.json``), watches the child's log for progress lines, kills the child
after ``--stall-seconds`` without one, relaunches at most ``--max-relaunches`` times, and
records every launch and kill with wall clocks. A killed launch is untested, never
negative: the report states ``N of 104 cells scanned`` and nothing more.

Spec: docs/specs/2026-09-07-h01-connectivity-completion.md (SP7a).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROGRESS = re.compile(r"^(cell \d+ \d+ \d+ c3 ids [\d.]+ s"
                      r"|(pre|post)_synaptic_cell \d+ annotations"
                      r"|\{.*\} wall [\d.]+ s)$")


def count_progress_lines(text):
    """Number of lines in ``text`` that the child prints once per finished unit of work."""
    return sum(1 for line in text.splitlines() if PROGRESS.match(line.strip()))


def checkpoint_cells(path):
    """Cells finished in the child's checkpoint, or 0 when it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return 0
    return len(json.loads(path.read_text())["cells"])


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def kill_tree(proc):
    """Kill the child and, on Windows, anything it spawned."""
    proc.kill()
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    proc.wait(timeout=60)


def child_command(args):
    """Command line that resumes the scan; the checkpoint is derived by the child from --output."""
    command = [args.python, "-u", str(args.script), "--limit", str(args.limit), "--cache", str(args.cache),
               "--archive", args.archive, "--output", str(args.output)]
    if args.cells:
        command += ["--cells", *args.cells]
    return command


def watch_launch(command, log_path, stall_seconds, poll_seconds, env, cwd):
    """Run one launch to exit or stall-kill; return its record."""
    record = {"started": now_iso(), "command": command, "log": str(log_path)}
    t0 = time.time()
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        record["pid"] = proc.pid
        seen, last_progress = 0, t0
        while True:
            code = proc.poll()
            lines = count_progress_lines(Path(log_path).read_text(errors="replace"))
            if lines > seen:
                seen, last_progress = lines, time.time()
            if code is not None:
                record.update(outcome="exit", exit_code=code)
                break
            if time.time()-last_progress > stall_seconds:
                kill_tree(proc)
                record.update(outcome="killed_stall", exit_code=None,
                              seconds_without_progress=round(time.time()-last_progress, 1))
                break
            time.sleep(poll_seconds)
    record.update(ended=now_iso(), wall_seconds=round(time.time()-t0, 1), progress_lines=seen)
    return record


def summarise(launches, checkpoint, cells_total, stall_seconds, max_relaunches):
    scanned = checkpoint_cells(checkpoint)
    complete = bool(launches) and launches[-1]["outcome"] == "exit" and launches[-1]["exit_code"] == 0
    return {"checkpoint": str(checkpoint), "cells_total": cells_total, "cells_scanned": scanned,
            "statement": f"{scanned} of {cells_total} cells scanned",
            "status": "complete" if complete else "partial; unscanned cells are untested, not negative",
            "stall_seconds": stall_seconds, "max_relaunches": max_relaunches,
            "launches": launches, "kills": sum(x["outcome"] == "killed_stall" for x in launches)}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Interpreter with cloudvolume.")
    parser.add_argument("--script", type=Path, default=Path("docs/evidence/h01_c3_edge_list.py"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--archive", default="proofread104.zip")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--cells", nargs="*", help="Restrict the child to these neuron ids (dry run).")
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/h01-resolved-edge-list-3000.json"))
    parser.add_argument("--record", type=Path, default=Path("docs/evidence/h01-c3-scan-3000.watchdog.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("docs/evidence"))
    parser.add_argument("--cells-total", type=int, default=104)
    parser.add_argument("--stall-seconds", type=float, default=600.)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--max-relaunches", type=int, default=3)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Worktree root (PYTHONPATH for the child).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.output.with_suffix(".cells.json")
    env = dict(os.environ, PYTHONPATH=str(args.root.resolve()), PYTHONUNBUFFERED="1")
    command = child_command(args)
    launches, failures_without_progress = [], 0
    for launch in range(args.max_relaunches+1):
        before = checkpoint_cells(checkpoint)
        log_path = args.log_dir/f"{args.output.stem}.launch{launch+1}.log"
        record = watch_launch(command, log_path, args.stall_seconds, args.poll_seconds, env, str(args.root))
        record.update(launch=launch+1, cells_before=before, cells_after=checkpoint_cells(checkpoint))
        launches.append(record)
        summary = summarise(launches, checkpoint, args.cells_total, args.stall_seconds, args.max_relaunches)
        args.record.write_text(json.dumps(summary, indent=1)+"\n")
        print(json.dumps({k: record[k] for k in ("launch", "outcome", "exit_code", "cells_after", "wall_seconds")}), flush=True)
        if record["outcome"] == "exit" and record["exit_code"] == 0:
            break
        failures_without_progress = (failures_without_progress+1
                                     if record["outcome"] == "exit" and record["cells_after"] == before else 0)
        if failures_without_progress >= 2:
            summary["status"] += "; stopped: two consecutive exits without progress"
            args.record.write_text(json.dumps(summary, indent=1)+"\n")
            break
    print(summary["statement"], "|", summary["status"], flush=True)
    return summary


if __name__ == "__main__":
    main()
