"""Register the kept-network driven-window runs: plan JSONs and the two-at-a-time queue script.

Spec: docs/specs/2026-09-16-h01-keep-drop.md, Step 2. Seven runs through
``var/h01-driven/run_h01.sh``: a 1 ms construction reference, four 50 ms matched controls
and the 10 ms refinement pair. Every kept cell receives its donor's primary current from
2 ms for 38 ms (``--currents-json``), replacing the population's 1 nA 2-5 ms probe.
"""

import argparse
import json
from pathlib import Path

SPEC = "docs/specs/2026-09-16-h01-keep-drop.md"
PULSE = dict(delay_ms=2., duration_ms=38.)
COMMON = dict(solver="h01_staggered_calcium_implicit", max_cv_um=10., wall_cap_seconds=14400)
RUNS = [
    dict(label="kept-bench-ei-1ms", control="ei", dt_ms=.000625, duration_ms=1., role="construction reference"),
    dict(label="kept-ctrl-ei-50ms", control="ei", dt_ms=.000625, duration_ms=50., role="control"),
    dict(label="kept-ctrl-e_only-50ms", control="e_only", dt_ms=.000625, duration_ms=50., role="control"),
    dict(label="kept-ctrl-i_only-50ms", control="i_only", dt_ms=.000625, duration_ms=50., role="control"),
    dict(label="kept-ctrl-disconnected-50ms", control="disconnected", dt_ms=.000625, duration_ms=50., role="control"),
    dict(label="kept-refine-ei-10ms-dt000625", control="ei", dt_ms=.000625, duration_ms=10., role="refinement coarse"),
    dict(label="kept-refine-ei-10ms-dt0003125", control="ei", dt_ms=.0003125, duration_ms=10., role="refinement fine"),
]


def plan(run, cells, inputs, source_commit):
    """One registered plan; ``cells`` is the kept count the runtime gate checks."""
    steps = int(round(run["duration_ms"]/run["dt_ms"]))
    return dict(status="registered before launch; untested", spec=SPEC, source_commit=source_commit,
                executor="Vast 50616476 RTX 4090, /workspace/venv314, branch campaign/h01-driven-window-20260914",
                cells=int(cells), current_na="per cell from "+inputs["currents"], pulse_window_ms=[PULSE["delay_ms"],
                PULSE["delay_ms"]+PULSE["duration_ms"]], topology=inputs["topology"], components=inputs["components"],
                control=run["control"], dt_ms=run["dt_ms"], duration_ms=run["duration_ms"], steps=steps, role=run["role"],
                qualification="finite driven runtime, matched controls and numerical refinement on the kept set only; "
                              "physiology and human provenance are not qualified", **COMMON)


def command(run, cells, inputs):
    """The receipted launcher line for one run (CPU set chosen by the queue)."""
    return (f"$R {run['label']} {COMMON['wall_cap_seconds']} -- --topology {inputs['topology']} --cache .cache/h01 "
            f"--solver {COMMON['solver']} --include-isolated --cells {int(cells)} --components {inputs['components']} "
            f"--currents-json {inputs['currents']} --pulse-delay-ms {PULSE['delay_ms']:g} "
            f"--pulse-duration-ms {PULSE['duration_ms']:g} --max-cv-um {COMMON['max_cv_um']:g} --heartbeat-s 60 "
            f"--duration-ms {run['duration_ms']:g} --dt-ms {run['dt_ms']:g} --control {run['control']}")


def queue_script(cells, inputs):
    """Bench first alone, then the remaining six two at a time; each pair waits for both terminals."""
    lines = ["#!/usr/bin/env bash", f"# Kept-network driven window; registered in {SPEC}.", "cd /workspace/braintrace",
             "R=var/h01-driven/run_h01.sh",
             "wait_done() { while [ ! -f var/h01-driven/$1/terminal.json ]; do sleep 60; done; }",
             "run() { CPUSET=$1 nohup $2 > /dev/null 2>&1 & }"]
    bench, rest = RUNS[0], RUNS[1:]
    lines += [f"CPUSET=0-63 {command(bench, cells, inputs)}"]
    for first, second in zip(rest[0::2], rest[1::2]):
        lines += [f'run 0-63 "{command(first, cells, inputs)}"', "sleep 600",
                  f'run 64-127 "{command(second, cells, inputs)}"',
                  f"wait_done {first['label']}", f"wait_done {second['label']}"]
    lines.append("echo QUEUE-DONE > var/h01-driven/kept-window.done")
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, required=True, help="kept cell count")
    parser.add_argument("--topology", default="docs/evidence/h01-kept-network.json")
    parser.add_argument("--components", default="docs/evidence/h01-kept-components.json")
    parser.add_argument("--currents", default="docs/evidence/h01-keep-drop/kept-currents.json")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = dict(topology=args.topology, components=args.components, currents=args.currents)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        path = args.out_dir/f"plan-{run['label']}.json"
        path.write_text(json.dumps(plan(run, args.cells, inputs, args.source_commit), indent=2)+"\n", newline="\n")
        print(path)
    script = args.out_dir/"kept-window-queue.sh"
    script.write_text(queue_script(args.cells, inputs), newline="\n")
    print(script)


if __name__ == "__main__":
    main()
