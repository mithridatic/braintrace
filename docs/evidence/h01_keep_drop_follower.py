"""Launch the dt-half repeat for each keep/drop candidate as soon as its primary run holds rules 1-4.

Spec: docs/specs/2026-09-16-h01-keep-drop.md. Runs on the Vast box beside the primary chains:
every ``--poll`` seconds it re-evaluates the primary phase, then runs (one at a time, blocking)
the dt-half command of every candidate that has no ``transfer-all-<cell>-dthalf/launch.json``
yet. It exits once every primary chain has written its done file and no candidate is pending.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from docs.evidence.h01_anatomy_transfer_decision import decide_keep
from docs.evidence.h01_keep_drop_chain import command, rows_from_types


def pending(folder, candidates, donors):
    """Candidates whose dt-half run has not been launched, with their donor keys."""
    folder = Path(folder)
    return [(cell, donors[cell]) for cell in candidates
            if not (folder/f"transfer-all-{cell}-dthalf"/"launch.json").exists()]


def chains_done(folder, n=4):
    return all((Path(folder)/f"keep-chain-{k}.done").exists() for k in range(1, n+1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="var/h01-driven")
    parser.add_argument("--types", default="docs/evidence/h01-population-types.json")
    parser.add_argument("--rests", default="docs/evidence/h01-keep-drop/donor-rest.json")
    parser.add_argument("--output", default="var/h01-driven/keep-primary-decision.json")
    parser.add_argument("--poll", type=float, default=120.)
    parser.add_argument("--cpuset", default="128-159")
    parser.add_argument("--memfrac", default=".12")
    args = parser.parse_args()
    donors = dict(rows_from_types(args.types))
    while True:
        decision = decide_keep(args.folder, args.types, args.rests, phase="primary")
        Path(args.output).write_text(json.dumps(decision, indent=2)+"\n")
        queue = pending(args.folder, decision["candidates"], donors)
        for cell, donor in queue:
            line = command(cell, donor, dt_half=True).replace("$R", f"{args.folder}/run_transfer.sh", 1)
            print(time.strftime("%FT%TZ", time.gmtime()), "dt-half", cell, donor, flush=True)
            subprocess.run(["bash", "-c", line], env=dict(CPUSET=args.cpuset, MEMFRAC=args.memfrac,
                                                          PATH="/usr/local/bin:/usr/bin:/bin", HOME="/root"), check=False)
        if chains_done(args.folder) and not queue:
            print(time.strftime("%FT%TZ", time.gmtime()), "follower done", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
