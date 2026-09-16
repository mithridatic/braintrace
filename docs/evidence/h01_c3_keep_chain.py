"""Write the two Vast chain scripts of the C3 test-to-failure gate (spec step C).

Spec: docs/specs/2026-09-16-h01-c3-partner-expansion.md. Every candidate cell gets its
donor step run (primary) immediately followed by its ramp run (0 to 3x the donor test
current); every kept cell gets the ramp only (its primary and dt-half runs are the
keep/drop receipts). Cells are dealt round-robin into ``n`` chains in this order:
interneuron candidates (rank order), pyramidal candidates (rank order), the 17 kept cells
(decision order), so the two runs of one cell stay adjacent in one chain. Lines are the
``h01_keep_drop_chain.command`` launcher lines; candidate lines carry the archive,
components and cell-table arguments, kept lines use the proofread defaults. The dt-half
repeat of a candidate that holds the measured rules is launched by
``h01_keep_drop_follower.py --gate failure``.

``--dry-run`` prints the first commands of each chain, the run count, and the wall-time
estimate from the earlier keep/drop campaign (190 s per run, launch-bound, independent of
the chain count) without writing anything.
"""

import argparse
import json
from pathlib import Path

from docs.evidence.h01_keep_drop_chain import CPU_SETS, command, rows_from_types

SPEC = "docs/specs/2026-09-16-h01-c3-partner-expansion.md"
SECONDS_PER_RUN_ESTIMATE = 190.   # aggregate per-run cost measured in the 2026-09-16 keep/drop campaign
DEFAULTS = dict(workdir="/workspace/braintrace-c3-keep", runner="docs/evidence/h01-c3-keep-drop/run_transfer.sh",
                runs="var/c3-keep", archive=".cache/h01/c3-candidates-20260916.zip",
                archive_sha256="75d15798e35cb1f111198f89c03e5210dec2cbd7f42f233783f46b1eb626faf7",
                components="docs/evidence/h01-c3-candidates-components.json",
                cell_table=".cache/h01/c3-segment-properties.json",
                cell_table_sha256="6178bfe1c0a876001d02f641dda1d5af3b1d3512935b6114d4765e7988c44b3f")


def candidate_extra(archive, archive_sha256, components, cell_table, cell_table_sha256=None):
    """The arguments every candidate line carries: the C3 archive, its inventory and its tag table."""
    extra = f"--archive {archive} --archive-sha256 {archive_sha256} --components {components} --cell-table {cell_table}"
    if cell_table_sha256:
        extra += f" --cell-table-sha256 {cell_table_sha256}"
    return extra


def _launched(skip_done, cell, ramp=False):
    return skip_done is not None and (Path(skip_done)/f"transfer-all-{cell}{'-ramp' if ramp else ''}"/"launch.json").exists()


def jobs(candidate_rows, kept_rows, extra, skip_done=None):
    """One job per cell: ``(cell, donor, kind, lines)``; candidates primary then ramp, kept ramp only.

    With ``skip_done`` (a run folder) a run whose ``launch.json`` exists there is omitted, so a
    re-queued chain continues where the last one stopped; a cell with nothing left is dropped.
    """
    ordered = []
    for cell, donor in candidate_rows:
        lines = [] if _launched(skip_done, cell) else [command(cell, donor, extra=extra)]
        if not _launched(skip_done, cell, ramp=True):
            lines.append(command(cell, donor, ramp=True, extra=extra))
        if lines:
            ordered.append((cell, donor, "candidate", lines))
    for cell, donor in kept_rows:
        if not _launched(skip_done, cell, ramp=True):
            ordered.append((cell, donor, "kept", [command(cell, donor, ramp=True)]))
    return ordered


def deal(items, n):
    """Round-robin ``items`` into ``n`` lists, order preserved within each."""
    return [items[k::n] for k in range(n)]


def chain_text(chain, k, n, workdir, runner, runs):
    """The script of chain ``k``: header, one line per run, the done file the follower waits for."""
    lines = ["#!/usr/bin/env bash", f"# C3 test-to-failure chain {k} of {n}; gate and protocol in {SPEC}.",
             f"# {sum(len(job[3]) for job in chain)} runs on {len(chain)} cells; receipts under {runs}/.",
             f"cd {workdir}", f"R={runner}", f"export CPUSET={CPU_SETS[k-1]} MEMFRAC=.2"]
    for _, _, _, commands in chain:
        lines += commands
    lines.append(f"echo CHAIN-DONE > {runs}/keep-chain-{k}.done")
    return "\n".join(lines)+"\n"


def follower_text(n, workdir, runner, runs, types, extra, python="/workspace/venv314/bin/python"):
    """The dt-half follower launch line for the candidate cells (same extra arguments as their chain lines)."""
    return "\n".join([
        "#!/usr/bin/env bash",
        f"# dt-half follower of the C3 test-to-failure gate; waits for {n} chain done files under {runs}/.",
        f"cd {workdir}",
        f"PYTHONPATH=. exec {python} docs/evidence/h01_keep_drop_follower.py --folder {runs} --types {types} "
        f"--gate failure --datums docs/evidence/h01-c3-keep-drop/human-datums.json --chains {n} --runner {runner} "
        f"--output {runs}/keep-primary-decision.json --cpuset 64-95 --memfrac .12 --extra-args \"{extra}\"",
    ])+"\n"


def write_chains(job_list, out_dir, n, workdir, runner, runs, types=None, extra=None):
    """Write keep-chain-<k>.sh for k = 1..n (and follower.sh when ``types`` is given); returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, chain in enumerate(deal(job_list, n), start=1):
        path = out_dir/f"keep-chain-{k}.sh"
        path.write_text(chain_text(chain, k, n, workdir, runner, runs), newline="\n")
        paths.append(path)
    if types is not None:
        path = out_dir/"follower.sh"
        path.write_text(follower_text(n, workdir, runner, runs, types, extra or ""), newline="\n")
        paths.append(path)
    return paths


def dry_run_report(job_list, n, first=3):
    """Per-chain first commands, run counts, and the wall-time estimate (190 s per run, earlier campaign)."""
    chains = deal(job_list, n)
    total = sum(len(job[3]) for job in job_list)
    is_ramp = lambda line: "--ramp-na" in line  # noqa: E731
    report = dict(chains=[], runs_total=total, cells_total=len(job_list),
                  runs_by_kind=dict(candidate_primary=sum(1 for j in job_list for l in j[3] if j[2] == "candidate" and not is_ramp(l)),
                                    candidate_ramp=sum(1 for j in job_list for l in j[3] if j[2] == "candidate" and is_ramp(l)),
                                    kept_ramp=sum(1 for j in job_list for l in j[3] if j[2] == "kept")),
                  seconds_per_run_estimate=SECONDS_PER_RUN_ESTIMATE,
                  wall_estimate_s=total*SECONDS_PER_RUN_ESTIMATE,
                  estimate_basis=("190 s per run: the aggregate per-cell cost measured in the 2026-09-16 keep/drop "
                                  "campaign (launch-bound, independent of chain count); an estimate, not a "
                                  "measurement of these runs; dt-half repeats of passing cells come on top"))
    for k, chain in enumerate(chains, start=1):
        commands = [line for job in chain for line in job[3]]
        report["chains"].append(dict(chain=k, cells=len(chain), runs=len(commands), first_commands=commands[:first]))
    return report


def print_report(report):
    for chain in report["chains"]:
        print(f"chain {chain['chain']}: {chain['runs']} runs on {chain['cells']} cells")
        for line in chain["first_commands"]:
            print("  "+line)
    kinds = report["runs_by_kind"]
    print(f"total {report['runs_total']} runs before dt-half ({kinds['candidate_primary']} candidate primary + "
          f"{kinds['candidate_ramp']} candidate ramp + {kinds['kept_ramp']} kept ramp) on {report['cells_total']} cells")
    hours = report["wall_estimate_s"]/3600.
    print(f"estimate: {report['runs_total']} x {report['seconds_per_run_estimate']:g} s = {report['wall_estimate_s']:.0f} s "
          f"({hours:.1f} h) -- {report['estimate_basis']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", type=Path, default=Path("docs/evidence/h01-c3-keep-drop/types.json"))
    parser.add_argument("--kept-types", type=Path, default=Path("docs/evidence/h01-c3-keep-drop/kept-types.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/evidence/h01-c3-keep-drop/chains"))
    parser.add_argument("--chains", type=int, default=2)
    for name, default in DEFAULTS.items():
        parser.add_argument("--"+name.replace("_", "-"), default=default)
    parser.add_argument("--skip-done", type=Path, help="run folder; runs already launched there (launch.json) are omitted")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="write the dry-run report JSON here")
    args = parser.parse_args(argv)
    extra = candidate_extra(args.archive, args.archive_sha256, args.components, args.cell_table, args.cell_table_sha256)
    job_list = jobs(rows_from_types(args.types), rows_from_types(args.kept_types), extra, skip_done=args.skip_done)
    report = dry_run_report(job_list, args.chains)
    print_report(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1)+"\n", newline="\n")
    if args.dry_run:
        return
    for path in write_chains(job_list, args.out_dir, args.chains, args.workdir, args.runner, args.runs,
                             types=args.types.as_posix(), extra=extra):
        print(path)


if __name__ == "__main__":
    main()
