# SP7: H01 connectivity scan completion

Status: spec written 2026-09-07 before code, under the programme spec
[2026-09-07-h01-population-programme.md](2026-09-07-h01-population-programme.md) (SP0 rules:
detached jobs via `Start-Process`, a progress line at least every 60 s, watchdog kill at
10 min without progress, killed = untested, measured estimate plus approval for anything
predicted over 15 min, never quote an unmeasured duration). Worktree `h01-connectivity`,
branch `feat/h01-connectivity`. Inputs: `docs/evidence/h01-measured-connectivity-progress.md`,
`h01-connectivity-audit.md`, `h01_c3_edge_list.py`, `h01_endpoint_recheck.py`,
`h01_connectivity_audit.py`, `h01_index_connectivity_audit.py`, W6c/W6d outcomes in
[the usable-circuit plan](2026-09-07-h01-usable-circuit-plan.md).

## State on opening

| Quantity | Value | Source |
| --- | --- | --- |
| 1,500-sample edge list | 104/104 cells, 123 candidates, 31 pairs, 3 endpoint-verified | `h01-resolved-edge-list.json` |
| 3,000-sample rescan | 26 of 104 cells, 98-110 s per cell, killed after 17 min without a cell line | `h01-resolved-edge-list-3000.cells.json` (sha256 `30648d8e...`, identical in `h01-braincell` and here) |
| Checkpoint key | `{"limit": 3000, "archive": "proofread104.zip"}` (`h01_c3_edge_list.py:98-113`); cells already present are skipped (`:161-162`) | code |
| Local C3 export shards | 9 (`export000000000000..08`, 197 MB each, 8,991,719 records), all nine already audited | `h01-connectivity-export*-audit.json` |
| `full_export_scanned` | `false`, hand-maintained in `h01-measured-connectivity-summary.json`; no script computes it | grep |
| Largest pair | `5654281423` <-> `4157825456`: 57 + 14 = 71 candidates, mixed type codes, 0 within the 2-voxel box | `h01-resolved-edge-list.json` |
| `c3_ids_shared_by_cells` | emitted by `h01_c3_edge_list.py:181,219`; `{}` in the released list, the pilot, and the 26-cell checkpoint | JSON |

The export listing under `data/20210729/c3/synapses/exported/` was never enumerated past
`maxResults=3` (`h01-c3-metadata-audit.json`), so the denominator of the export scan was
unknown. SP7b fixes that first.

## SP7a. Resume with watchdog

`docs/evidence/h01_c3_scan_watchdog.py` (+ `_test.py`) and the launcher
`docs/evidence/h01_c3_scan_watchdog.ps1`.

- Child: `h01_c3_edge_list.py --limit 3000 --cache <sibling .cache/h01> --archive proofread104.zip
  --output docs/evidence/h01-resolved-edge-list-3000.json`. The checkpoint path is derived by the
  child as `output.with_suffix(".cells.json")`, so this output name resumes from
  `h01-resolved-edge-list-3000.cells.json`. The child skips cells present in the checkpoint and
  saves after each cell, so a kill loses at most the cell in flight.
- Environment: `PYTHONPATH=<worktree root>` (the child imports `docs.evidence.*`),
  `PYTHONUNBUFFERED=1`, `CLOUD_FILES_DIR` left to the child's default under `--cache`.
- Progress line: `^cell \d+ \d+ \d+ c3 ids [\d.]+ s$` (`h01_c3_edge_list.py:180`). The
  two `..._synaptic_cell N annotations` lines and the final counts line also count as progress
  (they occur once the cell loop is complete).
- Watchdog: poll the child's stdout log every 5 s; if no new progress line for
  `--stall-seconds` (default 600) kill the child (`Process.kill()`, then `taskkill /T /F` on
  Windows so any grandchild dies). A kill is recorded as `killed_stall`; the cell in flight is
  untested, never negative.
- Relaunch: at most `--max-relaunches` (default 3) relaunches, i.e. up to 4 launches. Stop
  early when the child exits 0 (writes `h01-resolved-edge-list-3000.json`) or when a launch
  exits non-zero twice in a row without adding a cell (a deterministic failure is not a stall).
- Record: `docs/evidence/h01-c3-scan-3000.watchdog.json` with, per launch, wall-clock start
  and end (ISO 8601 local), pid, exit code or kill reason, cells in the checkpoint before and
  after, seconds since last progress at kill; plus `cells_scanned`, `cells_total` (104, read
  from the archive listing in the checkpoint's cell count is not enough, so the total is
  passed as `--cells-total 104` and cross-checked against `archive_cells` when the archive is
  readable), and the honest line `"N of 104 cells scanned"`.
- The child is launched through `sys.executable`-independent `--python` (the isolated
  cloudvolume interpreter `.worktrees/h01-braincell/.cache/h01-inspect/Scripts/python.exe`).
- Tests use a fake child script written to `tmp_path` that prints cell lines then either
  exits, stalls (sleeps), or fails; `--stall-seconds` is set to about one second. No GCS.
- Launcher: `Start-Process -NoNewWindow -PassThru -RedirectStandardOutput ... -RedirectStandardError ...`
  on the watchdog itself, writing the pid to `docs/evidence/h01-c3-scan-3000.pid` (ignored by
  git). Bash background jobs die at 10 min, so the launcher is PowerShell only.
- Duration (derived, needs approval): 78 remaining cells x 98-110 s (measured anchor) =
  2.1-2.4 h plus one relationship-index query and endpoint pass (950 s measured for the
  1,500-sample run including the resumed cells). Over 15 min, so the full rescan is NOT run
  in this SP without the user's approval. A one-cell dry run (`--cells <id>`) is allowed only
  if it completes under 3 min; a fresh output name is used so the 3,000 checkpoint is not
  mutated by the dry run.

## SP7b. Export-scan completion and the measured-estimate rule

`docs/evidence/h01_export_scan_coverage.py` (+ `_test.py`).

- `listing`: enumerate the export prefix through the JSON API (`maxResults=1000`, follow
  `nextPageToken`), keep `name` and `size` only, write
  `docs/evidence/h01-c3-export-listing.json`. Shards are the objects without a `.json`
  suffix. One small GET; done in this SP.
- `benchmark --shard export000000000000`: time the same fastavro pass and owner lookup that
  `h01_connectivity_audit.py:100-124` performs (imports nothing from CloudVolume; reuses the
  saved crosswalk's owner table by reading `proofread-base-membership.json`), reading the file
  from the sibling cache read-only. Records wall seconds, records per second, bytes.
- `coverage`: from the listing and the local `h01-connectivity-export*-audit.json` files,
  write `docs/evidence/h01-export-scan-coverage.json`: `shards_listed`, `shards_scanned`,
  `records_scanned`, `bytes_scanned`, `bytes_listed`, the statement
  `"N of M shards scanned"`, and `full_export_scanned = (scanned == listed)`. The summary's
  `full_export_scanned` is set from this file, never by hand.
- Measured-estimate rule: the estimate for the remaining shards is
  `remaining_shards x measured_read_seconds` plus `remaining_bytes / measured_download_rate`.
  Both factors must be measured in this session (one shard read; one shard download, if run)
  before the total is quoted, and the total is labelled derived. Anything over 15 min needs
  approval; the local nine are not re-run because rerunning them cannot change the flag.

## SP7c. Merge check on the 71-candidate pair

`docs/evidence/h01_pair_merge_check.py` (+ `_test.py`), offline, writes
`docs/evidence/h01-pair-merge-check.json` and `.md`.

- Identify the pair as the unordered pair with the most candidates in the edge list.
- Evidence per row (each of the 71 candidates): direction, type code, proofread label read at
  the pre and post voxel, nearest offsets, whether either endpoint reads the pre cell, the
  post cell, background, or a third label.
- Pair-level evidence: `c3_ids_shared_by_cells` restricted to the two cells (from the edge
  list and from any checkpoint given with `--checkpoint`), the cells' `identity_consistent`
  flags and sample label counts, direction and type-code mix, count of rows whose endpoints
  fall on one proofread label (the same label at both ends), count on background.
- Verdict rule:
  - `yes` when a C3 label is shared by the two cells, or when at least one row reads the same
    nonzero proofread label at both endpoints (a self-contact recorded as a two-cell contact).
  - `no` when no label is shared and at least one row reads the pre cell at the pre voxel and
    the post cell at the post voxel (an ordinary verified contact exists).
  - `undetermined` otherwise. Expected here: no shared label, 70 of 71 rows background at
    both endpoints, so the C3 ids that carried the candidates are not co-located with either
    proofread cell at the annotation voxel; whether they are merged with a third object cannot
    be read from the saved JSON. The decisive follow-up (online, not run here) is one
    `scattered_points` read of the C3 label at the 142 endpoints, checked against each cell's
    `c3_ids`.

## SP7d. Recheck trigger

Any edge that becomes endpoint-verified or within-box in `h01-resolved-edge-list-3000.json`
and was not in the 1,500-sample list is rechecked with
`h01_endpoint_recheck.py --edge-list docs/evidence/h01-resolved-edge-list-3000.json --ids <new ids>`
(17-19 s per edge measured, 300 s budget each); then `prepare_connectivity` -> topology for
SP8. Not triggered in this session because the rescan has not run.

## Gate and stop rule

- Gate: either `full_export_scanned` true in `h01-export-scan-coverage.json`, or an explicit
  coverage statement in both forms: `"N of 104 cells scanned"` at 3,000 samples and
  `"N of M shards scanned"` for the export. Absence of a contact is never established by
  either scan.
- Stop rule: 3 relaunches of the rescan; after the fourth launch ends the run is closed at
  whatever `N of 104` it reached and reported as partial, untested for the rest.
- Causal-model rule: the merge-check observation and the coverage state enter
  `docs/h01-causal-model.md` Y5 in the same commit as the JSON.

## Results (2026-09-07, same session)

- SP7a tooling delivered and tested (4 tests, fake child: finish, stall-kill x3, deterministic
  failure stop). Dry run through the launcher: 1 cell, 70.0 s, exit 0, no kill. Full resume
  not run (needs approval).
- SP7b: listing 166 shards; measured single-shard read 36.7 s, nine repeats 33.0-47.9 s
  (mean 38.4 s); measured download 36.3 MB/s (one shard). Derived remaining: about 6,886 s.
  `full_export_scanned` false = "9 of 166 shards scanned". The spec's "N-of-9" framing was
  wrong: the local nine were already scanned; the denominator is 166.
- SP7c: verdict undetermined (no shared C3 label; 70/71 rows background at both ends).
- SP7d: not triggered.
