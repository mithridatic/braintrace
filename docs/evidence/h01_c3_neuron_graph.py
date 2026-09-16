"""Stream all 166 H01 C3 synapse-export shards and keep the rows whose pre and
post ``neuron_id`` are both tagged ``neuron`` in the C3 cell table.

Layout (reused from ``h01_full_export_join.py``): curl download with a size
check, atomic progress JSON rewritten after every shard, shard deleted after
its scan, free-disk guard, resumable. New here: a process pool scans shards in
parallel (fastavro decoding is CPU-bound) while a small thread pool keeps the
next shards downloading. The pool uses the ``spawn`` context: with ``fork``,
a worker forked while a download thread sat inside ``subprocess.run`` inherited
that curl's pipe, its ``communicate()`` never saw EOF, and the run hung on one
shard (observed on shard 5 of the first box run, 2026-09-16). Each scanned shard writes its kept rows to a part file
under ``.cache/h01/c3-graph/``; the final merge concatenates the parts in shard
order into ``docs/evidence/h01-c3-neuron-graph.json.gz``.

Graph row layout (``columns`` in the output): ``pre, post, type, x, y, z,
pre_class, post_class, confidence, shard``; ``type`` 1 = inhibitory, 2 =
excitatory; ``x, y, z`` in 8/8/33 nm voxels; ``shard`` is the integer N of
``export{N:012d}``; confidence rounded to four decimals.

Run from the worktree root on the box (receipts under var/c3-graph/):
    /workspace/venv-c3/bin/python docs/evidence/h01_c3_neuron_graph.py --workers 6
"""

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import multiprocessing
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/exported/"
LISTING = ROOT / "docs/evidence/h01-c3-export-listing.json"
TABLE = ROOT / "docs/evidence/h01-c3-cell-table.json"
PROGRESS = ROOT / "docs/evidence/h01-c3-neuron-graph-progress.json"
GRAPH = ROOT / "docs/evidence/h01-c3-neuron-graph.json.gz"
CACHE = ROOT / ".cache/h01/c3-graph"
COLUMNS = ("pre", "post", "type", "x", "y", "z", "pre_class", "post_class", "confidence", "shard")
MIN_FREE_BYTES = 20 * 1024**3
WALL_LIMIT_S = 3 * 3600
CURL_MAX_S = 1500  # one shard; the object store throttles some streams to ~1.3 MB/s (150 s), never this slow
CURL_STALL = ("--speed-limit", "50000", "--speed-time", "120")  # abort under 50 kB/s for 120 s, then retry


def sha256_path(path):
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    """Atomic JSON rewrite (tmp file then replace)."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def shard_index(shard):
    """Integer N of ``export{N:012d}``."""
    if not shard.startswith("export") or not shard[6:].isdigit():
        raise ValueError(f"not a shard name: {shard}")
    return int(shard[6:])


def keep_row(record, neuron_ids, shard):
    """Graph row for one export record, or ``None`` when either side is not a neuron.

    Parameters
    ----------
    record : dict
        One Avro record of the export (``pre_synaptic_site``,
        ``post_synaptic_partner``, ``type``, ``location``, ``confidence``).
    neuron_ids : set of int
        C3 ids tagged ``neuron`` in the cell table.
    shard : int
        Shard index recorded on the row.

    Returns
    -------
    list or None
        Values in ``COLUMNS`` order.

    Examples
    --------
    .. code-block:: python

        >>> from docs.evidence.h01_c3_neuron_graph import keep_row
        >>> rec = {"pre_synaptic_site": {"neuron_id": 1, "class_label": "AXON"},
        ...        "post_synaptic_partner": {"neuron_id": 2, "class_label": "DENDRITE"},
        ...        "type": 2, "location": {"x": 10, "y": 20, "z": 3}, "confidence": 0.98765}
        >>> keep_row(rec, {1, 2}, 7)
        [1, 2, 2, 10, 20, 3, 'AXON', 'DENDRITE', 0.9877, 7]
        >>> keep_row(rec, {1}, 7) is None
        True
    """
    pre, post = record["pre_synaptic_site"], record["post_synaptic_partner"]
    pre_id, post_id = pre.get("neuron_id"), post.get("neuron_id")
    if pre_id is None or post_id is None:
        return None
    pre_id, post_id = int(pre_id), int(post_id)
    if pre_id not in neuron_ids or post_id not in neuron_ids:
        return None
    loc = record["location"]
    conf = record.get("confidence")
    return [pre_id, post_id, record.get("type"), loc["x"], loc["y"], loc["z"],
            pre.get("class_label"), post.get("class_label"),
            None if conf is None else round(float(conf), 4), shard]


def free_bytes(root=ROOT):
    return shutil.disk_usage(str(root)).free


def curl_command(shard, part):
    """curl argv for one shard: IPv4 only, fail on HTTP errors, abort a stalled stream, cap the wall time.

    ``-4`` because the box resolves storage.googleapis.com to IPv6 first and that
    path throttled to about 1 MB/s after ~33 GB on 2026-09-16 while the IPv4 path
    to the same object ran at 11.7 MB/s (probe recorded in the progress notes).
    """
    return ["curl", "-4", "-sS", "-L", "--fail", "--retry", "2", "--max-time", str(CURL_MAX_S), *CURL_STALL,
            "-o", str(part), BASE_URL + shard]


def download(shard, expected, attempts=3, cache=CACHE):
    """Fetch one shard with curl; return ``(path, seconds)`` or ``(None, error)``.

    Runs in a download thread. The scan workers are started with the ``spawn``
    context before any download begins, so no worker inherits a curl pipe.
    """
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / f"{shard}.avro"
    part = cache / f"{shard}.part"
    if final.exists() and final.stat().st_size == expected:
        return final, 0.0
    last = None
    for _ in range(attempts):
        t0 = time.monotonic()
        try:
            proc = subprocess.run(curl_command(shard, part), capture_output=True, text=True, timeout=CURL_MAX_S + 60)
        except subprocess.TimeoutExpired:
            last = f"curl exceeded {CURL_MAX_S + 60}s"
            part.unlink(missing_ok=True)
            continue
        if proc.returncode == 0 and part.exists() and part.stat().st_size == expected:
            part.replace(final)
            return final, time.monotonic() - t0
        last = (f"curl rc={proc.returncode} size={part.stat().st_size if part.exists() else None} "
                f"{proc.stderr.strip()}")
        part.unlink(missing_ok=True)
    return None, last


def part_path(shard, cache=CACHE):
    return Path(cache) / f"{shard}.kept.json"


def scan_shard(path, shard, neuron_ids, cache=CACHE):
    """Scan one downloaded shard in a worker process.

    Writes the kept rows to ``part_path(shard)`` and returns the receipt
    ``{id, bytes, sha256, records, kept, scan_seconds}``. Imports fastavro
    lazily so the module is importable where fastavro is absent (tests).
    """
    from fastavro import reader

    path = Path(path)
    t0 = time.monotonic()
    digest = sha256_path(path)
    idx = shard_index(shard)
    kept, records = [], 0
    with path.open("rb") as f:
        for record in reader(f):
            records += 1
            row = keep_row(record, neuron_ids, idx)
            if row is not None:
                kept.append(row)
    out = part_path(shard, cache)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(kept, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out)
    return {"id": shard, "bytes": path.stat().st_size, "sha256": digest, "records": records,
            "kept": len(kept), "scan_seconds": round(time.monotonic() - t0, 3)}


def load_neuron_ids(table=TABLE):
    return frozenset(int(r["id"]) for r in json.loads(Path(table).read_text(encoding="utf-8"))["rows"])


def new_progress(listing, workers, inputs):
    return {"scope": "H01 C3 synapse export (166 Avro shards) reduced to rows whose pre and post neuron_id "
                     "are both tagged neuron in h01-c3-cell-table.json; shards deleted after scanning",
            "source_prefix": BASE_URL, "input_hashes": inputs, "columns": list(COLUMNS),
            "units": {"type": "1 inhibitory, 2 excitatory", "xyz": "8/8/33 nm voxels",
                      "shard": "integer N of export{N:012d}", "confidence": "rounded to 4 decimals"},
            "workers": workers, "shards_total": len(listing), "shards": [], "totals": {},
            "started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_utc": None, "stopped_reason": None, "wall_seconds": 0.0}


def totals(entries):
    """Sum of per-shard receipts."""
    keys = ("bytes", "records", "kept", "download_seconds", "scan_seconds")
    out = {k: round(sum(e.get(k, 0) for e in entries), 3) for k in keys}
    out["shards"] = len(entries)
    return out


def pending_shards(listing, progress, cache=CACHE):
    """Shards not yet receipted with an existing part file, in name order."""
    done = {e["id"] for e in progress["shards"] if part_path(e["id"], cache).exists()}
    progress["shards"] = [e for e in progress["shards"] if e["id"] in done]
    return [s for s in sorted(listing) if s not in done]


def merge_parts(shards, out=GRAPH, cache=CACHE, progress=None):
    """Concatenate the part files in shard order into one gzip JSON document."""
    rows = 0
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as f:
        f.write('{"columns":' + json.dumps(list(COLUMNS)) + ',"rows":[')
        first = True
        for shard in shards:
            part = json.loads(part_path(shard, cache).read_text(encoding="utf-8"))
            for row in part:
                f.write(("" if first else ",") + json.dumps(row, separators=(",", ":")))
                first = False
            rows += len(part)
        f.write("]}\n")
    return {"path": str(Path(out).relative_to(ROOT)) if Path(out).is_relative_to(ROOT) else str(out),
            "rows": rows, "bytes": Path(out).stat().st_size, "sha256": sha256_path(out)}


def run(workers, downloads, listing, neuron_ids, progress, cache=CACHE, wall_limit=WALL_LIMIT_S):
    """Drive the download threads and scan processes until the queue is empty or a guard trips."""
    queue = collections.deque(pending_shards(listing, progress, cache))
    start = time.monotonic()
    wall_prior = progress.get("wall_seconds", 0.0)
    in_flight_limit = workers + 2
    dl_futs, scan_futs = {}, {}

    def checkpoint(reason=None):
        progress["stopped_reason"] = reason
        progress["wall_seconds"] = round(wall_prior + time.monotonic() - start, 3)
        progress["totals"] = totals(progress["shards"])
        write_json(PROGRESS, progress)

    reason = None
    spawn = multiprocessing.get_context("spawn")  # fork would copy a live curl's pipe fds into the workers
    with ProcessPoolExecutor(workers, mp_context=spawn) as scan_pool, ThreadPoolExecutor(downloads) as dl_pool:
        while queue or dl_futs or scan_futs:
            if reason is None and time.monotonic() - start > wall_limit:
                reason = f"wall clock limit {wall_limit}s reached"
            if reason is None and free_bytes() < MIN_FREE_BYTES:
                reason = f"free disk {free_bytes()/1024**3:.1f} GB under {MIN_FREE_BYTES/1024**3:.0f} GB"
            if reason is not None:
                queue.clear()
            while queue and len(dl_futs) + len(scan_futs) < in_flight_limit:
                shard = queue.popleft()
                dl_futs[dl_pool.submit(download, shard, listing[shard], 2, cache)] = shard
            if not dl_futs and not scan_futs:
                break
            done, _ = wait(list(dl_futs) + list(scan_futs), return_when=FIRST_COMPLETED)
            for fut in done:
                if fut in dl_futs:
                    shard = dl_futs.pop(fut)
                    path, dl = fut.result()
                    if path is None:
                        reason = reason or f"download of {shard} failed twice: {dl}"
                        continue
                    scan_futs[scan_pool.submit(scan_shard, str(path), shard, neuron_ids, cache)] = (shard, dl, path)
                else:
                    shard, dl, path = scan_futs.pop(fut)
                    try:
                        entry = fut.result()
                    except Exception as exc:  # noqa: BLE001 - recorded as the stop reason
                        reason = reason or f"scan of {shard} failed: {exc!r}"
                        continue
                    entry["download_seconds"] = round(dl, 3)
                    progress["shards"].append(entry)
                    progress["shards"].sort(key=lambda e: e["id"])
                    Path(path).unlink(missing_ok=True)
                    checkpoint()
                    print(json.dumps(entry), flush=True)
    complete = len(progress["shards"]) == len(listing)
    checkpoint(reason or ("all shards scanned" if complete else "queue exhausted"))
    return complete


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=6, help="scan processes")
    ap.add_argument("--downloads", type=int, default=3, help="download threads")
    ap.add_argument("--limit", type=int, default=None, help="scan at most this many pending shards (measurement)")
    args = ap.parse_args(argv)
    listing = json.loads(LISTING.read_text(encoding="utf-8"))["shards"]
    if args.limit is not None:
        listing = dict(list(sorted(listing.items()))[:args.limit])
    neuron_ids = load_neuron_ids()
    inputs = {str(p.relative_to(ROOT)): sha256_path(p) for p in (LISTING, TABLE)}
    progress = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else new_progress(
        listing, args.workers, inputs)
    progress["workers"] = args.workers
    progress["shards_total"] = len(listing)
    progress["neuron_ids"] = len(neuron_ids)
    complete = run(args.workers, args.downloads, listing, neuron_ids, progress)
    print("STOP:", progress["stopped_reason"], "wall", progress["wall_seconds"], flush=True)
    if not complete:
        return 1
    t0 = time.monotonic()
    progress["output"] = merge_parts(sorted(listing))
    progress["output"]["merge_seconds"] = round(time.monotonic() - t0, 3)
    progress["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(PROGRESS, progress)
    print("MERGED:", json.dumps(progress["output"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
