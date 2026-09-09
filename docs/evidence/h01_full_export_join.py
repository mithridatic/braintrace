"""Stream the unscanned H01 C3 synapse export shards through the exact
proofreading base-segment ownership join used for the nine cached shards.

Single process. One I/O thread prefetches the next shard while the main
thread decodes the current one. Every shard is deleted after it is scanned.
Progress is rewritten atomically after each shard so the run is resumable.

Run from the worktree root:
    .cache/h01-inspect/Scripts/python.exe docs/evidence/h01_full_export_join.py
"""

import collections
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastavro import reader

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/exported/"
LISTING = ROOT / "docs/evidence/h01-c3-export-listing.json"
MAPPING = ROOT / "docs/evidence/h01-cellmatrix-mapping-audit.json"
MEMBERSHIP = ROOT / ".cache/h01/proofread-base-membership.json"
PRIOR = ROOT / "docs/evidence/h01-cached-full-base-join.json"
PROGRESS = ROOT / "docs/evidence/h01-full-export-join-progress.json"
PAIRS = ROOT / "docs/evidence/h01-full-export-between-cell-pairs.json"
CACHE = ROOT / ".cache/h01/full-export"
WALL_LIMIT_S = 4 * 3600
MIN_FREE_BYTES = 50 * 1024**3
CACHED = {f"export{i:012d}" for i in range(9)}


def sha256_path(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_owners():
    owners = {r["record"]: r["existing_spatial_owners"][0]
              for r in json.loads(MAPPING.read_text())["rows"]}
    base = collections.defaultdict(set)
    for r in json.loads(MEMBERSHIP.read_text()):
        for ids in r["regions"].values():
            for b in ids:
                base[str(b)].add(owners[r["file"]])
    return {k: sorted(v) for k, v in base.items()}


def free_bytes():
    return shutil.disk_usage(str(ROOT)).free


def download(shard, expected, attempts=2):
    """Fetch one shard with curl; return (path, seconds) or (None, error)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    final = CACHE / f"{shard}.avro"
    part = CACHE / f"{shard}.part"
    if final.exists() and final.stat().st_size == expected:
        return final, 0.0
    last = None
    for _ in range(attempts):
        t0 = time.monotonic()
        proc = subprocess.run(
            ["curl", "-sS", "-L", "--fail", "--retry", "2", "-o", str(part), BASE_URL + shard],
            capture_output=True, text=True)
        if proc.returncode == 0 and part.exists() and part.stat().st_size == expected:
            part.replace(final)
            return final, time.monotonic() - t0
        last = f"curl rc={proc.returncode} size={part.stat().st_size if part.exists() else None} {proc.stderr.strip()}"
        part.unlink(missing_ok=True)
    return None, last


def scan(path, base):
    counts = collections.Counter()
    retained = []
    with path.open("rb") as f:
        for r in reader(f):
            pre = base.get(str(r["pre_synaptic_site"]["base_neuron_id"]), ())
            post = base.get(str(r["post_synaptic_partner"]["base_neuron_id"]), ())
            counts["records"] += 1
            if not pre and not post:
                continue
            if len(pre) > 1 or len(post) > 1:
                counts["ambiguous"] += 1
                retained.append(dict(kind="ambiguous", pre_owners=pre, post_owners=post, source=r))
            elif pre and post:
                if pre == post:
                    counts["same_cell"] += 1
                else:
                    counts["between_cell"] += 1
                    retained.append(dict(kind="between_cell", pre_owners=pre, post_owners=post, source=r))
            elif pre:
                counts["one_sided_pre"] += 1
            else:
                counts["one_sided_post"] += 1
    return dict(counts), retained


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def main():
    listing = json.loads(LISTING.read_text())["shards"]
    todo = [s for s in sorted(listing) if s not in CACHED]
    base = load_owners()
    inputs = {str(p.relative_to(ROOT)): sha256_path(p) for p in (MAPPING, MEMBERSHIP, LISTING)}
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text())
    else:
        progress = dict(
            scope="Exact proofreading region-base ownership join over export shards not covered by "
                  "h01-cached-full-base-join.json; shards deleted after scanning",
            source_prefix=BASE_URL, input_hashes=inputs,
            prior_nine_shard_result=str(PRIOR.relative_to(ROOT)),
            shards_total=len(listing), shards_previously_scanned=len(CACHED),
            shards=[], stopped_reason=None, wall_seconds=0.0)
    progress["stopped_reason"] = None
    done = {s["id"] for s in progress["shards"]}
    pairs = json.loads(PAIRS.read_text()) if PAIRS.exists() else dict(
        scope="Raw export records whose pre and post base segments are owned by two different "
              "population cells (or ambiguous ownership), from shards outside the cached nine",
        rows=[])
    queue = [s for s in todo if s not in done]
    start = time.monotonic()
    wall_prior = progress.get("wall_seconds", 0.0)

    prefetch = {}

    def fetch_into(shard):
        prefetch[shard] = download(shard, listing[shard])

    thread = None
    if queue:
        thread = threading.Thread(target=fetch_into, args=(queue[0],), daemon=True)
        thread.start()

    def finish(reason):
        progress["stopped_reason"] = reason
        progress["wall_seconds"] = wall_prior + time.monotonic() - start
        progress["shards_scanned_this_programme"] = len(progress["shards"])
        write_json(PROGRESS, progress)
        print("STOP:", reason, flush=True)

    for idx, shard in enumerate(queue):
        elapsed = time.monotonic() - start
        if elapsed > WALL_LIMIT_S:
            return finish(f"wall clock limit {WALL_LIMIT_S}s reached before {shard}")
        if free_bytes() < MIN_FREE_BYTES:
            return finish(f"free disk {free_bytes()/1024**3:.1f} GB under 50 GB before {shard}")
        thread.join()
        path, dl = prefetch.pop(shard)
        if path is None:
            return finish(f"download of {shard} failed twice: {dl}")
        nxt = queue[idx + 1] if idx + 1 < len(queue) else None
        if nxt is not None and elapsed < WALL_LIMIT_S:
            thread = threading.Thread(target=fetch_into, args=(nxt,), daemon=True)
            thread.start()
        t0 = time.monotonic()
        digest = sha256_path(path)
        counts, retained = scan(path, base)
        scan_s = time.monotonic() - t0
        entry = dict(id=shard, bytes=path.stat().st_size, sha256=digest,
                     download_seconds=round(dl, 3), scan_seconds=round(scan_s, 3),
                     records=counts.get("records", 0),
                     between_cell=counts.get("between_cell", 0),
                     same_cell=counts.get("same_cell", 0),
                     one_sided_pre=counts.get("one_sided_pre", 0),
                     one_sided_post=counts.get("one_sided_post", 0),
                     ambiguous=counts.get("ambiguous", 0), complete=True)
        for row in retained:
            row["shard"] = shard
        pairs["rows"].extend(retained)
        progress["shards"].append(entry)
        progress["wall_seconds"] = wall_prior + time.monotonic() - start
        progress["shards_scanned_this_programme"] = len(progress["shards"])
        write_json(PAIRS, pairs)
        write_json(PROGRESS, progress)
        path.unlink()
        print(json.dumps(entry), flush=True)
    finish("all shards scanned" if len(progress["shards"]) == len(todo) else "queue exhausted")


if __name__ == "__main__":
    sys.exit(main())
