"""Coverage of the H01 C3 synapse export scan, and the measured read benchmark behind it.

Three subcommands (spec: docs/specs/2026-09-07-h01-connectivity-completion.md, SP7b):

``listing``    enumerate the export prefix through the GCS JSON API (name and size only) and
               save ``h01-c3-export-listing.json``; one small GET, follows ``nextPageToken``.
``benchmark``  time the record pass that ``h01_connectivity_audit.py`` performs over one local
               shard (fastavro read plus base-segment owner lookup), read-only, no volume.
``coverage``   from the listing and the per-shard audit files, write
               ``h01-export-scan-coverage.json`` with ``full_export_scanned`` and the statement
               ``N of M shards scanned``. Absence of a contact is never established by a scan.

Run with the isolated h01-inspect Python for ``benchmark`` (needs fastavro); the other two
use the standard library only.
"""

import argparse
import collections
import glob
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

EXPORT_PREFIX = "data/20210729/c3/synapses/exported/"
LISTING_URL = ("https://storage.googleapis.com/storage/v1/b/h01-release/o?prefix="
               + urllib.parse.quote(EXPORT_PREFIX, safe="") + "&maxResults=1000&fields=items(name,size),nextPageToken")
SHARD = re.compile(r"export\d{12}$")


def shards_from_listing(items):
    """``{shard_name: bytes}`` for the avro objects in a listing (``.json`` sidecars excluded)."""
    out = {}
    for item in items:
        name = item["name"].rsplit("/", 1)[-1]
        if SHARD.fullmatch(name):
            out[name] = int(item["size"])
    return out


def fetch_listing(opener=urllib.request.urlopen):
    """All objects under the export prefix, following pagination."""
    items, token = [], None
    while True:
        url = LISTING_URL+(f"&pageToken={token}" if token else "")
        with opener(url, timeout=60) as response:
            page = json.loads(response.read())
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return items


def owners_from_membership(cells):
    """base segment id -> [(cell index, region)] exactly as the audit builds it."""
    owners = collections.defaultdict(list)
    for index, cell in enumerate(cells):
        for region, segments in cell["regions"].items():
            for segment in segments:
                owners[str(segment)].append((index, region))
    return owners


def scan_shard(stream, owners, reader):
    """The audit's per-record pass (``h01_connectivity_audit.py:100-107``) without report writing."""
    counts = collections.Counter()
    for synapse in reader(stream):
        counts["records"] += 1
        pre = owners.get(str(synapse["pre_synaptic_site"]["base_neuron_id"]), [])
        post = owners.get(str(synapse["post_synaptic_partner"]["base_neuron_id"]), [])
        if pre and post:
            counts["both_in_archive"] += 1
    return dict(counts)


def coverage(shards_listed, audits):
    """Coverage record from the listing and the per-shard audit summaries.

    Parameters
    ----------
    shards_listed : dict
        shard name -> bytes, from :func:`shards_from_listing`.
    audits : dict
        shard name -> ``counts`` dict of its audit file (must contain ``records``).
    """
    scanned = sorted(set(audits) & set(shards_listed))
    unlisted = sorted(set(audits) - set(shards_listed))
    bytes_scanned = sum(shards_listed[s] for s in scanned)
    bytes_listed = sum(shards_listed.values())
    return {"shards_listed": len(shards_listed), "shards_scanned": len(scanned), "shards_audited_but_unlisted": unlisted,
            "records_scanned": sum(audits[s]["records"] for s in scanned),
            "bytes_scanned": bytes_scanned, "bytes_listed": bytes_listed,
            "bytes_remaining": bytes_listed-bytes_scanned,
            "statement": f"{len(scanned)} of {len(shards_listed)} shards scanned",
            "full_export_scanned": bool(shards_listed) and len(scanned) == len(shards_listed),
            "scope": "Partial scan; absence of a contact is never established."}


def derived_estimate(cov, read_seconds_per_shard, download_bytes_per_second=None):
    """Derived (not measured) total for the remaining shards; download term only when measured."""
    remaining = cov["shards_listed"]-cov["shards_scanned"]
    out = {"label": "derived", "remaining_shards": remaining,
           "read_seconds": round(remaining*read_seconds_per_shard, 1)}
    if download_bytes_per_second:
        out["download_seconds"] = round(cov["bytes_remaining"]/download_bytes_per_second, 1)
        out["total_seconds"] = round(out["read_seconds"]+out["download_seconds"], 1)
    else:
        out["download_seconds"] = "unmeasured; no total quoted"
    return out


def benchmark_summary(records):
    """Mean and range of measured per-shard read times (repeat range per Hartshorne p199)."""
    seconds = [r["wall_seconds"] for r in records]
    return {"shards": [r["shard"] for r in records], "wall_seconds": seconds, "measured": True,
            "mean_seconds": round(sum(seconds)/len(seconds), 1), "min_seconds": min(seconds), "max_seconds": max(seconds),
            "repeat_range_seconds": round(max(seconds)-min(seconds), 1)}


def audit_counts(evidence_dir):
    out = {}
    for path in glob.glob(str(Path(evidence_dir)/"h01-connectivity-export*-audit.json")):
        shard = re.search(r"(export\d{12})", path).group(1)
        out[shard] = json.loads(Path(path).read_text())["counts"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["listing", "benchmark", "coverage"])
    parser.add_argument("--cache", type=Path, default=Path(".cache/h01"))
    parser.add_argument("--shard", default="export000000000000")
    parser.add_argument("--evidence", type=Path, default=Path("docs/evidence"))
    parser.add_argument("--listing", type=Path, default=Path("docs/evidence/h01-c3-export-listing.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/h01-export-scan-coverage.json"))
    parser.add_argument("--download-bytes-per-second", type=float, help="Only if measured this session.")
    parser.add_argument("--download-source", default="", help="How the download rate was measured.")
    args = parser.parse_args(argv)
    if args.command == "listing":
        items = fetch_listing()
        record = {"url": LISTING_URL, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "items": items,
                  "shards": shards_from_listing(items)}
        args.listing.write_text(json.dumps(record, indent=1)+"\n")
        print(len(record["shards"]), "shards", sum(record["shards"].values()), "bytes", flush=True)
        return record
    if args.command == "benchmark":
        from fastavro import reader
        owners = owners_from_membership(json.loads((args.cache/"proofread-base-membership.json").read_text()))
        shard = args.cache/(args.shard+".avro")
        t0 = time.time()
        with shard.open("rb") as stream:
            counts = scan_shard(stream, owners, reader)
        seconds = round(time.time()-t0, 1)
        record = {"shard": args.shard, "bytes": shard.stat().st_size, "counts": counts, "wall_seconds": seconds,
                  "records_per_second": round(counts["records"]/seconds, 1), "measured": True,
                  "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (args.evidence/f"h01-export-scan-benchmark-{args.shard}.json").write_text(json.dumps(record, indent=1)+"\n")
        print(json.dumps(record), flush=True)
        return record
    shards = json.loads(args.listing.read_text())["shards"]
    cov = coverage(shards, audit_counts(args.evidence))
    benchmarks = sorted(glob.glob(str(args.evidence/"h01-export-scan-benchmark-*.json")))
    if benchmarks:
        cov["benchmark"] = benchmark_summary([json.loads(Path(b).read_text()) for b in benchmarks])
        cov["estimate_for_remaining"] = derived_estimate(cov, cov["benchmark"]["mean_seconds"], args.download_bytes_per_second)
        if args.download_bytes_per_second:
            cov["estimate_for_remaining"]["download_rate"] = {"bytes_per_second": args.download_bytes_per_second,
                                                              "source": args.download_source, "measured": True}
    cov["listing"] = str(args.listing)
    args.output.write_text(json.dumps(cov, indent=1)+"\n")
    print(cov["statement"], "full_export_scanned", cov["full_export_scanned"], flush=True)
    return cov


if __name__ == "__main__":
    main()
