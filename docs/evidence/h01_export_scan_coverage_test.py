"""Tests for the export-scan coverage tool (pure functions and a fake listing; no GCS)."""

import io
import json

from docs.evidence.h01_export_scan_coverage import (benchmark_summary, coverage, derived_estimate, fetch_listing,
                                                    owners_from_membership, scan_shard, shards_from_listing)

PREFIX = "data/20210729/c3/synapses/exported/"


def test_shards_exclude_sidecars_and_directory_marker():
    items = [{"name": PREFIX, "size": "0"}, {"name": PREFIX+"export000000000000", "size": "10"},
             {"name": PREFIX+"export000000000000.json", "size": "3"}, {"name": PREFIX+"export000000000165", "size": "20"}]
    assert shards_from_listing(items) == {"export000000000000": 10, "export000000000165": 20}


def test_fetch_listing_follows_page_tokens():
    pages = [{"items": [{"name": "a", "size": "1"}], "nextPageToken": "t"}, {"items": [{"name": "b", "size": "2"}]}]
    urls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(url, timeout):
        urls.append(url)
        return Response(json.dumps(pages.pop(0)).encode())
    assert [x["name"] for x in fetch_listing(opener)] == ["a", "b"]
    assert "pageToken" not in urls[0] and urls[1].endswith("&pageToken=t")


def test_coverage_statement_and_flag():
    listed = {f"export{i:012d}": 100 for i in range(3)}
    audits = {"export000000000000": {"records": 5}, "export000000000002": {"records": 7}, "export000000000009": {"records": 1}}
    cov = coverage(listed, audits)
    assert cov["statement"] == "2 of 3 shards scanned" and cov["full_export_scanned"] is False
    assert cov["records_scanned"] == 12 and cov["bytes_remaining"] == 100
    assert cov["shards_audited_but_unlisted"] == ["export000000000009"]
    full = coverage(listed, {k: {"records": 1} for k in listed})
    assert full["full_export_scanned"] is True and full["statement"] == "3 of 3 shards scanned"
    assert coverage({}, {})["full_export_scanned"] is False


def test_derived_estimate_quotes_no_total_without_a_measured_download_rate():
    cov = coverage({f"export{i:012d}": 100 for i in range(3)}, {"export000000000000": {"records": 5}})
    partial = derived_estimate(cov, 30.)
    assert partial["label"] == "derived" and partial["read_seconds"] == 60. and "total_seconds" not in partial
    full = derived_estimate(cov, 30., download_bytes_per_second=50.)
    assert full["download_seconds"] == 4. and full["total_seconds"] == 64.


def test_scan_shard_counts_like_the_audit():
    owners = owners_from_membership([{"regions": {"axon": [1], "dendrite": [2]}}, {"regions": {"axon": [3]}}])
    assert owners["1"] == [(0, "axon")] and owners["3"] == [(1, "axon")]
    rows = [{"pre_synaptic_site": {"base_neuron_id": 1}, "post_synaptic_partner": {"base_neuron_id": 3}},
            {"pre_synaptic_site": {"base_neuron_id": 1}, "post_synaptic_partner": {"base_neuron_id": 9}},
            {"pre_synaptic_site": {"base_neuron_id": 7}, "post_synaptic_partner": {"base_neuron_id": 2}}]
    assert scan_shard(None, owners, lambda stream: iter(rows)) == {"records": 3, "both_in_archive": 1}


def test_benchmark_summary_reports_mean_and_repeat_range():
    summary = benchmark_summary([{"shard": "a", "wall_seconds": 30.}, {"shard": "b", "wall_seconds": 40.}])
    assert summary["mean_seconds"] == 35. and summary["repeat_range_seconds"] == 10. and summary["shards"] == ["a", "b"]
