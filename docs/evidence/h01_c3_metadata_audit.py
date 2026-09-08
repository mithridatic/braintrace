"""Fetch bounded public C3 metadata without downloading the synapse database."""

import hashlib
import json
from pathlib import Path
import urllib.request

folder = Path(__file__).parent
urls = {
    "export_listing": "https://storage.googleapis.com/storage/v1/b/h01-release/o?prefix=data%2F20210729%2Fc3%2Fsynapses%2Fexported%2F&maxResults=3",
    "annotations": "https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/precomputed/info",
}
results = {}
for name, url in urls.items():
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(1024*1024+1)
    if len(data) > 1024*1024:
        raise ValueError("Metadata exceeded the one-megabyte bound.")
    results[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "data": json.loads(data)}
sample_url = "https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/exported/export000000000000.json"
request = urllib.request.Request(sample_url, headers={"Range": "bytes=0-16383"})
with urllib.request.urlopen(request, timeout=30) as response:
    if response.status != 206:
        raise ValueError("Server did not honor the bounded range request.")
    sample = response.read(16385)
    content_range = response.headers.get("Content-Range")
if len(sample) > 16384:
    raise ValueError("Range response exceeded the requested bound.")
lines = sample.splitlines()
records = [json.loads(line) for line in lines[:-1]]
results["record_sample"] = {"url": sample_url, "content_range": content_range,
                            "range_sha256": hashlib.sha256(sample).hexdigest(),
                            "complete_records_in_range": len(records), "first_record": records[0]}
(folder/"h01-c3-metadata-audit.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results["record_sample"]))
