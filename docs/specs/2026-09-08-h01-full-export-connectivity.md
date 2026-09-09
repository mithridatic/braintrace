# H01 full-export connectivity join (2026-09-08)

Question: how many synapse records in the complete H01 C3 synapse export have
both a presynaptic and a postsynaptic base segment owned by one of the 104
proofread population cells, and what are those (pre, post, location) records?

Prior state (docs/evidence/h01-cached-full-base-join.json): 9 of 166 shards
scanned, 8,991,719 records, 470 pre-matched endpoints, 14,591 post-matched
endpoints, 39 same-cell pairs, 0 between-cell pairs. 157 shards unscanned.

## Shard source

- Bucket prefix: `https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/exported/`
- Objects: `export000000000000` .. `export000000000165` (166 Avro files, no
  extension; a `.json` twin of ~757 MB per shard also exists and is NOT used).
- Listing: docs/evidence/h01-c3-export-listing.json (fetched 2026-09-07).
  Total Avro bytes 32,859,876,087 (32.86 GB); per shard 197,344,616 to
  198,456,495 bytes (~197.5 MB).
- Public object, anonymous HTTPS GET returns 200; no credentials or gsutil
  needed. Downloader: `curl` (already installed).
- The nine cached shards live at `.cache/h01/export00000000000{0..8}.avro`
  and are the same objects (byte sizes match the listing; SHA256 recorded in
  h01-cached-full-base-join.json).

## Measured cost on the nine cached shards

- Scan: 412.7 s for 9 shards with the fastavro streaming reader
  (`.cache/h01-inspect/Scripts/python.exe`), 45.9 s per shard, ~999k records
  per shard.
- Download: not previously timed; measured on the first new shard and recorded
  in the progress file (`download_seconds`).

## Projection for 157 shards

- Download volume: 157 x 197.5 MB = 31.0 GB.
- Scan time: 157 x 45.9 s = 7,206 s = 2.0 h of CPU-bound decoding.
- Download time at 5 MB/s: 157 x 40 s = 1.7 h; overlapped with scanning
  (next shard fetched by an I/O thread inside the single process while the
  current shard is decoded), the wall time is bounded by max(scan, download)
  plus one download, so ~2.1 to 2.5 h if the link sustains >= 4.5 MB/s. The
  4-hour cap is expected to cover all 157 shards only if the link is that
  fast; otherwise coverage will be partial and recorded as such.
- Record count: 157 x 999k = ~157 M records; full export ~166 M.

## Exact filter (identical to the nine-shard scan)

Ownership map: for every proofreading record in
`.cache/h01/proofread-base-membership.json` (SHA256
3201df49269d31b8e788a91607f5b07ed14c3f0d7cd2dfbae7d51fef3c287a9f), every base
segment id in every `regions` list (cell_body, dendrite, axon, ...) maps to
the record's owner cell, where the owner is
`existing_spatial_owners[0]` of the matching row in
`docs/evidence/h01-cellmatrix-mapping-audit.json` (SHA256
360d8c334f8f1b280dd860c69a633cdee6bcb7b6de09a369f4225433e80a899e). This
yields one set of owners per base id (all singletons in the nine-shard run:
zero ambiguous matches).

Per record `r`:

- `pre = owners[r.pre_synaptic_site.base_neuron_id]`
- `post = owners[r.post_synaptic_partner.base_neuron_id]`
- if either set has more than one owner: `ambiguous` (record retained)
- elif both non-empty and `pre == post`: `same_cell`
- elif both non-empty: `between_cell` (record retained with raw fields)
- elif exactly one non-empty: `one_sided` (counted, split pre/post)
- else: unmatched.

## Procedure

Script: `docs/evidence/h01_full_export_join.py` (single process; one I/O
thread prefetches the next shard). Per shard: curl to
`.cache/h01/full-export/export<id>.part`, rename on success, stream-decode
with fastavro, append a per-shard entry to
`docs/evidence/h01-full-export-join-progress.json` (id, bytes, records,
download_seconds, scan_seconds, between_cell, same_cell, one_sided_pre,
one_sided_post, ambiguous, sha256), then delete the shard file. At most two
downloaded shards exist at once beyond the nine cached ones. The progress
file is rewritten atomically after every shard; a restart skips completed
shards.

Retained between-cell and ambiguous records (raw Avro record plus owners and
shard id) go to `docs/evidence/h01-full-export-between-cell-pairs.json`.

Hard limits: stop at 4 h wall clock, when free space on C: falls under
50 GB, or when a shard download fails twice. The stopping reason is written
to the progress file (`stopped_reason`).

Not in scope: no network code or topology JSON edits, no simulation, no
promotion of any record to a simulation contact.

## Outcome

Filled in after the run (see the bottom of this file).

## Outcome (2026-09-09, recorded after the scan agent died before committing)

All 157 remaining shards scanned, 5720 s wall, no incomplete shards. Totals for the
157: 157,224,349 records; **10 between-cell synapses** across 9 directed pairs (one
reciprocal pair 3519995546 <-> 3680152874); 760 same-cell; 8,166 one-sided pre;
253,974 one-sided post. Export `type` field: 5 records type 1, 5 type 2. With the
nine cached shards (0 between-cell) this is full-export coverage: 10 synapses among
the 104 population cells in total. Raw rows: docs/evidence/h01-full-export-between-cell-pairs.json.
Per-shard log: docs/evidence/h01-full-export-join-progress.json.

Enabling them as contacts would need, per pair: map each record's post-synaptic
location to a compartment of the postsynaptic cell (same placement step used for the
two enabled contacts), assign polarity from the export type field or presynaptic cell
type, and add the pair to the topology JSON. Not done here; the network code and
topology are unchanged.
