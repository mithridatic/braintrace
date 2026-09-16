"""Across-cell spread of human cells of each donor's type, from the Allen Cell Types database.

The C3 test-to-failure gate (spec 2026-09-16-h01-c3-partner-expansion, step C, J's decision
2026-09-16) scores the plausibility rules (count at the primary drive, pre-pulse rest,
post-pulse level) with the datum = the donor's own value and the tolerance = 2 sd of the
same quantity across the human cells of the donor's type in the Allen Cell Types database.
This module reads two sha-pinned pulls of the Allen API under ``.cache/h01/``:

* ``allen-human-ephys-features.json`` — ``model::ApiCellTypesSpecimenDetail`` for every human
  specimen (413 cells; ``tag__dendrite_type``, ``structure__layer``, ``ef__*``).
* ``allen-human-ephys-sweeps.json`` — ``model::EphysSweep`` long-square rows of the spiny L2/3,
  spiny L4 and aspiny populations (``stimulus_absolute_amplitude`` pA, ``num_spikes``,
  ``pre_vm_mv``, ``post_vm_mv``).

Type labels the API exposes for human cells: ``tag__dendrite_type`` (spiny / aspiny /
sparsely spiny) and ``structure__layer``. No fast-spiking / non-fast-spiking label exists
for human cells (no transgenic line), so both interneuron donors use the aspiny population.

Per cell, the readings are taken on that cell's 1 s long-square sweeps whose amplitude lies
within one sweep step (20 pA) of the donor's primary drive: ``num_spikes`` (null = no
spike detected = 0, verified on the L2 donor's sub-rheobase sweeps), ``pre_vm_mv`` (ipfx:
mean over the 500 ms before onset) and ``post_vm_mv`` (ipfx: mean over the last 500 ms of
the recording, 5-7 s after offset; the table has no field for the 10 ms ending 200 ms after
offset, so this is the closest documented post-stimulus level). Voltages are shifted by
the -14 mV liquid junction potential the recordings report uncorrected; the spread is
unaffected. Tolerance = 2 sd (sample sd across cells); the 5-95 percent range is recorded
beside it for reference and is not the tolerance.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

JUNCTION_MV = -14.
STEP_PA = 20.
AMPLITUDE_TOLERANCE_PA = .5
SD_FACTOR = 2.
POPULATIONS = {
    "spiny_l23": dict(dendrite_type="spiny", layers=("2", "3"), label="human spiny (pyramidal) layer 2/3"),
    "spiny_l4": dict(dendrite_type="spiny", layers=("4",), label="human spiny (pyramidal) layer 4"),
    "aspiny": dict(dendrite_type="aspiny", layers=None,
                   label="human aspiny (interneuron), all layers; the API exposes no fast-spiking label for human cells"),
}
DONOR_POPULATIONS = {
    "l2-pyramidal-allen-541563728": "spiny_l23",
    "l4-pyramidal-allen-527952884": "spiny_l4",
    "l5-pv-basket-hl5bn1": "aspiny",
    "l3-sst-interneuron-hl5mn1": "aspiny",
}
WINDOWS = dict(count="num_spikes on 1 s long-square sweeps within one step of primary_pa; null = 0 spikes",
               rest_mv="ipfx pre_vm_mv: mean over the 500 ms before onset, -14 mV LJP",
               after_mv="ipfx post_vm_mv: mean over the last 500 ms of the recording (5-7 s after offset), -14 mV LJP; "
                        "the Allen table has no field for the 10 ms ending 200 ms after offset")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def population_ids(features, name):
    """Specimen ids of one population from the specimen-detail rows."""
    spec = POPULATIONS[name]
    return sorted(r["specimen__id"] for r in features
                  if r.get("tag__dendrite_type") == spec["dendrite_type"]
                  and (spec["layers"] is None or r.get("structure__layer") in spec["layers"]))


def matched_sweeps(sweeps, ids, primary_pa, step_pa=STEP_PA):
    """``{specimen_id: [sweep rows]}``: 1 s long-square sweeps within one step of ``primary_pa``."""
    wanted = set(ids)
    out = {}
    for row in sweeps:
        amplitude, duration = row.get("stimulus_absolute_amplitude"), row.get("stimulus_duration")
        if (row.get("specimen_id") not in wanted or amplitude is None or duration is None
                or abs(duration-1.) > .05 or abs(amplitude-primary_pa) > step_pa+AMPLITUDE_TOLERANCE_PA):
            continue
        out.setdefault(row["specimen_id"], []).append(row)
    return out


def per_cell_readings(by_cell, shift_mv=JUNCTION_MV):
    """One count / pre / post value per cell (means over its matched sweeps); null spikes read as 0."""
    counts, pre, post = [], [], []
    for rows in by_cell.values():
        counts.append(float(np.mean([0 if r.get("num_spikes") is None else r["num_spikes"] for r in rows])))
        pres = [r["pre_vm_mv"] for r in rows if r.get("pre_vm_mv") is not None]
        posts = [r["post_vm_mv"] for r in rows if r.get("post_vm_mv") is not None]
        if pres:
            pre.append(float(np.mean(pres))+shift_mv)
        if posts:
            post.append(float(np.mean(posts))+shift_mv)
    return dict(count=counts, pre_mv=pre, post_mv=post)


def spread(values, sd_factor=SD_FACTOR):
    """n, mean, sd, tolerance (sd_factor x sd) and the 5-95 percent range; None below two cells."""
    x = np.asarray(values, float)
    if len(x) < 2:
        return dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None, sd=None, tolerance=None,
                    p5=None, p95=None, min=None, max=None)
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)), tolerance=float(sd_factor*x.std(ddof=1)),
                p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)), min=float(x.min()), max=float(x.max()))


def type_population(features, sweeps, name, primary_pa):
    """The three spreads of one population at one primary drive."""
    ids = population_ids(features, name)
    by_cell = matched_sweeps(sweeps, ids, primary_pa)
    readings = per_cell_readings(by_cell)
    return dict(
        population=name, label=POPULATIONS[name]["label"], cells_in_population=len(ids),
        cells_with_matched_sweeps=len(by_cell), primary_pa=primary_pa, step_pa=STEP_PA,
        matched_sweeps=int(sum(len(v) for v in by_cell.values())), tolerance_rule=f"{SD_FACTOR:g} sd across cells",
        count=spread(readings["count"]), rest_mv=spread(readings["pre_mv"]), after_mv=spread(readings["post_mv"]),
        windows=dict(WINDOWS))


def load(features_path, sweeps_path, expected=None):
    """Both cached pulls with their digests; ``expected`` = {name: sha256} pins them."""
    paths = dict(features=Path(features_path), sweeps=Path(sweeps_path))
    digests = {k: sha256(p) for k, p in paths.items()}
    for key, digest in (expected or {}).items():
        if digest and digests[key] != digest:
            raise ValueError(f"Allen {key} cache digest mismatch: {paths[key].name}")
    features = json.loads(paths["features"].read_text(encoding="utf-8"))
    sweeps = json.loads(paths["sweeps"].read_text(encoding="utf-8"))
    provenance = {key: dict(path=paths[key].name, sha256=digests[key], source=doc.get("source"),
                            fetched_utc=doc.get("fetched_utc"), rows=len(doc.get("rows", [])))
                  for key, doc in (("features", features), ("sweeps", sweeps))}
    return features, sweeps, provenance


def donor_type_populations(features_doc, sweeps_doc, primaries):
    """``{donor_key: type_population}`` for every donor in ``primaries`` ({donor_key: primary_pa})."""
    return {donor: type_population(features_doc["rows"], sweeps_doc["rows"], DONOR_POPULATIONS[donor], primary)
            for donor, primary in primaries.items()}
