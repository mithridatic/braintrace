# H01 diagnostic diagrams redrawn to Hartshorne's forms (2026-09-09)

## Problem

The H01 evidence used the names of Hartshorne's diagnostic pictures without the pictures.
Four "Youden" plots compared one model at two drives (no human axis, no tolerance, no
repeatability), no search tree was ever drawn although the causal model names one nine
times, and no energy-network (Thevenin four-box) diagram was drawn although the campaign
was run as an "energetic search". Raw traces are no longer on disk; every per-row
human-versus-model number survives in the usable-tier and decision JSON.

## Deliverables (all under docs/evidence, each a script with a co-located `_test.py`)

| Script | Output | Source of every number |
|---|---|---|
| `h01_youden_contract.py` | `h01-youden-contract-{e,i}.{png,svg}`, `h01-youden-contract.json` | `h01-e-usable/b3-all-inputs-usable.json`, `h01-usable-tier-i.json` |
| `h01_search_tree.py` | `h01-search-tree.{png,svg,json}` | curated node list; every cited decision file must exist (tested) |
| `h01_thevenin_boxes.py` | `h01-thevenin-i-clamp.{png,svg}`, `h01-thevenin-ie-pair.{png,svg}`, `h01-thevenin-boxes.json` | `h01-ie-synapse-literature.json`, `h01-i-campaign/a-into-G1a-027.json` (bias), `h01-ie-inhibition/decision.json` |

Youden: human on x, model on y, one point per contract row, 45 degree line, each row's
contract limit as a vertical bar centred on the diagonal, human repeat spread as a
horizontal bar, `o` pass and `x` fail. Search tree: one whole-programme tree (the book's
form), branch styles for established, eliminated, failed at cap, launched-untested and
open, each node carrying its decision file. Thevenin: four-box source, series impedance,
load and dissipation with the measured or assumed value in each box and "not measured"
written where the programme never measured one.

The four `h01-youden-i-candidate-*.png` files are deleted.

## Multivari (added the same day)

| Script | Output | Source |
|---|---|---|
| `h01_multivari_data.py` | `h01-multivari-data.json` | Allen recordings in `.cache` (every suprathreshold long-square sweep, repeats kept apart) and model traces; six traces rerun in the NEURON image into `h01-multivari-runs/` (gitignored, log committed) |
| `h01_multivari_chart.py` | `h01-multivari-{e,i}.{png,svg}`, `h01-multivari-chart.json` | the data JSON only |

Form follows the book's Figures 104 and 136: one measure per panel as deviation from the human
median, x nested cycle within sweep within input within machine, thin rules between inputs and a
heavy rule between human and model, family labels stacked under the axis, and each panel titled
with the family that varies most (range of family means within the parent family).

## Out of scope

No network code or topology changes and no new physiology claim. The only simulations are the six single-cell reruns for the multivari charts.
