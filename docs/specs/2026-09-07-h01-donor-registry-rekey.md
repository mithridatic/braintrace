# H01 donor registry re-key (SP6a)

Programme: `docs/specs/2026-09-07-h01-population-programme.md`, section SP6. Worktree
`h01-registry` / branch `feat/h01-registry`. Serial and small: it precedes SP2's E work and
SP8's builder changes.

## Problem

The donor physiology registry is keyed by polarity in four places, so adding a third human
donor (SP6b-d) would mean editing each of them by hand:

- `h01_cell_types.DONORS = {"E": ..., "I": ...}`; `donor_match(kind)` indexes
  `DONORS[kind.polarity]`.
- `h01_ei_profiles.get_ei_profile(polarity, mode)` picks name, digest, source, voltages and
  axial resistivity through E/I ternaries and looks the density table up with
  `getattr(parameters, polarity + "_" + mode.upper())`.
- `h01_ei_cell._paint_profile` hardcodes the channel prefix (`"H01L2" if E else "H01PV"`) and
  the sodium/potassium reversals (53/-107 mV for E, 50/-85 mV for I).
- `h01_network.py:141` maps the Dale sign to `"E"` or `"I"` and passes only that.

## Design

### Registry (`h01_cell_types.py`)

`DONORS` becomes keyed by a donor key. The two current donors keep their profile names and
sources; the keys are `l2-pyramidal-allen-541563728` and `l5-pv-basket-hl5bn1`. Each record
carries `layer`, `cell_class`, `modifiers` (tuple), `polarity`, `subtype`, `source`,
`profile`, `channel_prefix`, `sodium_reversal_mv`, `potassium_reversal_mv`.

`DONORS_BY_POLARITY` is a derived view `{polarity: record}` for the polarity default of each
sign. With two donors it equals the old `DONORS` mapping plus the new fields. A later donor with
the same polarity does not replace the default; the default is the first record of that polarity
in `DONORS` (`DEFAULT_DONOR_KEYS = {"E": key, "I": key}` is the explicit statement of it).

`donor_for(kind) -> str` resolves the donor key of an `H01CellType` in precedence order:

1. a donor whose `(layer, cell_class, modifiers)` all equal the type's;
2. a donor whose `(layer, cell_class)` equal the type's and whose `modifiers` are empty;
3. a donor whose `cell_class` equals the type's (first in registry order);
4. the polarity default.

`donor_for_tags(tags, polarity) -> str` is the tolerant form used by the network builder:
tags that read as a full type resolve through `donor_for`; tags without exactly one layer and
one class (the existing test fixtures use `("pyramidal",)`) resolve to the polarity default.
The Dale sign the builder already verified against the tags stays the authority on polarity.

`donor_match(kind)` keeps its output shape (`donor`, `match`, `note`) and its match-state
vocabulary (`matched`, and comma-joined `layer`, `class`, `modifier`), computed against the
resolved donor instead of `DONORS[kind.polarity]`, and gains `donor_key`. With the two current
donors every resolution lands on the polarity default, so the population table regenerates with
identical counts (30 matched, 47 layer, 13 layer+modifier, ...) plus a new `donor_key` column.
`type_rows` passes the new field through.

### Profiles (`h01_ei_profiles.py`, `_h01_ei_parameters.py`)

`EIProfile` gains three frozen fields: `channel_prefix`, `sodium_reversal_mv`,
`potassium_reversal_mv`, set from the donor record. Every other field keeps today's value.

`get_donor_profile(key, *, mode="candidate") -> EIProfile` is the primary constructor. The
per-donor constants that were ternaries move into a private `_PHYSIOLOGY` table keyed by donor
key: profile name, metadata digest, source string, `initial_mv`, `reversal_mv` per mode,
`axial_ohm_cm`. An unknown key raises `ValueError`.

`_h01_ei_parameters.py` gains `DONOR_REGIONS = {key: {"candidate": ..., "source": ...}}` so the
density table is looked up by key rather than by building an attribute name. The four existing
constants stay for any reader that imports them.

`get_ei_profile(polarity, mode)` becomes a thin alias:
`get_donor_profile(DEFAULT_DONOR_KEYS[polarity], mode=mode)` after the same argument check. The
test compares its four outputs (`E`/`I` x `candidate`/`source`) field by field against a
snapshot of today's values taken before this change; the three new fields are asserted equal to
the previously hardcoded literals.

`channel_controls` is unchanged in this SP (it branches on `profile.polarity`; moving the
candidate controls into the donor record is SP6c's concern when a third donor needs its own).

### Cell builder (`h01_ei_cell.py`)

`_paint_profile` reads `profile.channel_prefix`, `profile.sodium_reversal_mv` and
`profile.potassium_reversal_mv`. The ion-painting condition (`calcium is not None`) is not
changed here; a one-line comment marks it as SP2's fix.

`make_h01_ei_cell(..., polarity, donor=None)`: when `donor` is given, the profile is
`get_donor_profile(donor, mode=mode)` and its polarity must equal `polarity` (a mismatch is a
`ValueError`, since the region requirements and the network's Dale signs are keyed by polarity);
otherwise the polarity default is used exactly as today. Evidence gains `"donor": key`.

### Network builder (`h01_network.py:141`)

For each simulated cell the builder resolves `donor_for_tags(tags, polarity)` from the
annotations' tags and the verified Dale sign, passes `donor=` to `make_h01_ei_cell`, and
records `donors = {identity: key}` in the build evidence dict. Nothing else in the module
changes.

### Evidence

`docs/evidence/h01_population_types.py` renders the donor table from `DONORS` (key, polarity,
profile, source, layer, class) and adds a `Donor key` column to the cell table.
`h01-population-types.{json,md}` are regenerated; the match counts must be unchanged.

## Tests (co-located, written first)

- `h01_cell_types_test.py`: precedence (exact modifiers > layer+class > class > polarity
  default) exercised against a temporarily extended registry via `monkeypatch`; fallback to the
  polarity default for unmatched types; `donor_for_tags` on layer-less tags; `donor_match`
  unchanged shape plus `donor_key`; `DONORS_BY_POLARITY` equals the defaults.
- `h01_ei_profiles_test.py`: `get_ei_profile` equals the pre-change snapshot for all four
  combinations; new fields equal the old literals; `get_donor_profile` by key; unknown key and
  bad mode raise `ValueError`.
- `h01_ei_cell_test.py`: `donor=` selects the profile and records the key; mismatched donor
  polarity raises; default path unchanged.
- `h01_network_test.py`: evidence carries `donors` for every simulated cell.

Coverage per changed module stays at or above 90 %.

## Out of scope

Third-donor modules, acquisition audits, survey (SP6b-d); the ion-painting condition (SP2);
moving `channel_controls` into the donor record.
