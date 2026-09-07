# H01 experimental profile modes: `finalist` (I) and `b3` (E)

Programme: [population programme](2026-09-07-h01-population-programme.md), SP2 follow-up
under the SP0 rules. Parent spec: [SP2 transfer isolation](2026-09-07-h01-transfer-isolation.md),
"Discovered precondition" and "E side". Registry: [donor re-key](2026-09-07-h01-donor-registry-rekey.md).
Worktree `h01-transfer` / branch `feat/h01-transfer`, after merging `feat/h01-braincell`.
Written 2026-09-07 before code. No simulation is launched by anything written here.

## Problem

The SP2 runner refuses to score the I arms because the BrainCell I profile (mode
`candidate`) carries the pre-finalist somatic Kv3 closing factor 0.5 while the NEURON
finalist traces (`h01-i-energetic/e-kv3-close2-*`) carry 2.0. On the E side the frozen B3
profile exists only as an exported literal (`docs/evidence/h01-e-b3-experimental-profile.py`)
that the registry cannot select, and `_paint_profile` skips the sodium and potassium ions on
a region whose calcium tuple is `None`, so the B3 axon row (inserted NaTs, no calcium) cannot
be painted at all.

Both gaps are held-equal preconditions for SP2. Neither is a promotion: the default mode of
every donor stays `candidate`, and `examples/` output is unchanged.

## Design

### Two non-default modes in the registry

`_h01_ei_parameters.DONOR_REGIONS[key]` gains one extra mode per donor:

| Donor key | New mode | Density table | Source |
| --- | --- | --- | --- |
| `l5-pv-basket-hl5bn1` | `finalist` | `I_FINALIST`, the same object as `I_CANDIDATE` | finalist differs from the candidate by a phase control only (below) |
| `l2-pyramidal-allen-541563728` | `b3` | `E_B3_EXPERIMENTAL`, copied verbatim from `docs/evidence/h01-e-b3-experimental-profile.py` | `b3-sk035-ca-decay-sweep53.json`, sha256 `52f85ac9...92e1` |

The valid modes of a donor are exactly the keys of `DONOR_REGIONS[key]`; the literal
`("candidate", "source")` check in `get_donor_profile` / `get_ei_profile` is replaced by that
lookup. `get_ei_profile(polarity, mode=...)` therefore accepts `finalist` for I and `b3` for E
and rejects the cross pairs (`E`/`finalist`, `I`/`b3`) with `ValueError`.

`_PHYSIOLOGY[key]["reversal_mv"]` gains the new mode: I `finalist` -96.97510324827309 (equal
to candidate and source); E `b3` -87.97993469238281 (`E_B3_EXPERIMENTAL_REVERSAL_MV`, equal to
the E candidate: source -83.97993469238281 shifted by -4). `initial_mv` and `axial_ohm_cm` are
per donor and already equal the B3 export (-83.97993469238281 and 94.62299222737664) and the I
finalist run (-80.0 and 100.0); a test asserts these equalities against the evidence files.

Profile name: `<profile>:<mode>:v1` as today, so `h01-pv-regional-mesh-axon2187:finalist:v1`
and `h01-l2-kv3-ninety-ca133:b3:v1`.

`EIProfile.limitations` for the two new modes is the existing three-line tuple plus one line:

> Experimental mode `<mode>`: registered so the SP2 held-equal transfer arms can run against the
> NEURON finalist; not promoted, not a candidate; the default mode stays candidate.

The tuple for `candidate` and `source` is unchanged (snapshot test).

### Phase controls (`channel_controls`)

`channel_controls(profile, family, mechanism)` returns constructor keywords of the registered
`_PVChannel` family, whose vocabulary is `m_open, m_close, h_open, h_close, h_slope` (a factor
on the gate time constant while the gate moves toward equilibrium in the named direction;
`Kv3_1` names its directions the other way round in `_rate_accessor`). The mapping from the
NEURON flags, verified against the mod files and drivers:

| NEURON flag | Where applied in NEURON | BrainCell keyword | Scope in BrainCell |
| --- | --- | --- | --- |
| PV `somatic_kv3_close_factor` (`m_close_factor_Kv3_1`) | soma sections | `Kv3_1` `m_close` | `family == "soma"` |
| PV `somatic_kv3_tau_factor` | soma sections | `Kv3_1` `m_open` | `family == "soma"` |
| PV `sodium_h_tau_factor`, `sodium_h_recovery_factor`, `sodium_h_slope_mv` | every section with NaTg | `NaTg` `h_close`, `h_open`, `h_slope` | every family |
| L2 `sodium_opening_factor` (`NaTs.m_opening_factor`: `mTau *= f` when `mInf > m`) | soma sections only (`h01_l2_neuron_reference.py:250-255`) | `NaTs` `m_open` | `family == "soma"` |
| L2 `sodium_recovery_factor` (`NaTs.h_recovery_factor`: `hTau *= f` when `hInf > h`) | soma sections only (`:244-249`) | `NaTs` `h_open` | `family == "soma"` |
| L2 `kv3_closing_factor` (`Kv3_1.m_closing_factor`) | soma sections only (`:256-261`) | `Kv3_1` `m_close` | `family == "soma"` |

The export file spells the recovery control `h_recovery`; the channel constructor has no such
keyword, so the registry stores it as `h_open` (identity value 1.0 in B3). A test asserts the
registry's B3 controls equal the export's controls under that renaming.

Controls become a table keyed by `(polarity, mode)`; the candidate entries reproduce today's
return values exactly (E: `NaTs -> {"m_open": 2.}`, `Kv3_1 -> {"m_close": .9}` in every
family; I: `NaTg -> {"h_close": .15, "h_open": 1., "h_slope": 5.}` in every family, soma
`Kv3_1 -> {"m_open": .5, "m_close": .5}`; source: `{}` everywhere). New entries:

- I `finalist`: as I candidate except soma `Kv3_1 -> {"m_open": .5, "m_close": 2.}`.
- E `b3`: soma `NaTs -> {"m_open": 2., "h_open": 1.}`, soma `Kv3_1 -> {"m_close": .9}`; the
  axon `NaTs` row receives `{}` because NEURON applied the factors to soma sections only. This
  is the one place the E `b3` scope is narrower than the E `candidate` scope (which names no
  family because the candidate has NaTs and Kv3_1 in the soma alone).

B3 also carries two flags that are not constructor keywords: `calcium_decay_factor` 1.0 and
`distribute_ih` true. They are already realised inside `E_B3_EXPERIMENTAL` (soma calcium decay
494.01955262344603 ms = source x 1.0, and Ih 9.994138594205759e-05 S/cm2 on soma, dend and apic).
`channel_controls` cannot return them without breaking the channel constructor, so a new
`mode_flags(profile)` returns `{"calcium_decay_factor": 1., "distribute_ih": True}` for E `b3`
and `{}` for every other profile, and the test asserts the table realises both flags.

### Every difference between the I candidate and the I finalist

Compared: `h01-i-energetic/r-candidate.candidate.json` (the pre-finalist candidate that
`I_CANDIDATE` reproduces) against `h01-i-energetic/e-kv3-close2.candidate.json` and the run
metadata `e-kv3-close2-027.json`.

| Flag | candidate | finalist | Registry consequence |
| --- | --- | --- | --- |
| `somatic_kv3_close_factor` | 0.5 | 2.0 | soma `Kv3_1` `m_close` 0.5 -> 2.0 (the only change) |
| `somatic_kv3_tau_factor` | 0.5 | 0.5 | none |
| `somatic_kv3_factor` (run metadata) | 1.0 | 1.0 | none (soma Kv3 density = source) |
| `sodium_h_tau_factor` / `_recovery_factor` / `_slope_mv` | 0.15 / 1.0 / 5.0 | same | none |
| `conductance_factor` NaTg soma | 1.1 | 1.1 | none (0.4916517466528855 x 1.1 = 0.5408169213181742) |
| `somatic_calva_factor` | 0.5 | 0.5 | none (0.0265442718245302 x 0.5 = 0.0132721359122651) |
| `axon_calcium_decay_ms` / `_gamma` | 300 / 0.004 | same | none |
| `cvode_atol`, `nseg_factor`, `refine_region`, observe flags | equal | equal | not profile fields |
| initial voltage, 34 C, 270-1270 ms stimulus | equal | equal | not profile fields |

So `I_FINALIST is I_CANDIDATE` and the finalist mode is a control-only mode.

### Every difference between the E candidate and E `b3`

`E_B3_EXPERIMENTAL` vs `E_CANDIDATE`: soma NaTs 2.6406641498523276 (source x 0.9) vs
3.8142926608978067 (source x 1.3); soma SK 0.001049516063519434 (source x 0.35) vs
0.0029986173243412404 (source); soma calcium decay 494.01955262344603 ms (source) vs
657.0460049891833 ms (source x 1.33); axon channels `(("NaTs", 3.814),)` vs `()`. Equal: every
other density, capacitances, leaks, dend/apic Ih 9.994138594205759e-05, reversal -87.9799...,
initial -83.9799..., Ra 94.6229.... Controls: `b3` adds `NaTs h_open 1.0` (identity) and
narrows both controls to the soma. The B3 input convention "command plus recorded bias"
(`E_B3_EXPERIMENTAL_INPUT`) is a runner concern (`--include-recorded-bias`, already the
default of `h01_l2_braincell_reference.py`), not a profile field.

### `_paint_profile` ion fix (`h01_ei_cell.py`)

Today the sodium and potassium ions are painted only when the region's calcium tuple is not
`None`. New rule: paint `SodiumFixed` and `PotassiumFixed` when the calcium tuple is not
`None` **or** the region lists any channel other than `Ih` (`Ih` is the one registered
mechanism whose root type needs no ion; dend/apic rows carrying only Ih therefore stay exactly
as today). The calcium ion is still painted only with a calcium tuple. A region that lists a
calcium-sensing or calcium-carrying mechanism (`SK`, `Ca_HVA`, `Ca_LVA`) without a calcium
tuple raises `ValueError` instead of failing inside BrainCell.

Effect: the B3 axon row gets sodium, potassium and `H01L2_NaTs`; every candidate/source row
is painted with the same mechanisms as before.

### Runner and reference scripts

- `h01_transfer_isolation_runner.py`: `--check-alignment` / `--score` select the profile with
  `get_ei_profile(manifest["polarity"], mode=manifest["braincell_profile_key"])`; the manifest
  sets `braincell_profile_key` to `finalist` and names the spec of the mode. The refusal path
  stays for an unaligned key. `braincell_command` already passes `--mode <key>`.
- `h01_pv_braincell_reference.py`: `--mode` accepts `finalist`.
- `h01_l2_braincell_reference.py`: `--profile-key` accepts `b3`; `--mode` is accepted as an
  alias of `--profile-key` so both runners take the same flag.

### Causal model

No observation. One dated line under the Y2 registered prediction: the alignment is now
checkable and no longer refused; the arms remain untested.

## Tests (co-located, written first)

- `h01_ei_profiles_test.py`: four candidate/source profiles still equal the pre-re-key snapshot
  (existing test); `finalist` equals `candidate` field by field except `mode`, `name` and the
  extra limitation line; finalist `channel_controls` differ from candidate only at soma `Kv3_1`
  `m_close` 2.0; `b3` regions equal the export literal, `b3` reversal/initial/axial equal the
  export scalars, `b3` controls equal the export controls under `h_recovery -> h_open`, axon
  `NaTs` gets `{}`, `mode_flags` and the realised decay/Ih; cross pairs and unknown modes raise;
  new-mode limitations carry the experimental line and candidate/source do not.
- `_h01_ei_parameters`: `E_B3_EXPERIMENTAL == docs.evidence.h01-e-b3-experimental-profile.E_B3_EXPERIMENTAL`
  (loaded by path, the file name has hyphens) and `I_FINALIST is I_CANDIDATE`.
- `h01_ei_cell_test.py`: a fixture cell built with `polarity="E", mode="b3"` on a soma+axon
  partition paints `SodiumFixed`, `PotassiumFixed` and `H01L2_NaTs` on the axon and no
  calcium there, and runs finitely for a few steps; candidate/source paint lists unchanged
  (the existing runs); the calcium-less `SK` row raises.
- `h01_l2_cell_test.py`: `mode="b3"` runs with finite calcium.
- Runner test: the finalist profile is aligned with `e-kv3-close2-027.json`; the manifest key is
  `finalist`; `--check-alignment` prints `"aligned": true`; the refusal path is exercised through
  a temporary manifest with `braincell_profile_key` `candidate`.
- Reference-script tests: `--mode finalist` and `--profile-key b3` / `--mode b3` reach the
  builder.

Coverage per changed module at or above 90 %.

## Out of scope

Promotion of either mode; any simulation; moving `channel_controls` into the donor record
(SP6c); the E manifest and E arms (registered after the 100 ms benchmark per the SP2 spec).
