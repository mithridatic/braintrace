"""Assemble the anatomy-transfer term from the per-donor transfer runs on retained H01 anatomy."""

import argparse
import hashlib
import json
import math
from pathlib import Path

PRIMARY = {
    "l2-pyramidal-allen-541563728": "transfer-e310",
    "l4-pyramidal-allen-527952884": "transfer-l4-090",
    "l3-sst-interneuron-hl5mn1": "transfer-sst-100",
    "l5-pv-basket-hl5bn1": "transfer-pv-190",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def donor_verdict(primary, half):
    """One donor's pass/fail under the registered rule; missing dt-half means unconfirmed."""
    verdict = primary["verdict_vs_human"]
    band_ok = verdict["repeat_band"] == "held" if primary["repeat_counts"] else verdict["exact"] in ("held", "missed_within_one")
    finite = primary["output_site"]["finite"] and primary["soma_site"]["finite"]
    half_agrees = None if half is None else (half["output_site"]["count"] == primary["output_site"]["count"]
                                             and half["output_site"]["finite"])
    passed = bool(band_ok and finite and half_agrees)
    return dict(cell=primary["cell"], input_na=primary["current_na"], count=primary["output_site"]["count"],
                human_count=primary["registered_human_count"], repeat_counts=primary["repeat_counts"],
                donor_fit_count=primary["donor_model_count_on_own_anatomy"], finite=finite,
                rest_mv=primary["rest_before_pulse_mv"], peak_mv=primary["peak_mv"],
                count_rule=("repeat band" if primary["repeat_counts"] else "within one"),
                count_rule_held=bool(band_ok), dt_half_count=None if half is None else half["output_site"]["count"],
                dt_half_agrees=half_agrees, passed=passed)


def decide(folder):
    folder = Path(folder)
    donors, hashes = {}, {}
    for donor, label in PRIMARY.items():
        primary = json.loads((folder/label/"run.json").read_text())
        half_path = folder/(label+"-dthalf")/"run.json"
        half = json.loads(half_path.read_text()) if half_path.exists() else None
        donors[donor] = donor_verdict(primary, half)
        hashes[label] = sha256(folder/label/"run.json")
        if half is not None:
            hashes[label+"-dthalf"] = sha256(half_path)
    passed = sum(1 for d in donors.values() if d["passed"])
    return dict(
        anatomy_transfer=passed/len(donors), measured=True,
        verdict=f"{passed} of {len(donors)} deployed donors hold their human count on retained H01 anatomy",
        scope="Each deployed donor driven on one type-matched, production-imported H01 component under its own "
              "recording's step protocol; a transfer reading against another human cell, not a measurement "
              "of the H01 donor. dt 0.005 ms implicit with a dt-half count repeat. No channel provenance qualified.",
        rule="pass = finite traces, count inside the human repeat band (or within one where no repeats exist), "
             "and the dt-half repeat reproduces the count",
        donors=donors, input_hashes=hashes)


KEEP_RULE_NAMES = ("finite", "rest_and_return", "no_spike_before_pulse", "count_in_band", "dt_half_reproduces")
KEEP_RULE = ("keep = finite output and soma traces; mean output voltage over the 100 ms before the pulse and "
             "over the 10 ms ending 200 ms after it both within 10 mV of the donor recording's rest; no -20 mV "
             "crossing before the pulse; count within max(2, 30 percent) of the human count at the primary "
             "input; and the dt-half repeat (0.0025 ms) is finite and reproduces the count")
KEEP_SCOPE = ("A kept cell reproduces a count and a resting level from another human donor's recording on "
              "retained H01 anatomy under that donor's step protocol; it is not a measurement of the H01 "
              "donor's own cell, and no channel provenance is qualified.")
REST_TOLERANCE_MV = 10.


def count_band(human):
    """Integer count band from |count - human| <= max(2, 0.3 x human)."""
    width = max(2., .3*human)
    return [int(math.ceil(human-width)), int(math.floor(human+width))]


def keep_verdict(primary, half, donor_rest_mv):
    """One cell's five-rule keep verdict; a missing dt-half repeat is 'dt_half_missing', never a keep."""
    count, human = primary["output_site"]["count"], primary["registered_human_count"]
    band = count_band(human)
    return_mv = primary.get("return_mv")
    rest_mv = primary.get("rest_mean_mv")
    rest_ok = rest_mv is not None and abs(rest_mv-donor_rest_mv) <= REST_TOLERANCE_MV
    return_ok = bool(primary.get("return_available")) and return_mv is not None and abs(return_mv-donor_rest_mv) <= REST_TOLERANCE_MV
    rules = dict(
        finite=bool(primary["output_site"]["finite"] and primary["soma_site"]["finite"]),
        rest_and_return=bool(rest_ok and return_ok),
        no_spike_before_pulse=primary["pre_pulse_count"] == 0,
        count_in_band=band[0] <= count <= band[1],
        dt_half_reproduces=None if half is None else bool(half["output_site"]["finite"]
                                                          and half["output_site"]["count"] == count))
    failed = [name for name in KEEP_RULE_NAMES if rules[name] is False]
    if rules["dt_half_reproduces"] is None and not failed:   # the repeat only runs for cells holding rules 1-4
        failed.append("dt_half_missing")
    keep = not failed
    return dict(cell=primary["cell"], donor=primary["donor"], count=count, human_count=human, count_band=band,
                finite=rules["finite"], rest_mean_mv=primary["rest_mean_mv"], return_mv=return_mv,
                pre_pulse_count=primary["pre_pulse_count"],
                dt_half_count=None if half is None else half["output_site"]["count"],
                rules=rules, keep=keep, drop_reason=None if keep else ",".join(failed))


HARD_RULE_NAMES = ("finite", "no_spike_before_pulse", "recruitable_in_human_range", "fires_at_highest",
                   "no_block_in_recorded_range", "dt_half_reproduces")
RECORDED_READING_NAMES = ("rheobase_in_step",)   # recorded beside the verdict, not a rule (coordinator, 2026-09-16)
PLAUSIBILITY_RULE_NAMES = ("count_in_type_spread", "rest_in_type_spread", "return_in_type_spread")
FAILURE_RULE_NAMES = HARD_RULE_NAMES + PLAUSIBILITY_RULE_NAMES
DONOR_BAND_RULE_NAMES = ("count_in_repeat_range", "rest_in_donor_spread")
FAILURE_RULE = ("keep = every hard failure edge holds and every plausibility rule holds. Hard edges: finite output "
                "and soma traces; no -20 mV crossing before the pulse; recruitable in the human range (the ramp "
                "rheobase at or below the human's highest recorded amplitude and the step run at the primary drive "
                "firing at least once); the ramp still firing at the human's highest recorded amplitude; no "
                "depolarisation block at or below that amplitude (the block current above it is recorded, not "
                "scored); the dt-half repeat (0.0025 ms) finite and reproducing the count. Plausibility rules "
                "(J, 2026-09-16): the step count at the primary input, the mean output voltage over the 100 ms "
                "before the pulse, and over the 10 ms ending 200 ms after it, each within 2 across-cell sd of the "
                "same reading over the human cells of the donor's type in the Allen Cell Types database "
                "(human-datums.json type_population; datum = the donor's own value). The single-donor bands "
                "(count_band across the sweeps within one step of the primary; 3 across-sweep sd of the donor's own "
                "pre-pulse and post-pulse family levels) and the former 10 mV / 30 percent bands are recorded beside "
                "the verdict for comparison and are not the gate. rheobase_in_step (the 1 s ramp rheobase within one "
                "sweep step of the human long-square rheobase) is a recorded reading, not a rule: a 570 pA/s ramp and "
                "a step are different measurement chains and the donors' own Allen recordings document a step-to-ramp "
                "offset (spec 2026-09-16-h01-c3-partner-expansion, step C).")
REST_SD_FACTOR = 3.


def count_band_of(datums):
    """The human count band: ``count_band`` min/max when the datums carry it, else the registered repeats.

    Returns
    -------
    tuple
        ``(low, high, source)``; ``(None, None, source)`` when neither is available.
    """
    band = datums.get("count_band") or {}
    if band.get("min") is not None and band.get("max") is not None:
        return band["min"], band["max"], "count_band (sweeps within one step of the primary)"
    repeats = list(datums.get("repeat_counts") or [])
    if repeats:
        return min(repeats), max(repeats), "repeat_counts (registered repeats at the primary; no count_band)"
    return None, None, "unavailable"


def ramp_readings_of(run):
    """The ramp readings of a ramp run.json (``h01_anatomy_transfer_run.summarize`` nests them under ``ramp``).

    Returns the nested block with the run's ``pre_pulse_count`` and ``finite`` (output and
    soma traces, and the ramp analysis) folded in; a flat dict (already readings) passes through.
    """
    if run is None:
        return None
    if not isinstance(run.get("ramp"), dict):
        return run
    readings = dict(run["ramp"])
    readings.setdefault("pre_pulse_count", run.get("pre_pulse_count", 0))
    finite = [readings.get("finite", True)]
    for site in ("output_site", "soma_site"):
        if isinstance(run.get(site), dict) and "finite" in run[site]:
            finite.append(run[site]["finite"])
    readings["finite"] = bool(all(finite))
    return readings


def firing_range_rules(primary, ramp, datums):
    """The three firing-range readings against the human datums; None where a reading is unavailable."""
    count, repeats = primary["output_site"]["count"], list(datums.get("repeat_counts") or [])
    low, high, band_source = count_band_of(datums)
    band = datums.get("count_band") or {}
    rules = dict(rheobase_in_step=None, fires_at_highest=None,
                 count_in_repeat_range=(low <= count <= high) if low is not None else None)
    readings = dict(model_rheobase_pa=None, model_block_pa=None, model_last_spike_pa=None, ramp_max_pa=None,
                    human_rheobase_pa=datums.get("rheobase_pa"), human_step_pa=datums.get("sweep_step_pa"),
                    human_highest_firing_pa=datums.get("highest_firing_pa"), human_repeat_counts=repeats,
                    human_count_band=None if low is None else [low, high], human_count_band_source=band_source,
                    human_count_band_sweeps=band.get("sweeps"), human_count_band_amplitudes_pa=band.get("amplitudes_pa"),
                    human_count_band_counts=band.get("counts"), block_above_recorded_range=None)
    if ramp is None:
        return rules, readings
    rheobase = ramp.get("rheobase_na")
    block = ramp.get("block_na")
    spikes = ramp.get("ramp_spike_times_ms") or []
    ramp_max, pulse = ramp["ramp_max_na"]*1e3, primary["pulse_ms"]
    readings.update(ramp_max_pa=ramp_max, model_rheobase_pa=None if rheobase is None else rheobase*1e3,
                    model_block_pa=None if block is None else block*1e3)
    if spikes:
        readings["model_last_spike_pa"] = ramp_max*(spikes[-1]-pulse[0])/(pulse[1]-pulse[0])
    human_rheobase, step, highest = readings["human_rheobase_pa"], readings["human_step_pa"], readings["human_highest_firing_pa"]
    if human_rheobase is not None and step is not None:
        rules["rheobase_in_step"] = (readings["model_rheobase_pa"] is not None
                                     and abs(readings["model_rheobase_pa"]-human_rheobase) <= step)
    if highest is not None and ramp_max >= highest:
        last = readings["model_last_spike_pa"]
        rules["fires_at_highest"] = last is not None and last >= highest-1e-6
        readings["block_above_recorded_range"] = None if block is None else bool(block*1e3 > highest)
    return rules, readings


def _type_rule(value, datum, population, key):
    """One plausibility rule: |value - datum| <= the type population's tolerance; None when unavailable."""
    stats = (population or {}).get(key) or {}
    tolerance = stats.get("tolerance")
    if value is None or datum is None or tolerance is None:
        return None, tolerance
    return bool(abs(value-datum) <= tolerance), tolerance


def plausibility_rules(primary, donor):
    """The three plausibility rules against the donor's type population, with their readings."""
    population = donor.get("type_population")
    measured = list(donor.get("measured_counts_at_primary") or donor.get("repeat_counts") or [])
    count_datum = float(sum(measured))/len(measured) if measured else None
    rest_datum, after_datum = donor.get("rest_repeat_mean_mv"), donor.get("after_repeat_mean_mv")
    count, rest_mv = primary["output_site"]["count"], primary.get("rest_mean_mv")
    return_mv = primary.get("return_mv") if primary.get("return_available") else None
    count_ok, count_tol = _type_rule(count, count_datum, population, "count")
    rest_ok, rest_tol = _type_rule(rest_mv, rest_datum, population, "rest_mv")
    return_ok, after_tol = _type_rule(return_mv, after_datum, population, "after_mv")
    if return_ok is None and after_tol is not None and after_datum is not None:
        return_ok = False    # the trace does not reach the window: the leg is measured as absent
    rules = dict(count_in_type_spread=count_ok, rest_in_type_spread=rest_ok, return_in_type_spread=return_ok)
    readings = dict(type_population=None if population is None else population.get("population"),
                    type_population_label=None if population is None else population.get("label"),
                    type_cells=None if population is None else population.get("cells_with_matched_sweeps"),
                    type_tolerance_rule=None if population is None else population.get("tolerance_rule"),
                    count_datum=count_datum, count_tolerance=count_tol,
                    type_count_mean=None if population is None else population["count"].get("mean"),
                    rest_type_datum_mv=rest_datum, rest_type_tolerance_mv=rest_tol,
                    after_type_datum_mv=after_datum, after_type_tolerance_mv=after_tol,
                    type_windows=None if population is None else population.get("windows"))
    return rules, readings


def keep_verdict_failure(primary, ramp, half, donor):
    """One cell's test-to-failure verdict under the split gate; the single-donor bands beside it.

    Parameters
    ----------
    primary, ramp, half : dict or None
        Run JSON of the donor step run, the ramp run and the dt-half repeat.
    donor : dict
        ``rest_mv`` and ``sd_mv`` from donor-rest.json merged with the donor's human datums
        (``rheobase_pa``, ``sweep_step_pa``, ``highest_firing_pa``, ``count_band``,
        ``repeat_counts``, ``measured_counts_at_primary``, ``rest_repeat_mean_mv``,
        ``rest_repeat_sd_mv``, ``after_repeat_mean_mv``, ``after_repeat_sd_mv`` and
        ``type_population``). Hard edges use the human firing-range datums; plausibility
        rules use the donor's own value as the datum and 2 across-cell sd of the type
        population as the tolerance; the single-donor bands (``count_band``, 3 across-sweep
        sd of each rest leg) are written under ``donor_band_rules`` for comparison only.

    Returns
    -------
    dict
        ``rules`` (hard + plausibility), ``hard_rules``, ``plausibility_rules``,
        ``donor_band_rules``, readings, ``keep``, ``drop_reason``, ``pending`` and
        ``legacy_verdict``; a missing ramp or dt-half repeat is pending, never a keep; an
        unavailable datum is never a pass.
    """
    count = primary["output_site"]["count"]
    ramp = ramp_readings_of(ramp)
    repeat_sd = donor.get("rest_repeat_sd_mv")
    rest_sd_source = ("across-sweep repeat sd (human-datums rest_repeat_sd_mv)" if repeat_sd is not None
                      else "within-trace sd (donor-rest sd_mv; repeat sd absent)")
    tolerance = REST_SD_FACTOR*(donor["sd_mv"] if repeat_sd is None else repeat_sd)
    repeat_mean = donor.get("rest_repeat_mean_mv")
    rest_datum = donor["rest_mv"] if repeat_mean is None else repeat_mean   # datum and tolerance from one sweep set
    rest_mv, return_mv = primary.get("rest_mean_mv"), primary.get("return_mv")
    rest_ok = rest_mv is not None and abs(rest_mv-rest_datum) <= tolerance
    after_mean, after_sd = donor.get("after_repeat_mean_mv"), donor.get("after_repeat_sd_mv")
    return_scored = after_mean is not None and after_sd is not None
    after_tolerance = REST_SD_FACTOR*after_sd if return_scored else None
    if return_scored:
        return_ok = (bool(primary.get("return_available")) and return_mv is not None
                     and abs(return_mv-after_mean) <= after_tolerance)
        return_leg = "scored against after_repeat_mean_mv +- 3 after_repeat_sd_mv (the family's 10 ms ending 200 ms after offset)"
    else:
        return_ok = True
        return_leg = ("not scored: the recording carries no across-sweep datum for the 10 ms ending 200 ms after offset"
                      + (" (no sweep reaches it)" if after_mean is None else " (one sweep, no repeat spread)"))
    firing, readings = firing_range_rules(primary, ramp, donor)
    plausible, type_readings = plausibility_rules(primary, donor)
    readings.update(type_readings)
    readings.update(rest_datum_mv=rest_datum, rest_sd_mv=donor["sd_mv"] if repeat_sd is None else repeat_sd,
                    rest_sd_source=rest_sd_source, donor_single_sweep_rest_mv=donor["rest_mv"],
                    after_datum_mv=after_mean, after_sd_mv=after_sd, after_tolerance_mv=after_tolerance,
                    return_leg_scored=return_scored, return_leg=return_leg,
                    rest_leg_held=bool(rest_ok), return_leg_held=bool(return_ok) if return_scored else None)
    highest, block = readings["human_highest_firing_pa"], readings["model_block_pa"]
    no_block = None if ramp is None or highest is None else not (block is not None and block <= highest)
    model_rheobase = readings["model_rheobase_pa"]
    recruitable = (None if ramp is None or highest is None
                   else bool(model_rheobase is not None and model_rheobase <= highest and count >= 1))
    allen = donor.get("allen_ramp_threshold") or {}
    window_s = (primary["pulse_ms"][1]-primary["pulse_ms"][0])/1e3
    readings.update(rheobase_in_step=firing["rheobase_in_step"],
                    rheobase_in_step_note="recorded, not a rule: 1 s ramp rheobase vs the human long-square rheobase",
                    human_threshold_i_ramp_pa=allen.get("threshold_i_ramp_pa"),
                    human_threshold_i_long_square_pa=allen.get("threshold_i_long_square_pa"),
                    human_step_to_ramp_offset_pa=allen.get("step_to_ramp_offset_pa"),
                    human_ramp_slope_note=None if not allen else f"Allen slow ramp, rheobase {allen.get('peak_t_ramp_s')} s after onset",
                    model_ramp_slope_pa_per_s=None if ramp is None else readings["ramp_max_pa"]/window_s,
                    fires_at_primary_step=count >= 1)
    hard = dict(finite=bool(primary["output_site"]["finite"] and primary["soma_site"]["finite"]
                            and (ramp is None or ramp.get("finite", True))),
                no_spike_before_pulse=primary["pre_pulse_count"] == 0 and (ramp is None or ramp.get("pre_pulse_count", 0) == 0),
                recruitable_in_human_range=recruitable, fires_at_highest=firing["fires_at_highest"],
                no_block_in_recorded_range=no_block,
                dt_half_reproduces=None if half is None else bool(half["output_site"]["finite"]
                                                                  and half["output_site"]["count"] == count))
    donor_band = dict(count_in_repeat_range=firing["count_in_repeat_range"], rest_in_donor_spread=bool(rest_ok and return_ok))
    rules = dict(hard, **plausible)
    failed = [name for name in FAILURE_RULE_NAMES if rules[name] is False]
    pending = []
    if ramp is None:
        pending.append("ramp_missing")   # not yet measured: the ramp edges stay None
    elif any(rules[name] is None for name in ("recruitable_in_human_range", "fires_at_highest", "no_block_in_recorded_range")):
        failed.append("firing_datum_unavailable")   # an unavailable row is not a pass (qualification rules)
    if any(rules[name] is None for name in PLAUSIBILITY_RULE_NAMES):
        failed.append("plausibility_datum_unavailable")
    if rules["dt_half_reproduces"] is None and not failed:
        pending.append("dt_half_missing")   # the repeat only runs for cells holding the measured rules
    keep = not failed and not pending
    return dict(cell=primary["cell"], donor=primary["donor"], count=count, human_repeat_counts=readings["human_repeat_counts"],
                rest_mean_mv=rest_mv, return_mv=return_mv, rest_tolerance_mv=tolerance, donor_rest_mv=rest_datum,
                pre_pulse_count=primary["pre_pulse_count"], readings=readings,
                dt_half_count=None if half is None else half["output_site"]["count"],
                rules=rules, hard_rules=hard, plausibility_rules=plausible, donor_band_rules=donor_band,
                recorded_readings=dict(rheobase_in_step=firing["rheobase_in_step"]),
                keep=keep, drop_reason=",".join(failed) or None, pending=",".join(pending) or None,
                measured_rules=[name for name in FAILURE_RULE_NAMES if rules[name] is not None],
                unmeasured_rules=[name for name in FAILURE_RULE_NAMES if rules[name] is None],
                donor_band_failed=[name for name in DONOR_BAND_RULE_NAMES if donor_band[name] is False],
                legacy_verdict=keep_verdict(primary, half, donor["rest_mv"]))


def _primary_holds(verdict):
    """Every rule but the dt-half one holds (None counts as not held)."""
    return all(verdict["rules"][name] is True for name in verdict["rules"] if name != "dt_half_reproduces")


def _primary_failures(verdict):
    return [name for name in verdict["rules"] if name != "dt_half_reproduces" and verdict["rules"][name] is not True]


def _load_keep_runs(folder, cell, with_ramp=False, ramp_folder=None):
    """Primary, dt-half (and ramp) run JSON of one cell with their hashes; the ramp may live in ``ramp_folder``."""
    label = f"transfer-all-{cell}"
    primary_path = folder/label/"run.json"
    if not primary_path.exists():
        return (None, None, {}) + ((None,) if with_ramp else ())
    hashes = {label: sha256(primary_path)}
    extra = {}
    for suffix in ("-dthalf",)+(("-ramp",) if with_ramp else ()):
        path = (folder if suffix != "-ramp" or ramp_folder is None else Path(ramp_folder))/(label+suffix)/"run.json"
        extra[suffix] = None
        if path.exists():
            extra[suffix] = json.loads(path.read_text())
            hashes[label+suffix] = sha256(path)
    loaded = (json.loads(primary_path.read_text()), extra["-dthalf"], hashes)
    return loaded + ((extra["-ramp"],) if with_ramp else ())


def _failure_status(verdict, phase):
    """'held', 'dropped' (a measured rule is False) or 'pending' (a run is still missing), with the reason."""
    names = [name for name in verdict["rules"] if phase == "final" or name != "dt_half_reproduces"]
    failed = [name for name in names if verdict["rules"][name] is False]
    failed += [reason for reason in (verdict["drop_reason"] or "").split(",") if reason.endswith("_datum_unavailable")]
    if failed:
        return "dropped", ",".join(failed)
    if verdict["pending"] and (phase == "final" or "ramp_missing" in verdict["pending"]):
        return "pending", verdict["pending"]
    return "held", None


def decide_keep(folder, types_path, rests_path, phase="final", gate="bands", datums_path=None, ramp_folder=None):
    """Per-cell keep/drop over the population; 'primary' lists the dt-half candidates.

    ``gate="bands"`` is the former five-rule verdict; ``gate="failure"`` is the test-to-failure
    verdict, which also reads ``transfer-all-<cell>-ramp/run.json`` (under ``ramp_folder`` when
    given, else ``folder``) and the human datums file. Under the failure gate a cell whose ramp
    (or, in the final phase, dt-half repeat) has not run is ``pending``: neither kept nor
    dropped, its unmeasured rules ``None``.
    """
    folder = Path(folder)
    rows = json.loads(Path(types_path).read_text())["rows"]
    rests = json.loads(Path(rests_path).read_text())["donors"]
    datums = None
    if gate == "failure":
        if datums_path is None:
            raise ValueError("The test-to-failure gate needs --datums (human-datums.json)")
        datums = json.loads(Path(datums_path).read_text())["donors"]
    cells, dropped, pending, missing, hashes = {}, {}, {}, [], {}
    for row in rows:
        cell = row["cell_id"]
        if gate == "failure":
            primary, half, run_hashes, ramp = _load_keep_runs(folder, cell, with_ramp=True, ramp_folder=ramp_folder)
        else:
            primary, half, run_hashes = _load_keep_runs(folder, cell)
        if primary is None:
            missing.append(cell)
            continue
        hashes.update(run_hashes)
        if gate == "failure":
            donor = dict(datums[row["donor_key"]], rest_mv=rests[row["donor_key"]]["rest_mv"],
                         sd_mv=rests[row["donor_key"]]["sd_mv"])
            verdict = keep_verdict_failure(primary, ramp, half, donor)
            status, reason = _failure_status(verdict, phase)
            if status == "dropped":
                dropped[cell] = reason
            elif status == "pending":
                pending[cell] = reason
        else:
            verdict = keep_verdict(primary, half, rests[row["donor_key"]]["rest_mv"])
            held = _primary_holds(verdict) if phase == "primary" else verdict["keep"]
            if not held:
                dropped[cell] = ",".join(_primary_failures(verdict)) if phase == "primary" else verdict["drop_reason"]
        cells[cell] = verdict
    held_cells = sorted(cell for cell in cells if cell not in dropped and cell not in pending)
    result = dict(phase=phase, gate=gate, dropped=dropped, missing=sorted(missing), cells=cells, input_hashes=hashes)
    if gate == "failure":
        result["pending"] = pending
        result["ramp_folder"] = None if ramp_folder is None else str(ramp_folder)
    if phase == "primary":
        result["candidates"] = held_cells
        return result
    by_donor = {}
    for row in rows:
        entry = by_donor.setdefault(row["donor_key"], {"kept": 0, "dropped": 0})
        if row["cell_id"] in cells:
            entry["kept" if row["cell_id"] in held_cells else "dropped"] += 1
    result.update(kept=held_cells, rule=FAILURE_RULE if gate == "failure" else KEEP_RULE, scope=KEEP_SCOPE,
                  by_donor=by_donor,
                  counts=dict(kept=len(held_cells), dropped=len(dropped), missing=len(missing), total=len(rows)))
    if gate == "failure":
        result["counts"]["pending"] = len(pending)
        result["datums_sha256"] = sha256(datums_path)
        result["rule_counts"] = rule_counts(cells)
    return result


def rule_counts(cells):
    """Per-rule held / failed / unmeasured counts for the gate rules, the single-donor bands and the legacy bands."""
    def tally(name, pick):
        values = [pick(v).get(name) for v in cells.values()]
        return dict(held=sum(1 for x in values if x is True), failed=sum(1 for x in values if x is False),
                    unmeasured=sum(1 for x in values if x is None))
    return dict(hard={n: tally(n, lambda v: v["hard_rules"]) for n in HARD_RULE_NAMES},
                plausibility={n: tally(n, lambda v: v["plausibility_rules"]) for n in PLAUSIBILITY_RULE_NAMES},
                donor_band={n: tally(n, lambda v: v["donor_band_rules"]) for n in DONOR_BAND_RULE_NAMES},
                recorded={n: tally(n, lambda v: v["recorded_readings"]) for n in RECORDED_READING_NAMES},
                legacy={n: tally(n, lambda v: v["legacy_verdict"]["rules"]) for n in KEEP_RULE_NAMES})


def _print_keep(decision):
    pending = decision.get("pending") or {}
    tail = f"; {len(pending)} pending (not yet measured)" if pending else ""
    if decision["phase"] == "primary":
        print(f"{len(decision['candidates'])} candidates for the dt-half repeat; {len(decision['dropped'])} dropped; "
              f"{len(decision['missing'])} missing{tail}")
    else:
        counts = decision["counts"]
        print(f"kept {counts['kept']} / dropped {counts['dropped']} / missing {counts['missing']} of {counts['total']}{tail}")
    for cell, reason in sorted(decision["dropped"].items()):
        print(f"  drop {cell}: {reason}")
    for cell, reason in sorted(pending.items()):
        print(f"  pending {cell}: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep", action="store_true", help="Per-cell keep/drop decision instead of the four-donor term.")
    parser.add_argument("--types", default="docs/evidence/h01-population-types.json")
    parser.add_argument("--rests", default="docs/evidence/h01-keep-drop/donor-rest.json")
    parser.add_argument("--phase", choices=["primary", "final"], default="final")
    parser.add_argument("--gate", choices=["bands", "failure"], default="bands",
                        help="'failure' is the test-to-failure gate (needs --datums and the -ramp runs).")
    parser.add_argument("--datums", default="docs/evidence/h01-c3-keep-drop/human-datums.json")
    parser.add_argument("--ramp-folder", default=None,
                        help="Folder holding transfer-all-<cell>-ramp runs when they are not beside the primaries "
                             "(the 17 kept cells: primaries under h01-keep-drop/runs, ramps under h01-c3-keep-drop/runs).")
    args = parser.parse_args()
    if args.keep:
        decision = decide_keep(args.folder, args.types, args.rests, args.phase, gate=args.gate,
                               datums_path=args.datums if args.gate == "failure" else None,
                               ramp_folder=args.ramp_folder)
        Path(args.output).write_text(json.dumps(decision, indent=2)+"\n", newline="\n")
        _print_keep(decision)
        return
    decision = decide(args.folder)
    Path(args.output).write_text(json.dumps(decision, indent=2)+"\n", newline="\n")
    print(decision["verdict"])
    for donor, row in decision["donors"].items():
        print(f"  {donor}: count {row['count']} vs human {row['human_count']} ({row['count_rule']}) "
              f"dt-half {row['dt_half_count']} -> {'PASS' if row['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
