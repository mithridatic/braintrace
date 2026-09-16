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


FAILURE_RULE_NAMES = ("finite", "rheobase_in_step", "fires_at_highest", "count_in_repeat_range",
                      "rest_in_donor_spread", "no_spike_before_pulse", "dt_half_reproduces")
FAILURE_RULE = ("keep = finite output and soma traces; firing range: the ramp rheobase within one human sweep "
                "step of the human rheobase, the ramp still firing at the human's highest recorded amplitude, and "
                "the step count at the primary input inside the human repeat range; rest: mean output voltage "
                "over the 100 ms before the pulse and over the 10 ms ending 200 ms after it both within 3 across-sweep "
                "repeat sd of the donor recording's rest, no -20 mV crossing before the pulse; numerical: the dt-half "
                "repeat (0.0025 ms) is finite and reproduces the count. The block current above the recorded "
                "range is recorded, not scored (spec 2026-09-16-h01-c3-partner-expansion, step C).")
REST_SD_FACTOR = 3.


def firing_range_rules(primary, ramp, datums):
    """The three firing-range readings against the human datums; None where a reading is unavailable."""
    count, repeats = primary["output_site"]["count"], list(datums.get("repeat_counts") or [])
    rules = dict(rheobase_in_step=None, fires_at_highest=None,
                 count_in_repeat_range=(min(repeats) <= count <= max(repeats)) if repeats else None)
    readings = dict(model_rheobase_pa=None, model_block_pa=None, model_last_spike_pa=None, ramp_max_pa=None,
                    human_rheobase_pa=datums.get("rheobase_pa"), human_step_pa=datums.get("sweep_step_pa"),
                    human_highest_firing_pa=datums.get("highest_firing_pa"), human_repeat_counts=repeats,
                    block_above_recorded_range=None)
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


def keep_verdict_failure(primary, ramp, half, donor):
    """One cell's test-to-failure verdict; the former five-rule verdict is written beside it.

    Parameters
    ----------
    primary, ramp, half : dict or None
        Run JSON of the donor step run, the ramp run and the dt-half repeat.
    donor : dict
        ``rest_mv`` and ``sd_mv`` from donor-rest.json merged with the donor's human datums
        (``rheobase_pa``, ``sweep_step_pa``, ``highest_firing_pa``, ``repeat_counts``,
        ``rest_repeat_sd_mv``). The rest tolerance is 3 x the across-sweep repeat sd; the
        within-trace ``sd_mv`` is the fallback when no repeat sd is recorded.

    Returns
    -------
    dict
        Rules, readings, ``keep``, ``drop_reason`` and ``legacy_verdict``; a missing ramp or
        dt-half repeat is a named reason, never a keep.
    """
    count = primary["output_site"]["count"]
    repeat_sd = donor.get("rest_repeat_sd_mv")
    rest_sd_source = "across-sweep repeat sd (human-datums rest_repeat_sd_mv)" if repeat_sd is not None else         "within-trace sd (donor-rest sd_mv; repeat sd absent)"
    tolerance = REST_SD_FACTOR*(donor["sd_mv"] if repeat_sd is None else repeat_sd)
    rest_mv, return_mv = primary.get("rest_mean_mv"), primary.get("return_mv")
    rest_ok = rest_mv is not None and abs(rest_mv-donor["rest_mv"]) <= tolerance
    return_ok = (bool(primary.get("return_available")) and return_mv is not None
                 and abs(return_mv-donor["rest_mv"]) <= tolerance)
    firing, readings = firing_range_rules(primary, ramp, donor)
    readings.update(rest_sd_mv=donor["sd_mv"] if repeat_sd is None else repeat_sd, rest_sd_source=rest_sd_source)
    rules = dict(finite=bool(primary["output_site"]["finite"] and primary["soma_site"]["finite"]
                             and (ramp is None or ramp.get("finite", True))),
                 **firing,
                 rest_in_donor_spread=bool(rest_ok and return_ok),
                 no_spike_before_pulse=primary["pre_pulse_count"] == 0 and (ramp is None or ramp.get("pre_pulse_count", 0) == 0),
                 dt_half_reproduces=None if half is None else bool(half["output_site"]["finite"]
                                                                   and half["output_site"]["count"] == count))
    failed = [name for name in FAILURE_RULE_NAMES if rules[name] is False]
    if ramp is None:
        failed.append("ramp_missing")
    elif any(rules[name] is None for name in ("rheobase_in_step", "fires_at_highest", "count_in_repeat_range")):
        failed.append("firing_datum_unavailable")
    if rules["dt_half_reproduces"] is None and not failed:
        failed.append("dt_half_missing")
    keep = not failed
    return dict(cell=primary["cell"], donor=primary["donor"], count=count, human_repeat_counts=readings["human_repeat_counts"],
                rest_mean_mv=rest_mv, return_mv=return_mv, rest_tolerance_mv=tolerance, donor_rest_mv=donor["rest_mv"],
                pre_pulse_count=primary["pre_pulse_count"], readings=readings,
                dt_half_count=None if half is None else half["output_site"]["count"],
                rules=rules, keep=keep, drop_reason=None if keep else ",".join(failed),
                legacy_verdict=keep_verdict(primary, half, donor["rest_mv"]))


def _primary_holds(verdict):
    """Every rule but the dt-half one holds (None counts as not held)."""
    return all(verdict["rules"][name] is True for name in verdict["rules"] if name != "dt_half_reproduces")


def _primary_failures(verdict):
    return [name for name in verdict["rules"] if name != "dt_half_reproduces" and verdict["rules"][name] is not True]


def _load_keep_runs(folder, cell, with_ramp=False):
    label = f"transfer-all-{cell}"
    primary_path = folder/label/"run.json"
    if not primary_path.exists():
        return (None, None, {}) + ((None,) if with_ramp else ())
    hashes = {label: sha256(primary_path)}
    extra = {}
    for suffix in ("-dthalf",)+(("-ramp",) if with_ramp else ()):
        path = folder/(label+suffix)/"run.json"
        extra[suffix] = None
        if path.exists():
            extra[suffix] = json.loads(path.read_text())
            hashes[label+suffix] = sha256(path)
    loaded = (json.loads(primary_path.read_text()), extra["-dthalf"], hashes)
    return loaded + ((extra["-ramp"],) if with_ramp else ())


def decide_keep(folder, types_path, rests_path, phase="final", gate="bands", datums_path=None):
    """Per-cell keep/drop over the population; 'primary' lists the dt-half candidates.

    ``gate="bands"`` is the former five-rule verdict; ``gate="failure"`` is the test-to-failure
    verdict, which also reads ``transfer-all-<cell>-ramp/run.json`` and the human datums file.
    """
    folder = Path(folder)
    rows = json.loads(Path(types_path).read_text())["rows"]
    rests = json.loads(Path(rests_path).read_text())["donors"]
    datums = None
    if gate == "failure":
        if datums_path is None:
            raise ValueError("The test-to-failure gate needs --datums (human-datums.json)")
        datums = json.loads(Path(datums_path).read_text())["donors"]
    cells, dropped, missing, hashes = {}, {}, [], {}
    for row in rows:
        cell = row["cell_id"]
        if gate == "failure":
            primary, half, run_hashes, ramp = _load_keep_runs(folder, cell, with_ramp=True)
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
        else:
            verdict = keep_verdict(primary, half, rests[row["donor_key"]]["rest_mv"])
        cells[cell] = verdict
        held = _primary_holds(verdict) if phase == "primary" else verdict["keep"]
        if not held:
            dropped[cell] = ",".join(_primary_failures(verdict)) if phase == "primary" else verdict["drop_reason"]
    held_cells = sorted(cell for cell in cells if cell not in dropped)
    result = dict(phase=phase, gate=gate, dropped=dropped, missing=sorted(missing), cells=cells, input_hashes=hashes)
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
        result["datums_sha256"] = sha256(datums_path)
    return result


def _print_keep(decision):
    if decision["phase"] == "primary":
        print(f"{len(decision['candidates'])} candidates for the dt-half repeat; {len(decision['dropped'])} dropped; "
              f"{len(decision['missing'])} missing")
    else:
        counts = decision["counts"]
        print(f"kept {counts['kept']} / dropped {counts['dropped']} / missing {counts['missing']} of {counts['total']}")
    for cell, reason in sorted(decision["dropped"].items()):
        print(f"  drop {cell}: {reason}")


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
    args = parser.parse_args()
    if args.keep:
        decision = decide_keep(args.folder, args.types, args.rests, args.phase, gate=args.gate,
                               datums_path=args.datums if args.gate == "failure" else None)
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
