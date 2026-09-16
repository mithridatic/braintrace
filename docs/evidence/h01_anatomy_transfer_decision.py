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


def _primary_holds(verdict):
    """Rules 1-4 hold (the dt-half rule is not consulted)."""
    return all(verdict["rules"][name] for name in KEEP_RULE_NAMES[:-1])


def _load_keep_runs(folder, cell):
    label = f"transfer-all-{cell}"
    primary_path, half_path = folder/label/"run.json", folder/(label+"-dthalf")/"run.json"
    if not primary_path.exists():
        return None, None, {}
    hashes = {label: sha256(primary_path)}
    half = None
    if half_path.exists():
        half = json.loads(half_path.read_text())
        hashes[label+"-dthalf"] = sha256(half_path)
    return json.loads(primary_path.read_text()), half, hashes


def decide_keep(folder, types_path, rests_path, phase="final"):
    """Per-cell keep/drop over the population; 'primary' evaluates rules 1-4 to list dt-half candidates."""
    folder = Path(folder)
    rows = json.loads(Path(types_path).read_text())["rows"]
    rests = json.loads(Path(rests_path).read_text())["donors"]
    cells, dropped, missing, hashes = {}, {}, [], {}
    for row in rows:
        cell = row["cell_id"]
        primary, half, run_hashes = _load_keep_runs(folder, cell)
        if primary is None:
            missing.append(cell)
            continue
        hashes.update(run_hashes)
        verdict = keep_verdict(primary, half, rests[row["donor_key"]]["rest_mv"])
        cells[cell] = verdict
        held = _primary_holds(verdict) if phase == "primary" else verdict["keep"]
        if not held:
            reason = ",".join(n for n in KEEP_RULE_NAMES[:-1] if verdict["rules"][n] is False) if phase == "primary" else verdict["drop_reason"]
            dropped[cell] = reason
    held_cells = sorted(cell for cell in cells if cell not in dropped)
    result = dict(phase=phase, dropped=dropped, missing=sorted(missing), cells=cells, input_hashes=hashes)
    if phase == "primary":
        result["candidates"] = held_cells
        return result
    by_donor = {}
    for row in rows:
        entry = by_donor.setdefault(row["donor_key"], {"kept": 0, "dropped": 0})
        if row["cell_id"] in cells:
            entry["kept" if row["cell_id"] in held_cells else "dropped"] += 1
    result.update(kept=held_cells, rule=KEEP_RULE, scope=KEEP_SCOPE, by_donor=by_donor,
                  counts=dict(kept=len(held_cells), dropped=len(dropped), missing=len(missing), total=len(rows)))
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
    args = parser.parse_args()
    if args.keep:
        decision = decide_keep(args.folder, args.types, args.rests, args.phase)
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
