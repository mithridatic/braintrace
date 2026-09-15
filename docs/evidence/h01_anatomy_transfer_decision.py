"""Assemble the anatomy-transfer term from the per-donor transfer runs on retained H01 anatomy."""

import argparse
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = decide(args.folder)
    Path(args.output).write_text(json.dumps(decision, indent=2)+"\n", newline="\n")
    print(decision["verdict"])
    for donor, row in decision["donors"].items():
        print(f"  {donor}: count {row['count']} vs human {row['human_count']} ({row['count_rule']}) "
              f"dt-half {row['dt_half_count']} -> {'PASS' if row['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
