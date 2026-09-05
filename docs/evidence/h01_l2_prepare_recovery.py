"""Prepare an isolated, source-checked selective sodium-recovery mechanism."""

import hashlib
import json
from pathlib import Path


def patch_recovery(source):
    """Add a recovery-only time factor to the pinned NaTs mechanism.

    Parameters
    ----------
    source : bytes
        Unmodified NaTs source from the retrieved Allen template.

    Returns
    -------
    bytes
        Patched source with default recovery factor one.

    Raises
    ------
    ValueError
        If the source digest differs from the inspected version.
    """
    if hashlib.sha256(source).hexdigest() != "d75ed63d6367fe27c6e5671704594a510e8d83b57db835f30fc36e2f24bef6cf":
        raise ValueError("Unexpected source NaTs digest.")
    text = source.decode()
    newline = "\r\n" if "\r\n" in text else "\n"
    replacements = {
        "RANGE gbar, g, ina": "RANGE gbar, g, ina, h_recovery_factor",
        "gbar = 0.00001 (S/cm2)": "gbar = 0.00001 (S/cm2)" + newline + "\th_recovery_factor = 1",
        "hTau = (1/(hAlpha + hBeta))/qt": "hTau = (1/(hAlpha + hBeta))/qt" + newline +
            "\t\tif (hInf > h) { hTau = hTau * h_recovery_factor }",
    }
    for old, new in replacements.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text.encode()


def main():
    """Copy verified mechanism files and record the isolated source manifest."""
    evidence = Path(__file__).parent
    cache = evidence.parents[1] / ".cache/human-pyramidal-l2"
    destination = cache / "sodium-recovery-source"
    destination.mkdir(exist_ok=True)
    manifest = json.loads((evidence / "h01-pyramidal-allen-l2-model-source.json").read_text())
    records = []
    for row in manifest["files"]:
        if not row["file"].endswith(".mod"):
            continue
        original = (cache / "source-model" / row["file"]).read_bytes()
        assert hashlib.sha256(original).hexdigest() == row["sha256_local"]
        content = patch_recovery(original) if row["file"] == "NaTs.mod" else original
        output = destination / row["file"]
        if output.exists() and output.read_bytes() != content:
            raise ValueError("Refusing to replace a different isolated source: " + str(output))
        output.write_bytes(content)
        records.append({"file": row["file"], "original_sha256": row["sha256_local"],
                        "prepared_sha256": hashlib.sha256(content).hexdigest(), "changed": content != original})
    report = {"files": records, "destination": str(destination),
              "change": "NaTs hTau multiplied only when hInf > h; default factor one",
              "qualification": "Prepared diagnostic sources, not a validated kinetic or cell result."}
    (evidence / "h01-l2-sodium-recovery-source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": len(records), "changed": [r["file"] for r in records if r["changed"]]}))


if __name__ == "__main__":
    main()
