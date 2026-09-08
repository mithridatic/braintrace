"""Prepare an isolated Kv3 time-constant diagnostic with source hashes."""

import hashlib
import json
from pathlib import Path
import shutil


def main():
    """Copy sources into a new cache and change only the Kv3 time constant."""
    folder = Path(__file__).parent
    cache = folder.parents[1] / ".cache/human-pv"
    original = cache / "closure-recovery-reference"
    target = cache / "kv3-kinetics-reference"
    source = (original / "mod/Kv3_1.mod").read_text()
    modified = source
    for old, new in (("RANGE gbar, g, ik", "RANGE gbar, g, ik, m_tau_factor"),
                     ("PARAMETER\t{", "PARAMETER\t{\n\tm_tau_factor = 1"),
                     ("mTau =  0.2*20.000", "mTau = m_tau_factor * 0.2*20.000")):
        assert modified.count(old) == 1, old
        modified = modified.replace(old, new)
    assert not target.exists(), "Refuse to overwrite an existing diagnostic cache."
    target.mkdir()
    for name in ("source", "mod"):
        shutil.copytree(original / name, target / name)
    (target / "mod/Kv3_1.mod").write_text(modified)
    hashes = {}
    for path in sorted((target / "mod").glob("*.mod")):
        parent = original / "mod" / path.name
        hashes[path.name] = {"parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
                             "diagnostic_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.name != "Kv3_1.mod":
            assert path.read_bytes() == parent.read_bytes()
    report = {"parent_cache": str(original), "diagnostic_cache": str(target),
              "changes": "Kv3 mTau multiplied by positive factor; default 1; equilibrium unchanged",
              "mechanism_hashes": hashes, "qualification": "Uncompiled diagnostic sources; fixed-voltage verification required"}
    (folder / "h01-pv-kv3-kinetics-source.json").write_text(json.dumps(report, indent=2))
    print(target)


if __name__ == "__main__":
    main()
