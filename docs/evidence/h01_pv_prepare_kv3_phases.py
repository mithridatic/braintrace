"""Prepare an isolated closing-time override for the Kv3 diagnostic."""

import hashlib
import json
from pathlib import Path
import shutil


def main():
    """Copy the validated timing source and preserve all other mechanisms."""
    folder = Path(__file__).parent
    cache = folder.parents[1] / ".cache/human-pv"
    parent, target = cache / "kv3-kinetics-reference", cache / "kv3-phase-reference"
    source = (parent / "mod/Kv3_1.mod").read_text()
    modified = source
    for old, new in (("RANGE gbar, g, ik, m_tau_factor", "RANGE gbar, g, ik, m_tau_factor, m_close_factor"),
                     ("m_tau_factor = 1", "m_tau_factor = 1\n\tm_close_factor = -1"),
                     ("m' = (mInf-m)/mTau", "if (m_close_factor > 0 && mInf < m) {\n        m' = (mInf-m)/(mTau/m_tau_factor*m_close_factor)\n    } else {\n        m' = (mInf-m)/mTau\n    }")):
        assert modified.count(old) == 1, old
        modified = modified.replace(old, new)
    assert not target.exists(), "Refuse to overwrite the diagnostic cache."
    target.mkdir()
    for name in ("source", "mod"):
        shutil.copytree(parent / name, target / name)
    (target / "mod/Kv3_1.mod").write_text(modified)
    hashes = {}
    for path in sorted((target / "mod").glob("*.mod")):
        original = parent / "mod" / path.name
        hashes[path.name] = {"parent_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                             "diagnostic_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.name != "Kv3_1.mod":
            assert original.read_bytes() == path.read_bytes()
    report = {"parent_cache": str(parent), "diagnostic_cache": str(target),
              "changes": "Optional Kv3 closing-time override; disabled by default",
              "mechanism_hashes": hashes, "qualification": "Prepared diagnostic source; requires compile and analytic checks"}
    (folder / "h01-pv-kv3-phase-source.json").write_text(json.dumps(report, indent=2))
    print(target)


if __name__ == "__main__":
    main()
