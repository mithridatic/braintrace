"""Configured-decay contact evidence and unchanged reference trajectory."""

import hashlib
import json
from pathlib import Path
from docs.biology import default_contact_probe as probe


def test_configured_decay_and_retained_analytic_evidence(tmp_path):
    probe.main(tmp_path)
    report = json.loads((tmp_path/'default-contact.json').read_text())
    settings = json.loads((tmp_path/'settings.json').read_text())
    rows = json.loads((tmp_path/'analytic-sweep.json').read_text())
    assert report['decay'] == settings['decay'] == .99
    assert report['probe_sha256'] == hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
    assert len(rows) == 16
    assert settings['biology']['extracellular']['origin'] == 'synthetic'
