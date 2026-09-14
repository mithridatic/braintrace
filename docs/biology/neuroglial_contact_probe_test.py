"""Fixed-impulse contact learning and source identity acceptance."""

import hashlib
import json
from pathlib import Path

from docs.biology import neuroglial_contact_probe as probe


def test_delayed_contact_credit_and_evidence_identity(tmp_path):
    probe.main(tmp_path)
    report = json.loads((tmp_path/'contact-probe.json').read_text())
    settings = json.loads((tmp_path/'settings.json').read_text())
    topology = json.loads((tmp_path/'topology.json').read_text())
    probe.validate(report)
    assert report['probe_sha256'] == hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
    assert settings['biology']['extracellular']['origin'] == 'synthetic'
    assert len(topology['active_contacts']) == report['contact_count']
    assert all(row['synthetic'] and row['delay_ms'] == .5 for row in topology['contacts'].values())
