"""Frozen selection, exact metadata values, and regional candidate scope."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import pytest
from .h01_ei_profiles import get_ei_profile, channel_controls


@pytest.mark.parametrize("role,file", [("E", "h01-l2-kv3-ninety-ca133"), ("I", "h01-pv-regional-mesh-axon2187")])
def test_profile_is_frozen_and_pinned(role, file):
    p = get_ei_profile(role)
    reference = Path(__file__).resolve().parents[2]/"docs/evidence"/(file+".json")
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == p.metadata_sha256
    with pytest.raises(FrozenInstanceError):
        p.polarity = "I"
    with pytest.raises(TypeError):
        p.regions[0][3][0][1] = 0.
    assert p.mode == "candidate"


def test_e_densities_and_calcium_match_applied_genome():
    p = get_ei_profile("E")
    data = json.loads((Path(__file__).resolve().parents[2]/"docs/evidence/h01-l2-kv3-ninety-ca133.json").read_text())
    for family, cm, leak, channels, calcium in p.regions:
        rows = {r["name"]: r["value"] for r in data["applied_genome"] if r["section"] == family}
        assert leak == rows["g_pas"]
        for mechanism, density in channels:
            assert density == rows["gbar_"+mechanism]
        if calcium:
            assert calcium == (rows["decay_CaDynamics"], rows["gamma_CaDynamics"])


def test_i_region_scope_and_source_preservation():
    candidate, source = get_ei_profile("I"), get_ei_profile("I", mode="source")
    for c, s in zip(candidate.regions, source.regions):
        family = c[0]
        expected = dict(s[3])
        if family == "soma":
            expected["NaTg"] *= 1.1
            expected["Ca_LVA"] *= .5
        assert dict(c[3]) == expected
        assert c[4] == ((300., .004) if family == "axon" else s[4])
        assert channel_controls(source, family, "NaTg") == {}
        assert channel_controls(candidate, family, "NaTg")["h_close"] == .15
    assert channel_controls(candidate, "axon", "Kv3_1") == {}
    assert channel_controls(candidate, "soma", "Kv3_1") == {"m_open": .5, "m_close": .5}
    assert get_ei_profile("E", mode="source").reversal_mv == -83.97993469238281


@pytest.mark.parametrize("role,mode", [("bad", "candidate"), ("E", "bad"), ("e", "source")])
def test_invalid_profile_fails(role, mode):
    with pytest.raises(ValueError, match="Choose polarity"):
        get_ei_profile(role, mode=mode)
