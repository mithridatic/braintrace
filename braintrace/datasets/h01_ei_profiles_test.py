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


from dataclasses import asdict
from .h01_ei_profiles import get_donor_profile
from . import _h01_ei_parameters as parameters

# Field values of get_ei_profile before the donor re-key (SP6a), captured from the polarity-keyed code.
_BEFORE = {
    "E": dict(name="h01-l2-kv3-ninety-ca133", digest="926efb3d5ae93b324bd653698d1c7d3fd478d9b94db1143b4eb933c5fcfa0f3a",
              source="Allen specimen 541563728; model 626170538; fit SHA256 2ceca2317ccbd586adde4b1e72507ad4bdf2fc10fc26ad4b281484324dd5f0c3",
              initial_mv=-83.97993469238281, reversal_mv={"candidate": -87.97993469238281, "source": -83.97993469238281},
              axial_ohm_cm=94.62299222737664, regions={"candidate": parameters.E_CANDIDATE, "source": parameters.E_SOURCE},
              channel_prefix="H01L2", sodium_reversal_mv=53., potassium_reversal_mv=-107.),
    "I": dict(name="h01-pv-regional-mesh-axon2187", digest="d8d4d022457eb139a033ead73780f2eb12a0d58852503267912d961b1eb9d546",
              source="ModelDB 267587 HL5BN1; commit 82cdd91bc93942ba19315371330a2412e064baf5",
              initial_mv=-80., reversal_mv={"candidate": -96.97510324827309, "source": -96.97510324827309},
              axial_ohm_cm=100., regions={"candidate": parameters.I_CANDIDATE, "source": parameters.I_SOURCE},
              channel_prefix="H01PV", sodium_reversal_mv=50., potassium_reversal_mv=-85.),
}
_LIMITATIONS = ("Borrowed donor and mechanisms; not measured H01 physiology.",
                "Candidate selection is not physiological qualification.",
                "E onset and rising-phase errors remain; I early intervals remain too short.")


@pytest.mark.parametrize("role", ["E", "I"])
@pytest.mark.parametrize("mode", ["candidate", "source"])
def test_polarity_alias_is_identical_to_pre_rekey_profile(role, mode):
    before = _BEFORE[role]
    expected = dict(name=before["name"]+":"+mode+":v1", polarity=role, mode=mode, source=before["source"],
                    metadata_sha256=before["digest"], initial_mv=before["initial_mv"],
                    reversal_mv=before["reversal_mv"][mode], axial_ohm_cm=before["axial_ohm_cm"],
                    regions=before["regions"][mode], limitations=_LIMITATIONS,
                    channel_prefix=before["channel_prefix"], sodium_reversal_mv=before["sodium_reversal_mv"],
                    potassium_reversal_mv=before["potassium_reversal_mv"])
    assert asdict(get_ei_profile(role, mode=mode)) == expected


def test_donor_profile_by_key_matches_polarity_alias():
    assert get_donor_profile("l2-pyramidal-allen-541563728") == get_ei_profile("E")
    assert get_donor_profile("l5-pv-basket-hl5bn1", mode="source") == get_ei_profile("I", mode="source")
    assert parameters.DONOR_REGIONS["l5-pv-basket-hl5bn1"]["candidate"] is parameters.I_CANDIDATE


@pytest.mark.parametrize("key,mode", [("unknown-donor", "candidate"), ("l2-pyramidal-allen-541563728", "bad")])
def test_unknown_donor_key_or_mode_fails(key, mode):
    with pytest.raises(ValueError):
        get_donor_profile(key, mode=mode)
