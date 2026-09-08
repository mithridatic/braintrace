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


SST_KEY = "l3-sst-interneuron-hl5mn1"


@pytest.mark.parametrize("mode", ["candidate", "source"])
def test_sst_donor_profile_is_the_published_fit_in_both_modes(mode):
    p = get_donor_profile(SST_KEY, mode=mode)
    assert p.name == "h01-sst-l3-hl5mn1:"+mode+":v1" and p.polarity == "I"
    assert (p.initial_mv, p.reversal_mv, p.axial_ohm_cm) == (-81.5, -81.5, 100.)
    assert p.regions is parameters.SST_L3_HL5MN1_SOURCE
    assert (p.channel_prefix, p.sodium_reversal_mv, p.potassium_reversal_mv) == ("H01PV", 50., -85.)
    assert p.metadata_sha256 == "a9ce264f2733104ceb457d519678f447cb277db2dda7cedf0a0cc38fb6ffb0f4"
    acquisition = Path(__file__).resolve().parents[2]/"docs/evidence/h01-sst-acquisition.json"
    assert json.loads(acquisition.read_text())["files"]["biophys_HL5MN1.hoc"]["sha256"] == p.metadata_sha256
    for family in ("soma", "axon", "dend", "apic"):
        for mechanism in ("NaTg", "Kv3_1", "Ih"):
            assert channel_controls(p, family, mechanism) == {}


L4_KEY = "l4-pyramidal-allen-527952884"


@pytest.mark.parametrize("mode", ["candidate", "source"])
def test_l4_donor_profile_is_the_published_allen_fit_in_both_modes(mode):
    p = get_donor_profile(L4_KEY, mode=mode)
    assert p.name == "h01-l4-allen-527952884:"+mode+":v1" and p.polarity == "E"
    assert (p.initial_mv, p.reversal_mv, p.axial_ohm_cm) == (-80.81838607788086, -80.81838607788086, 14.9970627156)
    assert p.regions is parameters.L4_ALLEN_527952884_SOURCE
    assert (p.channel_prefix, p.sodium_reversal_mv, p.potassium_reversal_mv) == ("H01L2", 53., -107.)
    assert p.metadata_sha256 == "1aa0e2c59174c726f868422c7b152d95c727d0d962e92a6eb0356f7dfdb80c5a"
    acquisition = Path(__file__).resolve().parents[2]/"docs/evidence/h01-l4-acquisition.json"
    report = json.loads(acquisition.read_text())
    assert report["files"]["527952884_fit.json"]["sha256"] == p.metadata_sha256
    assert report["bundle_files"]["fit_parameters.json"]["sha256"] == p.metadata_sha256
    assert report["fit"]["conditions"]["v_init"] == p.initial_mv and report["fit"]["passive"]["ra"] == p.axial_ohm_cm
    for family in ("soma", "axon", "dend", "apic"):
        for mechanism in ("NaTs", "Kv3_1", "Ih"):
            assert channel_controls(p, family, mechanism) == {}
    with pytest.raises(ValueError):
        get_donor_profile(L4_KEY, mode="b3")


def test_pv_candidate_controls_are_unchanged_by_the_third_donor():
    candidate = get_ei_profile("I")
    assert channel_controls(candidate, "soma", "NaTg") == {"h_close": .15, "h_open": 1., "h_slope": 5.}
    assert channel_controls(candidate, "soma", "Kv3_1") == {"m_open": .5, "m_close": .5}
    assert channel_controls(get_ei_profile("E"), "soma", "NaTs") == {"m_open": 2.}
    assert channel_controls(get_ei_profile("E"), "soma", "Kv3_1") == {"m_close": .9}


import importlib.util
from .h01_cell_types import DEFAULT_DONOR_KEYS
from .h01_ei_profiles import mode_flags

_EVIDENCE = Path(__file__).resolve().parents[2]/"docs/evidence"


def _b3_export():
    spec = importlib.util.spec_from_file_location("h01_e_b3_export", _EVIDENCE/"h01-e-b3-experimental-profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finalist_mode_equals_candidate_except_the_somatic_kv3_closing_factor():
    finalist, candidate = get_ei_profile("I", mode="finalist"), get_ei_profile("I")
    f, c = asdict(finalist), asdict(candidate)
    assert f["name"] == "h01-pv-regional-mesh-axon2187:finalist:v1" and f["mode"] == "finalist"
    assert f["limitations"][:3] == c["limitations"] and "Experimental mode 'finalist'" in f["limitations"][3]
    for key in set(c)-{"name", "mode", "limitations"}:
        assert f[key] == c[key]
    assert parameters.I_FINALIST is parameters.I_CANDIDATE
    for family in ("soma", "axon", "dend", "apic"):
        for mechanism in ("NaTg", "Nap", "K_P", "Kv3_1", "SK", "Ih"):
            expected = channel_controls(candidate, family, mechanism)
            if (family, mechanism) == ("soma", "Kv3_1"):
                expected = {"m_open": .5, "m_close": 2.}
            assert channel_controls(finalist, family, mechanism) == expected


def test_finalist_controls_and_densities_match_the_finalist_run_metadata():
    meta = json.loads((_EVIDENCE/"h01-i-energetic/e-kv3-close2-027.json").read_text())
    finalist, source = get_ei_profile("I", mode="finalist"), get_ei_profile("I", mode="source")
    assert channel_controls(finalist, "soma", "Kv3_1") == {"m_open": meta["somatic_kv3_tau_factor"],
                                                           "m_close": meta["somatic_kv3_close_factor"]}
    for family in ("soma", "axon"):
        assert channel_controls(finalist, family, "NaTg") == {"h_close": meta["sodium_h_tau_factor"],
            "h_open": meta["sodium_h_recovery_factor"], "h_slope": meta["sodium_h_slope_mv"]}
    rows, base = {r[0]: r for r in finalist.regions}, {r[0]: r for r in source.regions}
    soma, soma_source = dict(rows["soma"][3]), dict(base["soma"][3])
    assert meta["conductance_intervention"] == {"mechanism": "NaTg", "factor": 1.1, "region": "soma"}
    assert soma["NaTg"] == soma_source["NaTg"]*meta["conductance_intervention"]["factor"]
    assert soma["Ca_LVA"] == soma_source["Ca_LVA"]*meta["somatic_calva_factor"]
    assert soma["Kv3_1"] == soma_source["Kv3_1"]*meta["somatic_kv3_factor"]
    assert rows["axon"][4] == (meta["axon_calcium_decay_ms"], meta["axon_calcium_gamma"])
    assert finalist.initial_mv == meta["initial_voltage_mv"]
    assert finalist.axial_ohm_cm == meta["geometry"][0]["ra_ohm_cm"]


def test_b3_mode_is_the_export_verbatim():
    export, b3, candidate = _b3_export(), get_ei_profile("E", mode="b3"), get_ei_profile("E")
    assert b3.regions == export.E_B3_EXPERIMENTAL and parameters.E_B3_EXPERIMENTAL == export.E_B3_EXPERIMENTAL
    assert (b3.reversal_mv, b3.initial_mv, b3.axial_ohm_cm) == (export.E_B3_EXPERIMENTAL_REVERSAL_MV,
        export.E_B3_EXPERIMENTAL_INITIAL_MV, export.E_B3_EXPERIMENTAL_AXIAL_OHM_CM)
    assert b3.name == "h01-l2-kv3-ninety-ca133:b3:v1" and b3.polarity == "E" and b3.mode == "b3"
    assert b3.limitations[:3] == _LIMITATIONS and "Experimental mode 'b3'" in b3.limitations[3]
    for key in ("metadata_sha256", "source", "channel_prefix", "sodium_reversal_mv", "potassium_reversal_mv"):
        assert getattr(b3, key) == getattr(candidate, key)
    assert get_donor_profile("l2-pyramidal-allen-541563728", mode="b3") == b3


def test_b3_controls_and_flags_match_the_export_under_the_channel_vocabulary():
    export, b3 = _b3_export(), get_ei_profile("E", mode="b3")
    nats = export.E_B3_EXPERIMENTAL_CONTROLS["NaTs"]
    values = export.E_B3_EXPERIMENTAL_SOURCE["candidate_json"]["values"]
    assert channel_controls(b3, "soma", "NaTs") == {"m_open": nats["m_open"], "h_open": nats["h_recovery"]}
    assert channel_controls(b3, "soma", "NaTs") == {"m_open": values["sodium_opening_factor"],
                                                    "h_open": values["sodium_recovery_factor"]}
    assert channel_controls(b3, "soma", "Kv3_1") == export.E_B3_EXPERIMENTAL_CONTROLS["Kv3_1"]
    assert channel_controls(b3, "soma", "Kv3_1") == {"m_close": values["kv3_closing_factor"]}
    assert channel_controls(b3, "axon", "NaTs") == {} and channel_controls(b3, "soma", "Ih") == {}
    assert mode_flags(b3) == {"calcium_decay_factor": values["calcium_decay_factor"],
                              "distribute_ih": values["distribute_ih"]}
    assert mode_flags(get_ei_profile("E")) == {} and mode_flags(get_ei_profile("I", mode="finalist")) == {}
    rows = {r[0]: r for r in b3.regions}
    source = {r[0]: r for r in get_ei_profile("E", mode="source").regions}
    assert rows["soma"][4][0] == source["soma"][4][0]*values["calcium_decay_factor"]
    assert {dict(rows[f][3])["Ih"] for f in ("soma", "dend", "apic")} == {9.994138594205759e-05}
    assert rows["axon"][3] == (("NaTs", 3.814),) and rows["axon"][4] is None


def test_candidate_controls_keep_their_family_scope_and_limitations():
    e = get_ei_profile("E")
    assert channel_controls(e, "axon", "NaTs") == {"m_open": 2.}
    assert channel_controls(e, "dend", "Kv3_1") == {"m_close": .9}
    for role in ("E", "I"):
        for mode in ("candidate", "source"):
            assert get_ei_profile(role, mode=mode).limitations == _LIMITATIONS


@pytest.mark.parametrize("role,mode", [("E", "finalist"), ("I", "b3"), ("E", "B3"), ("I", "")])
def test_experimental_modes_are_per_donor(role, mode):
    with pytest.raises(ValueError):
        get_ei_profile(role, mode=mode)
    with pytest.raises(ValueError):
        get_donor_profile(DEFAULT_DONOR_KEYS[role], mode=mode)
