"""Tests for the SP16 slow sodium inactivation patch."""

import hashlib
from pathlib import Path

import pytest

import h01_sodium_slow_inactivation as slow

CACHE = Path(__file__).resolve().parents[2]/".cache/worktree-recovery-2026-09-09/h01/.cache"
PINNED = {"NaTs": CACHE/"human-pyramidal-l2/kv3-closing-source/NaTs.mod",
          "NaTg": CACHE/"human-pv/kv3-phase-reference/mod/NaTg.mod"}


def _source(mechanism):
    path = PINNED[mechanism]
    if not path.is_file():
        pytest.skip(f"pinned {mechanism} source not in this checkout")
    return path.read_bytes()


@pytest.mark.parametrize("mechanism", sorted(slow.SOURCES))
def test_patch_adds_the_gate_to_every_block_and_keeps_the_newline_style(mechanism):
    source = _source(mechanism)
    patched = slow.patch_sodium(source, mechanism).decode()
    assert ("\r\n" in patched) == (b"\r\n" in source)
    assert "g = gbar*m*m*m*h*s" in patched
    assert "s' = (sInf-s)/sTau" in patched
    assert "\ts = sInf\n" in patched.replace("\r\n", "\n")
    assert "slow_inactivation = 0" in patched
    assert patched.count("sInf = 1 - slow_inactivation") == 1
    state = patched.replace("\r\n", "\n").split("STATE\t{")[1].split("}")[0]
    assert state.split() == ["m", "h", "s"]


@pytest.mark.parametrize("mechanism", sorted(slow.SOURCES))
def test_patch_only_rewrites_the_range_and_conductance_lines_and_otherwise_inserts(mechanism):
    import difflib
    before = _source(mechanism).decode().replace("\r\n", "\n").splitlines()
    after = slow.patch_sodium(_source(mechanism), mechanism).decode().replace("\r\n", "\n").splitlines()
    touched = [line for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, before, after).get_opcodes()
               if tag in ("replace", "delete") for line in before[i1:i2]]
    assert touched == [line for line in before
                       if "RANGE gbar" in line or line.strip() == "g = gbar*m*m*m*h"]
    assert len(after)-len(before) == 12  # 5 parameters, 2 assigned, 1 state, 1 derivative, 1 initial, 2 rates


def test_patch_refuses_an_unknown_mechanism_or_a_changed_source():
    with pytest.raises(ValueError, match="Unknown sodium mechanism"):
        slow.patch_sodium(b"anything", "NaV11")
    with pytest.raises(ValueError, match="digest"):
        slow.patch_sodium(b"not the pinned file", "NaTs")


def test_patch_refuses_a_source_whose_anchor_is_missing(monkeypatch):
    broken = _source("NaTs").replace(b"g = gbar*m*m*m*h", b"g = gbar*m*m*h")
    monkeypatch.setitem(slow.SOURCES, "NaTs", hashlib.sha256(broken).hexdigest())
    with pytest.raises(ValueError, match="Anchor"):
        slow.patch_sodium(broken, "NaTs")


def test_steady_state_is_one_when_the_gate_is_off_and_falls_to_one_minus_depth():
    assert all(slow.steady_state(v, 0.) == 1. for v in (-90., -60., -30., 0., 40.))
    assert slow.steady_state(-50., .4) == pytest.approx(.8)          # half-point: half the depth
    assert slow.steady_state(40., .4) == pytest.approx(.6, abs=1e-6)  # fully engaged
    assert slow.steady_state(-120., .4) == pytest.approx(1., abs=1e-4)  # released at rest
    assert slow.steady_state(-83., .4) > slow.steady_state(-65., .4)


def test_the_gate_is_shut_at_the_200_pa_plateau_and_engages_during_a_spike():
    # The E model sits at -65 mV at 200 pA before its first spike; the gate must not act there.
    assert 1.-slow.steady_state(-65., .6) < .05
    assert 1.-slow.steady_state(20., .6) == pytest.approx(.6, abs=1e-4)


def test_time_constant_is_slow_at_rest_and_fast_during_a_spike():
    assert slow.time_constant(-75.) == pytest.approx(1000., rel=.02)
    assert slow.time_constant(20.) == pytest.approx(10., rel=.02)
    assert slow.time_constant(-50.) == pytest.approx(505.)
    assert slow.time_constant(-50., entry_ms=30., recovery_ms=30.) == 30.


def test_the_pinned_digests_match_the_vendored_sources():
    for mechanism, digest in slow.SOURCES.items():
        assert hashlib.sha256(_source(mechanism)).hexdigest() == digest
