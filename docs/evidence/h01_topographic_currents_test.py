"""Tests for the stage-14 scorer (the rise decomposed by observation point)."""

import numpy as np
import pytest

import h01_topographic_currents as cur


def test_to_na_scales_density_by_area():
    assert cur._to_na(1., 100.) == pytest.approx(1.)          # 1 mA/cm2 x 100 um2 = 1 nA
    assert cur._to_na(np.array([0., 2.]), 65.8)[1] == pytest.approx(1.316)


def test_monotone_down():
    assert cur._monotone_down([.9, .5, .3])
    assert not cur._monotone_down([.5, .6, .4])
    assert not cur._monotone_down([.5])


def _state(na_above=6., site_above=1.):
    """A synthetic upstroke: V ramps -57 -> +20 over [1100, 1104] ms, contributions chosen for the test.

    i_cap is set to the exact sum of the contributions so the native KCL closes; the window is the
    whole span in milliseconds (closure/shares select by time, not index).
    """
    t = np.linspace(1100., 1104., 200)
    v = np.linspace(-57., 20., 200)
    above = v >= cur.SPLIT_MV
    contrib = {"na_drive": np.where(above, na_above, .2), "k_brake": np.full_like(v, -.5),
               "other_ionic": np.zeros_like(v), "intrasoma": np.where(above, na_above*.5, .1),
               "axon0": np.where(above, site_above, .8), "dend": np.where(above, -3., -.2),
               "clamp": np.full_like(v, .31)}
    icap = sum(contrib.values())                              # i_cap = axial_in + i_inj - ionic_out, so KCL closes
    return {"t": t, "v": v, "icap": icap, "contrib": contrib,
            "c_seg_pf": .66, "window": (1100., 1104.), "area_um2": 65.8, "cm": 1.}


def test_closure_passes_when_icap_is_the_sum_and_flags_a_broken_balance():
    st = _state()
    cl = cur.closure(st, st["window"])
    assert cl["kcl_residual_frac"] < 1e-9
    broken = _state()
    broken["icap"] = broken["icap"]+5.                        # inject a balance error
    assert cur.closure(broken, broken["window"])["kcl_residual_frac"] > cur.CLOSURE_TOL


def test_shares_split_above_and_below_forty():
    st = _state(na_above=6., site_above=1.)                   # somatic-total >> site above -40
    sh = cur.shares(st, st["window"])
    assert sh["above_40"]["somatic_total"] > sh["above_40"]["site_delivered"]
    assert set(sh) == {"above_40", "below_40"}


def test_decide_gates_on_closure_then_reads_a():
    good = {"closure": {"kcl_residual_frac": .01, "cap_residual_frac": .02},
            "shares": {"above_40": {"somatic_total": .8, "site_delivered": .3, "na_drive": .5, "intrasoma": .3, "dend": -.1, "k_brake": -.2, "other_ionic": .1},
                       "below_40": {"somatic_total": .2, "site_delivered": .7, "na_drive": .1, "intrasoma": .1, "dend": 0., "k_brake": 0., "other_ionic": .1}},
            "contrast": {"recorded_over_fit_at_bins": {"-40": .55, "-20": .48, "0": .4, "20": .38}}}
    d = cur.decide(good)
    assert d["verdict"] == "PASS" and d["dose_justified"]
    assert next(c for c in d["checks"] if c["reading"] == "a")["pass"] is True
    assert next(c for c in d["checks"] if c["reading"] == "c")["pass"] is True


def test_decide_refuses_dose_when_site_dominates_above_forty():
    site = {"closure": {"kcl_residual_frac": .01, "cap_residual_frac": .02},
            "shares": {"above_40": {"somatic_total": .3, "site_delivered": .8, "na_drive": .2, "intrasoma": .1, "dend": -.1, "k_brake": -.1, "other_ionic": .1},
                       "below_40": {"somatic_total": .2, "site_delivered": .7, "na_drive": .1, "intrasoma": .1, "dend": 0., "k_brake": 0., "other_ionic": .1}},
            "contrast": {"recorded_over_fit_at_bins": {"-40": .55, "-20": .48, "0": .4, "20": .38}}}
    d = cur.decide(site)
    assert not d["dose_justified"] and d["verdict"] == "MIXED"
    assert next(c for c in d["checks"] if c["reading"] == "a")["pass"] is False


def test_decide_voids_when_closure_fails():
    bad = {"closure": {"kcl_residual_frac": .2, "cap_residual_frac": .02},
           "shares": {"above_40": {}, "below_40": {}}, "contrast": {"recorded_over_fit_at_bins": {}}}
    assert cur.decide(bad)["verdict"] == "void"


def test_geometry_reads_the_run_and_classifies_neighbours():
    area, cm, groups = cur.geometry()
    assert 60. < area < 70. and cm == pytest.approx(1.)
    assert len(groups["axon0"]) == 1 and len(groups["intrasoma"]) == 2 and len(groups["dend"]) >= 7


def test_contrast_reproduces_the_stage13_peaks_and_widens(tmp_path):
    """Integration: the fit loop through the chain peaks at the stage-13 570 V/s, the recording at
    348, and the recorded/fit ratio widens above the -40 mV hand-over (exercises the loop reader)."""
    import json
    from h01_topographic_stage0 import OUT
    chain = json.loads((OUT/"stage-12.json").read_text(encoding="utf-8"))["E"]["chain"]
    con = cur.contrast(chain)
    assert con["fit_through_chain"]["peak_v_s"] == pytest.approx(570.4, abs=3.)
    assert con["recorded_raw"]["peak_v_s"] == pytest.approx(347.7, abs=3.)
    ratios = con["recorded_over_fit_at_bins"]
    assert ratios["-40"] > .85                                  # the loops still match at the hand-over
    above = [ratios[b] for b in ("-20", "0", "20")]
    assert all(np.isfinite(above)) and np.median(above) < .75 and cur._monotone_down([ratios["-40"]] + above)
