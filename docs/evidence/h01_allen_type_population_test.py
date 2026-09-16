"""Tests for the Allen type-population spreads."""

import hashlib
import json

import pytest

from h01_allen_type_population import (DONOR_POPULATIONS, POPULATIONS, donor_type_populations, load, matched_sweeps,
                                       per_cell_readings, population_ids, spread, type_population)

FEATURES = [dict(specimen__id=1, tag__dendrite_type="spiny", structure__layer="2"),
            dict(specimen__id=2, tag__dendrite_type="spiny", structure__layer="3"),
            dict(specimen__id=3, tag__dendrite_type="spiny", structure__layer="4"),
            dict(specimen__id=4, tag__dendrite_type="aspiny", structure__layer="5"),
            dict(specimen__id=5, tag__dendrite_type="aspiny", structure__layer="3"),
            dict(specimen__id=6, tag__dendrite_type="sparsely spiny", structure__layer="2"),
            dict(specimen__id=7, tag__dendrite_type="spiny", structure__layer=None)]


def _sweep(cell, amplitude, spikes, pre=-70., post=-70.5, duration=1., name="Long Square"):
    return dict(specimen_id=cell, stimulus_absolute_amplitude=amplitude, num_spikes=spikes, pre_vm_mv=pre,
                post_vm_mv=post, stimulus_duration=duration, stimulus_name=name)


SWEEPS = [_sweep(1, 290., 9), _sweep(1, 310., 10, pre=-69.), _sweep(1, 330., 12), _sweep(1, 350., 13),
          _sweep(2, 310.000002, None, pre=-72., post=-73.), _sweep(2, 250., 4),
          _sweep(3, 90., 12), _sweep(3, 110., 17), _sweep(3, 90., 12, duration=2.),
          _sweep(4, 190., 30, pre=-75., post=-76.), _sweep(5, 190., 60, pre=-77., post=-78.), _sweep(5, 200., 65),
          _sweep(6, 310., 8), _sweep(1, 310., 10, duration=None)]


def test_population_ids_follow_dendrite_type_and_layer():
    assert population_ids(FEATURES, "spiny_l23") == [1, 2]
    assert population_ids(FEATURES, "spiny_l4") == [3]
    assert population_ids(FEATURES, "aspiny") == [4, 5]   # all layers; sparsely spiny excluded
    assert set(DONOR_POPULATIONS.values()) <= set(POPULATIONS)


def test_matched_sweeps_take_one_step_around_the_primary_and_only_one_second_pulses():
    by = matched_sweeps(SWEEPS, [1, 2, 3], 310.)
    assert sorted(by) == [1, 2]   # cell 6 not in ids; cell 3 has no sweep near 310
    assert [r["stimulus_absolute_amplitude"] for r in by[1]] == [290., 310., 330.]   # 350 is two steps away
    assert len(by[2]) == 1 and by[2][0]["num_spikes"] is None
    assert [r["stimulus_absolute_amplitude"] for r in matched_sweeps(SWEEPS, [3], 90.)[3]] == [90., 110.]   # 2 s sweep dropped


def test_per_cell_readings_average_matched_sweeps_and_read_null_spikes_as_zero():
    readings = per_cell_readings(matched_sweeps(SWEEPS, [1, 2], 310.))
    assert readings["count"] == [pytest.approx(31/3), 0.]
    assert readings["pre_mv"] == [pytest.approx((-70.-69.-70.)/3-14.), pytest.approx(-86.)]
    assert readings["post_mv"] == [pytest.approx(-84.5), pytest.approx(-87.)]


def test_spread_is_two_sample_sd_with_percentiles_and_needs_two_cells():
    s = spread([1., 3., 5., 7.])
    assert s["n"] == 4 and s["mean"] == 4. and s["sd"] == pytest.approx(2.5819889) and s["tolerance"] == pytest.approx(5.1639778)
    assert s["p5"] == pytest.approx(1.3) and s["p95"] == pytest.approx(6.7) and (s["min"], s["max"]) == (1., 7.)
    assert spread([2.])["tolerance"] is None and spread([2.])["mean"] == 2. and spread([])["mean"] is None


def test_type_population_reports_counts_windows_and_the_missing_label():
    pop = type_population(FEATURES, SWEEPS, "aspiny", 190.)
    assert pop["cells_in_population"] == 2 and pop["cells_with_matched_sweeps"] == 2 and pop["matched_sweeps"] == 3
    assert pop["count"]["mean"] == pytest.approx((30.+62.5)/2) and pop["tolerance_rule"] == "2 sd across cells"
    assert pop["rest_mv"]["n"] == 2 and pop["after_mv"]["mean"] == pytest.approx((-90.+(-78.-70.5)/2-14.)/2)
    assert "no fast-spiking label" in pop["label"] and "last 500 ms" in pop["windows"]["after_mv"]
    assert pop["step_pa"] == 20. and pop["primary_pa"] == 190.


def test_load_pins_digests_and_donor_populations_use_each_donor_primary(tmp_path):
    features = tmp_path/"allen-human-ephys-features.json"
    sweeps = tmp_path/"allen-human-ephys-sweeps.json"
    features.write_text(json.dumps(dict(source="f", fetched_utc="t", rows=FEATURES)))
    sweeps.write_text(json.dumps(dict(source="s", fetched_utc="t", rows=SWEEPS)))
    digest = hashlib.sha256(features.read_bytes()).hexdigest()
    f, s, provenance = load(features, sweeps, expected=dict(features=digest, sweeps=None))
    assert provenance["features"]["sha256"] == digest and provenance["sweeps"]["rows"] == len(SWEEPS)
    with pytest.raises(ValueError, match="digest mismatch"):
        load(features, sweeps, expected=dict(features="0"*64))
    pops = donor_type_populations(f, s, {"l2-pyramidal-allen-541563728": 310., "l5-pv-basket-hl5bn1": 190.})
    assert pops["l2-pyramidal-allen-541563728"]["population"] == "spiny_l23" and pops["l5-pv-basket-hl5bn1"]["primary_pa"] == 190.
    assert pops["l2-pyramidal-allen-541563728"]["count"]["n"] == 2
