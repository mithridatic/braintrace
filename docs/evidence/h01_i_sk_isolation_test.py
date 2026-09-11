"""Band logic of the SP10 axonal-SK isolation scorer (fabricated rows, not real traces)."""

import numpy as np

import h01_i_sk_isolation as sk


def _rows(count, cycle2, troughs, delays, width=.221, peak=17., sk_current=0.01):
    rows = []
    for cycle in range(1, count+1):
        rows.append({"cycle": cycle, "cycle_ms": cycle2 if cycle == 2 else 30., "threshold_ms": 300.+30.*cycle,
                     "peak_mv": peak, "trough_ms": 305.+30.*cycle, "trough_mv": troughs[min(cycle, 3)-1],
                     "width_ms": width, "delay_to_next_threshold_ms": delays[min(cycle, len(delays))-1] if cycle < count else None,
                     "axon_sk_current_ma_cm2_at_sample": sk_current})
    return rows


def test_stage0_bands_hold_on_the_container_values():
    rows = _rows(14, 34.76, (-80.13, -80.54, -80.65), (30., 80.))
    bands = sk.stage0_bands("019", rows, {"axon_first": True})
    assert all(b["held"] for b in bands) and len(bands) == 6
    bands = sk.stage0_bands("019", _rows(15, 34.76, (-80.13, -80.54, -80.65), (30., 80.)), {"axon_first": True})
    assert [b["row"] for b in bands if not b["held"]] == ["count"]
    bands = sk.stage0_bands("027", _rows(37, 16.5, (-78.99, -79.30, -79.61), (13., 27.)), {"axon_first": False})
    assert sorted(b["row"] for b in bands if not b["held"]) == ["axon_first", "cycle2_ms"]


def test_stage1_bands_and_rejection():
    base = _rows(14, 34.76, (-80.13, -80.54, -80.65), (30., 80.))
    arm = _rows(20, 36., (-80.2, -80.6, -80.7), (28., 50.), sk_current=0.)
    bands = sk.stage1_bands("019", base, arm, {"axon_first": True}, {"longest_above_ms": .3, "blocked": False})
    assert all(b["held"] for b in bands), [b for b in bands if not b["held"]]
    unchanged = _rows(14, 34.76, (-80.13, -80.54, -80.65), (30., 78.), sk_current=0.)
    bands = sk.stage1_bands("019", base, unchanged, {"axon_first": True}, {"longest_above_ms": .3, "blocked": False})
    missed = {b["row"] for b in bands if not b["held"]}
    assert missed == {"late_delay_shortens_ms", "count"}
    bands_027 = sk.stage1_bands("027", base, unchanged, {"axon_first": True}, {"longest_above_ms": 9., "blocked": True})
    reject = sk.stage1_rejection(bands+bands_027)
    assert reject["met"] and reject["delay_not_shortened"] == ["0.19 nA", "0.27 nA"]
    assert reject["depolarisation_block"] == ["0.27 nA"] and reject["spikes_lost"] == []


def test_depolarisation_block_measures_the_longest_excursion():
    time = np.arange(0., 1500., .1)
    voltage = np.full_like(time, -80.)
    voltage[(time >= 500.) & (time < 507.)] = 0.
    voltage[(time >= 600.) & (time < 601.)] = 0.
    block = sk.depolarisation_block({"time_ms": time, "voltage_mv": voltage})
    assert block["blocked"] and abs(block["longest_above_ms"]-6.9) < .2
    voltage[(time >= 500.) & (time < 507.)] = -80.
    assert not sk.depolarisation_block({"time_ms": time, "voltage_mv": voltage})["blocked"]


def test_late_delay_is_the_last_measured_one():
    assert sk.late_delay(_rows(5, 30., (-80.,)*3, (30., 40., 50.))) == 50.
    assert sk.late_delay(_rows(1, 30., (-80.,)*3, (30.,))) is None
