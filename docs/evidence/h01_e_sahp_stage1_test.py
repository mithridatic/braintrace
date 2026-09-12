"""Tests for the SP13 stage-1 scorer."""

import numpy as np

import h01_e_sahp_stage1 as stage1


def _summary(peaks):
    return {"count": len(peaks), "peak_ms": list(peaks), "axon_first": True}


def test_250_bands_use_the_intervals_before_spikes_4_and_5():
    bands = stage1.bands_250(_summary([1078., 1112., 1332., 1640., 1913.]))
    assert [b["held"] for b in bands] == [True, True, True]
    assert bands[1]["observed"] == 308. and bands[2]["observed"] == 273.
    bands = stage1.bands_250(_summary([1075., 1096., 1143., 1327.]))
    assert [b["held"] for b in bands] == [False, False, False]
    assert bands[2]["observed"] is None


def test_310_bands_use_the_mean_of_the_last_three_intervals():
    peaks = [1050.+120.*k for k in range(10)]
    bands = stage1.bands_310(_summary(peaks))
    assert all(b["held"] for b in bands) and bands[1]["observed"] == 120.
    bands = stage1.bands_310(_summary(peaks[:7]))
    assert bands[0]["held"] is False and bands[1]["held"] is True
    assert stage1.bands_310(_summary(peaks[:2]))[1]["observed"] is None


def test_43_bands_compare_onset_rows_to_human_and_b3():
    time = np.arange(1000., 1200., .02)
    voltage = np.full_like(time, -84.)
    rows = [{"key": "sub_1019_mv", "kind": "subthreshold_voltage_mv", "human": -84.5, "model": -83.95},
            {"key": "sub_1520_mv", "kind": "subthreshold_voltage_mv", "human": -80., "model": -78.},
            {"key": "peak_1", "kind": "peak_sample_voltage_mv", "human": 30., "model": 31.}]
    bands = stage1.bands_43({"time_ms": time, "voltage_mv": voltage}, rows)
    assert [b["row"] for b in bands] == ["sub_1019_mv_vs_human", "sub_1019_mv_vs_b3"]
    assert bands[0]["held"] is True and bands[1]["held"] is True
    assert abs(bands[1]["observed"]-(-.05)) < 1e-9
