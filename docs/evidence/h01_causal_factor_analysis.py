"""Calculate paired H01 trace effects; no biological uncertainty is inferred."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
GRID_MS = .005
PAIRS = [
    ("physical", "sodium removal", "h01-active-05na", "h01-active-no-sodium", "sodium_ms_cm2"),
    ("physical", "potassium increase", "h01-active-aligned-2p5", "h01-active-aligned-k20", "potassium_ms_cm2"),
    ("physical", "sodium increase", "h01-active-aligned-k20", "h01-active-aligned-na27-k20", "sodium_ms_cm2"),
    ("numerical", "aligned time refinement", "h01-active-aligned-2p5", "h01-active-aligned-time-fine", "dt_ms"),
    ("numerical", "aligned space refinement", "h01-active-aligned-2p5", "h01-active-aligned-1p25", "max_cv_length_um"),
]


def _read(name):
    path = ROOT / (name + ".json")
    data = json.loads(path.read_text())
    config = {"pulse_count": 1, "period_ms": 25., "align_active_boundaries": False,
              **data["parameters"], **data["numerics"], "dt_ms": data["result"]["dt_ms"]}
    voltage = np.asarray(data["result"]["voltage_mv"]).ravel()
    stride = round(GRID_MS / config["dt_ms"])
    assert stride >= 1 and np.isclose(stride*config["dt_ms"], GRID_MS)
    assert np.isfinite(voltage).all()
    return data, config, voltage[stride-1::stride], hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(pair):
    group, name, left, right, factor = pair
    a, ac, av, ah = _read(left)
    b, bc, bv, bh = _read(right)
    assert ac.keys() == bc.keys()
    assert {k for k in ac if ac[k] != bc[k]} == {factor}
    assert a["measured_anatomy"] == b["measured_anatomy"]
    assert a["inferred_active_region"] == b["inferred_active_region"]
    assert a["borrowed_dynamics"] == b["borrowed_dynamics"]
    assert av.shape == bv.shape == (4000,)
    delta = bv-av
    rss = float(np.linalg.norm(delta))
    return {"group": group, "name": name, "factor": factor,
            "factor_before": ac[factor], "factor_after": bc[factor],
            "before": left+".json", "after": right+".json",
            "before_sha256": ah, "after_sha256": bh,
            "controlled_configuration_before": ac,
            "response_rss_mv": rss, "response_rms_mv": rss/np.sqrt(len(delta)),
            "max_absolute_change_mv": float(np.max(np.abs(delta))),
            "time_of_max_change_ms": float((np.argmax(np.abs(delta))+1)*GRID_MS),
            "signed_voltage_change_mv": delta.tolist()}


if __name__ == "__main__":
    effects = [_compare(pair) for pair in PAIRS]
    report = {"meaning": "root sum square of a controlled simulated voltage response change",
              "is_target_residual": False, "is_uncertainty_budget": False,
              "uncertainty_contributions": None,
              "datum": {"voltage": "inside minus outside, mV", "current": "positive inward",
                        "first_sample_ms": GRID_MS, "last_sample_ms": 20., "sample_count": 4000,
                        "sample_interval_ms": GRID_MS, "peak_alignment_applied": False},
              "limitations": ["Interventions have different sizes and operating points.",
                              "The sodium-removal pair uses the older unaligned mesh.",
                              "The physical ranking is only for these interventions, not universal sensitivity.",
                              "Time and space changes are numerical effects, not biological uncertainty.",
                              "No same-cell human pyramidal target waveform is available for a residual calculation.",
                              "Factor uncertainties and their covariance are not measured."],
              "effects": effects,
              "ranked_interventions_by_group": {
                  group: [{k: e[k] for k in ("name", "response_rss_mv", "response_rms_mv")}
                          for e in sorted(effects, key=lambda e: -e["response_rss_mv"]) if e["group"] == group]
                  for group in ("physical", "numerical")}}
    (ROOT / "h01-causal-factor-effects.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["ranked_interventions_by_group"], indent=2))
