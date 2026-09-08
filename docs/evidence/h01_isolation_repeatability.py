"""Isolation split with decision limits from repeats (Hartshorne 2020, Ch. 4-5).

The measurement branch's own spread comes from repeated human sweeps; the model's
from numerical repeats of one physical model. A human-to-model contrast is a search
row only when it exceeds five times the combined limit; smaller contrasts are parked
as "inside repeatability" rather than searched.
"""

import numpy as np

LANDMARKS = ("threshold_mv", "peak_mv", "max_rise_v_s", "max_fall_v_s", "minimum_mv", "cycle_ms")
DLF = {2: 2.95, 3: 1.81, 4: 1.47}
SEARCH_RATIO = 5.
REAL_RATIO = 1.


def first_spike_vector(rows, pulse_start_ms):
    """Landmarks of the first spike plus its latency, or ``None`` without a spike."""
    if not rows:
        return None
    vector = {key: rows[0][key] for key in LANDMARKS if key != "cycle_ms"}
    vector["latency_ms"] = None if rows[0]["threshold_ms"] is None else rows[0]["threshold_ms"]-pulse_start_ms
    return vector


def repeat_limits(vectors):
    """Range across repeats times the Dixon factor, per landmark key.

    Five or more repeats use the four-repeat factor as a conservative floor.
    """
    vectors = [v for v in vectors if v is not None]
    if len(vectors) < 2:
        raise ValueError("Decision limits need at least two repeats.")
    factor = DLF[min(len(vectors), 4)]
    limits = {}
    for key in set.intersection(*(set(v) for v in vectors)):
        values = [v[key] for v in vectors]
        if all(x is not None for x in values):
            limits[key] = (max(values)-min(values))*factor
    return {"repeats": len(vectors), "factor": factor, "limits": limits}


def positions(count, edge=3):
    """Spike positions compared: the first and last ``edge`` of a train."""
    early = list(range(min(edge, count)))
    late = [i for i in range(max(count-edge, edge), count)]
    return [("early", i) for i in early]+[("late", i) for i in late]


def combined_limit(key, human_limits, model_limits):
    """Root-sum-square of the two branch limits and which branches supplied it."""
    parts = {name: limits.get(key) for name, limits in (("human", human_limits), ("model", model_limits))}
    known = {name: value for name, value in parts.items() if value is not None}
    if not known:
        return None, "none"
    return float(np.sqrt(sum(v*v for v in known.values()))), " and ".join(sorted(known))


def tier(ratio):
    """Search, real, or parked, from the contrast-to-limit ratio."""
    if ratio is None:
        return "unlimited"
    if ratio >= SEARCH_RATIO:
        return "search"
    return "real" if ratio >= REAL_RATIO else "parked"


def contrast_rows(human_rows, model_rows, human_limits, model_limits):
    """Model-minus-human contrast per landmark at the early and late positions."""
    rows = []
    count = min(len(human_rows), len(model_rows))
    for phase, index in positions(count):
        human_index = index if phase == "early" else len(human_rows)-count+index
        model_index = index if phase == "early" else len(model_rows)-count+index
        for key in LANDMARKS:
            human, model = human_rows[human_index][key], model_rows[model_index][key]
            if human is None or model is None:
                continue
            limit, source = combined_limit(key, human_limits, model_limits)
            ratio = None if not limit else abs(model-human)/limit
            rows.append({"phase": phase, "spike": human_index+1, "key": key, "human": human,
                         "model": model, "contrast": model-human, "limit": limit, "limit_source": source,
                         "ratio": ratio, "tier": tier(ratio)})
    return rows


def branch_summary(rows):
    """Count search rows per landmark and name the parked landmarks."""
    keys = {}
    for row in rows:
        entry = keys.setdefault(row["key"], {"search": 0, "real": 0, "parked": 0, "unlimited": 0})
        entry[row["tier"]] += 1
    parked = sorted(k for k, v in keys.items() if v["search"] == 0 and v["unlimited"] == 0)
    searched = sorted(k for k, v in keys.items() if v["search"] > 0)
    return {"per_landmark": keys, "search_landmarks": searched, "parked_landmarks": parked}


def tree(cell, summary):
    """Mermaid isolation tree with parked landmarks struck out."""
    lines = ["flowchart TD", f"    R[{cell}: human-model contrast] --> IN[Inputs: recording and datum]",
             "    R --> FN[Function: model]", "    R --> ID[Interdependency]"]
    for key in summary["search_landmarks"]:
        lines.append(f"    FN --> {key}[{key}: search]")
    for key in summary["parked_landmarks"]:
        lines.append(f"    FN -.-> {key}[{key}: parked, inside repeatability]")
    return "\n".join(lines)


def subthreshold_contrast(human, model, human_limits, model_limits):
    """Contrast rows for the plateau and return voltages of a subthreshold input."""
    rows = []
    for key in ("plateau_mv", "return_mv"):
        limit, source = combined_limit(key, human_limits, model_limits)
        ratio = None if not limit else abs(model[key]-human[key])/limit
        rows.append({"phase": "subthreshold", "spike": 0, "key": key, "human": human[key],
                     "model": model[key], "contrast": model[key]-human[key], "limit": limit,
                     "limit_source": source, "ratio": ratio, "tier": tier(ratio)})
    return rows
