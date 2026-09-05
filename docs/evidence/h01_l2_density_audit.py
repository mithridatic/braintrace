"""Assess the prospective sodium-density prediction on direct events."""

import numpy as np


def assess_events(control, candidate, human):
    """Retain direct residuals and decide the full density prediction.

    Parameters
    ----------
    control, candidate, human : list of dict
        Complete events with onset, peak, and duration measurements.

    Returns
    -------
    dict
        Prediction status, separate conditions, and all matched residuals.
        Setup isolation must be verified separately before interpretation.
    """
    keys = ("rise_crossing_ms", "peak_sample_voltage_mv", "time_above_threshold_ms")
    reasons = []
    if len(human) != 5:
        reasons.append("human reference does not contain five events")
    for label, events in (("control", control), ("candidate", candidate), ("human", human)):
        if not events or any(not np.isfinite([e.get(k, np.nan) for k in keys]).all() for e in events):
            reasons.append(label + ": missing or nonfinite event data")
        elif (events[0]["peak_sample_voltage_mv"] <= 0 or
              any(e["time_above_threshold_ms"] <= 0 for e in events) or
              any(np.diff([e["rise_crossing_ms"] for e in events]) <= 0)):
            reasons.append(label + ": invalid event order, duration, or first peak")
    report = {"decision": "invalid", "invalid_reasons": reasons,
              "event_counts": [len(control), len(candidate), len(human)],
              "events": {"control": control, "candidate": candidate, "human": human}}
    if reasons:
        return report
    records = []
    human_intervals = np.diff([e["rise_crossing_ms"] for e in human])
    for events in (control, candidate):
        intervals = np.diff([e["rise_crossing_ms"] for e in events])
        length = min(len(intervals), len(human_intervals))
        records.append({"intervals_ms": intervals.tolist(),
                        "interval_errors_ms": (intervals[:length] - human_intervals[:length]).tolist(),
                        "event_residuals": [{k: a[k] - b[k] for k in keys} for a, b in zip(events, human)],
                        "unmatched_model_events": events[len(human):],
                        "unmatched_human_events": human[len(events):]})
    first = [r["event_residuals"][0] for r in records]
    complete = len(control) == len(candidate) == 5 and all(e["peak_sample_voltage_mv"] > 0 for e in candidate)
    interval_checks = [abs(b) <= abs(a) for a, b in zip(records[0]["interval_errors_ms"],
                                                       records[1]["interval_errors_ms"])]
    checks = {"first_onset_error_smaller": abs(first[1][keys[0]]) < abs(first[0][keys[0]]),
              "first_peak_error_smaller": abs(first[1][keys[1]]) < abs(first[0][keys[1]]),
              "five_positive_peak_events_retained": complete,
              "no_interval_error_worse": len(interval_checks) == 4 and all(interval_checks)}
    report.update(decision="supported" if all(checks.values()) else "rejected",
                  checks=checks, each_interval_not_worse=interval_checks, records=records)
    return report
