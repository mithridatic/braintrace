"""Spatially explicit local current observations from the H01 L2 recorder."""

import numpy as np

from docs.evidence.h01_direct_observations import observe, validate_trace


def local_currents(data, report):
    """Convert recorded densities and individual axial branches at soma(0.5).

    Parameters
    ----------
    data : mapping
        L2 recorder arrays, including voltage and named current probes.
    report : dict
        Recorder metadata with geometry and units for every density probe.

    Returns
    -------
    tuple
        Current arrays and spatial metadata. Missing geometry yields a
        voltage-only observation, never an inferred whole-soma balance.
    """
    t, v = validate_trace(data["time_ms"], data["voltage_mv"])
    geo = report.get("charge_balance_geometry")
    if not geo:
        return {}, {"status": "unavailable", "reason": "No current-boundary geometry."}
    area, cm = geo.get("area_um2"), geo.get("cm_uf_cm2")
    if not geo.get("location") or area is None or cm is None or not np.isfinite([area, cm]).all() or min(area, cm) <= 0:
        raise ValueError("Current boundary needs a location, positive area, and capacitance density.")
    units = report.get("soma_observation_units", {})
    currents, paths, missing = {}, {}, []
    for key in sorted(data):
        if not (key.startswith("soma_") and key.endswith("_ma_cm2")):
            continue
        if "mA/cm2" not in units.get(key, ""):
            raise ValueError(f"Missing density units for {key}.")
        name = key[5:-7]
        if name != "icap" and "outward positive" not in units[key]:
            raise ValueError(f"Missing current sign convention for {key}.")
        name = "storage" if name == "icap" else name
        sign = 1. if name == "storage" else -1.
        currents[name] = sign * np.asarray(data[key], float) * area * .01
        paths[name] = {"location": geo["location"], "units": "nA",
                           "sign": "positive charge storage" if name == "storage" else "inward positive"}
    if "applied_current_na" in data:
        currents["applied"] = np.asarray(data["applied_current_na"], float)
        paths["applied"] = {"location": geo["location"], "units": "nA", "sign": "inward positive"}
    else:
        missing.append("applied_current_na")
    for i, n in enumerate(geo.get("neighbours", [])):
        resistance = n.get("resistance_mohm")
        if not n.get("location") or resistance is None or not np.isfinite(resistance) or resistance <= 0:
            raise ValueError("Axial branch needs a location and positive resistance.")
        if n["voltage_key"] not in data:
            missing.append(n["voltage_key"])
            continue
        neighbour = np.asarray(data[n["voltage_key"]], float)
        if neighbour.shape != t.shape or not np.isfinite(neighbour).all():
            raise ValueError("Axial voltage must share the finite observation clock.")
        name = f"axial_{i}"
        currents[name] = (neighbour-v)/resistance
        paths[name] = {"location": geo["location"], "neighbour": n["location"], "units": "nA",
                           "sign": "into recorded compartment", "resistance_mohm": resistance,
                           "effort": "neighbour voltage minus local voltage, mV"}
    expected = [k for k in units if k.startswith("soma_") and k.endswith("_ma_cm2")]
    missing += [k for k in expected if k not in data]
    if "storage" not in currents:
        missing.append("soma_icap_ma_cm2")
    if not expected:
        missing.append("declared current inventory")
    if "neighbours" not in geo:
        missing.append("axial inventory")
    for name, a in currents.items():
        if a.shape != t.shape or not np.isfinite(a).all():
            raise ValueError(f"Current {name} must share the finite observation clock.")
    if not missing:
        currents["balance_residual"] = sum(a for k, a in currents.items() if k != "storage")-currents["storage"]
    return currents, {"status": "partial" if missing else "available", "missing": missing,
                          "location": geo["location"], "area_um2": area, "capacitance_nf": area*cm*1e-5,
                          "paths": paths, "limitation": "Local compartment only; human ionic currents and ion-reservoir energy unavailable."}


def extract_l2(data, report, pulse_ms=(1020., 2020.)):
    """Build small-cycle observations with voltage/current and voltage/charge pairs.

    Parameters
    ----------
    data : mapping
        Original L2 recorder samples.
    report : dict
        Matching recorder report, retained verbatim as provenance.
    pulse_ms : tuple of float
        Command start and stop in ms.

    Returns
    -------
    tuple
        Observation record and selected arrays, including individual branch power
        and capacitor charge checks when the required observations are available.
    """
    currents, boundary = local_currents(data, report)
    record, arrays = observe(data["time_ms"], data["voltage_mv"], pulse_ms, currents,
                             metadata={"boundary": boundary, "source_report": report})
    record["current_evidence"] = boundary["status"]
    for w in record["windows"]:
        if w["status"] != "available":
            continue
        prefix = w["array_prefix"]
        voltage = arrays[f"{prefix}_voltage_mv"]
        for name in report.get("soma_observation_units", {}):
            if name.startswith("soma_") and not name.endswith("_ma_cm2") and name in data:
                state = np.asarray(data[name], float)
                if state.shape != np.asarray(data["time_ms"]).shape or not np.isfinite(state).all():
                    raise ValueError(f"State {name} must share the finite observation clock.")
                arrays[f"{prefix}_state_{name}"] = np.interp(arrays[f"{prefix}_time_ms"], data["time_ms"], state)
        if "storage" in currents:
            expected = boundary["capacitance_nf"] * (voltage-voltage[0])
            arrays[f"{prefix}_capacitor_charge_pc"] = expected
            measured = arrays[f"{prefix}_charge_storage_pc"]
            w["max_abs_capacitor_charge_error_pc"] = float(np.max(np.abs(measured-expected)))
        if "balance_residual" in currents:
            w["max_abs_current_balance_residual_na"] = float(np.max(np.abs(arrays[f"{prefix}_current_balance_residual_na"])))
        for name, path in boundary.get("paths", {}).items():
            if "resistance_mohm" in path:
                current = arrays[f"{prefix}_current_{name}_na"]
                drop = current*path["resistance_mohm"]
                arrays[f"{prefix}_effort_{name}_mv"] = drop
                arrays[f"{prefix}_dissipation_{name}_pw"] = drop*current
    return record, arrays
