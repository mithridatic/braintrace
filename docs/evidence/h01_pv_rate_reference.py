"""Read algebraic rate outputs from compiled original NMODL mechanisms.

Only RANGE observation fields were added to the cached source copies.
This script calls compiled rate algebra; it does not advance membrane dynamics.
"""

import json
from pathlib import Path

import neuron
from neuron import h


voltage = [-119.3, -99.7, -86.2, -70.1, -60.001, -60., -59.999,
           -56.2, -40.1, -38.2, -27.2, -17.2, -1.2, 20.3, 50.4]
calcium = [0., .000099999, .0001, .000100001, .0002, .00043, .001, .01]
mechanisms = ["NaTg", "Nap", "K_P", "K_T", "Kv3_1", "Im", "Ih", "Ca_HVA", "Ca_LVA", "SK"]
tables = {}
sections = []
for name in mechanisms:
    section = h.Section(name="oracle_"+name)
    sections.append(section)
    section.insert(name)
    mechanism = getattr(section(.5), name)
    if name == "NaTg":
        mechanism.vshiftm = 0.
        mechanism.vshifth = 10.
        mechanism.slopem = 9.
        mechanism.slopeh = 6.
    if name == "Ih":
        for field, value in enumerate((262.07471272583734, 19.041504011554068,
            4.162478599311996, 75.56131984386032, 78.52193065082363, 1.3115816183914442), 1):
            setattr(mechanism, f"shift{field}", value)
    gates = ("z",) if name == "SK" else (("m",) if name in ("Kv3_1", "Im", "Ih") else ("m", "h"))
    values = {gate: {"inf": [], "tau_ms": []} for gate in gates}
    for point in calcium if name == "SK" else voltage:
        section(.5).v = -80. if name == "SK" else point
        # Synchronize NEURON's mechanism-local voltage before calling its
        # exported algebraic procedure. No membrane time step is advanced.
        h.fcurrent()
        if name == "SK":
            mechanism.rates(point)
        else:
            mechanism.rates()
        for gate in gates:
            values[gate]["inf"].append(float(getattr(mechanism, gate+"Inf")))
            values[gate]["tau_ms"].append(float(getattr(mechanism, gate+"Tau")))
    tables[name] = values
    if name != "SK":
        assert len(set(values["m"]["inf"])) > 1, "Rate oracle did not receive the voltage sweep."
report = {"neuron_version": neuron.__version__, "voltage_mv": voltage,
          "calcium_mm": calcium, "temperature_convention": "source fixed 34 Celsius",
          "parameters": "released HL5BN1 NaTg and Ih values; other source defaults",
          "instrumentation": "RANGE observation fields only; equations unchanged",
          "rates": tables}
Path("/evidence/h01-pv-rate-oracle.json").write_text(json.dumps(report, indent=2))
print("Exported", len(tables), "independent mechanism rate tables.")
