"""Read exposed rates from pinned NEURON mechanisms; no surrogate equations."""
import json
from pathlib import Path
import sys
from neuron import h

role = sys.argv[1]
h.load_file("stdrun.hoc")
h.celsius = 34.
cases = []
sections = []
for mechanism in sorted(p.stem for p in Path.cwd().glob("*.mod")):
    for voltage in (-154.9, -84., -66., -56., -40., -38., -27., -20., 20.):
        for mode in ("source", "candidate"):
            for phase, state in (("opening", 0.), ("closing", 1.)):
                sec = h.Section(name="clamp"+str(len(sections)))
                sec.insert(mechanism)
                sections.append(sec)
                cases.append((sec, mechanism, voltage, mode, phase, state))
h.finitialize(-80.)
for sec, _, voltage, _, _, _ in cases:
    sec.v = voltage
h.fcurrent()
rows = []
for sec, mechanism, voltage, mode, phase, state in cases:
    sec.v = voltage
    ch = getattr(sec(.5), mechanism)
    if role == "I" and mechanism == "NaTg":
        ch.vshifth, ch.slopem = 10., 9.
    if role == "I" and mechanism == "Ih":
        for key, value in enumerate((262.07471272583734, 19.041504011554068,
            4.162478599311996, 75.56131984386032, 78.52193065082363, 1.3115816183914442), 1):
            setattr(ch, "shift"+str(key), value)
    if mode == "candidate":
        if role == "E" and mechanism == "NaTs": ch.m_opening_factor = 2.
        if role == "E" and mechanism == "Kv3_1": ch.m_closing_factor = .9
        if role == "I" and mechanism == "NaTg":
            ch.h_tau_factor, ch.h_recovery_factor, ch.slopeh = .15, 1., 5.
        if role == "I" and mechanism == "Kv3_1":
            ch.m_tau_factor, ch.m_close_factor = .5, .5
    for g in ("m", "h", "z"):
        if hasattr(ch, g): setattr(ch, g, state)
    ca = 1e-4
    if mechanism == "SK": ch.rates(ca)
    else: ch.rates()
    gates = {}
    for g in ("m", "h", "z"):
        if not hasattr(ch, g): continue
        equilibrium = getattr(ch, g+"Inf")
        tau = getattr(ch, g+"Tau")
        # PV phase factors reside in DERIVATIVE, not in the rates procedure.
        if role == "I" and mode == "candidate":
            if mechanism == "NaTg" and g == "h" and equilibrium > state: tau /= .15
            if mechanism == "Kv3_1" and equilibrium < state: tau *= .5/.5
        gates[g] = dict(inf=equilibrium, tau_ms=tau)
    rows.append(dict(mechanism=mechanism, voltage_mv=voltage, mode=mode,
                     phase=phase, state=state, calcium_mm=ca, gates=gates))
Path("/evidence/h01-ei-"+role.lower()+"-gate-oracle.json").write_text(json.dumps(
    dict(role=role, source_hashes=json.loads(Path("source-hashes.json").read_text()),
         cases=rows, temperature_c=34., oracle="NEURON 9.0.2 exposed source rates"), indent=2)+"\n")
print(role, len(rows), "cases")
