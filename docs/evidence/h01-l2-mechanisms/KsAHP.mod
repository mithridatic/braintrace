: Spike-triggered slow outward potassium conductance.
: SP12, docs/specs/2026-09-11-h01-e-spike-triggered-outward.md.
:
: Phenomenological stand-in for a slow after-hyperpolarisation (sAHP) or a
: sodium-activated potassium (KNa) conductance whose carrier this fit cannot
: supply: the recorded B3 soma calcium at 200 pA rises from 1.0e-4 to 2.8e-4 mM
: and is not spike-shaped, so a calcium-gated gate would not be spike-triggered.
: The gate z opens only while the membrane is above vhalf (crossed during a
: spike) with time constant tau_on and closes with time constant tau_off. Below
: vhalf the steady-state activation is zero, so the mechanism is silent without
: a spike apart from the exponential tail of an earlier spike.

NEURON {
    SUFFIX KsAHP
    USEION k READ ek WRITE ik
    RANGE gbar, g, ik, vhalf, slope, tau_on, tau_off
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0 (S/cm2)
    vhalf = -20 (mV)
    slope = 2 (mV)
    tau_on = 1 (ms)
    tau_off = 1000 (ms)
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    g (S/cm2)
    zinf
    ztau (ms)
}

STATE {
    z
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar*z
    ik = g*(v-ek)
}

DERIVATIVE states {
    rates(v)
    z' = (zinf-z)/ztau
}

INITIAL {
    rates(v)
    z = zinf
}

PROCEDURE rates(v (mV)) {
    UNITSOFF
    zinf = 1/(1+exp(-(v-vhalf)/slope))
    ztau = tau_off+(tau_on-tau_off)*zinf
    UNITSON
}
