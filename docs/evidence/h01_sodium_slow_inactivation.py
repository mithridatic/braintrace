"""Add a slow inactivation gate to the pinned sodium mechanisms of both H01 cells.

SP16 (docs/specs/2026-09-12-h01-sodium-slow-inactivation.md). The E fit uses ``NaTs`` and
the I fit uses ``NaTg``; both are the same Colbert and Pan 2002 template, and both are
what the two donor fits share. The patch multiplies the conductance by one extra gate
``s`` whose steady state is

    sInf = 1 - slow_inactivation/(1 + exp(-(v - s_vhalf)/s_slope))

so that at ``slow_inactivation = 0`` (the default) ``sInf`` is identically one, ``s``
starts at one and never moves, and the patched mechanism is the original mechanism. That
default is what makes the unchanged candidate reproducible after the library is rebuilt,
which is the registered stage-0 check.

At a positive depth the gate removes up to that fraction of the sodium conductance while
the cell sits above the half-point, and recovers below it. Entry and recovery carry
separate time constants,

    sTau = s_tau_recovery_ms + (s_tau_entry_ms - s_tau_recovery_ms)/(1 + exp(-(v - s_vhalf)/s_slope))

because the recorded cells fix them separately: the pyramidal cell's threshold and rise
step between spike 1 and spike 2, inside 11 to 33 ms, and then hold for the rest of the
second, while spike 1 is the same in every sweep, so recovery takes longer than the
800 ms hold after a single spike at 200 pA and less than the gap between sweeps. The
half-point sits above the 200 pA plateau (-65 mV) so that the gate is shut below
threshold and engages during the spike. Its registered signature is cyclical, not
elemental: it must leave the first spike alone and step the threshold at spike 2.
"""

import hashlib

SOURCES = {
    "NaTs": "6ba797bada310a6b880f4ab2a86e915894cc205d411474c0ce78f806af532a1f",
    "NaTg": "fef2809e8f3edbeec5a1433e426925d5a9fe3ef82e8ebd2ee328d810a9a10dfe",
}
DEFAULTS = {"slow_inactivation": 0, "s_vhalf": -50, "s_slope": 6, "s_tau_entry_ms": 10, "s_tau_recovery_ms": 1000}

REPLACEMENTS = (
    ("RANGE gbar, g, ina",
     "RANGE gbar, g, ina, slow_inactivation, s_vhalf, s_slope, s_tau_entry_ms, s_tau_recovery_ms"),
    ("\tgbar = 0.00001 (S/cm2)",
     "\tgbar = 0.00001 (S/cm2)\n"
     "\tslow_inactivation = 0 : fraction of the conductance the slow gate can remove; 0 keeps the source mechanism\n"
     "\ts_vhalf = -50 (mV) : above the 200 pA plateau, so the gate is shut below threshold\n"
     "\ts_slope = 6 (mV)\n"
     "\ts_tau_entry_ms = 10 (ms) : above the half-point, during a spike\n"
     "\ts_tau_recovery_ms = 1000 (ms) : below the half-point, between sweeps"),
    ("\thBeta\n}", "\thBeta\n\tsInf\n\tsTau\n}"),
    ("STATE\t{\n\tm\n\th\n}", "STATE\t{\n\tm\n\th\n\ts\n}"),
    ("g = gbar*m*m*m*h", "g = gbar*m*m*m*h*s"),
    ("\tm' = (mInf-m)/mTau", "\tm' = (mInf-m)/mTau\n\ts' = (sInf-s)/sTau"),
    ("\tm = mInf\n\th = hInf\n}", "\tm = mInf\n\th = hInf\n\ts = sInf\n}"),
    ("hInf = hAlpha/(hAlpha + hBeta)",
     "hInf = hAlpha/(hAlpha + hBeta)\n"
     "\t\tsInf = 1 - slow_inactivation/(1 + exp(-(v - s_vhalf)/s_slope))\n"
     "\t\tsTau = s_tau_recovery_ms + (s_tau_entry_ms - s_tau_recovery_ms)/(1 + exp(-(v - s_vhalf)/s_slope))"),
)


def steady_state(voltage_mv, depth, vhalf_mv=DEFAULTS["s_vhalf"], slope_mv=DEFAULTS["s_slope"]):
    """The gate's steady state, the same expression the mechanism evaluates.

    Parameters
    ----------
    voltage_mv : float
        Membrane voltage in mV.
    depth : float
        ``slow_inactivation``: the fraction of the conductance the gate can remove.
    vhalf_mv, slope_mv : float, optional
        Half-point and slope of the gate in mV.

    Returns
    -------
    float
        One when ``depth`` is zero, falling towards ``1 - depth`` above the half-point.
    """
    import math
    return 1.-depth/(1.+math.exp(-(voltage_mv-vhalf_mv)/slope_mv))


def time_constant(voltage_mv, entry_ms=DEFAULTS["s_tau_entry_ms"], recovery_ms=DEFAULTS["s_tau_recovery_ms"],
                  vhalf_mv=DEFAULTS["s_vhalf"], slope_mv=DEFAULTS["s_slope"]):
    """The gate's time constant, the same expression the mechanism evaluates.

    Parameters
    ----------
    voltage_mv : float
        Membrane voltage in mV.
    entry_ms, recovery_ms : float, optional
        Time constants far above and far below the half-point, in ms.
    vhalf_mv, slope_mv : float, optional
        Half-point and slope shared with :func:`steady_state`.

    Returns
    -------
    float
        ``recovery_ms`` at rest, ``entry_ms`` during a spike, blended through the half-point.
    """
    import math
    return recovery_ms+(entry_ms-recovery_ms)/(1.+math.exp(-(voltage_mv-vhalf_mv)/slope_mv))


def patch_sodium(source, mechanism):
    """Add the slow inactivation gate to a pinned ``NaTs`` or ``NaTg`` source.

    Parameters
    ----------
    source : bytes
        The pinned mechanism file, in its original newline style.
    mechanism : str
        ``NaTs`` or ``NaTg``; selects the digest the source must carry.

    Returns
    -------
    bytes
        The patched source, in the same newline style.

    Raises
    ------
    ValueError
        If the mechanism is unknown, the digest differs from the inspected version, or an
        anchor is missing or ambiguous.
    """
    if mechanism not in SOURCES:
        raise ValueError(f"Unknown sodium mechanism {mechanism}; expected one of {sorted(SOURCES)}.")
    if hashlib.sha256(source).hexdigest() != SOURCES[mechanism]:
        raise ValueError(f"Unexpected {mechanism} source digest; the patch is pinned to the inspected version.")
    text = source.decode()
    newline = "\r\n" if "\r\n" in text else "\n"
    body = text.replace("\r\n", "\n")
    for anchor, replacement in REPLACEMENTS:
        if body.count(anchor) != 1:
            raise ValueError(f"Anchor {anchor!r} appears {body.count(anchor)} times in {mechanism}; expected once.")
        body = body.replace(anchor, replacement)
    return body.replace("\n", newline).encode()
