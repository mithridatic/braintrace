"""Prepare a pinned, closing-only Kv3 diagnostic mechanism."""

import hashlib


def patch_kv3_closing(source):
    """Add a default-one factor only when the Kv3 gate closes.

    Parameters
    ----------
    source : bytes
        Original pinned Kv3_1 mechanism.

    Returns
    -------
    bytes
        Source with the isolated closing-time intervention.

    Raises
    ------
    ValueError
        If the source digest differs from the inspected version.
    """
    if hashlib.sha256(source).hexdigest() != "e636c3437a9d9ccb69a0a7efc09fcfe1f16ad97f0468615b70fdbc352fe0855c":
        raise ValueError("Unexpected source Kv3 digest.")
    text = source.decode()
    newline = "\r\n" if "\r\n" in text else "\n"
    tau = "mTau =  0.2*20.000/(1+exp(((v -(-46.560 + vshift))/(-44.140))))"
    replacements = {
        "RANGE gbar, g, ik": "RANGE gbar, g, ik, m_closing_factor",
        "gbar = 0.00001 (S/cm2)": "gbar = 0.00001 (S/cm2)" + newline + "\tm_closing_factor = 1",
        tau: tau + newline + "\t\tif (mInf < m) { mTau = mTau * m_closing_factor }",
    }
    for old, new in replacements.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text.encode()
