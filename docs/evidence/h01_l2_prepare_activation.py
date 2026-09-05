"""Prepare a source-pinned, selective sodium-opening diagnostic patch."""

from docs.evidence.h01_l2_prepare_recovery import patch_recovery


def patch_activation(source):
    """Add opening-only time scaling to the checked recovery mechanism.

    Parameters
    ----------
    source : bytes
        Original pinned NaTs mechanism, before either diagnostic patch.

    Returns
    -------
    bytes
        Mechanism with default-one opening and recovery factors.

    Raises
    ------
    ValueError
        If the original mechanism has an unexpected digest.
    """
    text = patch_recovery(source).decode()
    replacements = {
        "RANGE gbar, g, ina, h_recovery_factor":
            "RANGE gbar, g, ina, h_recovery_factor, m_opening_factor",
        "\th_recovery_factor = 1":
            "\th_recovery_factor = 1\n\tm_opening_factor = 1",
        "mTau = (1/(mAlpha + mBeta))/qt":
            "mTau = (1/(mAlpha + mBeta))/qt\n"
            "\t\tif (mInf > m) { mTau = mTau * m_opening_factor }",
    }
    for old, new in replacements.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text.encode()
