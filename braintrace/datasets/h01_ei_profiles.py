"""Immutable starting profiles per donor; selection does not confer human validation."""

from dataclasses import dataclass
from . import _h01_ei_parameters as parameters
from .h01_cell_types import DEFAULT_DONOR_KEYS, DONORS


@dataclass(frozen=True)
class EIProfile:
    """Pinned borrowed dynamics and inferred candidate changes.

    Attributes
    ----------
    name, polarity, mode : str
        Versioned identity, E/I role, and candidate or source mode.
    source, metadata_sha256 : str
        Source identity and hash of the selected reference metadata.
    initial_mv, reversal_mv, axial_ohm_cm : float
        Initial voltage, leak reversal, and axial resistivity.
    regions : tuple
        Region name, capacitance, leak density, channels, calcium settings.
    limitations : tuple of str
        Known limitations; these are not H01 donor measurements.
    channel_prefix : str
        Registered mechanism family of the donor (``H01L2`` or ``H01PV``).
    sodium_reversal_mv, potassium_reversal_mv : float
        Fixed ion reversals painted with the donor's channels.
    """

    name: str
    polarity: str
    mode: str
    source: str
    metadata_sha256: str
    initial_mv: float
    reversal_mv: float
    axial_ohm_cm: float
    regions: tuple
    limitations: tuple
    channel_prefix: str
    sodium_reversal_mv: float
    potassium_reversal_mv: float


# Physiology pinned per donor key: reference digest, fit identity, voltages by mode.
_PHYSIOLOGY = {
    "l2-pyramidal-allen-541563728": dict(
        digest="926efb3d5ae93b324bd653698d1c7d3fd478d9b94db1143b4eb933c5fcfa0f3a",
        source="Allen specimen 541563728; model 626170538; fit SHA256 2ceca2317ccbd586adde4b1e72507ad4bdf2fc10fc26ad4b281484324dd5f0c3",
        initial_mv=-83.97993469238281, axial_ohm_cm=94.62299222737664,
        reversal_mv={"candidate": -87.97993469238281, "source": -83.97993469238281}),
    "l5-pv-basket-hl5bn1": dict(
        digest="d8d4d022457eb139a033ead73780f2eb12a0d58852503267912d961b1eb9d546",
        source="ModelDB 267587 HL5BN1; commit 82cdd91bc93942ba19315371330a2412e064baf5",
        initial_mv=-80., axial_ohm_cm=100.,
        reversal_mv={"candidate": -96.97510324827309, "source": -96.97510324827309}),
}
_LIMITATIONS = ("Borrowed donor and mechanisms; not measured H01 physiology.",
                "Candidate selection is not physiological qualification.",
                "E onset and rising-phase errors remain; I early intervals remain too short.")


def get_donor_profile(key, *, mode="candidate"):
    """Select the frozen profile of one registered donor.

    Parameters
    ----------
    key : str
        Donor key from ``h01_cell_types.DONORS``.
    mode : str, optional
        Candidate (default) or published source settings.

    Returns
    -------
    EIProfile
        Immutable physical parameters and qualification limits.

    Raises
    ------
    ValueError
        Unknown donor key or mode.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01_ei_profiles import get_donor_profile
        >>> get_donor_profile("l5-pv-basket-hl5bn1").name
        'h01-pv-regional-mesh-axon2187:candidate:v1'
    """
    if key not in _PHYSIOLOGY or key not in DONORS or mode not in ("candidate", "source"):
        raise ValueError(f"Unknown donor key {key!r} or mode {mode!r}; choose candidate or source.")
    donor, physiology = DONORS[key], _PHYSIOLOGY[key]
    return EIProfile(donor["profile"]+":"+mode+":v1", donor["polarity"], mode, physiology["source"],
                     physiology["digest"], physiology["initial_mv"], physiology["reversal_mv"][mode],
                     physiology["axial_ohm_cm"], parameters.DONOR_REGIONS[key][mode], _LIMITATIONS,
                     donor["channel_prefix"], donor["sodium_reversal_mv"], donor["potassium_reversal_mv"])


def get_ei_profile(polarity, *, mode="candidate"):
    """Select the polarity's default donor profile (alias of ``get_donor_profile``).

    Parameters
    ----------
    polarity : str
        Explicit E or I modeling role. Does not change anatomical labels.
    mode : str, optional
        Candidate (default) or published source settings.

    Returns
    -------
    EIProfile
        Immutable physical parameters and qualification limits.
    """
    if polarity not in DEFAULT_DONOR_KEYS or mode not in ("candidate", "source"):
        raise ValueError("Choose polarity E or I and mode candidate or source.")
    return get_donor_profile(DEFAULT_DONOR_KEYS[polarity], mode=mode)


def channel_controls(profile, family, mechanism):
    """Return explicit region-specific phase controls for a channel.

    Parameters
    ----------
    profile : EIProfile
        Selected immutable profile.
    family, mechanism : str
        Electrical region and source mechanism name.

    Returns
    -------
    dict
        Constructor keywords. Empty for unchanged source mechanisms.
    """
    if profile.mode == "source":
        return {}
    if profile.polarity == "E":
        if mechanism == "NaTs":
            return {"m_open": 2.}
        if mechanism == "Kv3_1":
            return {"m_close": .9}
    else:
        if mechanism == "NaTg":
            return {"h_close": .15, "h_open": 1., "h_slope": 5.}
        if mechanism == "Kv3_1" and family == "soma":
            return {"m_open": .5, "m_close": .5}
    return {}
