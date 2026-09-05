"""Immutable starting profiles; selection does not confer human validation."""

from dataclasses import dataclass
from . import _h01_ei_parameters as parameters


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


def get_ei_profile(polarity, *, mode="candidate"):
    """Select a frozen E or I profile, with candidate defaults.

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
    if polarity not in ("E", "I") or mode not in ("candidate", "source"):
        raise ValueError("Choose polarity E or I and mode candidate or source.")
    excitatory = polarity == "E"
    name = "h01-l2-kv3-ninety-ca133" if excitatory else "h01-pv-regional-mesh-axon2187"
    digest = ("926efb3d5ae93b324bd653698d1c7d3fd478d9b94db1143b4eb933c5fcfa0f3a" if excitatory
              else "d8d4d022457eb139a033ead73780f2eb12a0d58852503267912d961b1eb9d546")
    source = ("Allen specimen 541563728; model 626170538; fit SHA256 2ceca2317ccbd586adde4b1e72507ad4bdf2fc10fc26ad4b281484324dd5f0c3"
              if excitatory else "ModelDB 267587 HL5BN1; commit 82cdd91bc93942ba19315371330a2412e064baf5")
    return EIProfile(name+":"+mode+":v1", polarity, mode, source, digest,
                     -83.97993469238281 if excitatory else -80.,
                     (-87.97993469238281 if mode == "candidate" else -83.97993469238281)
                     if excitatory else -96.97510324827309,
                     94.62299222737664 if excitatory else 100.,
                     getattr(parameters, polarity+"_"+mode.upper()),
                     ("Borrowed donor and mechanisms; not measured H01 physiology.",
                      "Candidate selection is not physiological qualification.",
                      "E onset and rising-phase errors remain; I early intervals remain too short."))


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
