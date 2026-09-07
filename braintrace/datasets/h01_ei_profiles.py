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
        Versioned identity, E/I role, and mode (``candidate`` by default,
        ``source``, or an experimental mode registered for the donor).
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
        reversal_mv={"candidate": -87.97993469238281, "source": -83.97993469238281, "b3": -87.97993469238281}),
    "l5-pv-basket-hl5bn1": dict(
        digest="d8d4d022457eb139a033ead73780f2eb12a0d58852503267912d961b1eb9d546",
        source="ModelDB 267587 HL5BN1; commit 82cdd91bc93942ba19315371330a2412e064baf5",
        initial_mv=-80., axial_ohm_cm=100.,
        reversal_mv={"candidate": -96.97510324827309, "source": -96.97510324827309,
                     "finalist": -96.97510324827309}),
    "l3-sst-interneuron-hl5mn1": dict(
        digest="a9ce264f2733104ceb457d519678f447cb277db2dda7cedf0a0cc38fb6ffb0f4",
        source="ModelDB 267587 HL5MN1 (= Yao 2022 HL23SST); agmccrei/HumanL5Circuit_AGM2022 commit dd472f19a0d1bfbbba59677cfd82c6e9f8a80590; Allen specimen 571700636",
        initial_mv=-81.5, axial_ohm_cm=100.,
        reversal_mv={"candidate": -81.5, "source": -81.5}),
}
# Donors whose candidate and experimental modes carry phase controls in ``channel_controls``
# (looked up by polarity and mode below); donors without a candidate get none.
_CONTROLLED = {"h01-l2-kv3-ninety-ca133": "E", "h01-pv-regional-mesh-axon2187": "I"}
_LIMITATIONS = ("Borrowed donor and mechanisms; not measured H01 physiology.",
                "Candidate selection is not physiological qualification.",
                "E onset and rising-phase errors remain; I early intervals remain too short.")
_EXPERIMENTAL_MODES = ("finalist", "b3")
_EXPERIMENTAL_LIMITATION = ("Experimental mode {mode!r}: registered so the SP2 held-equal transfer arms can run "
                            "against the NEURON finalist; not promoted, not a candidate; the default mode "
                            "stays candidate.")

# Phase controls by (polarity, mode): (mechanism, family or None for every family) -> constructor
# keywords of the registered channel family. Candidate entries reproduce the pre-existing values;
# "finalist" differs from the I candidate by the somatic Kv3 closing factor (0.5 -> 2.0); "b3"
# maps the L2 NEURON soma-only factors (sodium_opening_factor -> NaTs m_open,
# sodium_recovery_factor -> NaTs h_open, kv3_closing_factor -> Kv3_1 m_close).
_I_NATG = {"h_close": .15, "h_open": 1., "h_slope": 5.}
_CONTROLS = {
    ("E", "candidate"): {("NaTs", None): {"m_open": 2.}, ("Kv3_1", None): {"m_close": .9}},
    ("E", "b3"): {("NaTs", "soma"): {"m_open": 2., "h_open": 1.}, ("Kv3_1", "soma"): {"m_close": .9}},
    ("I", "candidate"): {("NaTg", None): _I_NATG, ("Kv3_1", "soma"): {"m_open": .5, "m_close": .5}},
    ("I", "finalist"): {("NaTg", None): _I_NATG, ("Kv3_1", "soma"): {"m_open": .5, "m_close": 2.}},
}
# Flags of an experimental mode that are realised inside its density table rather than passed
# to a channel constructor (B3: soma calcium decay = source x 1.0; Ih spread over soma+dend+apic).
_MODE_FLAGS = {("E", "b3"): {"calcium_decay_factor": 1., "distribute_ih": True}}


def get_donor_profile(key, *, mode="candidate"):
    """Select the frozen profile of one registered donor.

    Parameters
    ----------
    key : str
        Donor key from ``h01_cell_types.DONORS``.
    mode : str, optional
        ``candidate`` (default), published ``source`` settings, or an
        experimental mode registered for the donor in
        ``_h01_ei_parameters.DONOR_REGIONS`` (``finalist`` for the PV donor,
        ``b3`` for the L2 donor). Experimental modes carry an extra limitation.

    Returns
    -------
    EIProfile
        Immutable physical parameters and qualification limits.

    Raises
    ------
    ValueError
        Unknown donor key, or a mode the donor does not register.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01_ei_profiles import get_donor_profile
        >>> get_donor_profile("l5-pv-basket-hl5bn1").name
        'h01-pv-regional-mesh-axon2187:candidate:v1'
        >>>
        >>> get_donor_profile("l5-pv-basket-hl5bn1", mode="finalist").limitations[-1][:28]
        "Experimental mode 'finalist'"
    """
    modes = parameters.DONOR_REGIONS.get(key, {})
    if key not in _PHYSIOLOGY or key not in DONORS or mode not in modes:
        raise ValueError(f"Unknown donor key {key!r} or mode {mode!r}; choose one of {tuple(modes) or 'a known key'}.")
    donor, physiology = DONORS[key], _PHYSIOLOGY[key]
    limitations = _LIMITATIONS
    if mode in _EXPERIMENTAL_MODES:
        limitations = _LIMITATIONS+(_EXPERIMENTAL_LIMITATION.format(mode=mode),)
    return EIProfile(donor["profile"]+":"+mode+":v1", donor["polarity"], mode, physiology["source"],
                     physiology["digest"], physiology["initial_mv"], physiology["reversal_mv"][mode],
                     physiology["axial_ohm_cm"], modes[mode], limitations,
                     donor["channel_prefix"], donor["sodium_reversal_mv"], donor["potassium_reversal_mv"])


def get_ei_profile(polarity, *, mode="candidate"):
    """Select the polarity's default donor profile (alias of ``get_donor_profile``).

    Parameters
    ----------
    polarity : str
        Explicit E or I modeling role. Does not change anatomical labels.
    mode : str, optional
        ``candidate`` (default), ``source``, or an experimental mode the
        default donor registers (``finalist`` for I, ``b3`` for E).

    Returns
    -------
    EIProfile
        Immutable physical parameters and qualification limits.
    """
    if polarity not in DEFAULT_DONOR_KEYS:
        raise ValueError("Choose polarity E or I and a mode the donor registers (candidate or source).")
    try:
        return get_donor_profile(DEFAULT_DONOR_KEYS[polarity], mode=mode)
    except ValueError as error:
        raise ValueError("Choose polarity E or I and a mode the donor registers: "+str(error)) from error


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
        Constructor keywords (``m_open``, ``m_close``, ``h_open``, ``h_close``,
        ``h_slope``). Empty for unchanged source mechanisms, for every
        mechanism the mode does not control in that family, and for donors
        without a candidate (the HL5MN1 import), whatever their polarity.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01_ei_profiles import channel_controls, get_ei_profile
        >>> channel_controls(get_ei_profile("I", mode="finalist"), "soma", "Kv3_1")
        {'m_open': 0.5, 'm_close': 2.0}
        >>>
        >>> channel_controls(get_ei_profile("E", mode="b3"), "axon", "NaTs")
        {}
    """
    if profile.name.split(":")[0] not in _CONTROLLED:
        return {}
    table = _CONTROLS.get((profile.polarity, profile.mode), {})
    controls = table.get((mechanism, family), table.get((mechanism, None), {}))
    return dict(controls)


def mode_flags(profile):
    """Return the flags of an experimental mode that its density table realises.

    Parameters
    ----------
    profile : EIProfile
        Selected immutable profile.

    Returns
    -------
    dict
        ``calcium_decay_factor`` and ``distribute_ih`` for the E ``b3`` mode;
        empty for every other profile. These are records of how the table was
        built, not constructor keywords.
    """
    return dict(_MODE_FLAGS.get((profile.polarity, profile.mode), {}))
