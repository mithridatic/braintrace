"""Check the audit against its retained legacy control and intervention records."""

from docs.evidence.h01_pv_onset_sodium_audit import main


def test_retained_control_and_regional_runs():
    """Accept the original control metadata from before the calcium option existed."""
    main()
