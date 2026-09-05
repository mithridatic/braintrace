"""Check refinement auditing against the retained tapered-cell geometry."""

from docs.evidence.h01_pv_failure_mesh_audit import main


def test_retained_tapered_geometry_refinement():
    """Audit the same physical cell despite changed discretized diameter values."""
    main()
