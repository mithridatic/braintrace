"""The circuit runner's interface: measured pair by default, illustrative pair only on request."""

import pytest

from examples.h01_ei_circuit import build_parser, default_i_current_na, identities_for


def test_measured_pair_is_the_default_and_uses_the_pinned_identities():
    args = build_parser().parse_args(["--output", "x"])
    assert args.connectivity == "measured" and args.control == "ei" and args.solver == "staggered"
    assert identities_for(args.connectivity) == (("E", "4157825456"), ("I", "5584343344"))
    assert args.e_delay_ms == 2. and args.i_pulse_ms == 3. and args.dt_ms == .005


def test_illustrative_pair_is_explicit_and_unknown_wiring_is_rejected():
    assert identities_for("illustrative") == (("E", "810151953"), ("I", "678539249"))
    with pytest.raises(ValueError):
        identities_for("reconstructed")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output", "x", "--connectivity", "reconstructed"])


def test_i_drive_defaults_differ_by_pair_and_explicit_values_win():
    assert default_i_current_na("measured", None) == 1.
    assert default_i_current_na("illustrative", None) == 0.
    assert default_i_current_na("measured", .25) == .25


def test_inhibitory_receptor_flags_default_to_the_assumed_values():
    from examples.h01_ei_circuit import build_parser
    args = build_parser().parse_args(["--output", "x"])
    assert (args.inhibitory_weight_us, args.inhibitory_reversal_mv, args.inhibitory_tau_ms) == (.02, -80., 5.)
    args = build_parser().parse_args(["--output", "x", "--inhibitory-weight-us", "0.0031", "--inhibitory-reversal-mv", "-75"])
    assert args.inhibitory_weight_us == .0031 and args.inhibitory_reversal_mv == -75.
