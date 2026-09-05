"""Row construction and no-pass rendering of the layer-2 acceptance table."""
from docs.evidence import h01_l2_acceptance_table as table


def _fixtures():
    events = [{"rise_crossing_ms": 100.+i*10, "fall_crossing_ms": 101.+i*10, "time_above_threshold_ms": 1.,
               "peak_sample_time_ms": 100.4+i*10, "peak_sample_voltage_mv": 30.} for i in range(2)]
    requirements = {"event_matching": "ordinal", "inputs": [{"role": "E", "events": events}]}
    recovery = {"definition": "min", "records": [{"reference": "E", "recovery_datums": [
        {"event_index": 0, "terminal_window": False, "minimum_sample_voltage_mv": -70., "delay_from_peak_ms": 2.},
        {"event_index": 1, "terminal_window": True, "minimum_sample_voltage_mv": -71., "delay_from_peak_ms": 2.1}]}]}
    subthreshold = {"records": [{"role": "E", "command_current_na": .11, "samples": [{"requested_time_ms": 1019., "voltage_mv": -84.5}]}]}
    candidate = {"events": events, "unmatched_events": [], "event_residuals": [{k: .5 for k in table.FIELDS}]*2,
                 "phase_residuals_ms": [[-.1, .1]]*2, "interval_errors_ms": [3.], "minimum_voltage_errors_mv": [.05]}
    limits = {"fields": {"rise_crossing_ms": .01, "peak_sample_voltage_mv": .001, "time_above_threshold_ms": .0004}, "phase_ms": .0004, "minimum_voltage_mv": .001}
    return requirements, recovery, subthreshold, candidate, limits


def test_every_required_observation_has_a_row_and_no_pass_rule():
    rows = table.build_rows(*_fixtures())
    names = [r["observation"] for r in rows]
    assert names[0] == "event count"
    assert sum(n.startswith("event 1 ") for n in names) == len(table.FIELDS)+2
    assert names.count("interval 1 ms") == 1
    assert "recovery minimum 2 voltage mV" in names and "subthreshold 1019 ms voltage mV" in names
    assert all(r["allowance"] == table.NO_ALLOWANCE for r in rows[1:])
    assert "terminal window" in next(r for r in rows if r["observation"] == "recovery minimum 2 voltage mV")["convention"]
    assert next(r for r in rows if r["observation"] == "recovery minimum 2 voltage mV")["candidate_residual"] == "not computed"


def test_numerical_limits_take_the_largest_recorded_change():
    tolerance = {"limits": {"a": 1.}, "max_absolute_differences": {"a": .1}, "max_phase_difference_ms": .0004}
    spatial = {"max_absolute_differences": {"a": .3}, "max_phase_difference_ms": .0003}
    minima = {"comparisons": {"x": {"individual_differences": [{"voltage_mv": -.002}, {"voltage_mv": .001}]}}}
    assert table.numerical_limits(tolerance, spatial, minima) == {"fields": {"a": .3}, "phase_ms": .0004, "minimum_voltage_mv": .002}


def test_render_states_no_pass_and_lists_sources():
    text = table.render(table.build_rows(*_fixtures()), ("a.json",))
    assert "issues no pass and no fail" in text and "[a.json](a.json)" in text
    assert text.count("\n| ") == len(table.build_rows(*_fixtures()))+2
