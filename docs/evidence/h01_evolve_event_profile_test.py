"""Tests for the pure parts of the H01 evolve event profile."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import h01_evolve_event_profile as profile   # noqa: E402

HLO = '''HloModule jit_x, is_scheduled=true

FileNames
1 "/repo/braintrace/datasets/h01_wilbers.py"
2 "/venv/saiunit/math/_fun_keep_unit.py"
3 "/repo/braintrace/datasets/h01_dhs_scan.py"
4 "/venv/brainstate/transform/_loop_collect_return.py"
5 "/repo/examples/pp_prop/h01_arc_model.py"

FunctionNames
1 "f_m_inf"
2 "where"
3 "level_step"
4 "scan"
5 "H01ArcModel.update"
6 "_fast_ind_exp_euler_step_selected"

FileLocations
1 {file_name_id=1 function_name_id=1 line=10 end_line=10 column=0 end_column=0}
2 {file_name_id=2 function_name_id=2 line=20 end_line=20 column=0 end_column=0}
3 {file_name_id=3 function_name_id=3 line=30 end_line=30 column=0 end_column=0}
4 {file_name_id=4 function_name_id=4 line=40 end_line=40 column=0 end_column=0}
5 {file_name_id=5 function_name_id=5 line=50 end_line=50 column=0 end_column=0}
6 {file_name_id=3 function_name_id=6 line=60 end_line=60 column=0 end_column=0}

StackFrames
1 {file_location_id=5 parent_frame_id=1}
2 {file_location_id=4 parent_frame_id=2}
3 {file_location_id=1 parent_frame_id=3}
4 {file_location_id=2 parent_frame_id=4}
5 {file_location_id=3 parent_frame_id=3}
6 {file_location_id=6 parent_frame_id=3}

%fused_gate (param_0: f64[10]) -> f64[10] {
  %param_0 = f64[10]{0} parameter(0)
  %exp.1 = f64[10]{0} exponential(%param_0), metadata={op_name="jit(x)/exp" stack_frame_id=4}
  ROOT %add.1 = f64[10]{0} add(%exp.1, %param_0), metadata={op_name="jit(x)/add" stack_frame_id=4}
}

%fused_solve (param_1: f64[10]) -> f64[10] {
  %param_1 = f64[10]{0} parameter(0)
  ROOT %scatter.1 = f64[10]{0} scatter(%param_1), metadata={op_name="jit(x)/scatter-add" stack_frame_id=5}
}

ENTRY %main (arg: f64[10]) -> f64[10] {
  %arg = f64[10]{0} parameter(0)
  %loop_gate_fusion = f64[10]{0} fusion(%arg), kind=kLoop, calls=%fused_gate, metadata={op_name="jit(x)/add" stack_frame_id=4}
  %input_scatter_fusion = f64[10]{0} fusion(%loop_gate_fusion), kind=kInput, calls=%fused_solve, metadata={op_name="jit(x)/scatter-add" stack_frame_id=5}
  %wrapped_exp = f64[10]{0} fusion(%input_scatter_fusion), kind=kLoop, calls=%missing, metadata={op_name="jit(x)/exp" stack_frame_id=6}
  ROOT %copy.3 = f64[10]{0} copy(%wrapped_exp), metadata={op_name="jit(x)/copy" stack_frame_id=2}
}
'''


def test_source_class_prefers_function_rules_over_file_rules():
    assert profile.source_class('/repo/braintrace/datasets/h01_dhs_scan.py::_fast_ind_exp_euler_step_selected') == 'channel'
    assert profile.source_class('/repo/braintrace/datasets/h01_dhs_scan.py::level_step') == 'axial-solve'
    assert profile.source_class('C:\\venv\\braincell\\network\\delivery.py::write_arrivals') == 'delivery'
    assert profile.source_class('/venv/saiunit/math/_fun_keep_unit.py::where') == 'other'


def test_parse_hlo_frames_applies_printed_parent_minus_one_rule():
    frames = profile.parse_hlo_frames(HLO)
    assert frames[1] == ('/repo/examples/pp_prop/h01_arc_model.py::H01ArcModel.update', 0)
    assert frames[4] == ('/venv/saiunit/math/_fun_keep_unit.py::where', 3)
    assert frames[6][1] == 2


def test_frame_class_walks_to_the_first_classified_parent():
    frames = profile.parse_hlo_frames(HLO)
    assert profile.frame_class(frames, 4) == ('channel', '/venv/saiunit/math/_fun_keep_unit.py::where')
    assert profile.frame_class(frames, 2) == ('loop-control', '/venv/brainstate/transform/_loop_collect_return.py::scan')
    assert profile.frame_class(frames, 99) == ('other', '')


def test_parse_hlo_kernels_classifies_fusions_by_majority_and_falls_back_to_own_frame():
    kernels = profile.parse_hlo_kernels(HLO)
    assert kernels['loop_gate_fusion'][0] == 'channel'
    assert kernels['input_scatter_fusion'][0] == 'axial-solve'
    assert kernels['wrapped_exp'][0] == 'channel'          # calls a missing computation: own frame 6
    assert kernels['copy.3'][0] == 'loop-control'
    assert 'arg' not in kernels


def test_parse_hlo_kernels_marks_command_buffers():
    text = HLO.replace('calls=%fused_gate', 'to_apply=%command_buffer_7')
    assert profile.parse_hlo_kernels(text)['loop_gate_fusion'] == ('command-buffer', 'command_buffer_7')


def test_kernel_class_handles_memcpy_suffixes_and_unknown_names():
    classes = {'loop_gate_fusion': ('channel', '')}
    assert profile.kernel_class('MemcpyD2D', {}, classes) == 'memcpy'
    assert profile.kernel_class('k', {'hlo_op': 'loop_gate_fusion.3'}, classes) == 'channel'
    assert profile.kernel_class('k', {'hlo_op': 'input_scatter_fusion.9'}, classes) == 'axial-solve'
    assert profile.kernel_class('k', {'hlo_op': 'while_cond'}, classes) == 'loop-control'
    assert profile.kernel_class('k', {'hlo_op': 'mystery'}, classes) == 'other'


def test_summarize_kernels_reports_shares_launches_and_idle_fraction():
    classes = {'gate': ('channel', ''), 'solve': ('axial-solve', '')}
    events = [('k1', 0, 1_000_000, {'hlo_op': 'gate'}), ('k2', 2_000_000, 1_000_000, {'hlo_op': 'solve'}),
              ('MemcpyD2D', 3_000_000, 1_000_000, {})]
    summary = profile.summarize_kernels(events, classes, executions=2, substeps=10)
    assert summary['launches_per_event'] == 1.5
    assert summary['launches_per_substep'] == pytest.approx(0.15)
    assert summary['busy_seconds_per_event'] == pytest.approx(1.5e-3)
    assert summary['span_seconds_per_event'] == pytest.approx(2e-3)
    assert summary['idle_fraction_of_span'] == pytest.approx(0.25)
    assert summary['classes']['channel']['share_of_busy'] == pytest.approx(1/3)
    assert summary['heaviest_ops'][0]['cls'] in ('channel', 'axial-solve', 'memcpy')


def test_summarize_kernels_with_no_events():
    summary = profile.summarize_kernels([], {}, executions=1)
    assert summary['classes'] == {} and summary['idle_fraction_of_span'] is None


def test_block_arithmetic_counts_initial_scoring_before_the_first_block():
    result = profile.block_arithmetic(2., 278.5, updates=128, fixed_seconds=300., cap_seconds=3600.,
                                      initial_events=116128, initial_seconds_per_event=1.)
    assert result['seconds_per_update'] == pytest.approx(557.)
    assert result['seconds_per_block'] == pytest.approx(557.*128)
    assert result['initial_scoring_seconds'] == pytest.approx(116128.)
    assert result['blocks_under_cap'] == 0.
    assert result['initial_scoring_fits_under_cap'] is False
    cheap = profile.block_arithmetic(.01, 100., updates=128, fixed_seconds=100., cap_seconds=3600.)
    assert cheap['blocks_under_cap'] == pytest.approx(3500./128)
    assert cheap['initial_scoring_fits_under_cap'] is True


def test_voltage_difference_uses_the_shared_window():
    reference = np.zeros((20, 3))
    other = np.ones((25, 3))*np.array([.1, .2, .3])
    result = profile.voltage_difference(reference, other)
    assert result['events'] == 20
    assert result['max_abs_mv'] == pytest.approx(.3)
    assert result['per_cell_max_abs_mv'] == pytest.approx([.1, .2, .3])
    with pytest.raises(ValueError):
        profile.voltage_difference(np.zeros((0, 3)), other)
    with pytest.raises(ValueError):
        profile.voltage_difference(np.zeros((2, 2)), other)


def test_corpus_denominators_counts_advancing_events_per_stage():
    def query(task, index, advancing):
        advances = np.zeros(705, dtype=bool)
        advances[:advancing] = True
        return SimpleNamespace(task_id=task, query_index=index, advances=advances)
    queries = [query('a', 0, 200), query('a', 1, 300), query('b', 0, 250)]
    entries = [SimpleNamespace(task_id='a', query_index=1), SimpleNamespace(task_id='b', query_index=0)]
    result = profile.corpus_denominators(queries, entries, ('a',))
    assert result['training_queries'] == 3 and result['training_tasks'] == 2
    assert result['corpus_advancing_events'] == 750
    assert result['corpus_padding_events'] == 3*705-750
    assert result['block_advancing_events'] == 550 and result['block_mean_advancing'] == 275.
    assert result['screen_queries'] == 2 and result['screen_advancing_events'] == 500
    assert result['stacked_events_bytes_float64'] == 3*705*441*8


def _arm(name, forward, learner=None, real=None, cells=17, dt=.000625, volts=None):
    report = dict(arm=name, cells=cells, compartments=100, settings=dict(dt_ms=dt, cell_index=None),
                  stages=dict(build_network=20., init_state=20., first_update_compile_and_run=40.),
                  forward=dict(seconds_per_event_median=forward, events=20), memory={},
                  learner=({'8': dict(seconds_per_event_naive=learner), 'seconds_per_event_slope': learner}
                           if learner is not None else {}),
                  real_update=(dict(seconds_per_advancing_event=real) if real is not None else None))
    return report, volts


def test_build_report_uses_real_update_when_present_and_skips_block_arithmetic_without_a_learner():
    corpus = dict(block_mean_advancing=278.5, corpus_advancing_events=116128, block_advancing_events=35648,
                  screen_advancing_events=770)
    arms = {'pinned': _arm('pinned', 1.0, learner=10., volts=np.zeros((20, 17))),
            'fast': _arm('fast', .1, learner=1., real=1.5, volts=np.full((20, 17), .5)),
            'fwd': _arm('fwd', .2), 'one': _arm('one', .01, cells=1, volts=np.zeros((400, 1)))}
    document = profile.build_report(arms, corpus, reference='pinned', cap_seconds=3600.)
    assert document['arithmetic']['pinned']['per_event_basis'] == 'truncated synthetic update slope'
    assert document['arithmetic']['pinned']['block_seconds'] == pytest.approx(356480.)
    assert document['arithmetic']['fast']['per_event_basis'] == 'real 705-event update_episode'
    assert document['arithmetic']['fast']['block_seconds'] == pytest.approx(1.5*35648)
    assert 'block_seconds' not in document['arithmetic']['fwd']
    assert document['arithmetic']['fwd']['corpus_scoring_fits_under_cap'] is False
    assert 'one' not in document['arithmetic'] and 'one' not in document['voltage_differences_vs_reference']
    assert document['voltage_differences_vs_reference']['fast']['max_abs_mv'] == pytest.approx(.5)
    assert 'fwd' not in document['voltage_differences_vs_reference']
