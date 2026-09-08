"""Tests for the per-cell network initialization progress and heartbeat."""

import time
from types import SimpleNamespace

import pytest

from . import h01_network_init as mod
from .h01_network_init import heartbeat, init_h01_network_states, process_rss_mb


class _Cell:
    def __init__(self, n_cv, delay=0., initialized=False):
        self.n_cv, self.delay, self._initialized, self.calls = n_cv, delay, initialized, 0

    def init_state(self, batch_size=None):
        del batch_size
        time.sleep(self.delay)
        self.calls += 1
        self._initialized = True


def _network(*cells):
    populations = {f"cell_{i}": SimpleNamespace(cell=c) for i, c in enumerate(cells)}
    net = SimpleNamespace(populations=populations, network_init_calls=0)
    net.init_state = lambda: setattr(net, "network_init_calls", net.network_init_calls+1)
    return net


def test_rss_is_positive_and_peak_at_least_current():
    current, peak = process_rss_mb(), process_rss_mb(peak=True)
    assert current > 0. and peak >= current*0.5


def test_rss_none_without_psutil(monkeypatch):
    monkeypatch.setattr(mod, "psutil", None)
    assert process_rss_mb() is None


def test_heartbeat_emits_while_block_is_silent_and_stops_after():
    lines = []
    with heartbeat("slow", lines.append, seconds=.02) as state:
        time.sleep(.15)
    count = len(lines)
    assert count >= 2 and state["beats"] == count
    assert all(line.startswith("heartbeat: slow elapsed ") and line.endswith(" MB") for line in lines)
    time.sleep(.06)
    assert len(lines) == count


def test_heartbeat_reports_unknown_rss_without_psutil(monkeypatch):
    monkeypatch.setattr(mod, "psutil", None)
    lines = []
    with heartbeat("x", lines.append, seconds=.01):
        time.sleep(.08)
    assert lines and all("RSS unknown MB" in line for line in lines)


def test_heartbeat_silent_for_fast_block_and_rejects_bad_period():
    lines = []
    with heartbeat("fast", lines.append, seconds=10.):
        pass
    assert lines == []
    with pytest.raises(ValueError):
        with heartbeat("bad", lines.append, seconds=0.):
            pass


def test_init_emits_per_cell_lines_skips_initialized_and_calls_network_init():
    a, b, c = _Cell(5), _Cell(7, initialized=True), _Cell(9)
    net, lines = _network(a, b, c), []
    result = init_h01_network_states(net, progress=lines.append, heartbeat_seconds=60.)
    assert (a.calls, b.calls, c.calls) == (1, 0, 1)
    assert net.network_init_calls == 1
    assert lines[0] == "Initializing cell_0 (1/3, 5 compartments)"
    assert lines[1].startswith("Initialized cell_0 in ")
    assert lines[2] == "Initializing cell_2 (3/3, 9 compartments)"
    assert len(lines) == 4
    assert result["initialized_populations"] == ["cell_0", "cell_2"]
    assert set(result["init_seconds_by_population"]) == {"cell_0", "cell_2"}
    assert result["init_state_seconds"] >= sum(result["init_seconds_by_population"].values())
    assert result["peak_rss_mb"] > 0.


def test_init_heartbeat_fires_inside_a_slow_cell():
    net, lines = _network(_Cell(3, delay=.12)), []
    init_h01_network_states(net, progress=lines.append, heartbeat_seconds=.03)
    beats = [line for line in lines if line.startswith("heartbeat: init_state cell_0 ")]
    assert len(beats) >= 2
    assert lines[-1].startswith("Initialized cell_0 in 0.1")


def test_init_without_progress_is_silent_and_still_initializes():
    cell = SimpleNamespace(init_state=lambda batch_size=None: None)
    net = _network(cell)
    result = init_h01_network_states(net)
    assert result["initialized_populations"] == ["cell_0"] and net.network_init_calls == 1
