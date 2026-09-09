"""Initialize a BrainCell network cell by cell with progress lines and a heartbeat.

``braincell.network.Network.init_state`` exposes no progress hook: it loops over
``network.populations`` and calls each ``Cell.init_state`` once. Both are public
and ``Network.init_state``/``Network.run`` skip already-initialized cells, so
this module performs the same loop while emitting one timestamped line per
population and a heartbeat (elapsed seconds, resident set size) while any single
call is silent. Nothing here alters the model or numerical settings.
"""

from contextlib import contextmanager
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a validation dependency
    psutil = None


def process_rss_mb(peak=False):
    """Return the current process resident set size in MiB.

    Parameters
    ----------
    peak : bool, optional
        Return the process-lifetime peak where the platform reports one
        (``peak_wset`` on Windows); otherwise the current value.

    Returns
    -------
    float or None
        Resident set size in MiB, or ``None`` when ``psutil`` is unavailable.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace.datasets.h01_network_init import process_rss_mb
        >>> process_rss_mb() > 0
        True
    """
    if psutil is None:
        return None
    info = psutil.Process().memory_info()
    value = getattr(info, "peak_wset", None) if peak else None
    return (info.rss if value is None else value)/2.**20


@contextmanager
def heartbeat(label, emit, *, seconds=60.):
    """Emit ``label`` heartbeat lines every ``seconds`` while the block runs.

    Parameters
    ----------
    label : str
        Names the silent call in each heartbeat line.
    emit : callable
        Receives each heartbeat message (a progress line for a silence watchdog).
    seconds : float, optional
        Heartbeat period; must be positive.

    Yields
    ------
    dict
        ``beats`` counts the heartbeats emitted so far.

    Examples
    --------
    .. code-block:: python

        >>> import time
        >>> from braintrace.datasets.h01_network_init import heartbeat
        >>> with heartbeat("sleep", print, seconds=.05) as state:
        ...     time.sleep(.12)
        heartbeat: sleep elapsed 0 s, RSS ... MB
        heartbeat: sleep elapsed 0 s, RSS ... MB
        >>> state["beats"] >= 1
        True
    """
    if not seconds > 0.:
        raise ValueError("Heartbeat period must be positive.")
    stop, state, started = threading.Event(), {"beats": 0}, time.perf_counter()

    def beat():
        while not stop.wait(seconds):
            rss = process_rss_mb()
            memory = "unknown" if rss is None else f"{rss:.0f}"
            state["beats"] += 1
            emit(f"heartbeat: {label} elapsed {time.perf_counter()-started:.0f} s, RSS {memory} MB")

    thread = threading.Thread(target=beat, name="h01-heartbeat-"+label, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        stop.set()
        thread.join()


def init_h01_network_states(network, *, progress=None, heartbeat_seconds=60.):
    """Initialize every population cell, one progress line per cell.

    Parameters
    ----------
    network : braincell.Network
        Constructed network; cells already initialized are skipped, exactly as
        ``Network.init_state`` does.
    progress : callable, optional
        Receives one line before and after each cell and the heartbeat lines.
    heartbeat_seconds : float, optional
        Heartbeat period while a single ``Cell.init_state`` call is silent.

    Returns
    -------
    dict
        ``init_state_seconds`` (all cells), ``init_seconds_by_population``,
        ``initialized_populations`` (names initialized here) and
        ``peak_rss_mb`` (process peak where reported, else current RSS).

    Examples
    --------
    .. code-block:: python

        >>> from types import SimpleNamespace
        >>> from braintrace.datasets.h01_network_init import init_h01_network_states
        >>> cell = SimpleNamespace(_initialized=False, n_cv=3, init_state=lambda batch_size=None: None)
        >>> network = SimpleNamespace(populations={"cell_1": SimpleNamespace(cell=cell)}, init_state=lambda: None)
        >>> init_h01_network_states(network, progress=print)["initialized_populations"]
        Initializing cell_1 (1/1, 3 compartments)
        Initialized cell_1 in 0.0 s
        ['cell_1']
    """
    emit = progress if progress is not None else lambda message: None
    started, seconds, names = time.perf_counter(), {}, list(network.populations)
    for index, name in enumerate(names, start=1):
        cell = network.populations[name].cell
        if getattr(cell, "_initialized", False):
            continue
        emit(f"Initializing {name} ({index}/{len(names)}, {getattr(cell, 'n_cv', 'unknown')} compartments)")
        cell_started = time.perf_counter()
        with heartbeat("init_state "+name, emit, seconds=heartbeat_seconds):
            cell.init_state()
        seconds[name] = time.perf_counter()-cell_started
        emit(f"Initialized {name} in {seconds[name]:.1f} s")
    network.init_state()
    import gc
    gc.collect()
    return dict(init_state_seconds=time.perf_counter()-started, init_seconds_by_population=seconds,
                initialized_populations=list(seconds), peak_rss_mb=process_rss_mb(peak=True))
