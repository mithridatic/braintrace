"""Real no-input physical waits, optionally carrying online eligibility."""

import brainstate
import jax.numpy as jnp
import numpy as np


def wait_event_count(duration_seconds):
    """Convert physical seconds to exact 0.1 ms H01 event intervals.

    Parameters
    ----------
    duration_seconds : float
        Nonnegative finite duration, divisible by the existing event clock.

    Returns
    -------
    int
        Number of silent physical events; biological rates are unchanged.
    """
    count = duration_seconds*10000.
    if not np.isfinite(count) or count < 0 or not np.isclose(count, round(count), rtol=0, atol=1e-8):
        raise ValueError('Physical wait must be a nonnegative multiple of 0.1 ms')
    return int(round(count))


class PhysicalWait(brainstate.nn.Module):
    """Carry a silent wait across bounded, event-aligned physical chunks.

    Parameters
    ----------
    model : H01ArcModel
        Actual cable model; each update advances exactly .1 ms.
    duration_seconds : float
        Physical duration. Matched experiment arms use 0, 1, 10 and 20 seconds.
    learner : object or None, optional
        Sparse online learner. Supply it while training so eligibility advances
        throughout the wait; no answer loss or optimizer update occurs here.

    Notes
    -----
    The cursor and model/learner states must be checkpointed together. Chunk
    boundaries occur between whole .1 ms events, not inside a cable substep.
    Calling this driver never resets model, queues, chemistry, RNG or traces.
    """

    def __init__(self, model, duration_seconds, *, learner=None):
        super().__init__()
        self.model, self.learner = model, learner
        self.total_events = wait_event_count(duration_seconds)
        self.elapsed_events = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    def update(self, *, max_events=100):
        """Advance at most the requested event count without output history.

        Parameters
        ----------
        max_events : int, optional
            Positive static bound for this compiled chunk, default 100 (10 ms).

        Returns
        -------
        scalar
            True when the requested physical wait is complete.
        """
        if type(max_events) is not int or max_events <= 0:
            raise ValueError('Chunk event bound must be a positive integer')
        event = jnp.zeros(441, dtype=self.model.input_weight.value.dtype)
        def step(_):
            def advance():
                if self.learner is None:
                    self.model.update(event)
                else:
                    self.learner(event)
                self.elapsed_events.value = self.elapsed_events.value+1
            brainstate.transform.cond(self.elapsed_events.value < self.total_events, advance, lambda: None)
        brainstate.transform.for_loop(step, jnp.arange(max_events))
        return self.elapsed_events.value == self.total_events
