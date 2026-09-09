"""Compiled ARC request loss and direct scoring on actual H01 cable state."""

import brainstate
import jax.numpy as jnp


def training_step(session, module):
    """Create the existing ARC request loss around sparse H01 pp-prop.

    Parameters
    ----------
    session : H01Session
        Actual model, learner and grouped Muon trainer.
    module : module
        Existing Example 21 ARC implementation.

    Returns
    -------
    callable
        Masked per-event request loss, using the unchanged ARC loss function.
    """
    def step(event, advance, request_kind, target_shape, target_rows, target_valid_mask):
        def active():
            session.learner(event)
            return module._supervised_request_loss(session.model.readout(), target_shape,
                target_rows, target_valid_mask, request_kind)
        return brainstate.transform.cond(advance, active, lambda: jnp.asarray(0., dtype=jnp.float32))
    return step


def update_episode(session, payload, module):
    """Apply one complete encoded ARC episode and grouped Muon update.

    Parameters
    ----------
    session : H01Session
        Trainable real-cell runtime.
    payload : dict
        Existing Example 21 encoded events, targets, request and padding masks.
    module : module
        Unchanged request-loss implementation and trainer.

    Returns
    -------
    tuple
        Episode loss and unclipped gradient norm.

    Notes
    -----
    Invoke inside BrainState jit or a compiled episode loop. Event stepping is
    owned by the existing finite-window eligibility sequence driver.
    """
    session.trainer.reset_episode()
    payload = dict(payload, events=jnp.asarray(payload['events'], dtype=jnp.float64))
    return session.trainer.update_episode(**payload, step_fn=training_step(session, module))


def score_episode(session, events, advances):
    """Directly score an encoded episode with unchanged 31-request readout.

    Parameters
    ----------
    session : H01Session
        Physical model; parameters and optimizer are not updated.
    events, advances : arrays
        ARC events and episode padding mask, with requests at the end.

    Returns
    -------
    tuple
        The final 31 readout vectors and mean absolute normalized soma activity.
        Activity is a voltage statistic, not a spike count.
    """
    session.model.reset_episode(session.learner)
    def step(event, advance):
        voltage = session.model.step(event, advance)
        features = jnp.tanh((voltage+65.)/20.)
        activity = jnp.where(advance, jnp.abs(features), jnp.zeros_like(features))
        return features @ session.model.readout_weight.value + session.model.readout_bias.value, activity
    logits, activity = brainstate.transform.for_loop(step, jnp.asarray(events, dtype=jnp.float64), advances)
    return logits[-31:], jnp.sum(activity, axis=0)/jnp.maximum(jnp.sum(advances), 1)
