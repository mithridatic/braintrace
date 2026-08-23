from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pp_prop.latent_workspace_protocol import (
    ArmStream,
    StepGates,
    build_batched_protocol_v2_arm,
    build_protocol_v2_arm,
    normalized_episode_weights,
)


def test_effort_zero_has_a_complete_equal_decoder_sweep() -> None:
    context = np.zeros((4, 2, 8), dtype=np.float32)
    context[:, :, 0] = 1.0
    arm = build_protocol_v2_arm(
        context,
        query_start=2,
        query_stop=4,
        decoder_rows=30,
        efforts=(0, 30, 60),
    )

    assert isinstance(arm, ArmStream)
    decode = np.asarray(arm.gates.decode_row)
    assert decode.sum(axis=0).tolist() == [90, 90]
    for effort in (0, 30, 60):
        start = arm.boundaries[f"decode_r{effort}_start"]
        stop = arm.boundaries[f"decode_r{effort}_stop"]
        assert stop - start == 30
        assert decode[start:stop].all()
    assert arm.metadata["candidate_policy"] == (
        "latest_checkpoint_factorized_global_top2_v2"
    )


def test_no_context_preserves_timing_and_disables_prequery_semantic_work() -> None:
    context = np.ones((5, 1, 6), dtype=np.float32)
    intact = build_protocol_v2_arm(
        context, query_start=3, query_stop=5, decoder_rows=30
    )
    no_context = build_protocol_v2_arm(
        context,
        query_start=3,
        query_stop=5,
        decoder_rows=30,
        no_context=True,
    )

    assert intact.boundaries == no_context.boundaries
    np.testing.assert_array_equal(
        intact.gates.advance_physics, no_context.gates.advance_physics
    )
    assert not np.asarray(no_context.events[:3]).any()
    assert not np.asarray(no_context.gates.latent_update[:3]).any()
    assert not np.asarray(no_context.gates.decode_row[:3]).any()
    assert not np.asarray(no_context.gates.answer_feedback[:3]).any()


def test_step_gates_reject_mismatched_shapes_and_answer_feedback() -> None:
    zeros = np.zeros((3, 2), dtype=np.bool_)
    with pytest.raises(ValueError, match="same shape"):
        StepGates(
            advance_physics=zeros,
            latent_update=zeros[:, :1],
            decode_row=zeros,
            answer_feedback=zeros,
            recurrent_enabled=zeros,
        )
    feedback = zeros.copy()
    feedback[0, 0] = True
    with pytest.raises(ValueError, match="answer feedback"):
        StepGates(
            advance_physics=zeros,
            latent_update=zeros,
            decode_row=zeros,
            answer_feedback=feedback,
            recurrent_enabled=zeros,
        )


def test_step_gates_reject_latent_without_physics_and_decoder_overlap() -> None:
    zeros = np.zeros((2, 1), dtype=np.bool_)
    latent = zeros.copy()
    latent[0, 0] = True
    with pytest.raises(ValueError, match="latent_update"):
        StepGates(zeros, latent, zeros, zeros, zeros)

    physics = zeros.copy()
    physics[0, 0] = True
    with pytest.raises(ValueError, match="(?i)decoder rows"):
        StepGates(physics, zeros, physics, zeros, zeros)


@pytest.mark.parametrize(
    ("events", "query_start", "query_stop", "decoder_rows", "efforts", "control", "message"),
    [
        (np.zeros((0, 1, 1)), 0, 0, 1, (0,), "intact", "nonempty"),
        (np.zeros((2, 1, 1)), 2, 2, 1, (0,), "intact", "query boundaries"),
        (np.zeros((2, 1, 1)), 0, 1, 0, (0,), "intact", "positive integer"),
        (np.zeros((2, 1, 1)), 0, 1, 1, (0,), "invalid", "protocol-v2 control"),
        (np.zeros((2, 1, 1)), 0, 1, 1, (30, 0), "intact", "begin at zero"),
    ],
)
def test_protocol_arm_rejects_malformed_inputs(
    events: np.ndarray,
    query_start: int,
    query_stop: int,
    decoder_rows: int,
    efforts: tuple[int, ...],
    control: str,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=f"(?i){message}"):
        build_protocol_v2_arm(
            events,
            query_start=query_start,
            query_stop=query_stop,
            decoder_rows=decoder_rows,
            efforts=efforts,
            control=control,  # type: ignore[arg-type]
        )


def test_arm_stream_validates_boundaries_and_metadata() -> None:
    events = np.zeros((2, 1, 1), dtype=np.float32)
    gates = StepGates(*(np.zeros((2, 1), dtype=np.bool_) for _ in range(5)))
    for boundaries, metadata, message in (
        ({}, {}, "nonempty"),
        ({"bad": 3, "total_steps": 2}, {}, "within"),
        ({"bad": 1, "total_steps": 2}, None, "mapping"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            ArmStream(events, gates, boundaries, metadata)  # type: ignore[arg-type]


def test_batched_protocol_rejects_invalid_boundaries_and_controls() -> None:
    events = np.zeros((3, 2, 1), dtype=np.float32)
    advance = np.ones((3, 2), dtype=np.bool_)
    with pytest.raises(ValueError, match="batches"):
        build_batched_protocol_v2_arm(events, advance[:, :1], np.array([2, 2]))
    with pytest.raises(ValueError, match="one integer"):
        build_batched_protocol_v2_arm(events, advance, np.array([2.0, 2.0]))
    with pytest.raises(ValueError, match="inside"):
        build_batched_protocol_v2_arm(events, advance, np.array([0, 4]))
    with pytest.raises(ValueError, match="protocol-v2 control"):
        build_batched_protocol_v2_arm(
            events, advance, np.array([2, 2]), control="invalid"  # type: ignore[arg-type]
        )


def test_normalized_episode_weights_reject_empty_and_zero_episodes() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        normalized_episode_weights(np.zeros((0, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="at least one"):
        normalized_episode_weights(np.zeros((2, 2), dtype=np.bool_))


def test_episode_weights_give_unequal_shapes_equal_batch_contribution() -> None:
    valid = np.zeros((2, 30, 30), dtype=np.bool_)
    valid[0, :1, :1] = True
    valid[1, :2, :3] = True

    weights = np.asarray(normalized_episode_weights(valid))

    np.testing.assert_allclose(weights.sum(axis=(1, 2)), (0.5, 0.5))
    assert weights[0, 0, 0] == pytest.approx(0.5)
    assert weights[1, 0, 0] == pytest.approx(1.0 / 12.0)


def test_weighted_batch_gradient_is_mean_of_independent_episode_gradients() -> None:
    valid = jnp.asarray(
        [
            [[True, False], [False, False]],
            [[True, True], [True, False]],
        ]
    )
    inputs = jnp.asarray(
        [[[2.0, 0.0], [0.0, 0.0]], [[1.0, 3.0], [4.0, 0.0]]]
    )
    targets = jnp.asarray(
        [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 2.0], [1.0, 0.0]]]
    )

    def batch_loss(parameter: jax.Array) -> jax.Array:
        error = jnp.square(parameter * inputs - targets)
        return jnp.sum(normalized_episode_weights(valid) * error)

    def episode_loss(parameter: jax.Array, index: int) -> jax.Array:
        error = jnp.square(parameter * inputs[index] - targets[index])
        weights = normalized_episode_weights(valid[index : index + 1])
        return jnp.sum(weights[0] * error)

    parameter = jnp.asarray(0.75)
    batched = jax.grad(batch_loss)(parameter)
    independent = jnp.mean(
        jnp.stack(
            [jax.grad(episode_loss, argnums=0)(parameter, index) for index in range(2)]
        )
    )
    np.testing.assert_allclose(batched, independent, rtol=1e-6, atol=1e-6)


def test_batched_arm_ignores_legacy_tail_length_disagreement() -> None:
    events = np.zeros((95, 2, 8), dtype=np.float32)
    advances = np.ones((65, 2), dtype=np.bool_)

    arm = build_batched_protocol_v2_arm(events, advances, np.array([4, 5]))

    assert arm.events.shape == (155, 2, 8)
    np.testing.assert_array_equal(arm.gates.advance_physics[:4], True)
