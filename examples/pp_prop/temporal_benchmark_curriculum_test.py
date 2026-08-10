"""Tests for checkpointed curriculum promotion."""

from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp

import temporal_benchmark_curriculum as curriculum
from temporal_benchmark_config import (
    CurriculumTraceHalfLives,
    TemporalBenchmarkConfig,
    TraceHalfLives,
)


@dataclass(frozen=True)
class _RuntimeStub:
    model: object
    learner: object
    topology: object


def _stable_dynamics() -> dict[str, float]:
    return {
        "mean_firing_spikes_per_neuron_step": 0.1,
        "silent_neuron_fraction": 0.0,
        "saturated_neuron_fraction": 0.0,
    }


def test_phase_promotes_only_after_two_stable_checkpoints(monkeypatch) -> None:
    config = SimpleNamespace(updates=3, evaluation_interval=1)
    spikes = jnp.zeros((3, 1, 1, 1))
    labels = jnp.zeros((3, 1), dtype=jnp.int32)
    accuracies = iter((0.79, 0.80, 0.81))
    monkeypatch.setattr(curriculum, "_training_batches", lambda *_: (spikes, labels))
    monkeypatch.setattr(
        curriculum,
        "_make_train_many",
        lambda *_: (
            lambda x, _y: (
                jnp.zeros(x.shape[0]),
                {"recurrent": jnp.zeros((x.shape[0], 9))},
            )
        ),
    )
    monkeypatch.setattr(
        curriculum,
        "_evaluate",
        lambda *_: (
            {"ensemble_accuracy": next(accuracies), "ensemble_nll": 0.5},
            _stable_dynamics(),
        ),
    )

    result = curriculum._run_phase(object(), config, object())

    assert result["promoted"] is True
    assert result["updates_completed"] == 3
    assert [entry["stability_passed"] for entry in result["history"]] == [
        True,
        True,
        True,
    ]


def test_curriculum_constants_match_sealed_phase_budgets() -> None:
    assert curriculum.CURRICULUM_PHASES == (
        ("short", 200),
        ("medium", 400),
        ("long", 800),
    )


def test_each_curriculum_phase_uses_its_independent_trace_pair(monkeypatch) -> None:
    selected = CurriculumTraceHalfLives(
        short=TraceHalfLives(5.0, 10.0),
        medium=TraceHalfLives(30.0, 10.0),
        long=TraceHalfLives(100.0, 30.0),
    )
    config = TemporalBenchmarkConfig(
        curriculum_trace_half_lives=selected,
        evaluation_interval=1,
        device="cpu",
    )
    observed: list[tuple[str, float, float]] = []
    runtime = _RuntimeStub(model=object(), learner=object(), topology=object())
    telemetry = {"recurrent": jnp.zeros((1, 9))}

    monkeypatch.setattr(curriculum, "_build_runtime", lambda *_: runtime)
    monkeypatch.setattr(curriculum, "_compile_learner", lambda *_: object())
    monkeypatch.setattr(
        curriculum,
        "_run_phase",
        lambda _runtime, phase_config, _bundle: (
            observed.append(
                (
                    phase_config.horizon,
                    phase_config.trace_half_life_x_steps,
                    phase_config.trace_half_life_f_steps,
                )
            )
            or {
                "promoted": True,
                "updates_completed": phase_config.updates,
                "history": [],
                "losses": [],
                "telemetry": telemetry,
            }
        ),
    )
    monkeypatch.setattr(
        curriculum,
        "_evaluate",
        lambda *_: ({"ensemble_accuracy": 1.0}, _stable_dynamics()),
    )
    monkeypatch.setattr(curriculum, "topology_metrics", lambda *_: {})
    bundle = SimpleNamespace(bundle_id="bundle")

    result = curriculum._execute_curriculum(config, bundle)

    assert observed == [
        ("short", 5.0, 10.0),
        ("medium", 30.0, 10.0),
        ("long", 100.0, 30.0),
    ]
    assert result["trace_half_lives"] == {
        "short": {"x": 5.0, "f": 10.0},
        "medium": {"x": 30.0, "f": 10.0},
        "long": {"x": 100.0, "f": 30.0},
    }
    assert result["trace_decays"]["medium"] == {
        "x": curriculum.half_life_decay(30.0),
        "f": curriculum.half_life_decay(10.0),
    }
