"""Finite-window pp-prop versus BPTT recurrent-gradient evidence."""

from __future__ import annotations

from dataclasses import replace

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from temporal_benchmark_config import HORIZONS, TemporalBenchmarkConfig
from temporal_benchmark_data import response_mask
from temporal_benchmark_manifest import SeedBundle
from temporal_benchmark_metrics import gradient_comparison
from temporal_benchmark_training import (
    _Runtime,
    _bptt_gradients,
    _build_runtime,
    _copy_parameters,
    _group_gradients,
    _make_train_many,
    _pp_prop_gradients,
    _reset,
    _training_batches,
    _tree_norm,
)

REFERENCE_NEURONS = 24
REFERENCE_DEGREE = 4
SMALL_UPDATE_RATIO = 0.01


def _flatten(tree) -> np.ndarray:
    leaves = [
        np.asarray(u.get_mantissa(leaf)).reshape(-1) for leaf in jax.tree.leaves(tree)
    ]
    return np.concatenate(leaves)


def _apply_parameter_step(runtime: _Runtime, gradients, update_ratio: float) -> None:
    weights = runtime.model.states(brainstate.ParamState)
    weight_norm = float(_tree_norm(weights.to_dict_values()))
    update_norm = update_ratio * weight_norm
    scale = update_norm / max(float(_tree_norm(gradients)), np.finfo(float).tiny)
    for path, state in weights.items():
        if path not in gradients:
            continue
        state.value = jax.tree.map(
            lambda value, gradient: value - scale * gradient,
            state.value,
            gradients[path],
        )


def _loss_change(config, bundle, source, spikes, labels, gradients) -> float:
    oracle_config = replace(config, arm="all_bptt")
    oracle = _build_runtime(oracle_config, bundle)
    _copy_parameters(source.model, oracle.model)
    mask = jnp.asarray(response_mask(HORIZONS[config.horizon]))
    _reset(oracle)
    _, before = _bptt_gradients(oracle, spikes, labels, mask)
    _apply_parameter_step(oracle, gradients, SMALL_UPDATE_RATIO)
    _reset(oracle)
    _, after = _bptt_gradients(oracle, spikes, labels, mask)
    return float(after - before)


def _probe(runtime, config, bundle, spikes, labels, seed) -> dict[str, float]:
    mask = jnp.asarray(response_mask(HORIZONS[config.horizon]))
    _reset(runtime)
    pp_gradients, pp_loss = _pp_prop_gradients(runtime, spikes, labels, mask)
    oracle_config = replace(config, arm="all_bptt")
    oracle = _build_runtime(oracle_config, bundle)
    _copy_parameters(runtime.model, oracle.model)
    _reset(oracle)
    bptt_gradients, bptt_loss = _bptt_gradients(oracle, spikes, labels, mask)
    pp_recurrent = _group_gradients(pp_gradients, runtime.groups["recurrent"])
    bptt_recurrent = _group_gradients(bptt_gradients, oracle.groups["recurrent"])
    pp_vector = _flatten(pp_recurrent)
    bptt_vector = _flatten(bptt_recurrent)
    comparison = gradient_comparison(pp_vector, bptt_vector)
    permuted = np.random.default_rng(seed).permutation(pp_vector)
    null_cosine = gradient_comparison(permuted, bptt_vector)["cosine_similarity"]
    return {
        **comparison,
        "pp_prop_loss": float(pp_loss),
        "bptt_loss": float(bptt_loss),
        "permuted_null_cosine_similarity": null_cosine,
        "cosine_advantage_over_permuted_null": comparison["cosine_similarity"]
        - null_cosine,
        "pp_prop_small_update_loss_change": _loss_change(
            config, bundle, runtime, spikes, labels, pp_gradients
        ),
        "bptt_small_update_loss_change": _loss_change(
            config, bundle, runtime, spikes, labels, bptt_gradients
        ),
    }


def _reference_config(config: TemporalBenchmarkConfig) -> TemporalBenchmarkConfig:
    if config.updates < 4 or config.updates % 4:
        raise ValueError(
            "Gradient-evidence updates must be a positive multiple of four. Set Gradient-evidence updates to a positive multiple of four."
        )
    return replace(
        config,
        arm="all_pp_prop",
        neurons=REFERENCE_NEURONS,
        degree=REFERENCE_DEGREE,
        curriculum=False,
    )


def run_gradient_evidence(
    config: TemporalBenchmarkConfig, bundle: SeedBundle
) -> dict[str, object]:
    """Measure finite-window gradient evidence at 0%, 25%, and 75% training."""
    reference = _reference_config(config)
    checkpoints = (0, reference.updates // 4, 3 * reference.updates // 4)
    with brainstate.environ.context(dt=reference.dt_seconds * u.second):
        runtime = _build_runtime(reference, bundle)
        spikes, labels = _training_batches(reference, bundle)
        trainer = _make_train_many(runtime, reference)
        probes: list[dict[str, object]] = []
        completed = 0
        for checkpoint in checkpoints:
            if checkpoint > completed:
                losses, _ = trainer(
                    spikes[completed:checkpoint], labels[completed:checkpoint]
                )
                jax.block_until_ready(losses)
                completed = checkpoint
            evidence = _probe(
                runtime,
                reference,
                bundle,
                spikes[checkpoint],
                labels[checkpoint],
                bundle.weight_seed + checkpoint,
            )
            probes.append({"update": checkpoint, **evidence})
    return {
        "reference_neurons": REFERENCE_NEURONS,
        "reference_degree": REFERENCE_DEGREE,
        "vjp_method": "single-step",
        "oracle": "full_window_reverse_mode_bptt",
        "small_update_to_weight_ratio": SMALL_UPDATE_RATIO,
        "probes": probes,
    }
