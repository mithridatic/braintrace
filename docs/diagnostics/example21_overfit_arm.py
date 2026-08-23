#!/Usr/bin/env python3
"""Step B: can Example 21's pipeline learn one ARC task, and does effort help?

Overfits a single ARC task through the production training path: the same
``learner.etrace_grad`` call, the same Adam optimizer, the same clipping, the
same terminal-mask supervision. The only things varied are the update count,
the learning rate, and the terminal effort.

Reads
-----
loss -> ~0 on both arms
    The pipeline learns; the 96-update budget was the whole story.
Loss -> ~0 but effort 32 about equals effort 0
    The recurrence is decorative: a second, separate defect.
Loss stuck near 9.1 at an adequate rate
    Something else is broken; return to Phase 1.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import msgspec
import pathlib
import sys

import numpy as np

REPO_ARG = "--worktree"


def load_example(worktree: pathlib.Path):
    sys.path.insert(0, str(worktree / "examples" / "pp_prop"))
    sys.path.insert(0, str(worktree))
    spec = importlib.util.spec_from_file_location(
        "ex21", worktree / "examples" / "pp_prop" / "21-latent-reasoning-in-context.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ex21"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(REPO_ARG, type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--task-id", default="007bbfb7")
    parser.add_argument("--effort", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--sparse-backend", default="jax_raw")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ex21 = load_example(args.worktree)

    import brainstate
    import braintools
    import jax
    import jax.numpy as jnp

    config = ex21.ExperimentConfig(
        source_manifest=args.manifest,
        output_dir=pathlib.Path("var/example21-overfit"),
        device="cpu",
        learning_rate=args.learning_rate,
    )
    row_config = ex21._row_config(config)
    data = ex21._load_data(config)
    origins = [o for o in data.training if o.task.task_id == args.task_id]
    if not origins:
        raise SystemExit(f"task {args.task_id} not found in the train split")
    task = origins[0].task

    # No augmentation: this is a memorisation test, the target must be fixed.
    encoded = ex21.encode_query_episode(task, 0, row_config)
    sequence = ex21._packed_events(encoded, config)
    advances = ex21._packed_advances(encoded, config)
    target = encoded.target
    padded = np.zeros((30, 30), dtype=np.int32)
    padded[: target.height, : target.width] = target.as_array()

    terminal = encoded.query_stop - 1 + args.effort
    if terminal >= sequence.shape[0]:
        raise SystemExit("effort exceeds packed sequence capacity")
    mask = np.zeros((sequence.shape[0],), dtype=np.float32)
    mask[terminal] = 1.0

    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        model_config = dataclasses.replace(
            ex21._model_config(config, row_config, batch_size=1),
            sparse_backend=args.sparse_backend,
        )
        with brainstate.random.seed_context(config.seed):
            model = ex21.LatentWorkspaceModel(model_config)
        learner = ex21.compile_pp_prop(model)
        rank = model.config.color_rank
        optimizer = braintools.optim.Adam(lr=config.learning_rate)
        optimizer.register_trainable_weights(learner.param_states)
        before = ex21.parameter_snapshot(model)

        heights = jnp.asarray([target.height], dtype=jnp.int32)
        widths = jnp.asarray([target.width], dtype=jnp.int32)
        colors = jnp.asarray(padded[None])
        events = jnp.asarray(sequence[:, None, :])
        gates = jnp.asarray(advances[:, None])
        mask_array = jnp.asarray(mask)

        def step_loss(event, advance_gate):
            compact = learner(event, advance_gate)
            return jnp.mean(
                ex21.arc_loss_per_example(
                    compact, heights, widths, colors, color_rank=rank
                )
            )

        @brainstate.transform.jit
        def one_update(_):
            model.reset_state()
            learner.reset_state(batch_size=1)
            gradients, objective = learner.etrace_grad(
                events,
                gates,
                step_fn=step_loss,
                mask=mask_array,
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            optimizer.update(brainstate.nn.clip_grad_norm(gradients, config.clip_norm))
            return objective

        losses = brainstate.transform.for_loop(
            one_update, jnp.arange(args.updates, dtype=jnp.int32)
        )
        losses = np.asarray(losses, dtype=np.float64)
        after = ex21.parameter_snapshot(model)

    changes = ex21._parameter_change_evidence(before, after)
    record = {
        "task_id": args.task_id,
        "effort": args.effort,
        "learning_rate": args.learning_rate,
        "updates": args.updates,
        "sparse_backend": args.sparse_backend,
        "target_shape": [int(target.height), int(target.width)],
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "loss_min": float(losses.min()),
        "loss_mean_last_50": float(losses[-50:].mean()),
        "loss_curve_every_100": [float(x) for x in losses[::100]],
        "parameter_changes": {
            key: {"l2_delta": value["l2_delta"]} for key, value in changes.items()
        },
    }
    args.output.write_text(msgspec.json.format(msgspec.json.encode(record), indent=1).decode())
    print(
        f"effort={args.effort} lr={args.learning_rate:g} updates={args.updates} "
        f"loss {record['loss_first']:.4f} -> {record['loss_last']:.4f} "
        f"(min {record['loss_min']:.4f}, mean last 50 {record['loss_mean_last_50']:.4f})"
    )


if __name__ == "__main__":
    main()
