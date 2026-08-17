#!/usr/bin/env python3
"""Step A: does the ARC grid loss reach the recurrent synapses at all?

Builds one real ARC training episode through Example 21's own production path,
calls the same ``learner.etrace_grad`` the training loop calls, and reports the
gradient L2 norm per parameter group at several effort levels.

H2 is confirmed if ``rec_syn.comm.weight`` receives an identically zero gradient,
or if its gradient is invariant to effort.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO = pathlib.Path(__file__).resolve()
WORKTREE = pathlib.Path(sys.argv[1])
MANIFEST = pathlib.Path(sys.argv[2])
EFFORTS = (0, 8, 16, 32)

sys.path.insert(0, str(WORKTREE / "examples" / "pp_prop"))
sys.path.insert(0, str(WORKTREE))

spec = importlib.util.spec_from_file_location(
    "ex21", WORKTREE / "examples" / "pp_prop" / "21-latent-reasoning-in-context.py"
)
assert spec is not None and spec.loader is not None
ex21 = importlib.util.module_from_spec(spec)
sys.modules["ex21"] = ex21
spec.loader.exec_module(ex21)

# Pull these from ex21's own namespace: it may have imported them under the
# ``examples.pp_prop.*`` package path, and a top-level import would create a
# second, non-identical class object that fails ex21's isinstance checks.
LatentWorkspaceModel = ex21.LatentWorkspaceModel
arc_loss_per_example = ex21.arc_loss_per_example
compile_pp_prop = ex21.compile_pp_prop
encode_query_episode = ex21.encode_query_episode


def pick_task(data) -> object:
    """Return the first training task whose query output shape differs from input."""
    for origin in data.training:
        pair = origin.task.test[0]
        if (
            pair.output is not None
            and pair.output.as_array().shape != pair.input.as_array().shape
        ):
            return origin
    return data.training[0]


def main() -> None:
    config = ex21.ExperimentConfig(
        source_manifest=MANIFEST,
        output_dir=pathlib.Path("var/example21-grad-probe"),
        device="cpu",
    )
    row_config = ex21._row_config(config)
    data = ex21._load_data(config)
    origin = pick_task(data)
    task = origin.task
    print(f"task={task.task_id} demos={len(task.train)} queries={len(task.test)}")
    print(
        f"query input shape={task.test[0].input.as_array().shape}"
        f" output shape={task.test[0].output.as_array().shape}"
    )

    encoded = encode_query_episode(task, 0, row_config)
    sequence = ex21._packed_events(encoded, config)
    advances = ex21._packed_advances(encoded, config)
    target = encoded.target
    padded = np.zeros((30, 30), dtype=np.int32)
    padded[: target.height, : target.width] = target.as_array()

    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        # The container image ships no numba, so the CPU sparse kernels must use
        # the pure-JAX path. ModelConfig already exposes this knob.
        import brainstate

        model_config = dataclasses.replace(
            ex21._model_config(config, row_config, batch_size=1),
            sparse_backend="jax_raw",
        )
        with brainstate.random.seed_context(config.seed):
            model = LatentWorkspaceModel(model_config)
        learner = compile_pp_prop(model)
        rank = model.config.color_rank

        heights = jnp.asarray([target.height], dtype=jnp.int32)
        widths = jnp.asarray([target.width], dtype=jnp.int32)
        colors = jnp.asarray(padded[None])

        def step_loss(event, advance_gate):
            compact = learner(event, advance_gate)
            return jnp.mean(
                arc_loss_per_example(compact, heights, widths, colors, color_rank=rank)
            )

        rows = {}
        for effort in EFFORTS:
            terminal = encoded.query_stop - 1 + effort
            if terminal >= sequence.shape[0]:
                print(f"effort {effort}: exceeds sequence capacity, skipped")
                continue
            mask = np.zeros((sequence.shape[0],), dtype=np.float32)
            mask[terminal] = 1.0
            model.reset_state()
            learner.reset_state(batch_size=1)
            gradients, objective = learner.etrace_grad(
                jnp.asarray(sequence[:, None, :]),
                jnp.asarray(advances[:, None]),
                step_fn=step_loss,
                mask=jnp.asarray(mask),
                reduction="mean",
                loss_output="scalar",
                return_value=True,
            )
            norms = {
                ".".join(str(part) for part in path)
                if isinstance(path, tuple)
                else str(path): float(
                    jnp.sqrt(
                        sum(
                            jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(value)
                        )
                    )
                )
                for path, value in gradients.items()
            }
            rows[effort] = norms
            print(f"\n--- effort {effort}  loss={float(objective):.6f}")
            for name, norm in sorted(norms.items()):
                flag = "  <== ZERO" if norm == 0.0 else ""
                print(f"    {name:<34} grad_l2={norm:.6e}{flag}")

        print("\n=== effort dependence (grad L2 by effort) ===")
        names = sorted(next(iter(rows.values())))
        header = "".join(f"{e:>14}" for e in rows)
        print(f"{'parameter':<34}{header}")
        for name in names:
            cells = "".join(f"{rows[e][name]:>14.4e}" for e in rows)
            varies = len({round(rows[e][name], 12) for e in rows}) > 1
            print(f"{name:<34}{cells}   {'varies' if varies else 'CONSTANT'}")


if __name__ == "__main__":
    main()
