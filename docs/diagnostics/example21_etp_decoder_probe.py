#!/usr/bin/env python3
"""Stage 2 prerequisite gate: how does the ETP compiler classify the new decoder?

`2026-08-18-example21-decoder-redesign.md` §7 asserts, from static reading only,
that (a) a ``brainstate.ShortTermState`` query buffer lands in ``_other_states``
and cannot form a hidden group, and (b) the five new decoder heads inherit the
existing heads' ``excluded_weights`` classification. Neither was ever executed.
If either fails, the decoder must instead thread the query grid through
``etrace_grad`` as an extra ``xs``, which is a materially larger diff.

This probe builds the *shape* of the Stage 2 decoder -- the same state kinds and
the same head fan-out, applied at the same point in ``compact_readout`` -- and
compiles it with the production ``compile_pp_prop``. It asserts:

  1. every new head weight path appears in ``report.excluded_weights``
  2. no new head weight path appears in ``report.etrace_weights``
  3. ``report.hidden_groups`` is identical to the legacy model's
  4. the query buffer is not a ``HiddenState`` and gains no eligibility trace

Usage:  example21_etp_decoder_probe.py WORKTREE
"""

from __future__ import annotations

import msgspec
import pathlib
import sys

import brainstate
import braintrace
import jax
import jax.numpy as jnp

WORKTREE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(WORKTREE / "examples" / "pp_prop"))
sys.path.insert(0, str(WORKTREE))

from latent_workspace_model import (  # noqa: E402
    COLOR_COUNT,
    MAX_GRID_SIZE,
    LatentWorkspaceModel,
    ModelConfig,
    compile_pp_prop,
)
from latent_workspace_task import (  # noqa: E402
    RowEventConfig,
    associative_memory_feature_indices,
)

NEURONS = 256
EDGES = 2048
BATCH = 2
MEMORY_WIDTH = 32
SHAPE_RULE_COUNT = 13


class ProbeDecoderModel(LatentWorkspaceModel):
    """Legacy model plus the Stage 2 decoder's states and head fan-out.

    Only the ETP-visible structure matters here: the head widths, the point in
    ``compact_readout`` where they are applied, and the state class holding the
    captured query grid. The arithmetic is deliberately a placeholder.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        width = config.readout_width
        # Strictly after every legacy head, per redesign spec §3: any draw
        # inserted earlier would shift every legacy weight.
        self.shape_rule_head = braintrace.nn.Linear(width, 2 * SHAPE_RULE_COUNT)
        self.shape_absolute_head = braintrace.nn.Linear(width, 2 * MAX_GRID_SIZE)
        self.copy_gate_head = braintrace.nn.Linear(
            width, MAX_GRID_SIZE * MAX_GRID_SIZE * 4
        )
        self.color_palette_head = braintrace.nn.Linear(width, COLOR_COUNT)
        self.color_explicit_head = braintrace.nn.Linear(
            width, MAX_GRID_SIZE * MAX_GRID_SIZE * COLOR_COUNT
        )
        self.query_grid = brainstate.ShortTermState(
            jnp.zeros(
                (config.batch_size, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=jnp.float32
            )
        )
        self.query_shape = brainstate.ShortTermState(
            jnp.zeros((config.batch_size, 2), dtype=jnp.float32)
        )

    def compact_readout(self, carrier: jax.Array | None = None) -> jax.Array:
        legacy = super().compact_readout(carrier)
        if carrier is None:
            carrier = (
                self.workspace_carrier.value
                if self.config.memory_enabled
                else self.spikes
            )
        carrier = jnp.asarray(carrier)
        hidden = jax.nn.gelu(self.readout_projection(carrier))
        query = self.query_grid.value.reshape(self.config.batch_size, -1)
        decoder = (
            self.shape_rule_head(hidden).sum(axis=-1)
            + self.shape_absolute_head(hidden).sum(axis=-1)
            + self.copy_gate_head(hidden).sum(axis=-1)
            + self.color_palette_head(hidden).sum(axis=-1)
            + self.color_explicit_head(hidden).sum(axis=-1)
            + query.sum(axis=-1)
            + self.query_shape.value.sum(axis=-1)
        )
        return legacy + decoder[..., None] * 0.0


NEW_HEADS = (
    "shape_rule_head",
    "shape_absolute_head",
    "copy_gate_head",
    "color_palette_head",
    "color_explicit_head",
)


def _config() -> ModelConfig:
    """Mirror ``_model_config`` from the entry point, at probe scale."""
    row_config = RowEventConfig()
    features = associative_memory_feature_indices(row_config)
    return ModelConfig(
        input_width=row_config.input_width,
        neuron_count=NEURONS,
        recurrent_edges=EDGES,
        batch_size=BATCH,
        context_memory_width=MEMORY_WIDTH,
        memory_decay=1.0,
        demonstration_phase_index=row_config.phase_slice.start,
        query_phase_index=row_config.phase_slice.start + 1,
        input_side_valid_index=row_config.side_valid_slice.start,
        output_side_valid_index=row_config.side_valid_slice.start + 1,
        memory_key_indices=features.key_indices,
        memory_value_indices=features.value_indices,
    )


def _path_text(path: object) -> str:
    if isinstance(path, (tuple, list)):
        return ".".join(str(part) for part in path)
    return str(path)


def _classification(model: LatentWorkspaceModel) -> dict[str, object]:
    learner = compile_pp_prop(model)
    report = learner.report
    return {
        "etrace_weights": sorted(_path_text(path) for path, _ in report.etrace_weights),
        "excluded_weights": sorted(
            _path_text(path) for path, _ in report.excluded_weights
        ),
        "hidden_group_count": len(report.hidden_groups),
        "hidden_state_paths": sorted(
            _path_text(path)
            for path in model.states(brainstate.HiddenState).keys()
        ),
    }


def main() -> int:
    legacy = _classification(LatentWorkspaceModel(_config()))
    probe_model = ProbeDecoderModel(_config())
    probe = _classification(probe_model)

    new_head_paths = {
        path
        for path in probe["excluded_weights"] + probe["etrace_weights"]
        if any(path.startswith(head + ".") for head in NEW_HEADS)
    }
    excluded = set(probe["excluded_weights"])
    etrace = set(probe["etrace_weights"])
    short_term = sorted(
        _path_text(path)
        for path in probe_model.states(brainstate.ShortTermState).keys()
    )

    checks = {
        "every_new_head_excluded": all(path in excluded for path in new_head_paths),
        "no_new_head_on_etrace_path": not (new_head_paths & etrace),
        "new_heads_seen_by_compiler": len(new_head_paths) >= len(NEW_HEADS),
        "hidden_groups_identical": (
            probe["hidden_group_count"] == legacy["hidden_group_count"]
        ),
        "hidden_states_identical": (
            probe["hidden_state_paths"] == legacy["hidden_state_paths"]
        ),
        "etrace_weights_identical": (
            probe["etrace_weights"] == legacy["etrace_weights"]
        ),
        "query_buffer_is_short_term": any(
            path.startswith("query_grid") for path in short_term
        ),
    }
    result = {
        "legacy": legacy,
        "probe": probe,
        "new_head_paths": sorted(new_head_paths),
        "short_term_state_paths": short_term,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    print(msgspec.json.format(msgspec.json.encode(result), indent=2).decode())
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
