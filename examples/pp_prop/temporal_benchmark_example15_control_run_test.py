"""Tests for the standalone fixed-profile Example 15 evidence runner."""

from pathlib import Path

import pytest

import temporal_benchmark_example15_control_run as runner


class _ExampleStub:
    SEEDS = (0, 1, 2)
    N_EPOCH = 5

    class _RunConfig:
        def __init__(self, seed, n_epochs, batch_size):
            self.n_rec = 96
            self.degree = 8
            self.n_step = 30
            self.final_window = 5
            self.learning_rate = 0.003
            self.decay_or_rank = 0.95
            self.clip_norm = 1.0
            self.sparse_backend = "jax_raw"
            self.recurrent_scale_basis = "neurons"

    def __init__(self):
        self.calls = []

    def main(self, *, n_epochs, batch_size, plot):
        self.calls.append((n_epochs, batch_size, plot))
        return {
            "seed_results": [],
            "mean_accuracy": 0.96,
            "minimum_accuracy": 0.95,
            "recurrent_nnz": 768,
        }


def test_fixed_config_pins_all_example15_numerical_defaults() -> None:
    assert runner.fixed_config_document() == {
        "seeds": [0, 1, 2],
        "n_epochs": 5,
        "batch_size": 32,
        "n_rec": 96,
        "degree": 8,
        "n_step": 30,
        "final_window": 5,
        "learning_rate": 0.003,
        "decay_or_rank": 0.95,
        "clip_norm": 1.0,
        "sparse_backend": "jax_raw",
        "recurrent_scale_basis": "neurons",
        "train_examples": 288,
        "validation_examples": 72,
    }


def test_run_wrapper_executes_only_the_fixed_profile(
    monkeypatch, tmp_path: Path
) -> None:
    example = _ExampleStub()
    script = tmp_path / "15.py"
    script.write_text("fixed", encoding="utf-8")
    monkeypatch.setattr(runner, "_load_example", lambda _: example)
    monkeypatch.setattr(
        runner,
        "_environment",
        lambda *_: {
            "source_commit": "abc",
            "source_dirty": True,
            "container_image_digest": "sha256:image",
        },
    )

    document = runner.run_fixed_example15(script, tmp_path, "sha256:image", "auto")

    assert example.calls == [(5, 32, False)]
    assert document["accepted_baseline"] is False
    assert document["fixed_config"] == runner.fixed_config_document()
    assert document["result"]["status"] == "completed"


def test_profile_verification_refuses_changed_defaults() -> None:
    changed = _ExampleStub()
    changed.N_EPOCH = 4

    with pytest.raises(RuntimeError, match="numerical defaults"):
        runner._verify_example_profile(changed)


def test_source_fingerprint_never_invents_missing_values(monkeypatch) -> None:
    monkeypatch.delenv("BRAINTRACE_SOURCE_COMMIT", raising=False)
    monkeypatch.delenv("BRAINTRACE_SOURCE_DIRTY", raising=False)
    monkeypatch.setattr(runner, "_git_output", lambda *_: None)

    assert runner.source_fingerprint(Path("missing")) == {
        "source_commit": None,
        "source_dirty": None,
    }
