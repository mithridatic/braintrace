"""End-to-end test for the configurable sparse pp-prop benchmark."""

import json
import pathlib
import subprocess
import sys

import jax
import pytest


SCRIPT = pathlib.Path(__file__).with_name("16-configurable-sparse-benchmark.py")
_TINY_RUN = (
    "--neurons", "12", "--degree", "3", "--steps", "3", "--final-window", "1",
    "--updates", "1", "--max-rss-gib", "4", "--min-available-gib", "1",
    "--max-wall-seconds", "120",
)


def test_tiny_supervised_worker_emits_learning_schema():
    command = [sys.executable, str(SCRIPT), *_TINY_RUN]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=150,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["metrics"]["recurrent_nnz"] == 36
    assert payload["metrics"]["updates_completed"] == 1
    assert payload["memory"]["peak_rss_bytes"] > 0
    assert "initial_accuracy=" in completed.stderr


def test_a_requested_gpu_is_refused_rather_than_run_on_the_host():
    if jax.default_backend() in {"gpu", "cuda", "rocm"}:
        pytest.skip("this host binds an accelerator, so the request is satisfiable")
    command = [sys.executable, str(SCRIPT), *_TINY_RUN, "--device", "gpu"]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=150)
    payload = json.loads(completed.stdout)

    assert completed.returncode != 0
    assert payload["status"] == "failed"
    assert "requested device gpu" in payload["error_output"]


def test_the_host_backend_can_be_pinned():
    command = [sys.executable, str(SCRIPT), *_TINY_RUN, "--device", "cpu"]

    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=150
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "completed"
    assert payload["environment"]["backend"] == "cpu"
    assert payload["config"]["device"] == "cpu"
