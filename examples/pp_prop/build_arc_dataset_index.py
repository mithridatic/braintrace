"""Build prevalidated ARC indexes for the Example 21 container image."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import msgspec

from examples.pp_prop.latent_workspace_task import (
    DatasetSource,
    assert_no_evaluation_leakage,
    load_dataset_source,
    write_dataset_index,
)


_LICENSE_REFERENCE = "https://github.com/fchollet/ARC-AGI/blob/master/LICENSE"


def build_arc_dataset_indexes(
    dataset_root: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    *,
    version: str,
    runtime_index_root: str = "/datasets/arc/index",
    training_exclusions: tuple[str, ...] = (),
    expected_training_tasks: int = 399,
    expected_evaluation_tasks: int = 400,
) -> dict[str, int]:
    """Validate ARC-AGI-1 once and emit runtime indexes and declarations.

    Parameters
    ----------
    dataset_root
        ARC-AGI-1 checkout containing ``data/training`` and ``data/evaluation``.
    output_root
        Build-time directory for generated indexes.
    manifest_path
        Build-time destination for the Example 21 source manifest.
    version
        Immutable ARC-AGI-1 revision recorded in both declarations.
    runtime_index_root
        Absolute in-image directory from which Example 21 reads the indexes.
    training_exclusions
        Explicit training fingerprints excluded to prevent evaluation leakage.
    expected_training_tasks
        Required accepted training count after exclusions and deduplication.
    expected_evaluation_tasks
        Required accepted evaluation count after deduplication.

    Returns
    -------
    dict
        Accepted training and evaluation task counts.

    Raises
    ------
    ValueError
        If validation, leakage checks, or expected-count checks fail.
    """

    dataset_path = Path(dataset_root).resolve()
    output_path = Path(output_root).resolve()
    declarations = (
        DatasetSource(
            name="ARC-AGI-1 training",
            role="train",
            version=version,
            path=str(dataset_path / "data" / "training"),
            license_reference=_LICENSE_REFERENCE,
            format="task_json",
            exclude_fingerprints=training_exclusions,
        ),
        DatasetSource(
            name="ARC-AGI-1 evaluation",
            role="evaluation",
            version=version,
            path=str(dataset_path / "data" / "evaluation"),
            license_reference=_LICENSE_REFERENCE,
            format="task_json",
        ),
    )
    training, evaluation = tuple(load_dataset_source(item) for item in declarations)
    assert_no_evaluation_leakage((training.manifest, evaluation.manifest))
    actual = (len(training.tasks), len(evaluation.tasks))
    expected = (expected_training_tasks, expected_evaluation_tasks)
    if actual != expected:
        raise ValueError(
            f"expected {expected[0]} training tasks and {expected[1]} evaluation "
            f"tasks, found {actual[0]} and {actual[1]}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    filenames = ("training.index.json", "evaluation.index.json")
    write_dataset_index(training, output_path / filenames[0])
    write_dataset_index(evaluation, output_path / filenames[1])
    runtime_root = PurePosixPath(runtime_index_root)
    sources = []
    for declaration, filename in zip(declarations, filenames, strict=True):
        sources.append(
            {
                "name": declaration.name,
                "role": declaration.role,
                "version": declaration.version,
                "path": str(runtime_root / filename),
                "license_reference": declaration.license_reference,
                "format": "indexed_json",
                "exclude_fingerprints": list(declaration.exclude_fingerprints),
            }
        )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(msgspec.json.encode({"sources": sources}, order="sorted"))
    return {
        "training_tasks": actual[0],
        "evaluation_tasks": actual[1],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--runtime-index-root", default="/datasets/arc/index")
    parser.add_argument("--training-exclusion", action="append", default=[])
    parser.add_argument("--expected-training-tasks", type=int, default=399)
    parser.add_argument("--expected-evaluation-tasks", type=int, default=400)
    return parser


def _main() -> None:
    arguments = _parser().parse_args()
    report = build_arc_dataset_indexes(
        arguments.dataset_root,
        arguments.output_root,
        arguments.manifest_path,
        version=arguments.version,
        runtime_index_root=arguments.runtime_index_root,
        training_exclusions=tuple(arguments.training_exclusion),
        expected_training_tasks=arguments.expected_training_tasks,
        expected_evaluation_tasks=arguments.expected_evaluation_tasks,
    )
    print(msgspec.json.encode(report, order="sorted").decode())


if __name__ == "__main__":
    _main()
