"""Run or resume the complete sealed Example 17 arm-by-bundle matrix."""

from __future__ import annotations

import argparse
import msgspec_json
import pathlib
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from temporal_benchmark_analysis import analyze_sealed_results
from temporal_benchmark_config import ARMS
from temporal_benchmark_manifest import load_manifest
from temporal_benchmark_reporting import write_result

CommandRunner = Callable[[Sequence[str], pathlib.Path], None]
Analyzer = Callable[[Iterable[pathlib.Path]], dict[str, object]]


@dataclass(frozen=True)
class SealedMatrixSettings:
    """Define paths and executable for one resumable sealed matrix.

    Attributes
    ----------
    output_directory : pathlib.Path
        Root for raw results and the final gate report.
    benchmark_script : pathlib.Path
        Example 17 child entry point.
    manifest : pathlib.Path
        Committed bundle manifest.
    frozen_selection : pathlib.Path
        Provenance-bound development selection.
    python_executable : str
        Interpreter used for isolated child runs.
    """

    output_directory: pathlib.Path
    benchmark_script: pathlib.Path
    manifest: pathlib.Path
    frozen_selection: pathlib.Path
    python_executable: str


def _bundle_ids(settings: SealedMatrixSettings) -> tuple[str, ...]:
    document = load_manifest(settings.manifest)
    bundles = document["bundles"]
    assert isinstance(bundles, list)
    return tuple(str(bundle["bundle_id"]) for bundle in bundles)


def result_path(
    settings: SealedMatrixSettings, arm: str, bundle_id: str
) -> pathlib.Path:
    """Return the stable final path for one sealed child result.

    Parameters
    ----------
    settings : SealedMatrixSettings
        Matrix execution settings.
    arm : str
        Paired benchmark arm.
    bundle_id : str
        Committed bundle identifier.

    Returns
    -------
    pathlib.Path
        Stable raw-result path.
    """
    return settings.output_directory / "raw" / arm / f"{bundle_id}.json"


def expected_result_paths(settings: SealedMatrixSettings) -> tuple[pathlib.Path, ...]:
    """Return all 84 final paths in deterministic arm-major order.

    Parameters
    ----------
    settings : SealedMatrixSettings
        Matrix execution settings.

    Returns
    -------
    tuple of pathlib.Path
        Complete expected result path set.
    """
    return tuple(
        result_path(settings, arm, bundle_id)
        for arm in ARMS
        for bundle_id in _bundle_ids(settings)
    )


def _command(
    settings: SealedMatrixSettings,
    arm: str,
    bundle_id: str,
    output: pathlib.Path,
) -> list[str]:
    command = [
        settings.python_executable,
        str(settings.benchmark_script),
        "--manifest", str(settings.manifest),
        "--frozen-selection", str(settings.frozen_selection),
        "--bundle-id", bundle_id,
        "--arm", arm,
        "--horizon", "long",
        "--updates", "800",
        "--device", "gpu",
        "--sealed-test",
        "--json-output", str(output),
    ]
    if arm == "all_pp_prop":
        command.append("--gradient-evidence")
    return command


def _run_subprocess(command: Sequence[str], source_root: pathlib.Path) -> None:
    completed = subprocess.run(command, cwd=source_root, check=False)
    if completed.returncode:
        raise RuntimeError(f"sealed child failed with exit {completed.returncode}")


def run_sealed_matrix(
    settings: SealedMatrixSettings,
    *,
    runner: CommandRunner = _run_subprocess,
    analyzer: Analyzer = analyze_sealed_results,
) -> dict[str, object]:
    """Run missing children, validate the full matrix, and write its gate report.

    Existing raw files are never promoted or reported without passing the same
    complete-matrix analyzer as newly produced results.

    Parameters
    ----------
    settings : SealedMatrixSettings
        Matrix execution settings.
    runner : CommandRunner
        Isolated child-process runner.
    analyzer : Analyzer
        Fail-closed complete-matrix analyzer.

    Returns
    -------
    dict
        Scientific gate report.
    """
    source_root = settings.benchmark_script.resolve().parents[2]
    for arm in ARMS:
        for bundle_id in _bundle_ids(settings):
            final = result_path(settings, arm, bundle_id)
            if final.is_file():
                continue
            partial = final.with_suffix(".partial.json")
            partial.unlink(missing_ok=True)
            partial.parent.mkdir(parents=True, exist_ok=True)
            runner(_command(settings, arm, bundle_id, partial), source_root)
            if not partial.is_file():
                raise RuntimeError(f"sealed child did not write {partial}")
            partial.replace(final)
    paths = expected_result_paths(settings)
    report = analyzer(paths)
    report_path = settings.output_directory / "episodic-gates.json"
    write_result(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = pathlib.Path(__file__).resolve().parent
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument(
        "--benchmark-script", type=pathlib.Path,
        default=base / "17-temporal-credit-benchmark.py",
    )
    parser.add_argument(
        "--manifest", type=pathlib.Path,
        default=base / "temporal_benchmark_manifest.json",
    )
    parser.add_argument("--frozen-selection", type=pathlib.Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sealed matrix CLI and print its final gate document.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments, excluding the executable name.

    Returns
    -------
    int
        Zero after successful complete-matrix validation.
    """
    values = _parser().parse_args(argv)
    settings = SealedMatrixSettings(
        values.output_directory.resolve(),
        values.benchmark_script.resolve(),
        values.manifest.resolve(),
        values.frozen_selection.resolve(),
        values.python_executable,
    )
    report = run_sealed_matrix(settings)
    print(msgspec_json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
