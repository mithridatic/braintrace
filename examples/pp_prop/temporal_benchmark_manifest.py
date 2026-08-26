"""Schema-versioned paired randomness manifest for Example 17."""

from __future__ import annotations

import hashlib
import msgspec_json
from dataclasses import asdict, dataclass
from pathlib import Path

from temporal_benchmark_config import SplitSizes
from temporal_benchmark_data import balanced_trial_specs, trial_commitment

MANIFEST_SCHEMA_VERSION = 1
MASTER_SEED = 20260810


@dataclass(frozen=True)
class SeedBundle:
    """Hold all domain-separated seeds for one paired arm bundle."""

    bundle_id: str
    split_seed: int
    topology_seed: int
    weight_seed: int
    training_order_seed: int
    training_encoding_seed: int
    evaluation_encoding_seeds: tuple[int, ...]
    test_commitment_sha256: str


def _seed(master_seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{domain}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def generate_manifest(
    master_seed: int = MASTER_SEED, sizes: SplitSizes = SplitSizes()
) -> dict[str, object]:
    """Generate the locked 3-by-2-by-2 paired seed matrix."""
    split_seeds = [_seed(master_seed, f"split:{index}") for index in range(3)]
    topology_seeds = [_seed(master_seed, f"topology:{index}") for index in range(2)]
    weight_seeds = [_seed(master_seed, f"weight:{index}") for index in range(2)]
    bundles: list[SeedBundle] = []
    for split_index, split_seed in enumerate(split_seeds):
        test_specs = balanced_trial_specs(sizes.test, _seed(split_seed, "test"))
        commitment = trial_commitment(test_specs)
        for topology_index, topology_seed in enumerate(topology_seeds):
            for weight_index, weight_seed in enumerate(weight_seeds):
                bundle_id = (
                    f"split{split_index}-topology{topology_index}-weight{weight_index}"
                )
                bundles.append(
                    SeedBundle(
                        bundle_id=bundle_id,
                        split_seed=split_seed,
                        topology_seed=topology_seed,
                        weight_seed=weight_seed,
                        training_order_seed=_seed(master_seed, f"{bundle_id}:order"),
                        training_encoding_seed=_seed(master_seed, f"{bundle_id}:train"),
                        evaluation_encoding_seeds=tuple(
                            _seed(master_seed, f"{bundle_id}:evaluate:{index}")
                            for index in range(8)
                        ),
                        test_commitment_sha256=commitment,
                    )
                )
    document: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "master_seed": master_seed,
        "split_sizes": asdict(sizes),
        "bundles": [asdict(bundle) for bundle in bundles],
    }
    validate_manifest(document)
    return document


def validate_manifest(document: dict[str, object]) -> None:
    """Fail closed on malformed, colliding, or incomplete seed manifests."""
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported temporal benchmark manifest schema. Use a supported option or change the configuration.")
    bundles = document.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 12:
        raise ValueError("Manifest must contain exactly 12 paired bundles. Add exactly 12 paired bundles to Manifest.")
    ids = [bundle.get("bundle_id") for bundle in bundles if isinstance(bundle, dict)]
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError("Bundle identifiers must be unique. Set Bundle identifiers to unique.")
    per_bundle_seeds: list[int] = []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            raise ValueError("Each bundle must be an object. Set Each bundle to an object.")
        local = [
            bundle.get("training_order_seed"),
            bundle.get("training_encoding_seed"),
        ]
        evaluation = bundle.get("evaluation_encoding_seeds")
        if not isinstance(evaluation, (list, tuple)) or len(evaluation) != 8:
            raise ValueError("Each bundle must have eight evaluation encoding seeds. Ensure Each bundle has eight evaluation encoding seeds.")
        local.extend(evaluation)
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in local):
            raise ValueError("All derived seeds must be integers. Set All derived seeds to integers.")
        per_bundle_seeds.extend(local)
    if len(per_bundle_seeds) != len(set(per_bundle_seeds)):
        raise ValueError("Order and encoding seed domains must not collide. Ensure Order and encoding seed domains does not collide.")


def write_manifest(path: Path, document: dict[str, object]) -> None:
    """Write a validated manifest with canonical stable formatting."""
    validate_manifest(document)
    path.write_text(
        msgspec_json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> dict[str, object]:
    """Load and validate a manifest from disk."""
    document = msgspec_json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Manifest root must be an object. Set Manifest root to an object.")
    validate_manifest(document)
    return document


def find_bundle(document: dict[str, object], bundle_id: str) -> SeedBundle:
    """Return one validated bundle by stable identifier."""
    validate_manifest(document)
    bundles = document["bundles"]
    assert isinstance(bundles, list)
    for raw in bundles:
        assert isinstance(raw, dict)
        if raw["bundle_id"] == bundle_id:
            return SeedBundle(
                **{
                    **raw,
                    "evaluation_encoding_seeds": tuple(
                        raw["evaluation_encoding_seeds"]
                    ),
                }
            )
    raise KeyError(f"Unknown bundle: {bundle_id}. Set the named field to one of the supported values, then rerun the operation.")


def split_specs(bundle: SeedBundle, split: str, sizes: SplitSizes) -> tuple:
    """Materialize train or validation specs; test requires the sealed path."""
    if split == "test":
        raise PermissionError("Test specs require materialize_sealed_test_specs. Provide the required value for Test specs.")
    if split not in {"train", "validation"}:
        raise ValueError("Split must be train or validation. Set Split to train or validation.")
    count = getattr(sizes, split)
    return balanced_trial_specs(count, _seed(bundle.split_seed, split))


def materialize_sealed_test_specs(
    bundle: SeedBundle, sizes: SplitSizes, *, sealed: bool
) -> tuple:
    """Materialize test specs only after checking the committed seal."""
    if not sealed:
        raise PermissionError("Test split is sealed until configuration is frozen. Fix the input condition named in the error, then rerun the operation.")
    specs = balanced_trial_specs(sizes.test, _seed(bundle.split_seed, "test"))
    if trial_commitment(specs) != bundle.test_commitment_sha256:
        raise ValueError("Test trial commitment mismatch. Use matching values and structures.")
    return specs
