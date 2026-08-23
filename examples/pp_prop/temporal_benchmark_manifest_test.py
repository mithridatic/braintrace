"""Tests for domain-separated seeds and sealed trial commitments."""

import msgspec_json

import pytest

from temporal_benchmark_config import SplitSizes
from temporal_benchmark_manifest import (
    find_bundle,
    generate_manifest,
    materialize_sealed_test_specs,
    validate_manifest,
)


def test_manifest_contains_locked_paired_matrix_and_seed_domains() -> None:
    document = generate_manifest()

    assert document["master_seed"] == 20260810
    assert len(document["bundles"]) == 12
    assert len({bundle["split_seed"] for bundle in document["bundles"]}) == 3
    assert len({bundle["topology_seed"] for bundle in document["bundles"]}) == 2
    assert len({bundle["weight_seed"] for bundle in document["bundles"]}) == 2
    derived = []
    for bundle in document["bundles"]:
        derived.extend(
            [
                bundle["training_order_seed"],
                bundle["training_encoding_seed"],
                *bundle["evaluation_encoding_seeds"],
            ]
        )
    assert len(derived) == len(set(derived))


def test_test_specs_remain_sealed_until_explicit_unlock() -> None:
    bundle = find_bundle(generate_manifest(), "split0-topology0-weight0")

    with pytest.raises(PermissionError, match="sealed"):
        materialize_sealed_test_specs(bundle, SplitSizes(), sealed=False)

    specs = materialize_sealed_test_specs(bundle, SplitSizes(), sealed=True)
    assert len(specs) == 512


def test_manifest_round_trip_is_schema_valid() -> None:
    document = generate_manifest()
    repeated = msgspec_json.loads(msgspec_json.dumps(document))

    validate_manifest(repeated)


def test_manifest_rejects_seed_collisions() -> None:
    document = generate_manifest()
    first, second = document["bundles"][:2]
    second["training_order_seed"] = first["training_order_seed"]

    with pytest.raises(ValueError, match="must not collide"):
        validate_manifest(document)
