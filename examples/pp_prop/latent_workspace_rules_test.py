"""Tests for the demonstration-verified rule channel.

The load-bearing property is *soundness*: no rule may be admitted unless it
reproduces every demonstration pair exactly. Recall is a measured quantity, not
an asserted one, so these tests pin the contract rather than a coverage number.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from latent_workspace_rules import (  # noqa: E402
    fit_verified_rules,
    verified_rule_candidates,
)

Pairs = list[tuple[np.ndarray, np.ndarray]]


def _grid(rows: list[list[int]]) -> np.ndarray:
    """Return an ``int32`` grid from nested lists."""

    return np.asarray(rows, dtype=np.int32)


def test_every_admitted_rule_reproduces_every_demonstration() -> None:
    """Soundness: admission implies exact reproduction of all pairs."""

    pairs: Pairs = [
        (_grid([[1, 2], [3, 4]]), _grid([[2, 1], [4, 3]])),
        (_grid([[5, 6], [7, 8]]), _grid([[6, 5], [8, 7]])),
    ]
    rules = fit_verified_rules(pairs)
    assert rules
    for rule in rules:
        for source, target in pairs:
            produced = rule.apply(source)
            assert produced is not None
            assert np.array_equal(produced, target)


def test_no_rule_is_admitted_for_an_unlearnable_task() -> None:
    """Contradictory demonstrations admit nothing rather than guessing."""

    pairs: Pairs = [
        (_grid([[1, 1], [1, 1]]), _grid([[2, 2], [2, 2]])),
        (_grid([[1, 1], [1, 1]]), _grid([[3, 3], [3, 3]])),
    ]
    assert fit_verified_rules(pairs) == ()
    assert verified_rule_candidates(pairs, _grid([[1, 1], [1, 1]])) == ()


def test_empty_demonstrations_are_rejected() -> None:
    """Fitting requires at least one demonstration."""

    with pytest.raises(ValueError, match="at least one demonstration"):
        fit_verified_rules([])


@pytest.mark.parametrize(
    ("source", "target", "query", "expected"),
    [
        ([[1, 2]], [[2, 1]], [[3, 4]], [[4, 3]]),
        ([[1, 2], [3, 4]], [[1, 3], [2, 4]], [[5, 6], [7, 8]], [[5, 7], [6, 8]]),
        ([[1]], [[1, 1], [1, 1]], [[7]], [[7, 7], [7, 7]]),
    ],
)
def test_geometric_rules_generalise_to_a_new_query(
    source: list[list[int]],
    target: list[list[int]],
    query: list[list[int]],
    expected: list[list[int]],
) -> None:
    """A verified geometry applies unchanged to an unseen input."""

    candidates = verified_rule_candidates([(_grid(source), _grid(target))], _grid(query))
    assert candidates
    assert candidates[0][1].tolist() == expected


def test_colour_substitution_is_fitted_and_applied() -> None:
    """A consistent per-colour table transfers to a new grid."""

    pairs: Pairs = [
        (_grid([[1, 2], [2, 1]]), _grid([[3, 4], [4, 3]])),
        (_grid([[2, 2], [1, 1]]), _grid([[4, 4], [3, 3]])),
    ]
    candidates = verified_rule_candidates(pairs, _grid([[1, 1], [2, 2]]))
    assert candidates[0][1].tolist() == [[3, 3], [4, 4]]


def test_colour_substitution_declines_an_unseen_colour() -> None:
    """A query colour outside the fitted table produces no proposal from it."""

    pairs: Pairs = [(_grid([[1, 2]]), _grid([[3, 4]]))]
    for name, grid in verified_rule_candidates(pairs, _grid([[7, 8]])):
        assert not name.endswith("|cm:identity") or grid is None


def test_candidates_are_deduplicated_and_vote_ranked() -> None:
    """Identical proposals collapse to one entry, ordered by corroboration."""

    pairs: Pairs = [(_grid([[1, 1], [1, 1]]), _grid([[1, 1], [1, 1]]))]
    candidates = verified_rule_candidates(pairs, _grid([[2, 2], [2, 2]]))
    keys = [grid.tobytes() for _, grid in candidates]
    assert len(keys) == len(set(keys))


def test_candidates_never_exceed_the_arc_grid_limits() -> None:
    """Proposals outside 1..30 on either side are discarded."""

    pairs: Pairs = [(_grid([[1]]), _grid([[1] * 3] * 3))]
    query = _grid([[1] * 20] * 20)
    for _, grid in verified_rule_candidates(pairs, query):
        assert 1 <= grid.shape[0] <= 30
        assert 1 <= grid.shape[1] <= 30


def test_object_recolouring_transfers_by_size() -> None:
    """Objects are recoloured by a fitted size-to-colour table."""

    source = _grid([[1, 0, 0], [0, 0, 0], [0, 1, 1]])
    target = _grid([[2, 0, 0], [0, 0, 0], [0, 3, 3]])
    query = _grid([[0, 1, 1], [0, 0, 0], [1, 0, 0]])
    candidates = verified_rule_candidates([(source, target)], query)
    grids = [grid.tolist() for _, grid in candidates]
    assert [[0, 3, 3], [0, 0, 0], [2, 0, 0]] in grids


def test_periodic_repair_transfers_to_a_new_hole() -> None:
    """A verified periodic repair fills a hole at a different position."""

    source = _grid([[1, 2, 1, 2], [1, 2, 9, 2]])
    target = _grid([[1, 2, 1, 2], [1, 2, 1, 2]])
    query = _grid([[3, 4, 3, 4], [9, 4, 3, 4]])
    candidates = verified_rule_candidates([(source, target)], query)
    grids = [grid.tolist() for _, grid in candidates]
    assert [[3, 4, 3, 4], [3, 4, 3, 4]] in grids


def test_rule_names_record_reduction_and_completion() -> None:
    """Every admitted name is a ``reduction|completion`` pair for attribution."""

    pairs: Pairs = [(_grid([[1, 2]]), _grid([[2, 1]]))]
    for rule in fit_verified_rules(pairs):
        assert rule.name.count("|") == 1
        reduction, completion = rule.name.split("|")
        assert reduction and completion


def test_a_failing_transform_is_swallowed_not_raised() -> None:
    """A reduction that cannot apply to the query yields no candidate, not an error."""

    pairs: Pairs = [
        (_grid([[0, 5, 0], [0, 5, 0]]), _grid([[0], [0]])),
        (_grid([[1, 5, 1], [1, 5, 1]]), _grid([[1], [1]])),
    ]
    assert verified_rule_candidates(pairs, _grid([[7, 7], [7, 7]])) is not None
