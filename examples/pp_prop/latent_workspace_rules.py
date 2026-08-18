"""Demonstration-verified grid rules that propose exact ARC candidates.

A rule is admitted only when it reproduces **every** demonstration pair exactly,
so the channel proposes nothing it has not already been shown to be right about.
Nothing here inspects a query target: only the demonstrations, which the ARC
protocol supplies at test time.

Coverage comes from composition. A *reduction* maps an input grid to an
intermediate grid (crop, panel selection, object selection, lattice collapse,
denoise); a *completion* maps that intermediate to the output (dihedral map,
colour substitution, tiling, magnification, periodic repair). Every
reduction/completion pair is fitted and verified, which multiplies the reachable
rule set without hand-writing each combination.

Candidates are reported separately from the spiking model's own candidates so a
solve is never mis-attributed to latent reasoning by the network.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from latent_workspace_rule_cells import CELL_FAMILIES
from latent_workspace_rule_edits import EDIT_FAMILIES
from latent_workspace_rule_parts import (
    COLOR_COUNT,
    DIHEDRAL_NAMES,
    MAX_SIDE,
    apply_dihedral,
    background_color,
    connected_components,
    crop_to_mask,
    is_valid_grid,
    lattice_cells,
    periodic_fill,
    split_panels,
)

IntArray = np.ndarray
GridMap = Callable[[IntArray], "IntArray | None"]
DemoPairs = Sequence[tuple[IntArray, IntArray]]

_SAFE_ERRORS = (ValueError, IndexError, ZeroDivisionError, TypeError)


@dataclass(frozen=True)
class GridRule:
    """A demonstration-verified transformation from an input grid to an output.

    Attributes
    ----------
    name
        Stable identifier reported alongside any solve this rule produces.
    apply
        Callable mapping one input grid to a proposed output, or ``None`` when
        the rule does not apply to that input.
    """

    name: str
    apply: GridMap


def _call(transform: GridMap, grid: IntArray) -> IntArray | None:
    """Apply a transform, converting expected failures into ``None``."""

    try:
        result = transform(grid)
    except _SAFE_ERRORS:
        return None
    return None if result is None or result.size == 0 else result


def _reproduces(rule: GridRule, pairs: DemoPairs) -> bool:
    """Return whether a rule reproduces every demonstration pair exactly."""

    for source, target in pairs:
        produced = _call(rule.apply, source)
        if produced is None or produced.shape != target.shape:
            return False
        if not np.array_equal(produced, target):
            return False
    return True


# --------------------------------------------------------------------------
# Reductions: input grid -> intermediate grid
# --------------------------------------------------------------------------


def _object_masks(grid: IntArray, background: int) -> list[tuple[IntArray, ...]]:
    """Return component mask lists under each connectivity convention."""

    return [
        tuple(
            connected_components(
                grid, background, diagonal=diagonal, same_color=same_color
            )
        )
        for diagonal in (False, True)
        for same_color in (False, True)
    ]


def _object_reduction(
    background: int, variant: int, criterion: str
) -> GridMap:
    """Build a reduction that crops to one connected component."""

    def reduce(grid: IntArray) -> IntArray | None:
        masks = _object_masks(grid, background)[variant]
        if not masks:
            return None
        if criterion == "largest":
            chosen = max(masks, key=lambda mask: int(mask.sum()))
        elif criterion == "smallest":
            chosen = min(masks, key=lambda mask: int(mask.sum()))
        elif criterion == "most_colors":
            chosen = max(masks, key=lambda mask: len(np.unique(grid[mask])))
        elif criterion == "unique_shape":
            shapes = collections.Counter(
                crop_to_mask(grid, mask).shape for mask in masks
            )
            singles = [
                mask for mask in masks if shapes[crop_to_mask(grid, mask).shape] == 1
            ]
            if len(singles) != 1:
                return None
            chosen = singles[0]
        else:
            raise ValueError(f"unknown object criterion: {criterion!r}")
        return crop_to_mask(grid, chosen)

    return reduce


def _panel_reduction(criterion: str, background: int) -> GridMap:
    """Build a reduction that selects one panel of a separator-split grid."""

    def reduce(grid: IntArray) -> IntArray | None:
        panels = split_panels(grid)
        if len(panels) < 2:
            return None
        if criterion == "most_colors":
            return max(panels, key=lambda panel: len(np.unique(panel)))
        if criterion == "fewest_colors":
            return min(panels, key=lambda panel: len(np.unique(panel)))
        if criterion == "most_filled":
            return max(panels, key=lambda panel: int((panel != background).sum()))
        if criterion == "fewest_filled":
            return min(panels, key=lambda panel: int((panel != background).sum()))
        if criterion == "first":
            return panels[0]
        if criterion == "last":
            return panels[-1]
        if criterion == "unique":
            keys = [panel.tobytes() + bytes(panel.shape) for panel in panels]
            counts = collections.Counter(keys)
            singles = [panel for panel, key in zip(panels, keys) if counts[key] == 1]
            return singles[0] if len(singles) == 1 else None
        if criterion == "repeated":
            keys = [panel.tobytes() + bytes(panel.shape) for panel in panels]
            counts = collections.Counter(keys)
            repeats = [panel for panel, key in zip(panels, keys) if counts[key] > 1]
            return repeats[0] if repeats else None
        raise ValueError(f"unknown panel criterion: {criterion!r}")

    return reduce


def _lattice_reduction(grid: IntArray) -> IntArray | None:
    """Collapse a uniform-cell lattice to one pixel per cell."""

    decomposed = lattice_cells(grid)
    if decomposed is None:
        return None
    cells, _ = decomposed
    collapsed = np.empty((len(cells), len(cells[0])), dtype=grid.dtype)
    for row, cell_row in enumerate(cells):
        for column, cell in enumerate(cell_row):
            values = np.unique(cell)
            if values.size != 1:
                return None
            collapsed[row, column] = values[0]
    return collapsed


def _dedup_reduction(grid: IntArray) -> IntArray | None:
    """Collapse runs of identical adjacent rows and columns to one each."""

    rows = np.concatenate(([True], (np.diff(grid, axis=0) != 0).any(axis=1)))
    reduced = grid[rows]
    columns = np.concatenate(([True], (np.diff(reduced, axis=1) != 0).any(axis=0)))
    return np.ascontiguousarray(reduced[:, columns])


def _denoise_reduction(background: int, diagonal: bool) -> GridMap:
    """Build a reduction that erases single-cell components."""

    def reduce(grid: IntArray) -> IntArray | None:
        masks = connected_components(
            grid, background, diagonal=diagonal, same_color=True
        )
        cleaned = grid.copy()
        for mask in masks:
            if int(mask.sum()) == 1:
                cleaned[mask] = background
        return cleaned

    return reduce


def _crop_bbox_reduction(background: int) -> GridMap:
    """Build a reduction that crops to the non-background bounding box."""

    def reduce(grid: IntArray) -> IntArray | None:
        mask = grid != background
        return crop_to_mask(grid, mask) if mask.any() else None

    return reduce


def _crop_color_reduction(color: int) -> GridMap:
    """Build a reduction that crops to the bounding box of one colour."""

    def reduce(grid: IntArray) -> IntArray | None:
        mask = grid == color
        return crop_to_mask(grid, mask) if mask.any() else None

    return reduce


def _reductions(pairs: DemoPairs) -> Iterator[tuple[str, GridMap]]:
    """Yield every named reduction considered for composition."""

    background = background_color(pairs)
    yield "id", lambda grid: grid
    for value in sorted({background, 0}):
        yield f"crop{value}", _crop_bbox_reduction(value)
    for color in range(COLOR_COUNT):
        yield f"cropc{color}", _crop_color_reduction(color)
    for criterion in (
        "most_colors",
        "fewest_colors",
        "most_filled",
        "fewest_filled",
        "first",
        "last",
        "unique",
        "repeated",
    ):
        yield f"panel:{criterion}", _panel_reduction(criterion, background)
    for variant in range(4):
        for criterion in ("largest", "smallest", "most_colors", "unique_shape"):
            yield f"obj{variant}:{criterion}", _object_reduction(
                background, variant, criterion
            )
    yield "lattice", _lattice_reduction
    yield "dedup", _dedup_reduction
    for diagonal in (False, True):
        yield f"denoise{int(diagonal)}", _denoise_reduction(background, diagonal)


# --------------------------------------------------------------------------
# Completions: intermediate grid -> output grid, fitted to reduced pairs
# --------------------------------------------------------------------------


def _fit_color_table(pairs: DemoPairs, geometry: str) -> dict[int, int] | None:
    """Fit a consistent per-colour substitution under a fixed geometry."""

    table: dict[int, int] = {}
    for source, target in pairs:
        moved = apply_dihedral(source, geometry)
        if moved.shape != target.shape:
            return None
        for before, after in zip(moved.ravel().tolist(), target.ravel().tolist()):
            if table.setdefault(before, after) != after:
                return None
    return table


def _completion_geometry(pairs: DemoPairs) -> Iterator[GridRule]:
    """Yield dihedral maps and dihedral-composed colour substitutions."""

    for geometry in DIHEDRAL_NAMES:
        yield GridRule(
            f"d:{geometry}", lambda grid, g=geometry: apply_dihedral(grid, g)
        )
        table = _fit_color_table(pairs, geometry)
        if table is None:
            continue
        lookup = np.arange(COLOR_COUNT, dtype=np.int32)
        for before, after in table.items():
            lookup[before] = after
        known = frozenset(table)

        def recolor(grid: IntArray, g=geometry, lookup=lookup, known=known) -> IntArray | None:
            moved = apply_dihedral(grid, g)
            if any(int(value) not in known for value in np.unique(moved)):
                return None
            return lookup[moved].astype(grid.dtype)

        yield GridRule(f"cm:{geometry}", recolor)


def _completion_constant(pairs: DemoPairs) -> Iterator[GridRule]:
    """Yield the constant completion when every reduced output is identical."""

    first = pairs[0][1]
    if all(
        target.shape == first.shape and np.array_equal(target, first)
        for _, target in pairs
    ):
        yield GridRule("const", lambda _grid, first=first: first.copy())


def _completion_scale(pairs: DemoPairs) -> Iterator[GridRule]:
    """Yield integer per-cell magnification and its exact inverse."""

    source, target = pairs[0]
    if not target.shape[0] % source.shape[0] and not target.shape[1] % source.shape[1]:
        factors = (
            target.shape[0] // source.shape[0],
            target.shape[1] // source.shape[1],
        )
        if factors != (1, 1):
            yield GridRule(
                f"up{factors[0]}x{factors[1]}",
                lambda grid, f=factors: np.kron(grid, np.ones(f, dtype=grid.dtype)),
            )
    if not source.shape[0] % target.shape[0] and not source.shape[1] % target.shape[1]:
        factors = (
            source.shape[0] // target.shape[0],
            source.shape[1] // target.shape[1],
        )
        if factors != (1, 1):

            def shrink(grid: IntArray, f=factors) -> IntArray | None:
                if grid.shape[0] % f[0] or grid.shape[1] % f[1]:
                    return None
                reduced = grid[:: f[0], :: f[1]]
                if not np.array_equal(
                    np.kron(reduced, np.ones(f, dtype=grid.dtype)), grid
                ):
                    return None
                return np.ascontiguousarray(reduced)

            yield GridRule(f"down{factors[0]}x{factors[1]}", shrink)


def _completion_tiling(pairs: DemoPairs) -> Iterator[GridRule]:
    """Yield block tilings whose blocks are per-position dihedral variants."""

    source, target = pairs[0]
    if target.shape[0] % source.shape[0] or target.shape[1] % source.shape[1]:
        return
    rows = target.shape[0] // source.shape[0]
    columns = target.shape[1] // source.shape[1]
    if (rows, columns) == (1, 1) or rows > MAX_SIDE or columns > MAX_SIDE:
        return
    variants = {
        name: apply_dihedral(source, name)
        for name in DIHEDRAL_NAMES
        if apply_dihedral(source, name).shape == source.shape
    }
    layout: list[list[str]] = []
    for row in range(rows):
        row_layout: list[str] = []
        for column in range(columns):
            block = target[
                row * source.shape[0] : (row + 1) * source.shape[0],
                column * source.shape[1] : (column + 1) * source.shape[1],
            ]
            match = next(
                (name for name, grid in variants.items() if np.array_equal(grid, block)),
                None,
            )
            if match is None:
                return
            row_layout.append(match)
        layout.append(row_layout)

    def tile(grid: IntArray, layout=layout) -> IntArray | None:
        blocks = [
            [apply_dihedral(grid, name) for name in row_layout] for row_layout in layout
        ]
        if any(block.shape != grid.shape for row in blocks for block in row):
            return None
        return np.block(blocks)

    yield GridRule(f"tile{len(layout)}x{len(layout[0])}", tile)


def _completion_fractal(pairs: DemoPairs) -> Iterator[GridRule]:
    """Yield self-similar stamping gated by the reduced input's own cells."""

    source, target = pairs[0]
    if target.shape != (source.shape[0] ** 2, source.shape[1] ** 2):
        return
    background = background_color(pairs)
    for invert in (False, True):

        def stamp(grid: IntArray, invert=invert) -> IntArray | None:
            height, width = grid.shape
            if height * height > MAX_SIDE or width * width > MAX_SIDE:
                return None
            gate = grid != background
            if invert:
                gate = ~gate
            out = np.full((height * height, width * width), background, dtype=grid.dtype)
            rows, columns = np.nonzero(gate)
            for row, column in zip(rows.tolist(), columns.tolist()):
                out[
                    row * height : (row + 1) * height,
                    column * width : (column + 1) * width,
                ] = grid
            return out

        yield GridRule(f"frac{int(invert)}", stamp)


def _repair_palette(pairs: DemoPairs) -> list[int]:
    """Return colours present in every source but absent from every target.

    A repair rule erases exactly one colour, so any colour surviving into the
    target cannot be the hole marker. Filtering here keeps the period search off
    nine tenths of the palette.
    """

    present = set(range(COLOR_COUNT))
    for source, target in pairs:
        present &= {int(v) for v in np.unique(source)}
        present -= {int(v) for v in np.unique(target)}
    return sorted(present)


def _completion_periodic(pairs: DemoPairs) -> Iterator[GridRule]:
    """Yield periodic repair, whole-grid and cropped to the repaired region."""

    for noise in _repair_palette(pairs):

        def repair(grid: IntArray, noise=noise) -> IntArray | None:
            return periodic_fill(grid, noise)

        def patch(grid: IntArray, noise=noise) -> IntArray | None:
            filled = periodic_fill(grid, noise)
            mask = grid == noise
            return crop_to_mask(filled, mask) if filled is not None and mask.any() else None

        yield GridRule(f"per{noise}", repair)
        yield GridRule(f"perp{noise}", patch)


def _completion_symmetry(pairs: DemoPairs) -> Iterator[GridRule]:
    """Yield mirror-symmetry repair, whole-grid and cropped."""

    for noise in _repair_palette(pairs):

        def repair(grid: IntArray, noise=noise) -> IntArray | None:
            return _symmetry_fill(grid, noise)

        def patch(grid: IntArray, noise=noise) -> IntArray | None:
            filled = _symmetry_fill(grid, noise)
            mask = grid == noise
            return crop_to_mask(filled, mask) if filled is not None and mask.any() else None

        yield GridRule(f"sym{noise}", repair)
        yield GridRule(f"symp{noise}", patch)


def _symmetry_fill(grid: IntArray, noise: int) -> IntArray | None:
    """Fill cells of one colour using whichever mirror symmetries hold."""

    known = grid != noise
    if known.all() or not known.any():
        return None
    filled = grid.copy()
    for name in ("flip_horizontal", "flip_vertical", "rot180", "transpose", "rot90"):
        moved = apply_dihedral(grid, name)
        if moved.shape != grid.shape:
            continue
        moved_known = apply_dihedral(known.astype(np.int8), name).astype(bool)
        agree = known & moved_known
        if agree.any() and not np.array_equal(grid[agree], moved[agree]):
            continue
        take = (~known) & moved_known
        filled = np.where(take, moved, filled)
        known = known | take
    return filled if known.all() else None


_COMPLETIONS: tuple[Callable[[DemoPairs], Iterator[GridRule]], ...] = (
    _completion_geometry,
    _completion_constant,
    _completion_scale,
    _completion_tiling,
    _completion_fractal,
    _completion_periodic,
    _completion_symmetry,
)

#: Reductions the edit families are composed with. Object-level recolouring of a
#: grid already reduced to a single object is degenerate, and connected-component
#: fitting is the most expensive step in the search, so the edit families run
#: only where an edit is meaningful.
_EDIT_REDUCTIONS = frozenset(
    {"id", "crop0", "dedup", "denoise0", "denoise1"}
)


def _completion_edits(pairs: DemoPairs) -> Iterator[GridRule]:
    """Adapt the same-shape edit families to the completion protocol."""

    for family in EDIT_FAMILIES + CELL_FAMILIES:
        try:
            named = list(family(pairs))
        except _SAFE_ERRORS:
            continue
        for name, apply in named:
            yield GridRule(name, apply)


# --------------------------------------------------------------------------
# Composition search
# --------------------------------------------------------------------------


def fit_verified_rules(demonstrations: DemoPairs) -> tuple[GridRule, ...]:
    """Return every reduction/completion composition that fits all demonstrations.

    Parameters
    ----------
    demonstrations
        Ordered ``(input, output)`` grid pairs. At least one pair is required.

    Returns
    -------
    tuple of GridRule
        Admitted rules in deterministic search order; empty when none fits.

    Raises
    ------
    ValueError
        If ``demonstrations`` is empty.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> pairs = [(np.array([[1, 2]]), np.array([[2, 1]]))]
        >>> "id|d:flip_horizontal" in {r.name for r in fit_verified_rules(pairs)}
        True
    """

    if not demonstrations:
        raise ValueError("fit_verified_rules requires at least one demonstration")
    pairs = [
        (np.asarray(source, np.int32), np.asarray(target, np.int32))
        for source, target in demonstrations
    ]
    admitted: list[GridRule] = []
    seen: set[str] = set()
    for reduction_name, reduction in _reductions(pairs):
        reduced = [(_call(reduction, source), target) for source, target in pairs]
        if any(source is None for source, _ in reduced):
            continue
        reduced_pairs = [(source, target) for source, target in reduced]
        completions = _COMPLETIONS
        if reduction_name in _EDIT_REDUCTIONS or reduction_name.startswith("crop"):
            completions = completions + (_completion_edits,)
        for completion in completions:
            for rule in _safe_iter(completion, reduced_pairs):
                name = f"{reduction_name}|{rule.name}"
                if name in seen:
                    continue
                composed = GridRule(
                    name,
                    lambda grid, r=reduction, c=rule.apply: (
                        None if (mid := _call(r, grid)) is None else _call(c, mid)
                    ),
                )
                if not _reproduces(composed, pairs):
                    continue
                seen.add(name)
                admitted.append(composed)
    return tuple(admitted)


def _safe_iter(
    completion: Callable[[DemoPairs], Iterator[GridRule]], pairs: DemoPairs
) -> Iterable[GridRule]:
    """Materialise a completion family, swallowing expected fitting failures."""

    try:
        return list(completion(pairs))
    except _SAFE_ERRORS:
        return ()


#: Cost at or above which a rule is treated as degenerate: it explains the
#: demonstrations without depending on the input, so it neither corroborates
#: another proposal nor outranks one that does depend on the input.
_DEGENERATE_COST = 20

_COMPLETION_COST: dict[str, int] = {"const": _DEGENERATE_COST, "cm": 2}


def rule_cost(name: str) -> int:
    """Return the parsimony cost of a ``reduction|completion`` rule name.

    A cheaper rule is preferred when several rules explain the demonstrations
    equally well. The identity reduction is free; every other reduction adds a
    step. The constant completion is charged :data:`_DEGENERATE_COST` because it
    ignores the input entirely and therefore fits any single demonstration.

    Parameters
    ----------
    name
        Rule name of the form ``"reduction|completion"``.

    Returns
    -------
    int
        Non-negative cost; lower is preferred.

    Examples
    --------
    .. code-block:: python

        >>> rule_cost("id|d:rot90") < rule_cost("id|const")
        True

        >>> rule_cost("id|d:rot90") < rule_cost("dedup|d:rot90")
        True
    """

    reduction, _, completion = name.partition("|")
    cost = 0 if reduction == "id" else 2
    return cost + _COMPLETION_COST.get(completion.partition(":")[0], 1)


_FIT_MEMO: dict[bytes, tuple[GridRule, ...]] = {}


def _demonstration_key(pairs: DemoPairs) -> bytes:
    """Return a content key identifying one demonstration set exactly."""

    parts: list[bytes] = []
    for source, target in pairs:
        parts.append(bytes(source.shape) + source.astype(np.int32).tobytes())
        parts.append(bytes(target.shape) + target.astype(np.int32).tobytes())
    return b"|".join(parts)


def fit_verified_rules_cached(demonstrations: DemoPairs) -> tuple[GridRule, ...]:
    """Fit verified rules, reusing the result for an identical demonstration set.

    The composition search is a pure function of the demonstrations, so the same
    task's several queries -- and the several evaluation arms that share a
    demonstration set -- need it computed once. This memoizes on grid content,
    never on task identity or arm name, so two arms with different
    demonstrations still fit independently.

    Parameters
    ----------
    demonstrations
        Ordered ``(input, output)`` grid pairs.

    Returns
    -------
    tuple of GridRule
        The same value :func:`fit_verified_rules` would return.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> pairs = [(np.array([[1, 2]]), np.array([[2, 1]]))]
        >>> fit_verified_rules_cached(pairs) == fit_verified_rules_cached(pairs)
        True
    """

    pairs = [
        (np.asarray(source, np.int32), np.asarray(target, np.int32))
        for source, target in demonstrations
    ]
    key = _demonstration_key(pairs)
    cached = _FIT_MEMO.get(key)
    if cached is None:
        cached = fit_verified_rules(pairs)
        _FIT_MEMO[key] = cached
    return cached


def clear_rule_cache() -> None:
    """Drop every memoized fit.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        >>> clear_rule_cache()
    """

    _FIT_MEMO.clear()


def verified_rule_candidates(
    demonstrations: DemoPairs, query_input: IntArray
) -> tuple[tuple[str, IntArray], ...]:
    """Propose deduplicated query candidates from demonstration-verified rules.

    Parameters
    ----------
    demonstrations
        Ordered ``(input, output)`` demonstration grid pairs.
    query_input
        Held-out query input grid.

    Returns
    -------
    tuple
        ``(rule_name, grid)`` pairs, most corroborated first. A grid proposed by
        several rules appears once, credited to the first rule that proposed it,
        and ranked ahead of proposals with fewer votes.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> pairs = [(np.array([[1, 2]]), np.array([[2, 1]]))]
        >>> verified_rule_candidates(pairs, np.array([[3, 4]]))[0][1].tolist()
        [[4, 3]]
    """

    source = np.asarray(query_input, np.int32)
    votes: collections.Counter[bytes] = collections.Counter()
    cheapest: dict[bytes, int] = {}
    first: dict[bytes, tuple[str, IntArray]] = {}
    order: list[bytes] = []
    for rule in fit_verified_rules_cached(demonstrations):
        proposed = _call(rule.apply, source)
        if not is_valid_grid(proposed):
            continue
        key = bytes(proposed.shape) + proposed.astype(np.int32).tobytes()
        cost = rule_cost(rule.name)
        if cost < _DEGENERATE_COST:
            votes[key] += 1
        if key not in first or cost < cheapest[key]:
            first[key] = (rule.name, proposed.astype(np.int32))
        if key not in cheapest:
            cheapest[key] = cost
            order.append(key)
        else:
            cheapest[key] = min(cheapest[key], cost)
    ranked = sorted(
        range(len(order)),
        key=lambda index: (cheapest[order[index]], -votes[order[index]], index),
    )
    return tuple(first[order[index]] for index in ranked)
