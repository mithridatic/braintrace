"""Surface-diversified training-only synthetic ARC curriculum (schema v4).

The eleven operator families of ``direct_synthetic_curriculum_v3`` are
retained with unchanged operator semantics. Only surface statistics are
diversified to span the measured public-training distribution: side lengths
3..30, foreground density 0.05..1.00, one to six foreground colors, four
scene painters (noise, rectangles, lines, blobs), and two to six
demonstrations per task. See
``docs/specs/2026-08-24-example21-diverse-curriculum-v47.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral

import brainstate
import numpy as np

try:
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        canonical_task_fingerprint,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script import fallback.
    from latent_workspace_task import (  # pyright: ignore[reportImplicitRelativeImport]
        ArcGrid,
        ArcPair,
        ArcTask,
        canonical_task_fingerprint,
    )

DIVERSE_CURRICULUM_SCHEMA_VERSION = "direct_synthetic_curriculum_v4"
FAMILIES = (
    "copy",
    "recolor",
    "dihedral",
    "crop",
    "upscale",
    "count",
    "pattern_label",
    "select_marked_region",
    "project_marker",
    "complete_corner",
    "mirror_concat",
)
FAMILY_SCHEDULE = (*FAMILIES, "pattern_label")
PAINTERS = ("noise", "rectangles", "lines", "blobs")


@dataclass(frozen=True)
class DiverseCurriculumConfig:
    """Configure the surface-diversified synthetic curriculum.

    Parameters
    ----------
    task_count : int
        Positive number of tasks generated in family round-robin order.
    max_grid_size : int, default=30
        Maximum generated input or output side length, from 6 through 30.
    min_demonstrations : int, default=2
        Minimum supervised demonstrations per task, from 2 through 10.
    max_demonstrations : int, default=6
        Maximum supervised demonstrations per task, at least the minimum and
        at most 10.
    """

    task_count: int
    max_grid_size: int = 30
    min_demonstrations: int = 2
    max_demonstrations: int = 6

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_count", _positive_integer(self.task_count, "task_count")
        )
        max_grid_size = _positive_integer(self.max_grid_size, "max_grid_size")
        if not 6 <= max_grid_size <= 30:
            raise ValueError("max_grid_size must be in 6..30.")
        object.__setattr__(self, "max_grid_size", max_grid_size)
        min_demos = _positive_integer(self.min_demonstrations, "min_demonstrations")
        max_demos = _positive_integer(self.max_demonstrations, "max_demonstrations")
        if not 2 <= min_demos <= 10:
            raise ValueError("min_demonstrations must be in 2..10.")
        if not min_demos <= max_demos <= 10:
            raise ValueError("max_demonstrations must be in min_demonstrations..10.")
        object.__setattr__(self, "min_demonstrations", min_demos)
        object.__setattr__(self, "max_demonstrations", max_demos)


@dataclass(frozen=True)
class DiverseCurriculum:
    """Hold generated tasks and stable provenance.

    Parameters
    ----------
    tasks : tuple of ArcTask
        Ordered generated tasks.
    family_counts : dict of str to int
        Exact generated count for every curriculum family.
    task_sha256 : str
        Digest of ordered canonical task fingerprints.
    schema_version : str
        Fixed generator contract identifier.
    """

    tasks: tuple[ArcTask, ...]
    family_counts: dict[str, int]
    task_sha256: str
    schema_version: str = DIVERSE_CURRICULUM_SCHEMA_VERSION


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _randint(
    rng: brainstate.random.RandomState,
    low: int,
    high: int,
    size: tuple[int, ...] | None = None,
) -> np.ndarray:
    return np.asarray(rng.randint(low, high, size=size), dtype=np.int32)


def _grid(array: np.ndarray) -> ArcGrid:
    return ArcGrid(tuple(tuple(int(cell) for cell in row) for row in array))


def _pair(input_array: np.ndarray, output_array: np.ndarray) -> ArcPair:
    return ArcPair(_grid(input_array), _grid(output_array))


def _task(family: str, task_index: int, pairs: list[ArcPair]) -> ArcTask:
    return ArcTask(
        train=tuple(pairs[:-1]),
        test=(pairs[-1],),
        task_id=f"synthetic-v4:{family}:{task_index:06d}",
    )


def _dihedral(array: np.ndarray, index: int) -> np.ndarray:
    transformed = np.fliplr(array) if index >= 4 else array
    return np.rot90(transformed, k=index % 4).copy()


def _foreground_box(array: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(array != 0)
    return array[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ].copy()


def _palette(rng: brainstate.random.RandomState, count: int) -> np.ndarray:
    return np.asarray(
        rng.permutation(np.arange(1, 10, dtype=np.int32)), dtype=np.int32
    )[:count]


def _side(rng: brainstate.random.RandomState, limit: int, low: int = 3) -> int:
    return int(_randint(rng, low, limit + 1))


def _noise_scene(
    rng: brainstate.random.RandomState, height: int, width: int
) -> np.ndarray:
    density = int(_randint(rng, 5, 101)) / 100.0
    colors = _palette(rng, int(_randint(rng, 1, 7)))
    mask = np.asarray(rng.random((height, width)), dtype=np.float32) < density
    mask[int(_randint(rng, 0, height)), int(_randint(rng, 0, width))] = True
    picks = _randint(rng, 0, len(colors), size=(height, width))
    return np.where(mask, colors[picks], 0).astype(np.int32)


def _rectangle_scene(
    rng: brainstate.random.RandomState, height: int, width: int
) -> np.ndarray:
    array = np.zeros((height, width), dtype=np.int32)
    colors = _palette(rng, int(_randint(rng, 1, 5)))
    for index in range(int(_randint(rng, 1, 5))):
        box_height = int(_randint(rng, 1, height + 1))
        box_width = int(_randint(rng, 1, width + 1))
        row = int(_randint(rng, 0, height - box_height + 1))
        column = int(_randint(rng, 0, width - box_width + 1))
        array[row : row + box_height, column : column + box_width] = colors[
            index % len(colors)
        ]
    return array


def _line_scene(
    rng: brainstate.random.RandomState, height: int, width: int
) -> np.ndarray:
    array = np.zeros((height, width), dtype=np.int32)
    colors = _palette(rng, int(_randint(rng, 1, 6)))
    for index in range(int(_randint(rng, 1, 6))):
        orientation = int(_randint(rng, 0, 3))
        row = int(_randint(rng, 0, height))
        column = int(_randint(rng, 0, width))
        if orientation == 0:
            delta = (0, 1 if column * 2 < width else -1)
        elif orientation == 1:
            delta = (1 if row * 2 < height else -1, 0)
        else:
            delta = (
                1 if row * 2 < height else -1,
                1 if column * 2 < width else -1,
            )
        length = int(_randint(rng, 1, max(height, width) + 1))
        for step in range(length):
            r, c = row + delta[0] * step, column + delta[1] * step
            if 0 <= r < height and 0 <= c < width:
                array[r, c] = colors[index % len(colors)]
    return array


def _blob_scene(
    rng: brainstate.random.RandomState, height: int, width: int
) -> np.ndarray:
    array = np.zeros((height, width), dtype=np.int32)
    colors = _palette(rng, int(_randint(rng, 1, 5)))
    for index in range(int(_randint(rng, 1, 5))):
        row = int(_randint(rng, 0, height))
        column = int(_randint(rng, 0, width))
        steps = int(_randint(rng, 2, max(3, min((height * width) // 6, 80))))
        for _ in range(steps):
            array[row, column] = colors[index % len(colors)]
            row = min(height - 1, max(0, row + int(_randint(rng, -1, 2))))
            column = min(width - 1, max(0, column + int(_randint(rng, -1, 2))))
    return array


def _scene(
    rng: brainstate.random.RandomState,
    config: DiverseCurriculumConfig,
    *,
    side_limit: int | None = None,
    height_limit: int | None = None,
    width_limit: int | None = None,
) -> np.ndarray:
    limit = min(config.max_grid_size, side_limit or config.max_grid_size)
    height = _side(rng, min(limit, height_limit or limit))
    width = _side(rng, min(limit, width_limit or limit))
    painter = PAINTERS[int(_randint(rng, 0, len(PAINTERS)))]
    if painter == "noise":
        return _noise_scene(rng, height, width)
    if painter == "rectangles":
        return _rectangle_scene(rng, height, width)
    if painter == "lines":
        return _line_scene(rng, height, width)
    return _blob_scene(rng, height, width)


def _demonstration_count(
    rng: brainstate.random.RandomState, config: DiverseCurriculumConfig
) -> int:
    return int(
        _randint(rng, config.min_demonstrations, config.max_demonstrations + 1)
    )


def _copy_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        array = _scene(rng, config)
        pairs.append(_pair(array, array.copy()))
    return _task("copy", task_index, pairs)


def _recolor_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    map_count = int(_randint(rng, 1, 4))
    colors = _palette(rng, 9)
    sources = colors[:map_count]
    targets = colors[map_count : map_count * 2]
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        array = _scene(rng, config)
        for _attempt in range(8):
            if np.isin(array, sources).any():
                break
            array = _scene(rng, config)
        output = array.copy()
        for source, target in zip(sources, targets):
            output = np.where(array == source, int(target), output)
        pairs.append(_pair(array, output.astype(np.int32)))
    return _task("recolor", task_index, pairs)


def _dihedral_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    transform = int(_randint(rng, 1, 8))
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        array = _scene(rng, config)
        pairs.append(_pair(array, _dihedral(array, transform)))
    return _task("dihedral", task_index, pairs)


def _crop_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    by_color = bool(int(_randint(rng, 0, 2)))
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        top = int(_randint(rng, 1, 4))
        left = int(_randint(rng, 1, 4))
        bottom = int(_randint(rng, 1, 4))
        right = int(_randint(rng, 1, 4))
        inner = _scene(
            rng,
            config,
            height_limit=config.max_grid_size - top - bottom,
            width_limit=config.max_grid_size - left - right,
        )
        canvas = np.zeros(
            (inner.shape[0] + top + bottom, inner.shape[1] + left + right),
            dtype=np.int32,
        )
        canvas[top : top + inner.shape[0], left : left + inner.shape[1]] = inner
        if by_color:
            present = np.unique(inner)
            present = present[present != 0]
            color = int(present[int(_randint(rng, 0, len(present)))])
            rows, columns = np.nonzero(canvas == color)
            output = canvas[
                int(rows.min()) : int(rows.max()) + 1,
                int(columns.min()) : int(columns.max()) + 1,
            ].copy()
        else:
            output = _foreground_box(canvas)
        pairs.append(_pair(canvas, output))
    return _task("crop", task_index, pairs)


def _upscale_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    factor = int(_randint(rng, 2, 5))
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        array = _scene(rng, config, side_limit=config.max_grid_size // factor)
        output = np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)
        pairs.append(_pair(array, output))
    return _task("upscale", task_index, pairs)


def _count_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    color = int(_randint(rng, 1, 10))
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        height = _side(rng, config.max_grid_size)
        width = _side(rng, config.max_grid_size)
        count = int(_randint(rng, 1, min(9, height * width) + 1))
        positions = np.asarray(rng.permutation(height * width), dtype=np.int32)[
            :count
        ]
        array = np.zeros((height, width), dtype=np.int32)
        array.reshape(-1)[positions] = color
        pairs.append(_pair(array, np.asarray([[count]], dtype=np.int32)))
    return _task("count", task_index, pairs)


def _pattern_label_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    class_count = int(_randint(rng, 2, 5))
    labels = _palette(rng, 9)[:class_count]
    patterns = []
    for _ in range(class_count):
        density = (int(_randint(rng, 20, 81))) / 100.0
        pattern = np.asarray(rng.random((3, 3)), dtype=np.float32) < density
        pattern[int(_randint(rng, 0, 3)), int(_randint(rng, 0, 3))] = True
        patterns.append(pattern)
    demonstrations = _demonstration_count(rng, config)
    pairs = []
    for index in range(demonstrations):
        class_index = index % class_count
        color = int(_randint(rng, 1, 10))
        input_array = np.where(patterns[class_index], color, 0).astype(np.int32)
        pairs.append(
            _pair(input_array, np.asarray([[labels[class_index]]], dtype=np.int32))
        )
    query_class = int(_randint(rng, 0, class_count))
    query_color = int(_randint(rng, 1, 10))
    query_input = np.where(patterns[query_class], query_color, 0).astype(np.int32)
    query = _pair(query_input, np.asarray([[labels[query_class]]], dtype=np.int32))
    return ArcTask(
        train=tuple(pairs),
        test=(query,),
        task_id=f"synthetic-v4:pattern_label:{task_index:06d}",
    )


def _select_marked_region_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        block_limit = min(15, config.max_grid_size // 2)
        block_height = int(_randint(rng, 3, block_limit + 1))
        block_width = int(_randint(rng, 3, block_limit + 1))
        colors = _palette(rng, 5)
        bases = colors[:4]
        marker = int(colors[4])
        counts = np.asarray(rng.permutation(np.arange(1, 5)), dtype=np.int32)
        array = np.zeros((2 * block_height, 2 * block_width), dtype=np.int32)
        for region_index, (row_start, column_start) in enumerate(
            (
                (0, 0),
                (0, block_width),
                (block_height, 0),
                (block_height, block_width),
            )
        ):
            region = np.full(
                (block_height, block_width), bases[region_index], dtype=np.int32
            )
            positions = np.asarray(
                rng.permutation(block_height * block_width), dtype=np.int32
            )[: int(counts[region_index])]
            region.reshape(-1)[positions] = marker
            array[
                row_start : row_start + block_height,
                column_start : column_start + block_width,
            ] = region
        winning_base = int(bases[int(np.argmax(counts))])
        pairs.append(_pair(array, np.asarray([[winning_base]], dtype=np.int32)))
    return _task("select_marked_region", task_index, pairs)


def _project_marker_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    colors = _palette(rng, 2)
    marker, connector = int(colors[0]), int(colors[1])
    direction_index = int(_randint(rng, 0, 4))
    row_delta, column_delta = ((1, 0), (-1, 0), (0, 1), (0, -1))[direction_index]
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        side = _side(rng, config.max_grid_size, low=4)
        connector_length = int(_randint(rng, 2, min(6, side - 2) + 1))
        if row_delta > 0:
            row = int(_randint(rng, 0, side - connector_length))
            column = int(_randint(rng, 0, side))
        elif row_delta < 0:
            row = int(_randint(rng, connector_length, side))
            column = int(_randint(rng, 0, side))
        elif column_delta > 0:
            row = int(_randint(rng, 0, side))
            column = int(_randint(rng, 0, side - connector_length))
        else:
            row = int(_randint(rng, 0, side))
            column = int(_randint(rng, connector_length, side))
        input_array = np.zeros((side, side), dtype=np.int32)
        input_array[row, column] = marker
        for offset in range(1, connector_length + 1):
            input_array[
                row + row_delta * offset,
                column + column_delta * offset,
            ] = connector
        output_array = input_array.copy()
        output_array[row, column] = 0
        output_array[
            row + row_delta * connector_length,
            column + column_delta * connector_length,
        ] = marker
        pairs.append(_pair(input_array, output_array))
    return _task("project_marker", task_index, pairs)


def _complete_corner_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    colors = _palette(rng, 2)
    shape_color, completion_color = int(colors[0]), int(colors[1])
    missing_index = int(_randint(rng, 0, 4))
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        side = _side(rng, config.max_grid_size, low=5)
        candidates = np.asarray(
            [
                (row, column)
                for row in range(0, side - 1, 3)
                for column in range(0, side - 1, 3)
            ],
            dtype=np.int32,
        )
        object_count = int(_randint(rng, 1, min(6, len(candidates)) + 1))
        selected = candidates[
            np.asarray(rng.permutation(len(candidates)), dtype=np.int32)[
                :object_count
            ]
        ]
        input_array = np.zeros((side, side), dtype=np.int32)
        output_array = np.zeros_like(input_array)
        for row, column in selected:
            input_block = np.full((2, 2), shape_color, dtype=np.int32)
            input_block.reshape(-1)[missing_index] = 0
            output_block = input_block.copy()
            output_block.reshape(-1)[missing_index] = completion_color
            input_array[row : row + 2, column : column + 2] = input_block
            output_array[row : row + 2, column : column + 2] = output_block
        pairs.append(_pair(input_array, output_array))
    return _task("complete_corner", task_index, pairs)


def _mirror_concat_task(
    task_index: int,
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    direction = int(_randint(rng, 0, 4))
    pairs = []
    for _ in range(_demonstration_count(rng, config) + 1):
        if direction < 2:
            array = _scene(rng, config, height_limit=config.max_grid_size // 2)
        else:
            array = _scene(rng, config, width_limit=config.max_grid_size // 2)
        if direction == 0:
            output_array = np.concatenate((np.flipud(array), array), axis=0)
        elif direction == 1:
            output_array = np.concatenate((array, np.flipud(array)), axis=0)
        elif direction == 2:
            output_array = np.concatenate((np.fliplr(array), array), axis=1)
        else:
            output_array = np.concatenate((array, np.fliplr(array)), axis=1)
        if output_array.shape[0] > config.max_grid_size or (
            output_array.shape[1] > config.max_grid_size
        ):  # pragma: no cover - guarded by side_limit.
            raise ValueError("mirror_concat output exceeds the grid bound.")
        pairs.append(_pair(array, output_array))
    return _task("mirror_concat", task_index, pairs)


_GENERATORS = {
    "copy": _copy_task,
    "recolor": _recolor_task,
    "dihedral": _dihedral_task,
    "crop": _crop_task,
    "upscale": _upscale_task,
    "count": _count_task,
    "pattern_label": _pattern_label_task,
    "select_marked_region": _select_marked_region_task,
    "project_marker": _project_marker_task,
    "complete_corner": _complete_corner_task,
    "mirror_concat": _mirror_concat_task,
}


def generate_diverse_curriculum(
    config: DiverseCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> DiverseCurriculum:
    """Generate surface-diversified synthetic ARC tasks for training only.

    Parameters
    ----------
    config : DiverseCurriculumConfig
        Validated curriculum dimensions.
    rng : brainstate.random.RandomState
        Sole random source used by every family.

    Returns
    -------
    DiverseCurriculum
        Ordered tasks, exact family counts, and canonical digest.
    """

    if not isinstance(config, DiverseCurriculumConfig):
        raise TypeError("config must be a DiverseCurriculumConfig instance.")
    if not isinstance(rng, brainstate.random.RandomState):
        raise TypeError("rng must be a brainstate.random.RandomState.")
    tasks = []
    family_counts = {family: 0 for family in FAMILIES}
    for task_index in range(config.task_count):
        family = FAMILY_SCHEDULE[task_index % len(FAMILY_SCHEDULE)]
        task = _GENERATORS[family](task_index, config, rng)
        tasks.append(task)
        family_counts[family] += 1
    fingerprints = "\n".join(canonical_task_fingerprint(task) for task in tasks)
    return DiverseCurriculum(
        tasks=tuple(tasks),
        family_counts=family_counts,
        task_sha256=hashlib.sha256(fingerprints.encode("ascii")).hexdigest(),
    )
