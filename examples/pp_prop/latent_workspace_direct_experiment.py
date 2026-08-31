"""Staged ARC corpus training and evaluation for direct model generation."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import subprocess
import time
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import brainstate
import jax
import msgspec
import numpy as np

try:
    from examples.pp_prop.latent_workspace_direct_curriculum import (
        SyntheticCurriculumConfig,
        generate_synthetic_curriculum,
    )
    from examples.pp_prop.latent_workspace_direct_generation import (
        DirectPredictionLogits,
        decode_first_candidate,
        first_prediction_bytes,
        strict_task_pass_at_1,
    )
    from examples.pp_prop.latent_workspace_direct_model import (
        MAX_GRID_SIZE,
        DirectARCGRU,
        DirectModelConfig,
    )
    from examples.pp_prop.latent_workspace_direct_training import (
        DirectBPTTTrainer,
        DirectEpisode,
        DirectTrainingChunk,
        encode_direct_episode,
        leave_one_out_tasks,
        load_direct_checkpoint,
        parameter_digest,
        save_direct_checkpoint,
        stack_direct_episodes,
    )
    from examples.pp_prop.latent_workspace_task import (
        ArcTask,
        AugmentationConfig,
        DatasetSource,
        LoadedDataset,
        RowEventConfig,
        assert_no_evaluation_leakage,
        associative_memory_feature_indices,
        augment_training_task,
        canonical_task_fingerprint,
        load_dataset_source,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script import fallback.
    from latent_workspace_direct_curriculum import (
        SyntheticCurriculumConfig,
        generate_synthetic_curriculum,
    )
    from latent_workspace_direct_generation import (
        DirectPredictionLogits,
        decode_first_candidate,
        first_prediction_bytes,
        strict_task_pass_at_1,
    )
    from latent_workspace_direct_model import (
        MAX_GRID_SIZE,
        DirectARCGRU,
        DirectModelConfig,
    )
    from latent_workspace_direct_training import (
        DirectBPTTTrainer,
        DirectEpisode,
        DirectTrainingChunk,
        encode_direct_episode,
        leave_one_out_tasks,
        load_direct_checkpoint,
        parameter_digest,
        save_direct_checkpoint,
        stack_direct_episodes,
    )
    from latent_workspace_task import (
        ArcTask,
        AugmentationConfig,
        DatasetSource,
        LoadedDataset,
        RowEventConfig,
        assert_no_evaluation_leakage,
        associative_memory_feature_indices,
        augment_training_task,
        canonical_task_fingerprint,
        load_dataset_source,
    )

EVALUATION_BATCH_SIZE = 10


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return integer


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real.")
    return number


@dataclass(frozen=True)
class DirectExperimentConfig:
    """Configure a staged direct-model ARC training run.

    Parameters
    ----------
    source_manifest : pathlib.Path
        Manifest containing public train and evaluation sources.
    output_dir : pathlib.Path
        Artifact directory.
    initial_checkpoint : pathlib.Path, optional
        Exact-schema direct-model checkpoint used to initialize parameters.
    device : {"cpu", "gpu"}, default="gpu"
        Required JAX platform.
    seed : int, default=2108
        Model, sampling, augmentation, and split seed.
    validation_task_count : int, default=80
        Number of ARC training tasks held out by fingerprint for validation.
    training_updates, training_chunk_size, training_batch_size : int
        Optimizer schedule and compiled chunk dimensions.
    synthetic_pretraining_updates, synthetic_task_count : int, default=0
        Optional training-only curriculum schedule. Both are zero or positive.
    synthetic_demonstrations : int, default=4
        Demonstrations per generated task.
    synthetic_max_grid_size : int, default=12
        Maximum generated grid side length.
    synthetic_seed : int, default=12108
        Independent BrainState curriculum-generation seed.
    learning_rate : float, default=0.001
        Adam learning rate.
    encoder_width, hidden_width, decoder_width, recurrent_layers : int
        Direct BrainTrace GRU architecture.
    augment : bool, default=True
        Enable task-independent color, dihedral, and demonstration-order
        augmentation on training episodes only.
    evaluate_complete_manifest : bool, default=False
        Score ARC evaluation only after explicit promotion; otherwise score the
        fixed training-task validation split.
    """

    source_manifest: pathlib.Path
    output_dir: pathlib.Path
    initial_checkpoint: pathlib.Path | None = None
    device: str = "gpu"
    seed: int = 2108
    validation_task_count: int = 80
    training_updates: int = 100
    training_chunk_size: int = 5
    training_batch_size: int = 8
    synthetic_pretraining_updates: int = 0
    synthetic_task_count: int = 0
    synthetic_demonstrations: int = 4
    synthetic_max_grid_size: int = 12
    synthetic_seed: int = 12108
    learning_rate: float = 0.001
    encoder_width: int = 128
    hidden_width: int = 256
    decoder_width: int = 128
    recurrent_layers: int = 2
    augment: bool = True
    evaluate_complete_manifest: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_manifest", pathlib.Path(self.source_manifest))
        object.__setattr__(self, "output_dir", pathlib.Path(self.output_dir))
        if self.initial_checkpoint is not None:
            object.__setattr__(
                self, "initial_checkpoint", pathlib.Path(self.initial_checkpoint)
            )
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'.")
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "synthetic_pretraining_updates",
            _nonnegative_integer(
                self.synthetic_pretraining_updates,
                "synthetic_pretraining_updates",
            ),
        )
        object.__setattr__(
            self,
            "synthetic_task_count",
            _nonnegative_integer(self.synthetic_task_count, "synthetic_task_count"),
        )
        object.__setattr__(
            self,
            "synthetic_seed",
            _nonnegative_integer(self.synthetic_seed, "synthetic_seed"),
        )
        for name in (
            "validation_task_count",
            "training_updates",
            "training_chunk_size",
            "training_batch_size",
            "encoder_width",
            "hidden_width",
            "decoder_width",
            "recurrent_layers",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if self.training_updates % self.training_chunk_size:
            raise ValueError("training_chunk_size must divide training_updates.")
        if bool(self.synthetic_pretraining_updates) != bool(self.synthetic_task_count):
            raise ValueError(
                "synthetic_pretraining_updates and synthetic_task_count must "
                "both be zero or both be positive."
            )
        if self.synthetic_pretraining_updates % self.training_chunk_size:
            raise ValueError(
                "training_chunk_size must divide synthetic_pretraining_updates."
            )
        curriculum_validation = SyntheticCurriculumConfig(
            task_count=max(1, self.synthetic_task_count),
            demonstrations=self.synthetic_demonstrations,
            max_grid_size=self.synthetic_max_grid_size,
        )
        object.__setattr__(
            self, "synthetic_demonstrations", curriculum_validation.demonstrations
        )
        object.__setattr__(
            self, "synthetic_max_grid_size", curriculum_validation.max_grid_size
        )
        object.__setattr__(
            self,
            "learning_rate",
            _positive_real(self.learning_rate, "learning_rate"),
        )
        if not isinstance(self.augment, bool) or not isinstance(
            self.evaluate_complete_manifest, bool
        ):
            raise TypeError("augment and evaluate_complete_manifest must be boolean.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible experiment configuration.

        Returns
        -------
        dict
            Stable configuration mapping.
        """

        value = asdict(self)
        value["source_manifest"] = str(self.source_manifest)
        value["output_dir"] = str(self.output_dir)
        value["initial_checkpoint"] = (
            None if self.initial_checkpoint is None else str(self.initial_checkpoint)
        )
        return value


@dataclass(frozen=True)
class DirectCorpora:
    """Hold resolved train/evaluation tasks and source evidence.

    Parameters
    ----------
    training, evaluation : tuple of ArcTask
        Role-separated public ARC tasks.
    loaded : tuple of LoadedDataset
        Complete resolved source evidence.
    """

    training: tuple[ArcTask, ...]
    evaluation: tuple[ArcTask, ...]
    loaded: tuple[LoadedDataset, ...]


def source_declarations(path: pathlib.Path) -> tuple[DatasetSource, ...]:
    """Parse fail-closed ARC source declarations.

    Parameters
    ----------
    path : pathlib.Path
        JSON manifest with a nonempty ``sources`` list.

    Returns
    -------
    tuple of DatasetSource
        Resolved declarations with paths relative to the manifest.
    """

    path = pathlib.Path(path)
    try:
        payload = msgspec.json.decode(path.read_bytes())
    except (OSError, msgspec.DecodeError) as error:
        raise ValueError(f"Cannot read source manifest {path}: {error}") from error
    values = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("Source manifest must contain a nonempty sources list.")
    declarations = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise TypeError(f"sources[{index}] must be an object.")
        required = {"name", "role", "version", "path", "license_reference"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"sources[{index}] is missing {sorted(missing)}.")
        source_path = pathlib.Path(str(value["path"]))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        declarations.append(
            DatasetSource(
                name=value["name"],
                role=value["role"],
                version=value["version"],
                path=str(source_path),
                license_reference=value["license_reference"],
                format=value.get("format", "auto"),
                exclude_fingerprints=tuple(value.get("exclude_fingerprints", ())),
            )
        )
    return tuple(declarations)


def load_corpora(path: pathlib.Path) -> DirectCorpora:
    """Load role-separated ARC corpora and verify split isolation.

    Parameters
    ----------
    path : pathlib.Path
        Source manifest path.

    Returns
    -------
    DirectCorpora
        Training tasks, evaluation tasks, and source evidence.
    """

    loaded = tuple(load_dataset_source(item) for item in source_declarations(path))
    assert_no_evaluation_leakage(item.manifest for item in loaded)
    training = tuple(
        task
        for item in loaded
        if item.manifest.source.role == "train"
        for task in item.tasks
    )
    evaluation = tuple(
        task
        for item in loaded
        if item.manifest.source.role == "evaluation"
        for task in item.tasks
    )
    if not training or not evaluation:
        raise ValueError("Both train and evaluation sources are required.")
    return DirectCorpora(training, evaluation, loaded)


def deterministic_task_split(
    tasks: tuple[ArcTask, ...], validation_task_count: int
) -> tuple[tuple[ArcTask, ...], tuple[ArcTask, ...]]:
    """Split tasks deterministically by content fingerprint.

    Parameters
    ----------
    tasks : tuple of ArcTask
        Public ARC training tasks.
    validation_task_count : int
        Positive number held out from fitting.

    Returns
    -------
    tuple
        ``(fit_tasks, validation_tasks)`` with no shared fingerprint.
    """

    count = _positive_integer(validation_task_count, "validation_task_count")
    if count >= len(tasks):
        raise ValueError("validation_task_count must be smaller than task count.")
    ordered = tuple(sorted(tasks, key=canonical_task_fingerprint))
    validation = ordered[:count]
    fitting = ordered[count:]
    return fitting, validation


def training_episode_catalog(tasks: tuple[ArcTask, ...]) -> tuple[ArcTask, ...]:
    """Build target-isolated LOO and official-query training episodes.

    Parameters
    ----------
    tasks : tuple of ArcTask
        Fitting-only ARC tasks.

    Returns
    -------
    tuple of ArcTask
        Episodes whose query targets remain out of model inputs.
    """

    catalog = []
    for task in tasks:
        if len(task.train) >= 2:
            catalog.extend(leave_one_out_tasks(task))
        for query in task.test:
            if query.output is not None:
                catalog.append(
                    ArcTask(train=task.train, test=(query,), task_id=task.task_id)
                )
    if not catalog:
        raise ValueError("Training episode catalog is empty.")
    return tuple(catalog)


def evaluation_episodes(
    tasks: tuple[ArcTask, ...], row_config: RowEventConfig
) -> tuple[DirectEpisode, ...]:
    """Encode official task queries without augmenting or fitting them.

    Parameters
    ----------
    tasks : tuple of ArcTask
        Validation or evaluation tasks with scorer-side outputs.
    row_config : RowEventConfig
        Fixed row-event layout.

    Returns
    -------
    tuple of DirectEpisode
        Manifest-ordered target-free inputs and scorer targets.
    """

    episodes = []
    for task in tasks:
        for query_index, query in enumerate(task.test):
            if query.output is None:
                raise ValueError("Evaluation queries require scorer-side targets.")
            episodes.append(encode_direct_episode(task, query_index, row_config))
    return tuple(episodes)


def sample_training_chunk(
    catalog: tuple[ArcTask, ...],
    row_config: RowEventConfig,
    rng: brainstate.random.RandomState,
    *,
    updates: int,
    batch_size: int,
    augment: bool,
) -> DirectTrainingChunk:
    """Encode update-major batches using only BrainState randomness.

    Parameters
    ----------
    catalog : tuple of ArcTask
        Fitting-only target-isolated episode catalog.
    row_config : RowEventConfig
        Fixed row-event layout.
    rng : brainstate.random.RandomState
        Sampling and augmentation stream.
    updates, batch_size : int
        Positive chunk dimensions.
    augment : bool
        Whether to apply training-only relation-preserving augmentation.

    Returns
    -------
    DirectTrainingChunk
        Host-encoded batches for one compiled optimizer loop.
    """

    update_count = _positive_integer(updates, "updates")
    batch_count = _positive_integer(batch_size, "batch_size")
    if not isinstance(rng, brainstate.random.RandomState):
        raise TypeError("rng must be a brainstate.random.RandomState.")
    if not isinstance(augment, bool):
        raise TypeError("augment must be boolean.")
    indices = np.asarray(
        rng.randint(0, len(catalog), size=(update_count, batch_count)),
        dtype=np.int32,
    )
    batches = []
    augmentation = AugmentationConfig(
        permute_colors=augment,
        dihedral=augment,
        shuffle_demonstrations=augment,
    )
    for update_indices in indices:
        episodes = []
        for index in update_indices:
            task = catalog[int(index)]
            if augment:
                task = augment_training_task(
                    task, rng, role="train", config=augmentation
                )
            episodes.append(encode_direct_episode(task, 0, row_config))
        batches.append(stack_direct_episodes(tuple(episodes)))
    return DirectTrainingChunk(
        events=np.stack([batch.events for batch in batches]),
        query_features=np.stack([batch.query_features for batch in batches]),
        shape_features=np.stack([batch.shape_features for batch in batches]),
        target_colors=np.stack([batch.target_colors for batch in batches]),
        target_mask=np.stack([batch.target_mask for batch in batches]),
        target_heights=np.stack([batch.target_heights for batch in batches]),
        target_widths=np.stack([batch.target_widths for batch in batches]),
    )


def evaluate_model(
    model: DirectARCGRU, episodes: tuple[DirectEpisode, ...]
) -> dict[str, object]:
    """Run one compiled recurrent evaluation and score strict task pass-at-one.

    Parameters
    ----------
    model : DirectARCGRU
        Frozen trained model.
    episodes : tuple of DirectEpisode
        Complete query population in manifest order.

    Returns
    -------
    dict
        Exact strict score, memberships, candidates, and candidate digest.
    """

    if not episodes:
        raise ValueError("Evaluation episodes must be nonempty.")
    batch_size = min(EVALUATION_BATCH_SIZE, len(episodes))
    padding = (-len(episodes)) % batch_size
    padded_episodes = episodes + (episodes[-1],) * padding
    batches = tuple(
        stack_direct_episodes(
            padded_episodes[index : index + batch_size]
        )
        for index in range(0, len(padded_episodes), batch_size)
    )
    brainstate.nn.init_all_states(model, batch_size=batch_size)

    def run_batch(
        events: jax.Array,
        query_features: jax.Array,
        shape_features: jax.Array,
    ):
        brainstate.nn.reset_all_states(model, batch_size=batch_size)
        return model.run(events, query_features, shape_features)

    @brainstate.transform.jit
    def run_all(
        events: jax.Array,
        query_features: jax.Array,
        shape_features: jax.Array,
    ):
        return brainstate.transform.for_loop(
            run_batch,
            events,
            query_features,
            shape_features,
        )

    height, width, colors = run_all(
        jax.numpy.asarray(np.stack([batch.events for batch in batches])),
        jax.numpy.asarray(np.stack([batch.query_features for batch in batches])),
        jax.numpy.asarray(np.stack([batch.shape_features for batch in batches])),
    )
    height_array = np.asarray(height).reshape(-1, MAX_GRID_SIZE)[: len(episodes)]
    width_array = np.asarray(width).reshape(-1, MAX_GRID_SIZE)[: len(episodes)]
    color_array = np.asarray(colors).reshape(
        -1, MAX_GRID_SIZE, MAX_GRID_SIZE, 10
    )[: len(episodes)]
    dependencies = tuple(
        ".".join(map(str, path)) for path in model.states(brainstate.ParamState)
    )
    candidates = []
    predictions = []
    targets = []
    task_ids = []
    for index, episode in enumerate(episodes):
        candidate = decode_first_candidate(
            DirectPredictionLogits(
                height=height_array[index],
                width=width_array[index],
                colors=color_array[index],
                parameter_dependencies=dependencies,
            )
        )
        candidates.append(candidate)
        predictions.append(candidate["grid"])
        target_height = episode.target_height + 1
        target_width = episode.target_width + 1
        targets.append(episode.target_colors[:target_height, :target_width].tolist())
        task_ids.append(episode.task_id)
    strict = strict_task_pass_at_1(predictions, targets, task_ids)
    candidate_bytes = first_prediction_bytes(candidates)
    return {
        **strict,
        "query_count": len(episodes),
        "task_count": len(set(task_ids)),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes_size": len(candidate_bytes),
        "candidates": candidates,
    }


def _source_revision() -> tuple[str, bool]:
    explicit_revision = os.environ.get("BRAINTRACE_SOURCE_REVISION")
    explicit_dirty = os.environ.get("BRAINTRACE_SOURCE_DIRTY")
    if explicit_revision is not None or explicit_dirty is not None:
        if explicit_revision is None or explicit_dirty is None:
            raise ValueError(
                "BRAINTRACE_SOURCE_REVISION and BRAINTRACE_SOURCE_DIRTY must be set together."
            )
        if len(explicit_revision) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in explicit_revision
        ):
            raise ValueError(
                "BRAINTRACE_SOURCE_REVISION must be a 40-character hex hash."
            )
        normalized_dirty = explicit_dirty.lower()
        if normalized_dirty not in {"true", "false"}:
            raise ValueError("BRAINTRACE_SOURCE_DIRTY must be true or false.")
        return explicit_revision.lower(), normalized_dirty == "true"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    return revision, dirty


def run_experiment(config: DirectExperimentConfig) -> dict[str, object]:
    """Train one direct model and write its validation or full-evaluation result.

    Parameters
    ----------
    config : DirectExperimentConfig
        Fully declared experiment configuration.

    Returns
    -------
    dict
        Bound training, data, model, and exact-score evidence.
    """

    if not isinstance(config, DirectExperimentConfig):
        raise TypeError("config must be a DirectExperimentConfig instance.")
    devices = jax.devices(config.device)
    if not devices:
        raise RuntimeError(f"Requested JAX {config.device} backend is unavailable.")
    device = devices[0]
    with jax.default_device(device):
        corpora = load_corpora(config.source_manifest)
        fitting, validation = deterministic_task_split(
            corpora.training, config.validation_task_count
        )
        row_config = RowEventConfig(max_demonstrations=10, max_grid_size=30)
        memory_features = associative_memory_feature_indices(row_config)
        catalog = training_episode_catalog(fitting)
        scored_tasks = (
            corpora.evaluation if config.evaluate_complete_manifest else validation
        )
        scored_episodes = evaluation_episodes(scored_tasks, row_config)
        model_config = DirectModelConfig(
            input_width=row_config.input_width,
            encoder_width=config.encoder_width,
            hidden_width=config.hidden_width,
            decoder_width=config.decoder_width,
            recurrent_layers=config.recurrent_layers,
            seed=config.seed,
            memory_key_indices=memory_features.key_indices,
            memory_value_indices=memory_features.value_indices,
            memory_key_color_block_width=(
                row_config.max_grid_size * row_config.color_count
            ),
        )
        initial_checkpoint_evidence = None
        if config.initial_checkpoint is None:
            model = DirectARCGRU(model_config)
        else:
            model, initial_metadata = load_direct_checkpoint(config.initial_checkpoint)
            if model.config != model_config:
                raise ValueError(
                    "initial checkpoint architecture does not match the experiment."
                )
            initial_checkpoint_evidence = {
                "path": str(config.initial_checkpoint),
                "file_sha256": hashlib.sha256(
                    config.initial_checkpoint.read_bytes()
                ).hexdigest(),
                "parameter_sha256": initial_metadata["parameter_sha256"],
            }
        before = parameter_digest(model)
        evaluation_before_training = evaluate_model(model, scored_episodes)
        trainer = DirectBPTTTrainer(
            model,
            batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
        )
        synthetic_pretraining = None
        if config.synthetic_pretraining_updates:
            curriculum = generate_synthetic_curriculum(
                SyntheticCurriculumConfig(
                    task_count=config.synthetic_task_count,
                    demonstrations=config.synthetic_demonstrations,
                    max_grid_size=config.synthetic_max_grid_size,
                ),
                brainstate.random.RandomState(config.synthetic_seed),
            )
            synthetic_catalog = training_episode_catalog(curriculum.tasks)
            synthetic_rng = brainstate.random.RandomState(config.synthetic_seed + 1)
            synthetic_losses = []
            synthetic_started = time.perf_counter()
            synthetic_chunk_count = (
                config.synthetic_pretraining_updates // config.training_chunk_size
            )
            for chunk_index in range(synthetic_chunk_count):
                chunk = sample_training_chunk(
                    synthetic_catalog,
                    row_config,
                    synthetic_rng,
                    updates=config.training_chunk_size,
                    batch_size=config.training_batch_size,
                    augment=False,
                )
                observed = np.asarray(trainer.train_chunk(chunk), dtype=np.float64)
                synthetic_losses.extend(observed.tolist())
                print(
                    f"[direct-synthetic] chunk={chunk_index + 1}/"
                    f"{synthetic_chunk_count} loss={observed[-1]:.6f}",
                    flush=True,
                )
            synthetic_pretraining = {
                "schema_version": curriculum.schema_version,
                "seed": config.synthetic_seed,
                "task_count": len(curriculum.tasks),
                "family_counts": curriculum.family_counts,
                "task_sha256": curriculum.task_sha256,
                "training_episode_count": len(synthetic_catalog),
                "losses": synthetic_losses,
                "finite": bool(np.all(np.isfinite(synthetic_losses))),
                "wall_seconds": time.perf_counter() - synthetic_started,
            }
        rng = brainstate.random.RandomState(config.seed + 1)
        losses = []
        started = time.perf_counter()
        chunk_count = config.training_updates // config.training_chunk_size
        for chunk_index in range(chunk_count):
            chunk = sample_training_chunk(
                catalog,
                row_config,
                rng,
                updates=config.training_chunk_size,
                batch_size=config.training_batch_size,
                augment=config.augment,
            )
            observed = np.asarray(trainer.train_chunk(chunk), dtype=np.float64)
            losses.extend(observed.tolist())
            print(
                f"[direct-arc] chunk={chunk_index + 1}/{chunk_count} "
                f"loss={observed[-1]:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - started
        after = parameter_digest(model)
        evaluation = evaluate_model(model, scored_episodes)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.output_dir / "checkpoint.npz"
        checkpoint_digest = save_direct_checkpoint(model, checkpoint_path)
        checkpoint_file_digest = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
    revision, dirty = _source_revision()
    result = {
        "schema_version": 1,
        "configuration": config.to_dict(),
        "source_revision": revision,
        "source_dirty": dirty,
        "device": {
            "platform": str(device.platform),
            "kind": str(getattr(device, "device_kind", device)),
        },
        "data": {
            "training_task_count": len(corpora.training),
            "fitting_task_count": len(fitting),
            "validation_task_count": len(validation),
            "evaluation_task_count": len(corpora.evaluation),
            "training_episode_count": len(catalog),
            "scored_split": (
                "complete_evaluation"
                if config.evaluate_complete_manifest
                else "training_task_validation"
            ),
            "sources": [item.manifest.to_dict() for item in corpora.loaded],
        },
        "model": {
            "architecture": asdict(model_config),
            "parameter_sha256_before": before,
            "parameter_sha256_after": after,
            "parameters_moved": before != after,
        },
        "training": {
            "losses": losses,
            "finite": bool(np.all(np.isfinite(losses))),
            "wall_seconds": training_seconds,
        },
        "checkpoint": {
            "filename": checkpoint_path.name,
            "parameter_sha256": checkpoint_digest,
            "file_sha256": checkpoint_file_digest,
        },
        "initial_checkpoint": initial_checkpoint_evidence,
        "synthetic_pretraining": synthetic_pretraining,
        "evaluation_before_training": evaluation_before_training,
        "evaluation": evaluation,
    }
    result_path = config.output_dir / "result.json"
    result_path.write_bytes(msgspec.json.encode(result, order="sorted"))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--initial-checkpoint", type=pathlib.Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--validation-task-count", type=int, default=80)
    parser.add_argument("--training-updates", type=int, default=100)
    parser.add_argument("--training-chunk-size", type=int, default=5)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--synthetic-pretraining-updates", type=int, default=0)
    parser.add_argument("--synthetic-task-count", type=int, default=0)
    parser.add_argument("--synthetic-demonstrations", type=int, default=4)
    parser.add_argument("--synthetic-max-grid-size", type=int, default=12)
    parser.add_argument("--synthetic-seed", type=int, default=12108)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--decoder-width", type=int, default=128)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--evaluate-complete-manifest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the direct ARC experiment command-line interface.

    Parameters
    ----------
    argv : list of str, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    int
        Zero after writing a complete result artifact.
    """

    args = _parser().parse_args(argv)
    config = DirectExperimentConfig(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        initial_checkpoint=args.initial_checkpoint,
        device=args.device,
        seed=args.seed,
        validation_task_count=args.validation_task_count,
        training_updates=args.training_updates,
        training_chunk_size=args.training_chunk_size,
        training_batch_size=args.training_batch_size,
        synthetic_pretraining_updates=args.synthetic_pretraining_updates,
        synthetic_task_count=args.synthetic_task_count,
        synthetic_demonstrations=args.synthetic_demonstrations,
        synthetic_max_grid_size=args.synthetic_max_grid_size,
        synthetic_seed=args.synthetic_seed,
        learning_rate=args.learning_rate,
        encoder_width=args.encoder_width,
        hidden_width=args.hidden_width,
        decoder_width=args.decoder_width,
        recurrent_layers=args.recurrent_layers,
        augment=not args.no_augment,
        evaluate_complete_manifest=args.evaluate_complete_manifest,
    )
    result = run_experiment(config)
    summary = {
        "strict_task_pass_at_1_count": result["evaluation"][
            "strict_task_pass_at_1_count"
        ]
    }
    print(msgspec.json.encode(summary, order="sorted").decode())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
