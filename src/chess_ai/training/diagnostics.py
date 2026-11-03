"""Read-only source diagnostics on the trainer's exact grouped holdout."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chess_ai.data.examples import TrainingExample, load_dataset, split_examples_by_group
from chess_ai.model.checkpoint import load_checkpoint
from chess_ai.storage.artifacts import atomic_json, sha256_file, signature
from chess_ai.training.trainer import (
    Trainer,
    TrainingConfig,
    TrainingError,
    _training_weight,
    training_sampling_weights,
)


def sampling_summary(
    examples: Sequence[TrainingExample], config: TrainingConfig, source_key: str
) -> dict[str, dict[str, float | int]]:
    """Expected source coverage under the existing sampler, before actual draws."""
    groups = Trainer.source_indices(examples, source_key)
    mass = training_sampling_weights(examples, config)
    weighted = [
        value * _training_weight(item, config.sample_weight_key)
        for value, item in zip(mass, examples, strict=True)
    ]
    result: dict[str, dict[str, float | int]] = {}
    for source, indices in sorted(groups.items()):
        rows = [examples[index] for index in indices]
        result[source] = {
            "examples": len(rows),
            "games": len({item.game_id for item in rows}),
            "example_share": len(rows) / len(examples),
            "expected_draw_share": sum(mass[index] for index in indices) / sum(mass),
            "expected_weighted_draw_share": sum(weighted[index] for index in indices)
            / sum(weighted),
            "mean_target_value": sum(item.target_value for item in rows) / len(rows),
            "draw_value_share": sum(item.target_value == 0.0 for item in rows) / len(rows),
        }
    return result


def diagnose_training(
    config: TrainingConfig,
    checkpoints: Sequence[Path],
    output: Path,
    *,
    source_key: str = "composition_source",
) -> dict[str, Any]:
    """Evaluate existing checkpoints; never train, sample new data, or edit inputs."""
    if output.exists():
        raise TrainingError(f"Diagnostic report already exists: {output}")
    if config.dataset_path is None or not checkpoints:
        raise TrainingError("Diagnostics require a dataset and at least one checkpoint")
    dataset = load_dataset(config.dataset_path)
    if dataset.metadata.get("value_target_spec") != config.value_target_spec:
        raise TrainingError("Dataset value semantics differ from diagnostic training config")
    if config.use_dataset_split:
        if dataset.metadata.get("explicit_split") is not True:
            raise TrainingError("Diagnostics require the explicit evaluation dataset split")
        training = [item for item in dataset if item.metadata.get("split") == "train"]
        validation = [item for item in dataset if item.metadata.get("split") == "validation"]
    else:
        training, validation = split_examples_by_group(
            dataset.examples,
            config.validation_fraction,
            seed=config.seed,
            group_key=config.validation_group_key,
        )
    if not training or not validation:
        raise TrainingError("Source diagnostics require non-empty training and held-out sets")
    sampling = sampling_summary(training, config, source_key)
    Trainer.source_indices(validation, source_key)
    masks = Trainer._build_legal_masks(validation) if config.legal_policy_mask else None
    results = []
    for checkpoint in checkpoints:
        loaded = load_checkpoint(checkpoint)
        trainer = Trainer(loaded.model, config)
        results.append(
            {
                "checkpoint": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "epoch": loaded.epoch,
                "validation_by_source": trainer.evaluate_by_source(
                    validation, source_key=source_key, legal_masks=masks
                ),
            }
        )
    group_key = config.validation_group_key
    validation_groups = sorted(
        {
            item.game_id if group_key == "game_id" else str(item.metadata[group_key])
            for item in validation
        }
    )
    report = {
        "format": "self-improving-chess-ai.source-diagnostics",
        "version": 1,
        "training_updates": 0,
        "dataset": str(config.dataset_path),
        "dataset_sha256": sha256_file(config.dataset_path),
        "dataset_examples": len(dataset),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "source_key": source_key,
        "split_seed": config.seed,
        "validation_group_key": group_key,
        "validation_group_digest": signature({"groups": validation_groups}),
        "training_config": config.checkpoint_metadata(),
        "sampling": sampling,
        "checkpoints": results,
        "metric_note": (
            "top1/top5 match the target argmax, not any acceptable teacher move. "
            "Unweighted metrics count every held-out row once. Weighted metrics use the configured "
            "loss weights. Expected weighted draw share is a ratio of expected masses, not an "
            "exact share of normalized minibatch gradients. Source labels do not imply benchmark skill."
        ),
    }
    atomic_json(output, report)
    return report
