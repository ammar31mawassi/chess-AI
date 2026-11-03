"""Bounded, split-preserving replay for Stockfish evaluation refinement."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from chess_ai.data.examples import TrainingExample, load_dataset, save_dataset
from chess_ai.data.value_targets import SHARDED_DATASET_FORMAT, validate_value_spec
from chess_ai.storage.artifacts import sha256_file, signature


def compose_refinement_replay(
    fresh_path: Path, replay_manifest: Path, output_path: Path, *, seed: int
) -> dict[str, Any]:
    """Draw one old example per fresh example in each split, without importing audits.

    All old validation inputs remain protected, including those not sampled for this
    candidate. New positions shared with old training replace that replay copy.
    """
    if output_path.exists():
        raise ValueError(f"Replay output exists; keep it and proceed to training: {output_path}")
    fresh = load_dataset(fresh_path)
    spec = validate_value_spec(fresh.metadata.get("value_target_spec"))
    manifest = json.loads(replay_manifest.read_text(encoding="utf-8"))
    if (
        fresh.format_version != 2
        or not fresh.metadata.get("explicit_split")
        or manifest.get("format") != SHARDED_DATASET_FORMAT
        or manifest.get("version") != 1
        or manifest.get("value_target_spec") != spec
    ):
        raise ValueError("Replay requires matching version-2 evaluation datasets with fixed splits")
    paths = []
    old_validation: set[str] = set()
    seen: set[str] = set()
    groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    count = 0
    for entry in manifest["shards"]:
        path = (replay_manifest.parent / entry["path"]).resolve()
        if (
            not path.is_relative_to(replay_manifest.parent.resolve())
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("Replay shard path/hash mismatch")
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (
            raw.get("format_version") != 2
            or raw["metadata"].get("value_target_spec") != spec
            or raw["example_count"] != entry["examples"]
        ):
            raise ValueError("Replay shard semantics/count mismatch")
        for encoded in raw["example_metadata_json"]:
            meta = json.loads(encoded)
            key, split = meta["position_key"], meta["split"]
            if key in seen or split not in groups:
                raise ValueError("Replay has duplicate positions or invalid splits")
            seen.add(key)
            groups[split].add(meta["group_id"])
            if split == "validation":
                old_validation.add(key)
            count += 1
        paths.append(path)
    if count != manifest["examples"] or groups["train"] & groups["validation"]:
        raise ValueError("Replay manifest has a count mismatch or split leakage")
    examples: list[TrainingExample] = []
    fresh_keys: set[str] = set()
    fresh_groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    removed = 0
    old_training_in_fresh_validation = 0
    for example in fresh:
        key, split = str(example.metadata["position_key"]), str(example.metadata["split"])
        if key in fresh_keys or split not in fresh_groups:
            raise ValueError("Fresh data has duplicate positions or invalid splits")
        fresh_keys.add(key)
        fresh_groups[split].add(str(example.metadata["group_id"]))
        if split == "train" and key in old_validation:
            removed += 1
            continue
        if split == "validation" and key in seen and key not in old_validation:
            old_training_in_fresh_validation += 1
            continue
        examples.append(
            replace(
                example,
                metadata={
                    **example.metadata,
                    "composition_source": "fresh",
                    "group_id": f"fresh:{example.metadata['group_id']}",
                },
            )
        )
    if fresh_groups["train"] & fresh_groups["validation"]:
        raise ValueError("Fresh data leaks an opening group across splits")
    targets = Counter(str(example.metadata["split"]) for example in examples)
    if not targets["train"] or not targets["validation"]:
        raise ValueError(
            "Fresh data needs both train and validation positions after holdout protection"
        )
    # Reservoirs store only row references, keeping the old corpus off the RAM heap.
    rng = random.Random(seed)
    pools: dict[str, list[tuple[Path, int]]] = {"train": [], "validation": []}
    eligible: Counter[str] = Counter()
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for index, encoded in enumerate(raw["example_metadata_json"]):
            meta = json.loads(encoded)
            if meta["position_key"] in fresh_keys:
                continue
            split = meta["split"]
            eligible[split] += 1
            pool = pools[split]
            if len(pool) < targets[split]:
                pool.append((path, index))
            else:
                chosen = rng.randrange(eligible[split])
                if chosen < len(pool):
                    pool[chosen] = (path, index)
    if any(len(pools[split]) != targets[split] for split in pools):
        raise ValueError("Not enough distinct old replay examples for a 50/50 split")
    selected: dict[Path, list[int]] = {}
    for pool in pools.values():
        for path, index in pool:
            selected.setdefault(path, []).append(index)
    for path, indices in sorted(selected.items()):
        shard = load_dataset(path, mmap=True)
        for index in sorted(indices):
            example = shard.examples[index]
            examples.append(
                replace(
                    example,
                    metadata={
                        **example.metadata,
                        "composition_source": "replay",
                        "group_id": f"replay:{example.metadata['group_id']}",
                    },
                )
            )
    binding = {
        "fresh_sha256": sha256_file(fresh_path),
        "replay_manifest_sha256": sha256_file(replay_manifest),
        "seed": seed,
        "fresh_fraction": 0.5,
    }
    metadata = {
        "kind": "stockfish_refinement_replay",
        "version": 1,
        "value_target_spec": spec,
        "explicit_split": True,
        "dataset_signature": signature(binding),
        "composition_binding": binding,
        "fresh_by_split": dict(targets),
        "replay_by_split": {split: len(pool) for split, pool in pools.items()},
        "fresh_training_rows_excluded_for_old_validation": removed,
        "fresh_validation_rows_excluded_for_old_training": old_training_in_fresh_validation,
    }
    save_dataset(output_path, examples, metadata=metadata, format_version=2)
    return {"output": str(output_path), "examples": len(examples), **metadata}
