"""Reserve uniquely named configs for one experiment without starting expensive work."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.run_id) is None:
        parser.error("run-id must be a safe 1-81 character directory name")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("source-commit must be the exact 40-character Git SHA")
    name = f"stockfish_teacher_v1_{args.run_id}"
    root = Path("data/curricula") / name
    root.mkdir(parents=True, exist_ok=False)
    configs = root / "configs"
    configs.mkdir()
    replacements = {
        "data/curricula/stockfish_teacher_v1/": f"{root.as_posix()}/",
        "data/curricula/stockfish_teacher_v1_smoke/": f"{root.as_posix()}/smoke/",
        "data/datasets/stockfish_teacher_v1_replay.pt": f"data/datasets/{name}_replay.pt",
        "checkpoints/stockfish_teacher_v1_candidate": f"checkpoints/{name}_candidate",
        "data/metrics/stockfish_teacher_v1_candidate.jsonl": f"data/metrics/{name}_candidate.jsonl",
        "data/games/evaluation/stockfish_teacher_v1_selection": f"data/games/evaluation/{name}_selection",
    }
    for stage, filename in (
        ("collect", "stockfish_teacher_v1.yaml"),
        ("smoke", "stockfish_teacher_v1_smoke.yaml"),
        ("compose", "stockfish_teacher_v1_compose.yaml"),
        ("train", "stockfish_teacher_v1_train.yaml"),
        ("select", "stockfish_teacher_v1_select.yaml"),
    ):
        text = (Path("configs") / filename).read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        config = yaml.safe_load(text)
        if "stockfish_batch" in config:
            config["stockfish_batch"]["stockfish"]["executable"] = (
                args.executable.resolve().as_posix()
            )
            config["stockfish_batch"]["stockfish"]["source_commit"] = args.source_commit
        (configs / f"{stage}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    print(configs.as_posix())


if __name__ == "__main__":
    main()
