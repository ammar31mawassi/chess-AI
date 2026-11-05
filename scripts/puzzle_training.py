"""Prepare, label and train the two explicitly authorized puzzle experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from chess_ai.data import puzzle_training as workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="puzzles_250k_v1_20261105_01")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "labels", "finalize", "train", "status", "evaluate"),
    )
    parser.add_argument("--count", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=5112041)
    parser.add_argument("--confirm-training-import", action="store_true")
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--arm", choices=("scratch", "finetune"), default="scratch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--split", choices=("validation", "confirmation"), default="validation")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.run_id) is None:
        raise ValueError("Use a safe, nonempty run ID")
    root = Path("data/experiments") / args.run_id
    if args.stage == "prepare":
        result = workflow.prepare(
            root,
            count=args.count,
            seed=args.seed,
            confirm_training_import=args.confirm_training_import,
        )
    elif args.stage == "labels":
        result = workflow.label(root, args.max_positions)
    elif args.stage == "finalize":
        result = workflow.finalize(root)
    elif args.stage == "status":
        result = workflow._verified(root)["summary"]
        progress = root / "label_progress.json"
        if progress.exists():
            result["label_progress"] = json.loads(progress.read_text(encoding="utf-8"))
    elif args.stage == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        result = workflow.evaluate(
            root, args.checkpoint, limit=args.limit, device=args.device, split=args.split
        )
    else:
        workflow.train(root, args.arm, resume=args.resume, device=args.device)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
