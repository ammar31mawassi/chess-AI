"""Prepare bounded 50/50 replay after the new collection is fully finalized."""

import argparse
import json
from pathlib import Path

from chess_ai.config import load_config
from chess_ai.data.refinement_replay import compose_refinement_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_config(args.config)["refinement_replay"]
    print(
        json.dumps(
            compose_refinement_replay(
                Path(settings["fresh_path"]),
                Path(settings["replay_manifest"]),
                Path(settings["output_path"]),
                seed=settings["seed"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
