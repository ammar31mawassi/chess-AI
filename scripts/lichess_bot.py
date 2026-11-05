"""Configure and run the explicitly authorized Lichess BOT integration."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path

from chess_ai.online.credentials import load_token, store_token
from chess_ai.online.lichess_api import LichessAPI, LichessError
from chess_ai.online.lichess_bot import BotConfig, BotRunner, ClockedPUCT, verify_account


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("store-token", "check", "upgrade", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/lichess_bot.yaml"))
    parser.add_argument("--confirm-bot-upgrade", action="store_true")
    args = parser.parse_args()
    if args.stage == "store-token":
        token = getpass.getpass("Lichess token (hidden): ")
        path = store_token(token)
        print(f"Encrypted credential saved for this Windows user: {path}")
        return
    api = LichessAPI(load_token())
    config = BotConfig.read(args.config)
    checkpoint = config.checkpoint()
    account = verify_account(config, api, require_bot=args.stage == "run")
    if args.stage == "upgrade" and account.get("title") != "BOT":
        if not args.confirm_bot_upgrade:
            raise ValueError("Upgrade is permanent; supply --confirm-bot-upgrade")
        if account.get("count", {}).get("all") != 0:
            raise ValueError("Only a new account with zero games can be upgraded")
        api.post("/api/bot/account/upgrade")
        account = verify_account(config, api)
    if args.stage == "run":
        agent = ClockedPUCT(
            checkpoint,
            device=config.device,
            seed=config.seed,
            max_seconds=config.max_search_seconds,
        )
        import chess

        agent.evaluate(chess.Board())  # Warm up before accepting any challenge.
        BotRunner(config, api, agent).run()
        return
    ongoing = api.get("/api/account/playing").get("nowPlaying", [])
    print(
        json.dumps(
            {
                "id": account["id"],
                "username": account["username"],
                "title": account.get("title"),
                "games": account.get("count", {}).get("all"),
                "bot_play_scope": True,
                "ready": account.get("title") == "BOT",
                "profile": "https://lichess.org/@/" + account["username"],
                "ongoing_games": len(ongoing),
                "checkpoint": str(checkpoint),
                "device": config.device,
                "simulations_cap": 32,
                "casual_standard_humans_only": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, LichessError) as exc:
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        print("Bot stopped. Restart the run command to recover any ongoing game.")
