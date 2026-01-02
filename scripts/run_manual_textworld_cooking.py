#!/usr/bin/env python
"""Manual interactive script to run TextWorldEnv with cooking games."""

import argparse
import random

from data import get_cooking_game, prepare_twcooking_data
from textworld_cooking_env import TextWorldEnv, tw_intro_text


def main():
    parser = argparse.ArgumentParser(
        description="Run TextWorldEnv interactively with a cooking game"
    )
    parser.add_argument(
        "--game-id",
        type=int,
        default=None,
        help="Game file index to use. If not provided, randomly samples a game.",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=1,
        choices=range(1, 11),
        help="Difficulty level (1-10). Default: 1",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use. Default: test",
    )
    args = parser.parse_args()

    # Prepare data if not already downloaded
    print("Preparing TextWorld cooking data...")
    prepare_twcooking_data(force=False)

    # Get available game files
    game_files = sorted(get_cooking_game(args.difficulty, split=args.split))

    if not game_files:
        print(f"No games found for difficulty {args.difficulty} in {args.split} split")
        return

    print(f"Found {len(game_files)} games for difficulty {args.difficulty}")

    # Select game file
    if args.game_id is not None:
        if args.game_id < 0 or args.game_id >= len(game_files):
            print(f"Invalid game_id {args.game_id}. Must be in range [0, {len(game_files) - 1}]")
            return
        game_file = game_files[args.game_id]
        print(f"Using game_id {args.game_id}")
    else:
        game_file = random.choice(game_files)
        game_idx = game_files.index(game_file)
        print(f"Randomly selected game_id {game_idx}")

    print(f"Game file: {game_file}")

    # Create and reset environment
    env = TextWorldEnv(game_file, admissible_commands=True)
    obs, infos = env.reset()

    # Display initial state
    clean_obs = obs.replace(tw_intro_text, "").strip()
    print("\n" + "=" * 60)
    print("INITIAL OBSERVATION:")
    print("=" * 60)
    print(clean_obs)
    print("\n" + "=" * 60)
    print("GAME INFO:")
    print("=" * 60)
    print(f"Max score: {infos['max_score']}")
    print(f"Walkthrough: {infos['extra.walkthrough']}")
    print(f"Admissible commands: {infos.get('admissible_commands', [])}")

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Type commands to play. Type 'quit' to exit.")
    print("=" * 60)

    total_reward = 0
    while True:
        try:
            action = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if action.lower() == "quit":
            break

        if not action:
            continue

        obs, reward, done, infos = env.step(action)
        total_reward += reward

        print(f"\n{obs}")
        print(f"[Reward: {reward}, Total: {total_reward}, Score: {infos['score']}/{infos['max_score']}]")

        if done:
            if infos.get("won", False):
                print("\n*** YOU WON! ***")
            elif infos.get("lost", False):
                print("\n*** GAME OVER ***")
            break

    print(f"\nFinal score: {infos['score']}/{infos['max_score']}")


if __name__ == "__main__":
    main()
