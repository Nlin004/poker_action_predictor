#!/usr/bin/env python3
"""
Preprocess the PHH dataset into structured model-ready data.

Usage:
    python scripts/preprocess_phh_dataset.py \
        --input_dir data/raw/phh-dataset/data/pluribus \
        --output data/processed/phh_holdem.parquet \
        --limit 10000
"""

import os
import re
import ast
import argparse
from tqdm import tqdm
import pandas as pd

def parse_phh_file(file_path):
    """Parse a single .phh file into a dictionary."""
    game = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [x.strip() for x in line.split("=", 1)]
            # Safely evaluate Python-like literals
            try:
                val = ast.literal_eval(val)
            except Exception:
                pass
            game[key] = val
    return game

def extract_features_from_game(game, recent_n: int = 10):
    """Extract features, including recent action history, for model training."""
    # Only consider No-Limit Texas Hold’em hands
    if game.get("variant") != "NT":
        return []

    actions = game.get("actions", [])
    players = game.get("players", [])
    blinds = game.get("blinds_or_straddles", [])
    stacks = game.get("starting_stacks", [])
    finishing = game.get("finishing_stacks", [])

    # Initialize pot and track which players remain active
    pot = 0
    active = set(range(len(players)))

    for act in actions:
        if "f" in act:  # player folded
            m = re.search(r"\bp(\d+)\b", act)
            if m:
                active.discard(int(m.group(1)) - 1)
        for d in re.findall(r"\b(\d+)\b", act):  # add numeric amounts
            pot += int(d)

    # Identify blinds (may be missing in some datasets)
    try:
        small_blind = blinds[0]
        big_blind = blinds[1]
    except Exception:
        small_blind, big_blind = None, None

    data_points = []
    # Iterate over actions, including limited action history
    for i, act_str in enumerate(actions):
        tokens = act_str.split()
        if len(tokens) < 2:
            continue

        # Player token: e.g., "p5"
        actor_token = next((t for t in tokens if t.startswith("p")), None)
        action_token = None
        amount = None

        # Parse action (fold, call, bet, raise, etc.)
        for tok in tokens:
            if re.match(r"^[fcbhr]+$", tok):  # fold/call/bet/raise/check patterns
                action_token = tok
            elif tok.isdigit():
                amount = int(tok)

        if not actor_token or not action_token:
            continue

        actor_idx = int(actor_token[1:]) - 1
        player_name = players[actor_idx] if actor_idx < len(players) else None

        # Take up to recent_n actions before this one
        recent = actions[max(0, i - recent_n):i]

        features = {
            "player": player_name,
            "player_idx": actor_idx,
            "num_players": len(players),
            "small_blind": small_blind,
            "big_blind": big_blind,
            "starting_stack": stacks[actor_idx] if actor_idx < len(stacks) else None,
            "finishing_stack": finishing[actor_idx] if actor_idx < len(finishing) else None,
            "action_str": act_str,
            "recent_actions": recent,
            "action_code": action_token,
            "amount": amount,
            "pot_size": pot,
            "num_active": len(active),
        }

        label_map = {
            "f": 0,      # fold
            "c": 1,      # call
            "r": 2,      # raise
            "h": 3,      # check
            "b": 4,      # bet
            "cbr": 2,    # treat call-bet-raise as raise
        }
        features["action_label"] = label_map.get(action_token, 5)  # 5 = "other/unknown"    

        data_points.append(features)

    return data_points

    

def main(args):
    rows = []
    all_phh_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".phh"):
                all_phh_files.append(os.path.join(root, f))
    all_phh_files.sort()

    print(f"Found {len(all_phh_files)} .phh files")

    for i, path in enumerate(tqdm(all_phh_files, desc="Parsing PHH files")):
        if args.limit and i >= args.limit:
            break
        try:
            game = parse_phh_file(path)
            rows.extend(extract_features_from_game(game))
        except Exception as e:
            # You can log bad files if needed
            continue

    if not rows:
        print("No valid Texas Hold'em hands found.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} action rows to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to phh-dataset/data/pluribus/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files for testing")
    args = parser.parse_args()
    main(args)
