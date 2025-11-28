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


ACTION_SET = ["fold", "call", "raise", "check", "bet", "other"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_SET)}

def normalize_action_token_simple(a):
    """Map noisy PHH action codes into clean canonical categories."""
    if not isinstance(a, str):
        return "other"
    a = a.lower()
    if "f" in a and len(a) <= 3:
        return "fold"
    if "cc" in a or a == "c" or "call" in a:
        return "call"
    if "r" in a or "br" in a or "cbr" in a:
        return "raise"
    if "check" in a or a == "x":
        return "check"
    if "bet" in a:
        return "bet"
    return "other"

def extract_features_from_game(game, game_path = None, recent_n: int = 10):
    """Extract features, including recent action history, for model training."""
    # if game.get("variant") != "NT":
        # return []
    variant = str(game.get("variant", "")).lower()
    if not any(x in variant for x in ["nt", "nlh", "holdem", "no-limit"]):
        return []

    # --- Fix 2: Safe game_id extraction ---
    if game_path:
        base = os.path.basename(game_path)
        game_id = os.path.splitext(base)[0]  # works for .phh or .phhs
    else:
        game_id = f"game_{os.urandom(3).hex()}"

    # helps establish a "context" for each game, so taht we know in which game each action row appeared in.

    actions = game.get("actions", [])
    players = game.get("players", [])
    blinds = game.get("blinds_or_straddles", [])
    stacks = game.get("starting_stacks", [])
    finishing = game.get("finishing_stacks", [])

    pot = 0
    active = set(range(len(players)))

    data_points = []
    for i, act_str in enumerate(actions):
        tokens = act_str.split()
        if len(tokens) < 2:
            continue

        actor_token = next((t for t in tokens if t.startswith("p")), None)
        if not actor_token:
            continue

        action_token = None
        amount = None
        for tok in tokens:
            if re.match(r"^[fcbhr]+$", tok):  # fold/call/bet/raise/check
                action_token = tok
            elif tok.isdigit():
                amount = int(tok)

        if not action_token:
            continue

        actor_idx = int(actor_token[1:]) - 1
        player_name = players[actor_idx] if actor_idx < len(players) else None

        normalized_token = normalize_action_token_simple(action_token)
        label_idx = ACTION_TO_IDX[normalized_token]

        features = {
            "game_id": game_id,
            "action_index": i,
            "player": player_name,
            "player_idx": actor_idx,
            "num_players": len(players),
            "small_blind": blinds[0] if len(blinds) > 0 else None,
            "big_blind": blinds[1] if len(blinds) > 1 else None,
            "starting_stack": stacks[actor_idx] if actor_idx < len(stacks) else None,
            "finishing_stack": finishing[actor_idx] if actor_idx < len(finishing) else None,
            "action_str": act_str,
            "action_code": action_token,
            "action_token": normalized_token,
            "action_label": label_idx,
            "amount": amount,
            "pot_size": pot,
            "num_active": len(active),
        }

        # compute pot incrementally
        for d in re.findall(r"\b(\d+)\b", act_str):
            pot += int(d)
        if "f" in act_str:
            m = re.search(r"\bp(\d+)\b", act_str)
            if m:
                active.discard(int(m.group(1)) - 1)

        data_points.append(features)

    return data_points

    

def main(args):
    rows = []
    all_phh_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".phh") or f.endswith(".phhs"):
                all_phh_files.append(os.path.join(root, f))
    all_phh_files.sort()

    print(f"Found {len(all_phh_files)} .phh files")

    for i, path in enumerate(tqdm(all_phh_files, desc="Parsing PHH files")):
        if args.limit and i >= args.limit:
            break
        try:
            game = parse_phh_file(path)
            rows.extend(extract_features_from_game(game, game_path = path))
        except Exception as e:
            # You can log bad files if needed
            continue

    if not rows:
        print("No valid Texas Hold'em hands found.")
        return

    df = pd.DataFrame(rows)
    print("\n=== Sample of preprocessed data ===")
    print(df.head(10).to_string(index=False))


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
