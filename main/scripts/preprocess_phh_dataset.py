import os
import re
import ast
import argparse
from tqdm import tqdm
import pandas as pd
from collections import deque

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
# assume we are able to get all 6 actions (even if the actual dataset might only really show like 3 or 4)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_SET)}

def normalize_action_token_simple(a):
    """Map noisy PHH action codes into clean canonical categories.

    Minimal, robust rules:
      - fold: tokens like 'f', 'fold'
      - call: 'c', 'cc', 'call'
      - raise: 'r', 'br', 'cbr', 'raise'
      - check: 'x', 'check'
      - bet: 'b', 'bet' or tokens containing 'bet'
      - other: fallback
    """
    if not isinstance(a, str):
        return "other"
    tok = a.strip().lower()

    # normalize punctuation
    tok = re.sub(r"[^a-z0-9]", "", tok)

    # Exact matches first (avoid substring mistakes like searching for 'r' everywhere)
    if tok in ("f", "fold"):
        return "fold"
    if tok in ("cc", "c", "call"):
        return "call"
    if tok in ("r", "br", "cbr", "raise"):
        return "raise"
    if tok in ("x", "check", "ch"):
        return "check"
    # detect explicit bet tokens or tokens that include the word "bet"
    if tok == "b" or "bet" in tok:
        return "bet"
    # anything else -> other
    return "other"


def extract_features_from_game(game, game_path=None, recent_n=10):
    """Extract features, including recent action history, for model training.

    Minimal edits made to:
     - robustly extract action tokens (strip punctuation)
     - treat unknown tokens as 'other' instead of dropping the action
     - recognize 'x' as check and 'b'/'bet' as bet
     - preserve existing structure / features
    """
    variant = str(game.get("variant", "")).lower()
    if not any(x in variant for x in ["nt", "nlh", "holdem", "no-limit"]):
        return []

    # Generate game_id
    if game_path:
        base = os.path.basename(game_path)
        game_id = os.path.splitext(base)[0]
    else:
        game_id = f"game_{os.urandom(3).hex()}"

    actions = game.get("actions", [])
    players = game.get("players", [])
    blinds = game.get("blinds_or_straddles", [])
    stacks = game.get("starting_stacks", [])
    finishing = game.get("finishing_stacks", [])

    # Helper function to safely convert to float
    def safe_float(val):
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None
    
    # Get blinds with fallback
    sb = safe_float(blinds[0]) if len(blinds) > 0 else 0.5
    bb = safe_float(blinds[1]) if len(blinds) > 1 else 1.0
    
    # Ensure bb is never 0
    if bb is None or bb <= 0:
        bb = 1.0
    if sb is None or sb <= 0:
        sb = bb / 2.0

    # Track game state
    pot = sb + bb  # Start with blinds posted
    active = set(range(len(players)))
    history = deque(maxlen=recent_n)
    
    # Track betting state for enhanced features
    current_street_bets = {}
    current_bet_to_call = bb
    raises_this_street = 0
    community_cards = 0
    button_idx = 0

    data_points = []
    for i, act_str in enumerate(actions):
        tokens = act_str.split()
        
        # Track community cards (board dealing)
        if "d db" in act_str or "d dh" in act_str:
            cards_in_action = len([t for t in tokens if len(t) == 2 and t[0] in 'AKQJT98765432' and t[1] in 'shdc'])
            if "d db" in act_str:
                community_cards += cards_in_action
                current_street_bets = {}
                current_bet_to_call = 0
                raises_this_street = 0
            history.append(act_str)
            continue
        
        if len(tokens) < 2:
            history.append(act_str)
            continue

        actor_token = next((t for t in tokens if t.startswith("p")), None)
        if not actor_token:
            history.append(act_str)
            continue

        # --- Improved token extraction ---
        action_token = None
        amount = None

        for tok in tokens:
            # strip punctuation from token (e.g., "c," or "r.")
            tok_clean = re.sub(r"[^a-zA-Z0-9]", "", tok).lower()

            # skip dealer card tokens (d, dh, db) and card strings (like As, Kh)
            if tok_clean in ("d", "dh", "db"):
                continue
            if len(tok_clean) == 2 and tok_clean[0] in 'AKQJT98765432' and tok_clean[1] in 'shdc':
                continue

            # numeric => amount
            if tok_clean.isdigit():
                amount = int(tok_clean)
                continue

            # player token 'p1' etc handled earlier; skip
            if tok_clean.startswith("p"):
                continue

            # check for common action encodings
            # allow tokens like 'f', 'cc', 'c', 'r', 'br', 'cbr', 'x', 'b', 'bet', 'call', 'check', or more
            if re.match(r"^[a-z]{1,5}$", tok_clean):
                # choose the first plausible action-like token
                # but prefer multi-letter tokens like 'cc', 'call', 'check', 'bet' over single letter noisy tokens
                # we keep the first plausible; if multiple exist they will be normalized below
                action_token = tok_clean
                # don't break — still collect amount if present later tokens
                # break

        # If we couldn't find a token by heuristic, fall back to 'other' (do NOT drop) thi sis our 5th action in 0,1,2,3,4,5
        if not action_token:
            action_token = "other"

        actor_idx = None
        try:
            actor_idx = int(actor_token[1:]) - 1
        except Exception:
            actor_idx = 0

        player_name = players[actor_idx] if actor_idx < len(players) else None

        # Normalize to canonical label (fold/call/raise/check/bet/other)
        normalized_token = normalize_action_token_simple(action_token)
        label_idx = ACTION_TO_IDX.get(normalized_token, ACTION_TO_IDX["other"])

        # Get player's stack
        player_stack = safe_float(stacks[actor_idx]) if actor_idx is not None and actor_idx < len(stacks) else None
        if player_stack is None or player_stack <= 0:
            player_stack = 100.0  # Default fallback
        
        # Current player's contribution this street
        player_bet_this_street = current_street_bets.get(actor_idx, 0)
        
        # Position
        position = (actor_idx - button_idx) % len(players) if len(players) > 0 else 0
        is_sb = (position == 1)
        is_bb = (position == 2)
        is_button = (position == 0)
        
        # Street
        if community_cards == 0:
            street = 0
        elif community_cards == 3:
            street = 1
        elif community_cards == 4:
            street = 2
        else:
            street = 3
        
        # Amount to call
        to_call = max(0, current_bet_to_call - player_bet_this_street)
        
        # SAFE pot odds calculation
        if to_call > 0 and (pot + to_call) > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0
        
        # SAFE SPR calculation
        if pot > 0:
            spr = player_stack / pot
        else:
            spr = 100.0  # Default high SPR when pot is 0
        
        # Cap SPR at reasonable value
        spr = min(spr, 100.0)

        # Create feature row
        features = {
            "game_id": game_id,
            "action_index": i,
            "player": player_name,
            "player_idx": actor_idx,
            "num_players": len(players),
            "small_blind": sb,
            "big_blind": bb,
            "starting_stack": player_stack,
            "finishing_stack": safe_float(finishing[actor_idx]) if actor_idx is not None and actor_idx < len(finishing) else None,
            "action_str": act_str,
            "action_code": action_token,
            "action_token": normalized_token,
            "action_label": label_idx,
            "amount": amount,
            "pot_size": pot,
            "num_active": len(active),
            "recent_actions": list(history),
            # Enhanced features
            "to_call": to_call,
            "to_call_norm": to_call / bb if bb > 0 else 0,
            "pot_odds": pot_odds,
            "spr": spr,
            "position": position,
            "is_button": int(is_button),
            "is_sb": int(is_sb),
            "is_bb": int(is_bb),
            "street": street,
            "community_cards": community_cards,
            "raises_this_street": raises_this_street,
        }

        data_points.append(features)

        # Update history AFTER creating the feature row
        history.append(act_str)

        # Update pot and betting state
        if amount and amount > 0:
            pot += amount
            current_street_bets[actor_idx] = player_bet_this_street + amount
            if amount > current_bet_to_call:
                current_bet_to_call = amount
                raises_this_street += 1
        
        # Update active players
        # Note: using the raw act_str check for 'f' is okay; we updated token logic to still record fold as label
        if "f" in act_str:
            m = re.search(r"\bp(\d+)\b", act_str)
            if m:
                active.discard(int(m.group(1)) - 1)

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
            game_rows = extract_features_from_game(game, game_path=path, recent_n=10)
            rows.extend(game_rows)
        except ZeroDivisionError as e:
            print(f"Division by zero in {path}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    if not rows:
        print("No valid Texas Hold'em hands found.")
        return

    df = pd.DataFrame(rows)
    
    # Clean numeric columns - coerce errors to NaN
    numeric_cols = ['starting_stack', 'finishing_stack', 'small_blind', 
                    'big_blind', 'pot_size', 'amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("\n=== Sample of preprocessed data ===")
    print(df.head(10).to_string(index=False))
    print(f"\n=== Dataset shape: {df.shape} ===")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for any remaining type issues
    print("\n=== Data types ===")
    print(df.dtypes)
    print(f"\n=== Missing values ===")
    print(df.isnull().sum())

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
