import torch
import ast
import re
import numpy as np
from train_next_action import (
    LSTMModel, MLPBaseline, normalize_action_token, parse_phh_file, ACTION_SET
)

import argparse

def build_state_from_phh(phh_path, recent_n=10):
    """Simulate reaching a point in the game and build a single state input."""
    game = parse_phh_file(phh_path)
    actions = game.get("actions", [])
    players = game.get("players", [])
    stacks = game.get("starting_stacks", [])
    blinds = game.get("blinds_or_straddles", [50, 100])
    bb = blinds[1] if len(blinds) > 1 else 100

    # choose a random mid-point in the hand to evaluate (simulate "mid-game")
    mid = len(actions) // 2
    history = actions[:mid]
    next_action = actions[mid] if mid < len(actions) else None

    # guess which player acts next (from 'pX' in next_action)
    actor_idx = None
    if next_action:
        m = re.search(r"\bp(\d+)\b", next_action)
        if m:
            actor_idx = int(m.group(1)) - 1

    # basic numeric features
    pot_est = sum(int(d) for a in history for d in re.findall(r"\b(\d+)\b", a))
    stack = stacks[actor_idx] if actor_idx is not None and actor_idx < len(stacks) else stacks[0]
    num_active = len(players)

    # convert to model input
    pot_norm = pot_est / bb
    stack_to_bb = stack / bb
    num_vec = np.array([pot_norm, stack_to_bb, num_active], dtype=np.float32)

    # encode last few actions
    from train_next_action import NextActionDataset
    token_map = {"fold":0,"call":1,"raise":2,"check":3,"bet":4,"other":5,"DEAL":6}
    def encode_history(history):
        ids = []
        for a in history[-recent_n:]:
            label = normalize_action_token(a)
            ids.append(token_map.get(label, token_map["other"]))
        pad_len = recent_n - len(ids)
        return [token_map["DEAL"]]*pad_len + ids

    hist_tokens = np.array(encode_history(history), dtype=np.int64)
    return num_vec, hist_tokens, actor_idx, next_action

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--phh_file", required=True)
    p.add_argument("--recent_n", type=int, default=10)
    p.add_argument("--model_type", choices=["mlp","lstm"], default="lstm")
    args = p.parse_args()

    num_vec, hist_tokens, actor_idx, true_next_action = build_state_from_phh(args.phh_file, args.recent_n)
    print(f"Evaluating hand: {args.phh_file}")
    print(f"Next actual action: {true_next_action}")

    # Load model
    if args.model_type == "mlp":
        model = MLPBaseline()
    else:
        model = LSTMModel()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    # Prepare tensors
    num_tensor = torch.from_numpy(num_vec).unsqueeze(0)
    hist_tensor = torch.from_numpy(hist_tokens).unsqueeze(0)

    with torch.no_grad():
        logits = model(num_tensor, hist_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_action = ACTION_SET[pred_idx]
        print(f"Predicted next action: {pred_action}")
        print("Probabilities:")
        for i, a in enumerate(ACTION_SET):
            print(f"  {a:>8}: {probs[0,i].item():.3f}")

if __name__ == "__main__":
    main()