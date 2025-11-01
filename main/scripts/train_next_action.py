# How do we define our DECISION POINT / ANCHOR?

# iterate through the actions sequence from the PHH file.
# If an action in actions is a player decision, you can make a training example where:
# - input (full public state just before that action occurred.)
# - target (the action that player took at that decision point.like fold, call, raise, bet, check)

import os
import argparse
import math
import ast
import re
from collections import deque, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---- Utilities: normalize action token from PHH action string ----
def normalize_action_token(action_str):
    # action_str examples: "p5 cbr 225", "p3 f", "d dh p1 3c9s"
    tok = action_str.strip().split()
    # find first player-action-like token (skip deals)
    # we'll classify based on substrings
    s = " ".join(tok)
    if " f" in s or s.endswith(" f") or re.search(r"\b(f)\b", s):
        return "fold"
    if re.search(r"\b(cb|call|c)\b", s):
        return "call"
    # 'r', 'raise', 'br' may indicate raise; 'cbr' contains 'br' too
    if re.search(r"\b(r|raise|br|bet)\b", s):
        # differentiate bet vs raise: if there is numeric amount and previous bet exists,
        # we treat as raise/bet uniformly as 'raise' here
        if "check" in s:
            return "check"
        return "raise"
    if "check" in s:
        return "check"
    if "bet" in s:
        return "bet"
    # fallback: try letters
    if "c" in tok:
        return "call"
    if "f" in tok:
        return "fold"
    return "other"

# mapping to label ids
ACTION_SET = ["fold", "call", "raise", "check", "bet", "other"]
ACTION_TO_IDX = {a:i for i,a in enumerate(ACTION_SET)}

# ---- PHH parser (simple .phh file parser; similar to earlier script) ----
def parse_phh_file(path):
    game = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [x.strip() for x in line.split("=", 1)]
            try:
                # handle values like lists, ints, booleans
                val_parsed = ast.literal_eval(val)
            except Exception:
                val_parsed = val.strip().strip("'").strip('"')
            game[key] = val_parsed
    return game

# ---- Build examples directly from PHH directory (sliding window) ----
def build_examples_from_phh_dir(phh_dir, limit_files=None, recent_n=10):
    """
    Walk phh_dir, open .phh files, and create examples:
        for each player decision action in actions list:
            - construct state features using prior actions up to this point
            - label the action
    Returns: list of dict rows.
    """
    rows = []
    files = []
    for root, _, files_here in os.walk(phh_dir):
        for f in files_here:
            if f.endswith(".phh"):
                files.append(os.path.join(root, f))
    files.sort()
    if limit_files:
        files = files[:limit_files]
    for fp in tqdm(files, desc="PHH files"):
        try:
            game = parse_phh_file(fp)
        except Exception:
            continue
        if game.get("variant") != "NT":  # only Texas Hold'em
            continue
        players = game.get("players", [])
        starting_stacks = game.get("starting_stacks", [])
        blinds = game.get("blinds_or_straddles", [None, None])
        big_blind = blinds[1] if len(blinds) > 1 else None
        pot_size = game.get("pot", None)  # sometimes not present; we'll compute via actions
        actions = game.get("actions", [])
        # sliding window over actions: keep last recent_n action labels
        history = deque(maxlen=recent_n)
        # for stacks contributions we could compute running stacks; for now, use starting_stacks
        for act in actions:
            # If this action is a player decision, make example with current state (history) BEFORE this action
            # Determine if it's a player move: contains 'p\d'
            if re.search(r"\bp\d+\b", act):
                # find actor index
                m = re.search(r"\bp(\d+)\b", act)
                actor_idx = int(m.group(1)) - 1 if m else None
                actor_name = players[actor_idx] if actor_idx is not None and actor_idx < len(players) else None
                # label
                label_str = normalize_action_token(act)
                label = ACTION_TO_IDX.get(label_str, ACTION_TO_IDX["other"])
                # simple numeric features
                # compute number of active players up to this point (players who have not folded earlier in actions)
                # naive active players: start with all, then remove those with fold before this point
                active = set(range(len(players)))
                for h in list(history):
                    if isinstance(h, str) and "f" in h:
                        mm = re.search(r"\bp(\d+)\b", h)
                        if mm:
                            idx = int(mm.group(1)) - 1
                            active.discard(idx)
                num_active = len(active)
                pot_est = 0
                # simplified estimate: look for numeric amounts in prior actions and sum them
                for h in history:
                    digits = re.findall(r"\b(\d+)\b", h)
                    for d in digits:
                        pot_est += int(d)
                # features
                row = {
                    "player_idx": actor_idx,
                    "player_name": actor_name,
                    "num_players": len(players),
                    "num_active": num_active,
                    "big_blind": big_blind,
                    "pot_size": pot_est,
                    "starting_stack": starting_stacks[actor_idx] if actor_idx is not None and actor_idx < len(starting_stacks) else None,
                    "community_count": 0,   # advanced: parse board dealing events to set this
                    "recent_actions": list(history),
                    "action_label": label,
                    "action_str": act,
                    "source_file": fp
                }
                rows.append(row)
            # push action onto history AFTER (so state is history-before-action)
            history.append(act)
    return rows

# ---- Dataset that uses either a parquet file or prebuilt rows ----
class NextActionDataset(Dataset):
    def __init__(self, rows, action_vocab=ACTION_TO_IDX, recent_n=10, max_history_tokens=10):
        """
        rows: a list of dicts or a dataframe with fields:
          - numeric features: pot_size, big_blind, starting_stack, num_active, ...
          - recent_actions: list[str]
          - action_label: int
        """
        if isinstance(rows, pd.DataFrame):
            self.df = rows
        else:
            self.df = pd.DataFrame(rows)
        self.recent_n = recent_n
        self.max_hist = max_history_tokens
        self.action_vocab = {"fold":0,"call":1,"raise":2,"check":3,"bet":4,"other":5}
        self.token_map = {"fold":0,"call":1,"raise":2,"check":3,"bet":4,"other":5,"DEAL":6}
        # Precompute numeric arrays
        # fillna
        self.df["pot_size"] = self.df["pot_size"].fillna(0).astype(float)
        self.df["big_blind"] = self.df["big_blind"].fillna(1).astype(float)
        self.df["starting_stack"] = self.df["starting_stack"].fillna(0).astype(float)
        self.df["num_active"] = self.df["num_active"].fillna(1).astype(int)
        # compute normalized features
        self.df["pot_norm"] = self.df["pot_size"] / (self.df["big_blind"].replace(0,1))
        self.df["stack_to_bb"] = self.df["starting_stack"] / (self.df["big_blind"].replace(0,1))

    def __len__(self):
        return len(self.df)

    def _encode_history(self, history_list):
        # history_list is list of action strings (most recent last)
        # we convert each string to a token id based on normalized_action_token
        ids = []
        for a in history_list[-self.max_hist:]:
            label = normalize_action_token(a)
            ids.append(self.token_map.get(label, self.token_map["other"]))
        # pad left to max_hist
        pad_len = self.max_hist - len(ids)
        return [self.token_map["DEAL"]]*pad_len + ids  # DEAL as pad token

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        # numeric vector (pot_norm, stack_to_bb, num_active)
        num_vec = np.array([r["pot_norm"], r["stack_to_bb"], r["num_active"]], dtype=np.float32)
        hist_tokens = self._encode_history(r["recent_actions"] if isinstance(r["recent_actions"], list) else [])
        hist_tokens = np.array(hist_tokens, dtype=np.int64)
        label = int(r["action_label"])
        return {"num": num_vec, "hist": hist_tokens, "label": label}

def collate_fn(batch):
    nums = np.array([b["num"] for b in batch], dtype=np.float32)
    hists = np.array([b["hist"] for b in batch], dtype=np.int64)
    labels = np.array([b["label"] for b in batch], dtype=np.int64)

    nums = torch.from_numpy(nums)
    hists = torch.from_numpy(hists)
    labels = torch.from_numpy(labels)
    return nums, hists, labels

# ---- Models ----
class MLPBaseline(nn.Module):
    def __init__(self, num_input=3, hist_vocab=7, hist_emb_dim=8, hist_len=10, hidden=64, num_classes=len(ACTION_SET)):
        super().__init__()
        self.hist_emb = nn.Embedding(hist_vocab, hist_emb_dim, padding_idx=None)
        self.hist_len = hist_len
        self.fc = nn.Sequential(
            nn.Linear(num_input + hist_emb_dim*hist_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, num, hist):
        # num: [B, num_input], hist: [B, hist_len]
        be = self.hist_emb(hist)  # [B, hist_len, emb]
        be_flat = be.view(be.size(0), -1)
        x = torch.cat([num, be_flat], dim=1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, num_input=3, hist_vocab=7, hist_emb_dim=16, lstm_hidden=64, num_classes=len(ACTION_SET)):
        super().__init__()
        self.hist_emb = nn.Embedding(hist_vocab, hist_emb_dim)
        self.lstm = nn.LSTM(hist_emb_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(num_input + lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, num_classes)
        )
    def forward(self, num, hist):
        emb = self.hist_emb(hist)  # [B, L, E]
        out, (hn, cn) = self.lstm(emb)  # out [B, L, H]
        last = hn[-1]  # [B, H]
        x = torch.cat([num, last], dim=1)
        return self.fc(x)

# ---- Training / Eval loops ----
def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for nums, hists, labels in loader:
        nums, hists, labels = nums.to(device), hists.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(nums, hists)
        loss = crit(logits, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss/total, correct/total

def eval_model(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for nums, hists, labels in loader:
            nums, hists, labels = nums.to(device), hists.to(device), labels.to(device)
            logits = model(nums, hists)
            loss = crit(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    acc = correct/total
    return total_loss/total, acc, all_preds, all_labels

# ---- Main CLI ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["parquet","phh_dir"], required=True)
    p.add_argument("--parquet_path", default=None)
    p.add_argument("--phh_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", choices=["mlp","lstm"], default="lstm")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--limit_files", type=int, default=None)
    p.add_argument("--recent_n", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOGGING!
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join(results_dir, "log.txt")
    metrics_file = os.path.join(results_dir, "metrics.csv")

    with open(log_file, "w") as f:
        f.write(f"Training run started: {run_name}\n")
        f.write(json.dumps(vars(args), indent=2) + "\n\n")

    # Build / load dataset
    if args.mode == "parquet":
        assert args.parquet_path, "parquet_path required in parquet mode"
        df = pd.read_parquet(args.parquet_path)
        # Expect df to have: recent_actions (list), pot_size, big_blind, starting_stack, num_active, action_label
        rows = df  # DataFrame
    else:
        assert args.phh_dir, "phh_dir required in phh_dir mode"
        print("Building examples from PHH directory (this can take time)...")
        rows = build_examples_from_phh_dir(args.phh_dir, limit_files=args.limit_files, recent_n=args.recent_n)
        rows = pd.DataFrame(rows)

    # Split
    train_df = rows.sample(frac=0.8, random_state=42)
    val_df = rows.drop(train_df.index)
    train_ds = NextActionDataset(train_df)
    val_ds = NextActionDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # Model
    if args.model == "mlp":
        model = MLPBaseline()
    else:
        model = LSTMModel()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, opt, crit, device)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, crit, device)



        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)



        line = (f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        print(line)
        with open(log_file, "a") as f:
            f.write(line + "\n")

        # print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
    print("Best val acc:", best_val)

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        "epoch": list(range(1, args.epochs + 1)),
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    })
    metrics_df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print(f"Saved metrics to {results_dir}/metrics.csv")

    #GENERATE PLOTS:
    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train_acc"], label="Train Acc")
    plt.plot(metrics_df["epoch"], metrics_df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))
    print(f"Saved plot to {results_dir}/accuracy_curve.png")

if __name__ == "__main__":
    main()