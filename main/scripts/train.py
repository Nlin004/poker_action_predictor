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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---- Action mapping ----
ACTION_SET = ["fold", "call", "raise", "check", "bet", "other"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_SET)}

def normalize_action_token(action_str):
    """Normalize action string to canonical label."""
    tok = action_str.strip().split()
    s = " ".join(tok)
    if " f" in s or s.endswith(" f") or re.search(r"\b(f)\b", s):
        return "fold"
    if re.search(r"\b(cb|call|c)\b", s): # any of these options (cb, call, c) count as call.
        return "call"
    if re.search(r"\b(r|raise|br|bet)\b", s): # phh format calls r, raise, br, bet as forms of  checking OR raising.
        if "check" in s:
            return "check"
        return "raise"
    if "check" in s:
        return "check"
    if "bet" in s:
        return "bet"
    if "c" in tok:
        return "call"
    if "f" in tok:
        return "fold"
    return "other"




class NextActionDataset(Dataset):
    """Dataset for poker action prediction."""
    
    def __init__(self, df, grouped=False, max_hist=10):
        self.grouped = grouped
        self.max_hist = max_hist

        self.token_map = {
            "fold": 0, "call": 1, "raise": 2,
            "check": 3, "bet": 4, "other": 5,
            "DEAL": 6  # padding token
        }

        df = df.copy()

        # Ensure required columns exist
        numeric_cols = [
            "pot_norm", "stack_to_bb", "num_active", 
            "to_call_norm", "pot_odds", "spr",
            "position", "is_button", "is_sb", "is_bb",
            "street", "community_cards", "raises_this_street"
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            df[col] = df[col].fillna(0)

        # Sanity check specific columns
        df["pot_odds"] = df["pot_odds"].clip(0, 1)  # Pot odds must be 0-1


        # Clean numeric columns
        df["pot_size"] = df["pot_size"].fillna(0).astype(float)
        df["big_blind"] = df["big_blind"].replace(0, 1).fillna(1).astype(float)
        df["starting_stack"] = df["starting_stack"].fillna(0).astype(float)
        df["num_active"] = df["num_active"].fillna(1).astype(int)

        # Normalize features
        df["pot_norm"] = df["pot_size"] / df["big_blind"]
        df["stack_to_bb"] = df["starting_stack"] / df["big_blind"]

        # Ensure action_index exists
        if "action_index" not in df.columns:
            df["action_index"] = df.groupby("game_id").cumcount()
        
        df["action_index"] = df["action_index"].fillna(0).astype(int)
        df = df.sort_values(["game_id", "action_index"])

        # Handle recent_actions column
        if "recent_actions" not in df.columns:
            print("WARNING: recent_actions column not found, creating empty lists")
            df["recent_actions"] = [[] for _ in range(len(df))]

        self.df = df
        self.groups = None

        if self.grouped: # if we CHOOSE to group by game_id:
            groups_out = []
            grouped = df.groupby("game_id")

            for gid, g in grouped:
                g = g.sort_values("action_index").copy()
                groups_out.append(g)

            self.groups = groups_out

    def __len__(self):
        return len(self.groups) if self.grouped else len(self.df)

    def encode_history(self, actions):
        # Encode action history into token IDs."""
        # basically we convert the recent actions into tokens: 
        # history: ["p3 cbr 225", "p2 cc", "p4 f"]
        # encoded: [2, 1, 0]  # raise, call, fold

        
        if not isinstance(actions, list):
            actions = []
        
        tokens = []
        for a in actions[-self.max_hist:]:
            label = normalize_action_token(str(a)) # actions like "call, fold,etc" -> [0,2,1,etc]
            tokens.append(self.token_map.get(label, self.token_map["other"]))

        pad = self.max_hist - len(tokens)
        return [self.token_map["DEAL"]] * pad + tokens # we pad it to be the SAME LENGTH FOR ENCODED HISTORY
        # so if we only have 3 recent actions but our expected histories are 5 actions long we padd with 6
        # [0,2,1] -> [6,6,0,2,1] so its 5 actions long regardless of whether or not there are more or less

    def __getitem__(self, idx):
        # MLP/Transformer mode: return single row
        if not self.grouped:
            r = self.df.iloc[idx] # FETCH ONE ROW
            # num = np.array([r["pot_norm"], r["stack_to_bb"], r["num_active"]], dtype=np.float32)
            num = np.array([
                r["pot_norm"],
                r["stack_to_bb"],
                r["num_active"],
                r["to_call_norm"],
                r["pot_odds"],
                r["spr"],
                r["position"],
                r["is_button"],
                r["is_sb"],
                r["is_bb"],
                r["street"],
                r["community_cards"],
                r["raises_this_street"],
            ], dtype=np.float32)

            hist = np.array(self.encode_history(r["recent_actions"]), dtype=np.int64)
            label = int(r["action_label"])

            # EXAMPLE: returns: {
            #     "num": [25.5, 100.0, 4, 3, ... ,5,3,3],  # pot_norm, stack_to_bb, num_active, etc. There are 13 numeric features now.
            #     "hist": [6, 6, 6, 0, 2, 1, 3, 4, 1, 2],  # Recent actions
            #     "label": 1  # Player called
            # } 
            # this is ONE ROW!!!! every entry specifies numerical features, as well as HISTORY TOKENS in ORDER.
            # label is what we would begin to start predicting (the player did action 1, given the above)
            return {"num": num, "hist": hist, "label": label}




        # Sequence mode (LSTM) - we care about the ENTIRE GAME. 
        g = self.groups[idx]

        # numeric features (T, 13)
        num_seq = np.stack([
            np.array([
                r["pot_norm"], r["stack_to_bb"], r["num_active"],
                r["to_call_norm"], r["pot_odds"], r["spr"],
                r["position"], r["is_button"], r["is_sb"], r["is_bb"],
                r["street"], r["community_cards"], r["raises_this_street"]
            ], dtype=np.float32)
            for _, r in g.iterrows() # EVERY NUMERICAL TUPLE IN ONE GAME. T is the length of the game.
        ])

        # history tokens (T, max_hist) so we could pad to something like 10
        hist_seq = np.stack([
            np.array(self.encode_history(r["recent_actions"]), dtype=np.int64)
            for _, r in g.iterrows()
        ])

        # labels (T,)
        labels = g["action_label"].astype(int).values

        return {
            "num_seq": torch.tensor(num_seq, dtype=torch.float32),
            "hist_seq": torch.tensor(hist_seq, dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }


def collate_fn_mlp(batch):
    """Collate for MLP/Transformer single-decision batches."""
    nums = torch.stack([torch.from_numpy(b["num"]) for b in batch])
    hists = torch.stack([torch.from_numpy(b["hist"]) for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.int64)
    return nums, hists, labels


def collate_fn_seq(batch):
    """Collate for LSTM sequence batches."""
    num_seqs = [b["num_seq"] for b in batch]
    hist_seqs = [b["hist_seq"] for b in batch]
    labels = [b["labels"] for b in batch]

    num_seqs = pad_sequence(num_seqs, batch_first=True, padding_value=0)
    hist_seqs = pad_sequence(hist_seqs, batch_first=True, padding_value=6)  # DEAL=6
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return num_seqs, hist_seqs, labels


# ---- Models ----
class MLPBaseline(nn.Module):
    def __init__(self, num_input=3, hist_vocab=7, hist_emb_dim=8, hist_len=10, 
                 hidden=64, num_classes=3):
        super().__init__()
        self.hist_emb = nn.Embedding(hist_vocab, hist_emb_dim, padding_idx=3)
        self.hist_len = hist_len
        self.fc = nn.Sequential(
            nn.Linear(num_input + hist_emb_dim * hist_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, num, hist):
        be = self.hist_emb(hist)
        be_flat = be.view(be.size(0), -1)
        x = torch.cat([num, be_flat], dim=1)
        return self.fc(x)


# class TransformerActionModel(nn.Module):
#     """Transformer for per-decision action prediction."""
    
#     def __init__(self, 
#                  num_input=13,           #  num of numeric columns in our adjusted dataset
#                  hist_vocab=7, 
#                  hist_emb_dim=64,        
#                  hist_len=10,              # After modifications to our first transformer iteration (0.59 val acc):
#                  d_model=256,            # INCREASED from 128
#                  nhead=8,                # INCREASED from 4
#                  num_encoder_layers=3,   # INCREASED from 2
#                  dim_feedforward=512,    # INCREASED from 256
#                  dropout=0.2,            # INCREASED from 0.1
#                  num_classes=6):
#         super().__init__()
#         self.hist_len = hist_len
        
#         # Token embedding
#         self.token_emb = nn.Embedding(hist_vocab, hist_emb_dim, padding_idx=6)
#         self.token_proj = nn.Linear(hist_emb_dim, d_model)
#         self.pos_emb = nn.Embedding(hist_len, d_model)
        
#         # Layer norm before transformer
#         self.pre_norm = nn.LayerNorm(d_model)
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu'  # GELU instead of ReLU
#         )
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer, 
#             num_layers=num_encoder_layers,
#             norm=nn.LayerNorm(d_model)  # Final layer norm
#         )
        
#         # Numeric feature processor (deeper network)
#         self.num_proj = nn.Sequential(
#             nn.Linear(num_input, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
        
#         # Classifier head (deeper)
#         self.head = nn.Sequential(
#             nn.Linear(d_model + 128, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, num, hist):
#         B, L = hist.shape
        
#         # Token embedding + positional
#         t = self.token_emb(hist)
#         t = self.token_proj(t)
#         pos_ids = torch.arange(0, L, device=hist.device).unsqueeze(0).expand(B, L)
#         p = self.pos_emb(pos_ids)
#         x = t + p
        
#         # Pre-normalization
#         x = self.pre_norm(x)
        
#         # Mask padding
#         key_padding_mask = (hist == 6)
#         all_padding = key_padding_mask.all(dim=1)
        
#         if all_padding.any():
#             enc = self.transformer(x)
#         else:
#             enc = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
#         # Pool over sequence
#         if key_padding_mask is not None and not all_padding.all():
#             mask_inv = (~key_padding_mask).float().unsqueeze(-1)
#         else:
#             mask_inv = torch.ones(B, L, 1, device=x.device)
        
#         enc_sum = (enc * mask_inv).sum(dim=1)
#         denom = mask_inv.sum(dim=1).clamp(min=1.0)
#         pooled = enc_sum / denom
        
#         # Process numeric features
#         num_proj = self.num_proj(num)
        
#         # Combine and classify
#         combined = torch.cat([pooled, num_proj], dim=1)
#         logits = self.head(combined)
#         return logits







# class ImprovedTransformerActionModel(nn.Module):
#     """
#     Enhanced Transformer with:
#     1. Cross-attention between history and numeric features
#     2. Better pooling (max + mean)
#     3. Residual connections in classifier
#     4. Learnable CLS token
#     5. Stochastic depth (layer dropout)
#     """
    
#     def __init__(self, 
#                  num_input=13,
#                  hist_vocab=7, 
#                  hist_emb_dim=64,
#                  hist_len=10,
#                  d_model=256,
#                  nhead=8,
#                  num_encoder_layers=3,
#                  dim_feedforward=512,
#                  dropout=0.2,
#                  stochastic_depth=0.1,  # NEW: layer dropout
#                  num_classes=6):
#         super().__init__()
#         self.hist_len = hist_len
#         self.d_model = d_model
        
#         # ===== IMPROVEMENT 1: Learnable CLS token =====
#         # Acts as a "summary" token that aggregates information
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
#         # Token embedding
#         self.token_emb = nn.Embedding(hist_vocab, hist_emb_dim, padding_idx=6)
#         self.token_proj = nn.Linear(hist_emb_dim, d_model)
        
#         # ===== IMPROVEMENT 2: Sinusoidal positional encoding =====
#         # Better than learned embeddings for generalization
#         self.register_buffer('pos_encoding', 
#                             self._create_sinusoidal_encoding(hist_len + 1, d_model))
        
#         # Layer norm
#         self.pre_norm = nn.LayerNorm(d_model)
        
#         # ===== IMPROVEMENT 3: Stochastic Depth =====
#         # Randomly drops entire transformer layers during training
#         encoder_layers = []
#         dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_encoder_layers)]
        
#         for i in range(num_encoder_layers):
#             layer = nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=dim_feedforward,
#                 dropout=dropout,
#                 batch_first=True,
#                 activation='gelu',
#                 norm_first=True  # Pre-LN (more stable)
#             )
#             encoder_layers.append(StochasticDepthLayer(layer, drop_prob=dpr[i]))
        
#         self.transformer_layers = nn.ModuleList(encoder_layers)
#         self.final_norm = nn.LayerNorm(d_model)
        
#         # ===== IMPROVEMENT 4: Better numeric feature processing =====
#         # Separate processing for different feature groups
#         self.position_proj = nn.Sequential(
#             nn.Linear(4, 32),  # position, is_button, is_sb, is_bb
#             nn.LayerNorm(32),
#             nn.GELU(),
#         )
        
#         self.betting_proj = nn.Sequential(
#             nn.Linear(6, 64),  # pot_norm, stack_to_bb, to_call_norm, pot_odds, spr, raises
#             nn.LayerNorm(64),
#             nn.GELU(),
#         )
        
#         self.game_state_proj = nn.Sequential(
#             nn.Linear(3, 32),  # num_active, street, community_cards
#             nn.LayerNorm(32),
#             nn.GELU(),
#         )
        
#         # Combine all numeric features
#         self.num_proj = nn.Sequential(
#             nn.Linear(32 + 64 + 32, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
        
#         # ===== IMPROVEMENT 5: Cross-attention =====
#         # Allow numeric features to attend to history
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=128,
#             num_heads=4,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.cross_norm = nn.LayerNorm(128)
        
#         # ===== IMPROVEMENT 6: Better classifier with residual connections =====
#         self.pre_classifier = nn.Linear(d_model + 128, 256)
        
#         self.classifier_blocks = nn.ModuleList([
#             ResidualBlock(256, 256, dropout),
#             ResidualBlock(256, 256, dropout),
#         ])
        
#         self.final_classifier = nn.Linear(256, num_classes)
        
#         # Initialize weights
#         self._init_weights()
    
#     def _create_sinusoidal_encoding(self, max_len, d_model):
#         """Create sinusoidal positional encoding."""
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0)  # (1, max_len, d_model)
    
#     def _init_weights(self):
#         """Better weight initialization."""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#             elif isinstance(module, nn.Embedding):
#                 nn.init.normal_(module.weight, mean=0, std=0.02)
    
#     def forward(self, num, hist):
#         B, L = hist.shape
        
#         # ===== Process history with CLS token =====
#         # Token embeddings
#         t = self.token_emb(hist)  # (B, L, emb_dim)
#         t = self.token_proj(t)    # (B, L, d_model)
        
#         # Add CLS token
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
#         t = torch.cat([cls_tokens, t], dim=1)  # (B, L+1, d_model)
        
#         # Add positional encoding
#         t = t + self.pos_encoding[:, :L+1, :]
        
#         # Pre-norm
#         t = self.pre_norm(t)
        
#         # Create padding mask (include CLS token position as valid)
#         key_padding_mask = torch.cat([
#             torch.zeros(B, 1, dtype=torch.bool, device=hist.device),  # CLS is never masked
#             (hist == 6)  # Mask padding tokens
#         ], dim=1)
        
#         # Transformer layers with stochastic depth
#         x = t
#         for layer in self.transformer_layers:
#             x = layer(x, src_key_padding_mask=key_padding_mask)
        
#         x = self.final_norm(x)
        
#         # ===== IMPROVEMENT 7: Multi-head pooling =====
#         # Use CLS token + mean pool + max pool
#         cls_output = x[:, 0, :]  # (B, d_model)
        
#         # Mean pooling over non-padded tokens (exclude CLS)
#         mask_inv = (~key_padding_mask[:, 1:]).float().unsqueeze(-1)  # (B, L, 1)
#         mean_pool = (x[:, 1:, :] * mask_inv).sum(dim=1) / mask_inv.sum(dim=1).clamp(min=1.0)
        
#         # Max pooling
#         masked_x = x[:, 1:, :].masked_fill(key_padding_mask[:, 1:].unsqueeze(-1), float('-inf'))
#         max_pool = masked_x.max(dim=1)[0]
#         max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)
        
#         # Combine pooling strategies (learnable weighted combination)
#         pooled = (cls_output + mean_pool + max_pool) / 3.0  # Simple average
        
#         # ===== Process numeric features in groups =====
#         # Split features into semantic groups
#         position_features = num[:, 6:10]  # position, is_button, is_sb, is_bb
#         betting_features = num[:, [0,1,3,4,5,12]]  # pot_norm, stack_to_bb, to_call_norm, pot_odds, spr, raises
#         game_state_features = num[:, [2,10,11]]  # num_active, street, community_cards
        
#         pos_emb = self.position_proj(position_features)
#         bet_emb = self.betting_proj(betting_features)
#         state_emb = self.game_state_proj(game_state_features)
        
#         num_combined = torch.cat([pos_emb, bet_emb, state_emb], dim=1)
#         num_proj = self.num_proj(num_combined)  # (B, 128)
        
#         # ===== Cross-attention: numeric features attend to history =====
#         # Allow betting decisions to be informed by action history
#         num_attended, _ = self.cross_attention(
#             query=num_proj.unsqueeze(1),  # (B, 1, 128)
#             key=x[:, 1:, :],  # History tokens (exclude CLS)
#             value=x[:, 1:, :],
#             key_padding_mask=key_padding_mask[:, 1:]
#         )
#         num_attended = num_attended.squeeze(1)  # (B, 128)
        
#         # Residual connection
#         num_proj = self.cross_norm(num_proj + num_attended)
        
#         # ===== Combine and classify =====
#         combined = torch.cat([pooled, num_proj], dim=1)  # (B, d_model + 128)
        
#         # Pre-classifier projection
#         x = self.pre_classifier(combined)  # (B, 256)
#         x = F.gelu(x)
        
#         # Residual blocks
#         for block in self.classifier_blocks:
#             x = block(x)
        
#         # Final classification
#         logits = self.final_classifier(x)  # (B, num_classes)
        
#         return logits
# ======================





class TransformerActionModel(nn.Module):
    """Transformer for per-decision action prediction."""
    
    def __init__(self, 
                 num_input=13,
                 hist_vocab=7, 
                 hist_emb_dim=64,
                 hist_len=10,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=3,
                 dim_feedforward=512,
                 dropout=0.2,
                 num_classes=3):
        super().__init__()
        self.hist_len = hist_len
        
        # Token embedding
        self.token_emb = nn.Embedding(hist_vocab, hist_emb_dim, padding_idx=6)
        self.token_proj = nn.Linear(hist_emb_dim, d_model)
        self.pos_emb = nn.Embedding(hist_len, d_model)
        
        # Layer norm before transformer
        self.pre_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Numeric feature processor
        self.num_proj = nn.Sequential(
            nn.Linear(num_input, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(d_model + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, num, hist):
        # B, L = hist.shape
        
        # # Token embedding + positional
        # t = self.token_emb(hist)
        # t = self.token_proj(t)
        # pos_ids = torch.arange(0, L, device=hist.device).unsqueeze(0).expand(B, L)
        # p = self.pos_emb(pos_ids)
        # x = t + p
        
        # # Pre-normalization
        # x = self.pre_norm(x)
        
        # # Mask padding 
        # # key_padding_mask: True where token is padding
        # key_padding_mask = (hist == 6)  # (B, L)
        # all_padding = key_padding_mask.all(dim=1)
        # # ALWAYS pass the src_key_padding_mask; TransformerEncoder supports rows that are all padding.
        # enc = self.transformer(x, src_key_padding_mask=key_padding_mask)
        # # +===================================================================
        # # key_padding_mask = (hist == 6)
        # # all_padding = key_padding_mask.all(dim=1)
        
        # # if all_padding.any():
        # #     enc = self.transformer(x)
        # # else:
        # #     enc = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # # Pool over sequence
        # if key_padding_mask is not None and not all_padding.all():
        #     mask_inv = (~key_padding_mask).float().unsqueeze(-1)
        # else:
        #     mask_inv = torch.ones(B, L, 1, device=x.device)
        
        # enc_sum = (enc * mask_inv).sum(dim=1)
        # denom = mask_inv.sum(dim=1).clamp(min=1.0)
        # pooled = enc_sum / denom
        
        # # Process numeric features
        # num_proj = self.num_proj(num)
        
        # # Combine and classify
        # combined = torch.cat([pooled, num_proj], dim=1)
        # logits = self.head(combined)
        # return logits

        B, L = hist.shape

        # If sequence length is 0 (edge case), force a dummy PAD token
        if L == 0:
            hist = torch.full((B, 1), 6, device=hist.device, dtype=torch.long)
            L = 1

        t = self.token_emb(hist)
        t = self.token_proj(t)

        pos_ids = torch.arange(L, device=hist.device).unsqueeze(0).expand(B, L)
        p = self.pos_emb(pos_ids)
        x = t + p

        x = self.pre_norm(x)

        key_padding_mask = (hist == 6)
        all_padding = key_padding_mask.all(dim=1)

        # Safe Transformer call: if ALL rows in batch are full padding, skip mask
        if all_padding.all():
            enc = self.transformer(x)  # No mask
        else:
            enc = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Pooling safely
        mask_inv = (~key_padding_mask).float().unsqueeze(-1)
        enc_sum = (enc * mask_inv).sum(dim=1)
        denom = mask_inv.sum(dim=1).clamp(min=1.0)  # Avoid division by zero
        pooled = enc_sum / denom

        num_proj = self.num_proj(num)
        combined = torch.cat([pooled, num_proj], dim=1)
        logits = self.head(combined)
        return logits


class StochasticDepthLayer(nn.Module):
    """Wrapper that randomly drops entire transformer layers."""
    def __init__(self, layer, drop_prob=0.0):
        super().__init__()
        self.layer = layer
        self.drop_prob = drop_prob
    
    def forward(self, x, **kwargs):
        if not self.training or self.drop_prob == 0.0:
            return self.layer(x, **kwargs)
        
        # Random drop with survival probability
        if torch.rand(1).item() < self.drop_prob:
            return x  # Skip this layer (identity)
        else:
            return self.layer(x, **kwargs)


class ResidualBlock(nn.Module):
    """Residual block for classifier."""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.norm2(out)
        
        out = out + identity  # Residual connection
        out = F.gelu(out)
        
        return out






















# =====================================
class ContextualLSTM(nn.Module):
    """LSTM for sequence-based action prediction."""
    
    def __init__(self, num_input=3, hist_vocab=7, hist_emb_dim=16, 
                 lstm_hidden=64, num_classes=3):
        super().__init__()
        self.hist_emb = nn.Embedding(hist_vocab, hist_emb_dim, padding_idx=6)
        self.hist_lstm = nn.LSTM(hist_emb_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(num_input + lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, num_seq, hist_seq):
        B, T, L = hist_seq.shape

        # Embed all history tokens
        emb = self.hist_emb(hist_seq.view(B * T, L))
        _, (hn, _) = self.hist_lstm(emb)
        hist_context = hn[-1].view(B, T, -1)

        x = torch.cat([num_seq, hist_context], dim=-1)
        logits = self.fc(x)
        return logits
    

# ==========================

# HEPER FUNCTIONS FOR TRAINING:

# ===== TRAINING IMPROVEMENTS =====

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Learning rate schedule:
    - Warmup: 0 -> lr over first N steps
    - Cosine decay: lr -> 0 over remaining steps
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # pred: [B, C]
        # target: [B]
        num_classes = pred.size(1)

        # --- SAFE: Clamp target to valid range in case something slipped ---
        target = target.clamp(min=0, max=num_classes - 1)

        with torch.no_grad():
            smooth = self.smoothing / (num_classes - 1)
            one_hot = torch.full_like(pred, smooth)
            one_hot.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(pred, dim=1)
        loss = (-one_hot * log_probs).sum(dim=1).mean()
        return loss
# class LabelSmoothingCrossEntropy(nn.Module):
#     """
#     Label smoothing helps prevent overconfidence.
#     Instead of [0, 1, 0, 0, 0, 0], target becomes [0.02, 0.88, 0.02, 0.02, 0.02, 0.02]
#     """
#     def __init__(self, smoothing=0.1):
#         super().__init__()
#         self.smoothing = smoothing
    
#     def forward(self, pred, target):
#         n_class = pred.size(-1)
#         one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
#         smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
#         log_prob = F.log_softmax(pred, dim=1)
#         loss = -(smooth_one_hot * log_prob).sum(dim=1)
#         return loss.mean()
    
def create_improved_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """
    AdamW with weight decay (better than Adam for transformers).
    """
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


# ---- Training / Eval ----
def train_epoch(model, loader, opt, crit, device, scheduler=None, gradient_clip=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        if isinstance(batch[0], torch.Tensor) and batch[0].dim() == 2:
            # MLP/Transformer mode
            nums, hists, labels = batch
            nums = nums.to(device)
            hists = hists.to(device)
            labels = labels.to(device)
            
            opt.zero_grad()
            logits = model(nums, hists)
            loss = crit(logits, labels)
            
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            opt.step() # update weights here.
            
            if scheduler:
                scheduler.step()
            
            preds = logits.argmax(dim=1)
            mask = (labels != -100)
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
            total_loss += loss.item() * mask.sum().item()
        
        else:
            # LSTM mode
            num_seq, hist_seq, labels = batch
            num_seq = num_seq.to(device)
            hist_seq = hist_seq.to(device)
            labels = labels.to(device)
            
            opt.zero_grad()
            logits = model(num_seq, hist_seq)
            
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            labels = labels.view(B*T)
            
            loss = crit(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            opt.step()
            
            if scheduler:
                scheduler.step()
            
            mask = (labels != -100)
            preds = logits.argmax(dim=1)
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
            total_loss += loss.item() * mask.sum().item()
    
    return total_loss/total, correct/total

# def train_epoch_with_mixup(model, loader, opt, crit, device, scheduler=None, 
#                            use_mixup=True, mixup_alpha=0.2):
#     """Enhanced training with mixup augmentation."""
#     model.train()
#     total_loss = 0.0
#     correct = 0
#     total = 0
    
#     mixup = MixupTransform(alpha=mixup_alpha) if use_mixup else None
    
#     for batch_idx, batch in enumerate(loader):
#         if isinstance(batch[0], torch.Tensor) and batch[0].dim() == 2:
#             nums, hists, labels = batch
#             nums = nums.to(device)
#             hists = hists.to(device)
#             labels = labels.to(device)
            
#             # Apply mixup with 50% probability
#             if use_mixup and torch.rand(1).item() < 0.5 and batch_idx > 0:
#                 # Get random indices for second batch
#                 indices = torch.randperm(nums.size(0))
                
#                 # Mixup numeric features and histories
#                 lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample()
#                 mixed_nums = lam * nums + (1 - lam) * nums[indices]
                
#                 # For discrete history tokens, randomly choose from one or the other
#                 mask = (torch.rand_like(hists.float()) < lam).long()
#                 mixed_hists = mask * hists + (1 - mask) * hists[indices]
                
#                 # Soft labels
#                 y1_onehot = F.one_hot(labels, num_classes=6).float()
#                 y2_onehot = F.one_hot(labels[indices], num_classes=6).float()
#                 mixed_labels = lam * y1_onehot + (1 - lam) * y2_onehot
                
#                 opt.zero_grad()
#                 logits = model(mixed_nums, mixed_hists)
                
#                 # KL divergence loss for soft labels
#                 log_probs = F.log_softmax(logits, dim=1)
#                 loss = -(mixed_labels * log_probs).sum(dim=1).mean()
#             else:
#                 opt.zero_grad()
#                 logits = model(nums, hists)
#                 loss = crit(logits, labels)
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
            
#             if scheduler:
#                 scheduler.step()
            
#             preds = logits.argmax(dim=1)
#             mask = (labels != -100)
#             correct += (preds[mask] == labels[mask]).sum().item()
#             total += mask.sum().item()
#             total_loss += loss.item() * mask.sum().item()
    
#     return total_loss/total, correct/total

def eval_model(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch[0], torch.Tensor) and batch[0].dim() == 2:
                nums, hists, labels = batch
                nums = nums.to(device)
                hists = hists.to(device)
                labels = labels.to(device)

                logits = model(nums, hists)
                loss = crit(logits, labels)

                preds = logits.argmax(dim=1)
                mask = (labels != -100)

                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
                total_loss += loss.item() * mask.sum().item()

            else:
                num_seq, hist_seq, labels = batch
                num_seq = num_seq.to(device)
                hist_seq = hist_seq.to(device)
                labels = labels.to(device)

                logits = model(num_seq, hist_seq)
                B, T, C = logits.shape

                logits = logits.view(B * T, C)
                labels = labels.view(B * T)

                loss = crit(logits, labels)
                preds = logits.argmax(dim=1)

                mask = (labels != -100)
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
                total_loss += loss.item() * mask.sum().item()

    return total_loss / total, correct / total, None, None


def split_by_game_id(df, frac=0.8, seed=42):
    """Split dataset by game_id to prevent data leakage."""
    np.random.seed(seed)
    game_ids = df["game_id"].unique()
    np.random.shuffle(game_ids)
    
    cut = int(len(game_ids) * frac)
    train_gids = set(game_ids[:cut])
    val_gids = set(game_ids[cut:])
    
    train_df = df[df["game_id"].isin(train_gids)].reset_index(drop=True)
    val_df = df[df["game_id"].isin(val_gids)].reset_index(drop=True)
    return train_df, val_df


# ---- Main ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", choices=["mlp", "lstm", "transformer"], default="transformer")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    log_file = os.path.join(results_dir, "log.txt")
    print(f"Using device: {device}")
    
    with open(log_file, "w") as f:
        f.write(json.dumps(vars(args), indent=2) + "\n\n")

    # Load data
    print(f"Loading {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    print(f"Loaded {len(df)} rows")

    # Split
    train_df, val_df = split_by_game_id(df, frac=0.8)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")


    # DEBUGGING TO TEST IF THERE ARE ACTUALLY 6 CLASSES, I'M ONLY SEEING 3:
    vals, counts = np.unique(train_df["action_label"].astype(int).values, return_counts=True)
    # df_num_classes = len(vals)
    # print(df_num_classes)
    unique_labels = np.unique(train_df["action_label"].values)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    train_df["action_label"] = train_df["action_label"].map(label_map)
    val_df["action_label"] = val_df["action_label"].map(label_map)

    df_num_classes = len(unique_labels)
    print(f"Unique labels (mapped): {list(range(df_num_classes))}")


    print("label -> count")
    for v,c in zip(vals,counts):
        print(v, c, f"{c/len(train_df):.4f}")
    maj_idx = vals[np.argmax(counts)]
    maj_acc = counts.max()/len(train_df)
    print("Majority class:", maj_idx, "freq:", maj_acc)

    # NEXT: quick confusion test by predicting majority on val set - is it just choosing biggest predictor?
    maj_pred = maj_idx
    val_acc_major = (val_df["action_label"].astype(int).values == maj_pred).mean()
    print("Val acc if always predict majority:", val_acc_major)





    # Dataset ========
    if args.model == "lstm":
        train_ds = NextActionDataset(train_df, grouped=True)
        val_ds   = NextActionDataset(val_df, grouped=True)
        collate_train = collate_fn_seq
        collate_val = collate_fn_seq
    else:  # mlp or transformer predict per-decision
        train_ds = NextActionDataset(train_df, grouped=False)
        val_ds   = NextActionDataset(val_df, grouped=False)
        collate_train = collate_fn_mlp
        collate_val = collate_fn_mlp

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_train, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_val, num_workers=4)




    # train_ds = NextActionDataset(train_df, grouped=False)
    # val_ds = NextActionDataset(val_df, grouped=False)
    
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
    #                          shuffle=True, collate_fn=collate_fn_mlp, num_workers=4)
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, 
    #                        collate_fn=collate_fn_mlp, num_workers=4)



    # Model selection:
    if args.model == "transformer":
        print("Using TransformerActionModel")
        model = TransformerActionModel(
            num_input=13,
            hist_vocab=7,
            hist_emb_dim=64,
            hist_len=10,
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            num_classes=df_num_classes
        )
    elif args.model == "lstm":
        print("Using ContextualLSTM")
        model = ContextualLSTM(num_input=13, num_classes=df_num_classes) # because 6 classes to predict / classify into
    else:
        print("Using BaselineMLP")
        model = MLPBaseline(num_input=13, hidden=128)

    model.to(device)
     

    # LOSS WITH WEIGHTS: account for less than 6 actions, even if there are in total possibly 6.
    # So far I've noticed usually only {0,1,2} show up (fold, call, raise).
    # Based on their documentation: 

    # x (check) is only logged if the system considers it a real option
    # but in many PHH logs, checks are merged with “call 0”, or omitted entirely
    # bet is logged using the same “raise” opcode in no-limit games, so bets become raises

    # so we literally just have:
    # - fold (f)
    # - call (c, cc)
    # - raise (r, b, br, cbr, bet)     All other cases - bet, check - are simply not contained in raw phh
    #  
    labels = train_df["action_label"].astype(int).values
    unique = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique,
        y=labels
    )
    # Then map back into full action size for the loss
    
    WEIGHTED_CRITERION = False
    if WEIGHTED_CRITERION:
        full_weights = np.ones(len(unique), dtype=np.float32)
        for cls, w in zip(unique, class_weights):
            full_weights[int(cls)] = w

        class_weights = torch.tensor(full_weights, dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weights)
    else:
        # crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        crit = LabelSmoothingCrossEntropy(smoothing=0.05)
        # loss function measures difference between predicted logits and true labels but 
        # "eases" hard values of 0,1 to values like 0.05, 0.95 to prevent overconfidence.
 
    # SIMPLE OPTIMIZER
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt = create_improved_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    # change the given weights of the model internally. I think generally AdamW is better for transformers

    SCHEDULER_ENABLED = True 
    # schedulers change the learning rate AS THE EPOCHS CHANGE.
    # this particular one follows a cosine curve, but as it goes on the 'height' of the wave decreases
    # effectively makes LR super small towards epochs 20-25
    if SCHEDULER_ENABLED:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        scheduler = None




    # Training loop 
    best_val = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []


    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, opt, crit, device, scheduler)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, crit, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # lr_now = scheduler.get_last_lr()[0]

        msg = (f"Epoch {epoch}/{args.epochs}  |  "
               f"train_loss={train_loss:.4f}  |  train_acc={train_acc:.4f}  |  "
               f"val_loss={val_loss:.4f}  |  val_acc={val_acc:.4f}")
        print(msg)

        with open(log_file, "a") as f:
            f.write(msg + "\n")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"====> New best: {best_val:.4f}")

    print(f"\nBest val accuracy: {best_val:.4f}")

    # Save metrics
    metrics = pd.DataFrame({
        "epoch": list(range(1, args.epochs + 1)),
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
    })
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(metrics["epoch"], metrics["train_acc"], label="Train")
    ax1.plot(metrics["epoch"], metrics["val_acc"], label="Val")
    ax1.set_title("Accuracy")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(metrics["epoch"], metrics["train_loss"], label="Train")
    ax2.plot(metrics["epoch"], metrics["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    print(f"\nSaved results to {results_dir}/")

if __name__ == "__main__":
    main()