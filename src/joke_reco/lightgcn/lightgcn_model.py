from __future__ import annotations

"""
Defines the original LightGCN recommender model.
Stores the model configuration, performs graph propagation, and computes BPR loss.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Configuration
# Stores the main model settings for LightGCN.
# ---------------------------------------------------------
@dataclass
class LightGCNConfig:
    # Number of users in the dataset
    num_users: int

    # Number of joke items in the dataset
    num_items: int

    # Size of each embedding vector
    embedding_dim: int = 64

    # Number of graph propagation layers
    num_layers: int = 3


# ---------------------------------------------------------
# 2. Main model
# Defines the original LightGCN recommender.
# ---------------------------------------------------------
class LightGCN(nn.Module):
    # Create the model and its trainable embeddings
    def __init__(self, cfg: LightGCNConfig):
        super().__init__()
        self.cfg = cfg

        # Create learnable user embeddings
        self.user_emb = nn.Embedding(cfg.num_users, cfg.embedding_dim)

        # Create learnable item embeddings
        self.item_emb = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        # Initialise user embeddings with small random values
        nn.init.normal_(self.user_emb.weight, std=0.1)

        # Initialise item embeddings with small random values
        nn.init.normal_(self.item_emb.weight, std=0.1)

    # ---------------------------------------------------------
    # 3. Graph propagation
    # Runs LightGCN message passing and layer averaging.
    # ---------------------------------------------------------
    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine user and item embeddings into one matrix
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        # Store the layer 0 embeddings
        embs = [all_emb]

        # Repeatedly propagate through the graph
        for _ in range(self.cfg.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        # Average embeddings across all layers
        out = torch.stack(embs, dim=0).mean(dim=0)

        # Split the combined matrix back into users and items
        users_out = out[: self.cfg.num_users]
        items_out = out[self.cfg.num_users:]

        return users_out, items_out

    # ---------------------------------------------------------
    # 4. BPR loss
    # Computes the pairwise ranking loss used for training.
    # ---------------------------------------------------------
    @staticmethod
    def bpr_loss(
        u: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        # Compute scores for positive items
        pos_scores = (u * pos).sum(dim=1)

        # Compute scores for negative items
        neg_scores = (u * neg).sum(dim=1)

        # Encourage positive items to rank above negative items
        return -F.logsigmoid(pos_scores - neg_scores).mean()