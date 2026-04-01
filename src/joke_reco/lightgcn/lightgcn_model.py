from __future__ import annotations

"""
LightGCN model definition for the joke recommender project.

Purpose:
- Store the core LightGCN configuration
- Define learnable user and item embeddings
- Propagate embeddings through the user-item graph
- Compute BPR loss for pairwise ranking
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Config: LightGCN settings
# ---------------------------------------------------------
@dataclass
class LightGCNConfig:
    """
    Stores the main configuration for the LightGCN model.
    """

    # Total number of users in the dataset
    num_users: int

    # Total number of joke items in the dataset
    num_items: int

    # Size of each embedding vector
    embedding_dim: int = 64

    # Number of graph propagation layers
    num_layers: int = 3


# ---------------------------------------------------------
# Model: LightGCN recommender
# ---------------------------------------------------------
class LightGCN(nn.Module):
    """
    A simple LightGCN recommender model.

    Main idea:
    - Learn embeddings for users and items
    - Pass information through the user-item graph
    - Average embeddings from all propagation layers
    - Use the final embeddings for recommendation
    """

    # ---------------------------------------------------------
    # Initialise model embeddings
    # ---------------------------------------------------------
    def __init__(self, cfg: LightGCNConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable embedding for each user
        self.user_emb = nn.Embedding(cfg.num_users, cfg.embedding_dim)

        # Learnable embedding for each joke/item
        self.item_emb = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        # Initialise embeddings with small random values
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    # ---------------------------------------------------------
    # Propagate embeddings through the graph
    # ---------------------------------------------------------
    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run LightGCN propagation over the normalised user-item graph.

        This method:
        1) combines user and item embeddings
        2) repeatedly propagates them through the graph
        3) stores embeddings from every layer
        4) averages all layer outputs
        5) splits the result back into user and item embeddings
        """

        # Combine user and item embeddings into one matrix
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        # Store embeddings from every layer, starting with layer 0
        embs = [all_emb]

        # Repeatedly pass information through the graph
        for _ in range(self.cfg.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        # Average embeddings from all layers
        out = torch.stack(embs, dim=0).mean(dim=0)

        # Split back into final user embeddings and final item embeddings
        users_out = out[: self.cfg.num_users]
        items_out = out[self.cfg.num_users :]
        return users_out, items_out

    # ---------------------------------------------------------
    # Static helper: BPR loss
    # ---------------------------------------------------------
    @staticmethod
    def bpr_loss(
        u: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Bayesian Personalised Ranking (BPR) loss.

        The goal is to encourage the model to score
        positive items higher than negative items
        for the same user.
        """

        # Score the positive item for the user
        pos_scores = (u * pos).sum(dim=1)

        # Score the negative item for the user
        neg_scores = (u * neg).sum(dim=1)

        # Encourage positive items to rank above negative items
        return -F.logsigmoid(pos_scores - neg_scores).mean()