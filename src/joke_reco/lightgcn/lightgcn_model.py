from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LightGCNConfig:
    # Total number of users in the dataset
    num_users: int

    # Total number of joke items in the dataset
    num_items: int

    # Size of each embedding vector for users and items
    embedding_dim: int = 64

    # Number of graph propagation layers to apply
    num_layers: int = 3


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LightGCN(nn.Module):
    """
    A simple LightGCN recommender model.

    Main idea:
    - Learn embeddings for users and items
    - Pass information through the user-item graph
    - Average the embeddings from each layer
    - Use the final embeddings for recommendation
    """

    #---------------------------------------------------------------------------------

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Learnable embedding for each user
        self.user_emb = nn.Embedding(cfg.num_users, cfg.embedding_dim)

        # Learnable embedding for each item/joke
        self.item_emb = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        # Initialise embeddings with small random values
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    # ---------------------------------------------------------------------------------

    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine user and item embeddings into one matrix
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        # Store embeddings from every layer, starting with original embeddings
        embs = [all_emb]

        # Repeatedly pass information through the graph
        for _ in range(self.cfg.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        # Average embeddings from all layers
        out = torch.stack(embs, dim=0).mean(dim=0)

        # Split back into user embeddings and item embeddings
        users_out = out[: self.cfg.num_users]
        items_out = out[self.cfg.num_users :]
        return users_out, items_out

    # ---------------------------------------------------------------------------------
    @staticmethod
    def bpr_loss(u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        # Score the positive item for the user
        pos_scores = (u * pos).sum(dim=1)

        # Score the negative item for the user
        neg_scores = (u * neg).sum(dim=1)

        # Encourage positive items to score higher than negative ones
        return -F.logsigmoid(pos_scores - neg_scores).mean()