from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LightGCNConfig:
    num_users: int
    num_items: int
    embedding_dim: int = 64
    num_layers: int = 3


class LightGCN(nn.Module):
    """
    Minimal LightGCN (pure PyTorch).

    - Learnable user/item embeddings
    - Graph propagation on normalized bipartite adjacency
    - Final embedding = mean of embeddings across layers (including layer 0)
    """

    def __init__(self, cfg: LightGCNConfig):
        super().__init__()
        self.cfg = cfg

        self.user_emb = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_emb = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # (U+I, D)
        embs = [all_emb]

        for _ in range(self.cfg.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        out = torch.stack(embs, dim=0).mean(dim=0)  # (U+I, D)

        users_out = out[: self.cfg.num_users]
        items_out = out[self.cfg.num_users :]
        return users_out, items_out

    @staticmethod
    def bpr_loss(u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()