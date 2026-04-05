from __future__ import annotations

"""
Text-augmented LightGCN model definition for the joke recommender project.

Purpose:
- Store the main LightGCN configuration
- Define learnable user embeddings
- Define learned item embeddings
- Encode standalone joke text vectors into the graph model
- Build initial item representations from text features and/or learned embeddings
- Propagate embeddings through the user-item graph
- Compute BPR loss for pairwise ranking
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Config: Text-augmented LightGCN settings
# ---------------------------------------------------------
@dataclass
class LightGCNConfig:
    """
    Stores the main configuration for the text-augmented LightGCN model.
    """

    # Total number of users in the dataset
    num_users: int

    # Total number of joke items in the dataset
    num_items: int

    # Size of each final embedding vector used in the graph model
    embedding_dim: int = 64

    # Number of graph propagation layers
    num_layers: int = 3

    # Dimension of the standalone joke text vectors
    text_feature_dim: int = 0

    # How to initialise item representations:
    # "learned_only" -> normal LightGCN item embeddings only
    # "text_only"    -> projected text vectors only
    # "add"          -> learned item embeddings + projected text vectors
    item_init_mode: str = "add"


# ---------------------------------------------------------
# Helper: text feature encoder / projector
# ---------------------------------------------------------
class JokeTextEncoder(nn.Module):
    """
    Projects standalone joke text vectors into the same embedding space
    used by the graph model.

    Input shape:
        (num_items, text_feature_dim)

    Output shape:
        (num_items, embedding_dim)
    """

    def __init__(self, text_feature_dim: int, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(text_feature_dim, embedding_dim)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.proj(text_features)


# ---------------------------------------------------------
# Model: Text-augmented LightGCN recommender
# ---------------------------------------------------------
class LightGCN(nn.Module):
    """
    A text-augmented LightGCN recommender model.

    Main idea:
    - Learn user embeddings
    - Optionally learn item embeddings
    - Encode standalone joke text vectors into the item space
    - Build initial item representations
    - Pass information through the user-item graph
    - Average embeddings from all propagation layers
    - Use the final embeddings for recommendation
    """

    # ---------------------------------------------------------
    # Initialise model embeddings and optional text encoder
    # ---------------------------------------------------------
    def __init__(
        self,
        cfg: LightGCNConfig,
        item_text_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # Learnable embedding for each user
        self.user_emb = nn.Embedding(cfg.num_users, cfg.embedding_dim)

        # Learnable embedding for each joke/item
        self.item_emb = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        # Initialise learned embeddings with small random values
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        # Optional standalone text features for jokes/items
        self.has_text_features = item_text_features is not None and cfg.text_feature_dim > 0

        if self.has_text_features:
            if item_text_features.size(0) != cfg.num_items:
                raise ValueError(
                    f"item_text_features has {item_text_features.size(0)} rows, "
                    f"but cfg.num_items is {cfg.num_items}."
                )
            if item_text_features.size(1) != cfg.text_feature_dim:
                raise ValueError(
                    f"item_text_features has dim {item_text_features.size(1)}, "
                    f"but cfg.text_feature_dim is {cfg.text_feature_dim}."
                )

            # Register raw text features as a non-trainable tensor stored with the model
            self.register_buffer("item_text_features", item_text_features)

            # Small encoder / projector from text-vector space -> graph embedding space
            self.text_encoder = JokeTextEncoder(
                text_feature_dim=cfg.text_feature_dim,
                embedding_dim=cfg.embedding_dim,
            )
        else:
            self.item_text_features = None
            self.text_encoder = None

    # ---------------------------------------------------------
    # Build initial item representations
    # ---------------------------------------------------------
    def get_initial_item_repr(self) -> torch.Tensor:
        """
        Build the initial item representation before graph propagation.

        Modes:
        - learned_only: use only learned item embeddings
        - text_only: use only encoded joke text vectors
        - add: combine learned item embeddings with encoded joke text vectors
        """
        learned_item_repr = self.item_emb.weight

        if not self.has_text_features or self.text_encoder is None:
            return learned_item_repr

        text_item_repr = self.text_encoder(self.item_text_features)

        if self.cfg.item_init_mode == "learned_only":
            return learned_item_repr

        if self.cfg.item_init_mode == "text_only":
            return text_item_repr

        if self.cfg.item_init_mode == "add":
            return learned_item_repr + text_item_repr

        raise ValueError(
            f"Unknown item_init_mode: {self.cfg.item_init_mode}. "
            f"Use 'learned_only', 'text_only', or 'add'."
        )

    # ---------------------------------------------------------
    # Propagate embeddings through the graph
    # ---------------------------------------------------------
    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run LightGCN propagation over the normalised user-item graph.

        This method:
        1) combines user embeddings with initial item representations
        2) repeatedly propagates them through the graph
        3) stores embeddings from every layer
        4) averages all layer outputs
        5) splits the result back into user and item embeddings
        """

        # Build initial user and item representations
        user_init = self.user_emb.weight
        item_init = self.get_initial_item_repr()

        # Combine user and item representations into one matrix
        all_emb = torch.cat([user_init, item_init], dim=0)

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