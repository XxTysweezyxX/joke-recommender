from __future__ import annotations

"""
Text-augmented LightGCN model definition for the joke recommender project.

This version is simplified so that item representations always use:

    learned item embedding + projected text embedding

The original LightGCN parts are still present:
- learnable user embeddings
- learnable item embeddings
- graph propagation
- layer averaging
- BPR loss

The new text-augmented part is:
- encode joke text features into embedding space
- add them to the learned item embeddings before propagation
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Configuration
# Same idea as original LightGCN, with one extra field
# for the dimension of joke text features
# ---------------------------------------------------------
@dataclass
class LightGCNConfig:
    """
    Stores the configuration for the text-augmented LightGCN model.
    """

    # Number of users in the dataset
    num_users: int

    # Number of joke items in the dataset
    num_items: int

    # Size of the embedding vectors used by the model
    embedding_dim: int = 64

    # Number of graph propagation layers
    num_layers: int = 3

    # Size of each raw joke text feature vector
    text_feature_dim: int = 0


# ---------------------------------------------------------
# NEW: Text augmentation component
# Converts joke text features into the same embedding space
# used by the LightGCN model
# ---------------------------------------------------------
class JokeTextEncoder(nn.Module):
    """
    Projects joke text feature vectors into LightGCN embedding space.

    Input shape:
        (num_items, text_feature_dim)

    Output shape:
        (num_items, embedding_dim)
    """

    def __init__(self, text_feature_dim: int, embedding_dim: int):
        super().__init__()

        # Linear layer that maps raw text features
        # into the graph embedding space
        self.proj = nn.Linear(text_feature_dim, embedding_dim)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        # Convert joke text features into embedding-space vectors
        return self.proj(text_features)


# ---------------------------------------------------------
# Model: Text-augmented LightGCN
# ---------------------------------------------------------
class LightGCN(nn.Module):
    """
    A simplified text-augmented LightGCN recommender.

    Structure:
    1) Create normal user embeddings         (same as original LightGCN)
    2) Create normal item embeddings         (same as original LightGCN)
    3) Encode joke text features             (new)
    4) Add text embedding to item embedding  (new)
    5) Propagate through the graph           (same as original LightGCN)
    6) Average all layer outputs             (same as original LightGCN)
    7) Train with BPR loss                   (same as original LightGCN)
    """

    def __init__(
        self,
        cfg: LightGCNConfig,
        item_text_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg

        # =========================================================
        # SAME AS ORIGINAL LIGHTGCN:
        # trainable user and item embeddings
        # =========================================================
        self.user_emb = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_emb = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        # Initialise embeddings with small random values
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        # =========================================================
        # NEW: TEXT-AUGMENTED PART
        # store joke text features and create text encoder
        # =========================================================
        self.has_text_features = item_text_features is not None and cfg.text_feature_dim > 0

        if self.has_text_features:
            # Check that the number of joke rows matches the number of items
            if item_text_features.size(0) != cfg.num_items:
                raise ValueError(
                    f"item_text_features has {item_text_features.size(0)} rows, "
                    f"but cfg.num_items is {cfg.num_items}."
                )

            # Check that the text feature width matches config
            if item_text_features.size(1) != cfg.text_feature_dim:
                raise ValueError(
                    f"item_text_features has dim {item_text_features.size(1)}, "
                    f"but cfg.text_feature_dim is {cfg.text_feature_dim}."
                )

            # Store raw joke text features with the model
            # These are fixed inputs, not trainable embeddings
            self.register_buffer("item_text_features", item_text_features)

            # Text encoder: converts raw joke text features
            # into the same embedding space as LightGCN
            self.text_encoder = JokeTextEncoder(
                text_feature_dim=cfg.text_feature_dim,
                embedding_dim=cfg.embedding_dim,
            )
        else:
            self.item_text_features = None
            self.text_encoder = None

    # ---------------------------------------------------------
    # NEW: Build initial item representations
    # This is the key text-augmented difference
    # ---------------------------------------------------------
    def get_initial_item_repr(self) -> torch.Tensor:
        """
        Build the initial item representation before graph propagation.

        If text features are available:
            initial_item = learned_item_embedding + projected_text_embedding

        Otherwise:
            initial_item = learned_item_embedding
        """
        # SAME AS ORIGINAL LIGHTGCN:
        # start from normal learned item embeddings
        learned_item_repr = self.item_emb.weight

        # If no text features exist, fall back to normal LightGCN behaviour
        if not self.has_text_features or self.text_encoder is None:
            return learned_item_repr

        # NEW:
        # project joke text vectors into embedding space
        text_item_repr = self.text_encoder(self.item_text_features)

        # NEW:
        # combine learned item embeddings with projected text embeddings
        return learned_item_repr + text_item_repr

    # ---------------------------------------------------------
    # SAME AS ORIGINAL LIGHTGCN:
    # graph propagation and layer averaging
    # ---------------------------------------------------------
    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run LightGCN propagation over the normalised user-item graph.

        Steps:
        1) build initial user and item representations
        2) combine them into one matrix
        3) propagate through the graph for several layers
        4) average all layer outputs
        5) split back into user and item embeddings
        """

        # SAME AS ORIGINAL LIGHTGCN:
        # normal trainable user embeddings
        user_init = self.user_emb.weight

        # DIFFERENCE:
        # item side may include added text information
        item_init = self.get_initial_item_repr()

        # Combine users and items into one matrix for propagation
        all_emb = torch.cat([user_init, item_init], dim=0)

        # Store layer 0 embeddings before propagation
        embs = [all_emb]

        # SAME AS ORIGINAL LIGHTGCN:
        # repeated graph propagation
        for _ in range(self.cfg.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        # SAME AS ORIGINAL LIGHTGCN:
        # average embeddings from all layers
        out = torch.stack(embs, dim=0).mean(dim=0)

        # Split back into final user and item embeddings
        users_out = out[: self.cfg.num_users]
        items_out = out[self.cfg.num_users:]
        return users_out, items_out

    # ---------------------------------------------------------
    # SAME AS ORIGINAL LIGHTGCN:
    # BPR loss for pairwise ranking
    # ---------------------------------------------------------
    @staticmethod
    def bpr_loss(
        u: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Bayesian Personalised Ranking (BPR) loss.

        The goal is to make the positive item score higher
        than the negative item score for the same user.
        """

        # Score the user against a positive item
        pos_scores = (u * pos).sum(dim=1)

        # Score the user against a negative item
        neg_scores = (u * neg).sum(dim=1)

        # Encourage positive items to rank above negative items
        return -F.logsigmoid(pos_scores - neg_scores).mean()