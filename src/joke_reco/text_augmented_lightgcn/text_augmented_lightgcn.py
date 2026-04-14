from __future__ import annotations

"""
Defines the text-augmented LightGCN recommender.
Combines learned item embeddings with projected joke text features before graph propagation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Configuration
# Stores the main model settings, including text feature size.
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

    # Size of each raw joke text feature vector
    text_feature_dim: int = 0


# ---------------------------------------------------------
# 2. Text encoder
# Projects joke text features into the model embedding space.
# ---------------------------------------------------------
# AI-assisted section:
# ChatGPT helped with the design of this text augmentation part.
# Prompt summary: "Help me implement a text-augmented LightGCN
# where TF-IDF joke features are projected into the embedding
# space and added to item embeddings."
class JokeTextEncoder(nn.Module):
    # Convert raw text features into embedding-space vectors
    def __init__(self, text_feature_dim: int, embedding_dim: int):
        super().__init__()

        # Linear projection from text space to embedding space
        self.proj = nn.Linear(text_feature_dim, embedding_dim)

    # Run the projection layer on the text features
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.proj(text_features)


# ---------------------------------------------------------
# 3. Main model
# Defines the text-augmented LightGCN recommender.
# ---------------------------------------------------------
class LightGCN(nn.Module):
    # Create the model and its components
    def __init__(
        self,
        cfg: LightGCNConfig,
        item_text_features: Optional[torch.Tensor] = None,
    ):
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

        # Check whether text features are available
        self.has_text_features = item_text_features is not None and cfg.text_feature_dim > 0

        # Set up the text augmentation components if text exists
        if self.has_text_features:
            # Check that the number of joke rows matches the number of items
            if item_text_features.size(0) != cfg.num_items:
                raise ValueError(
                    f"item_text_features has {item_text_features.size(0)} rows, "
                    f"but cfg.num_items is {cfg.num_items}."
                )

            # Check that the text feature width matches the config
            if item_text_features.size(1) != cfg.text_feature_dim:
                raise ValueError(
                    f"item_text_features has dim {item_text_features.size(1)}, "
                    f"but cfg.text_feature_dim is {cfg.text_feature_dim}."
                )

            # Store the fixed joke text features with the model
            self.register_buffer("item_text_features", item_text_features)

            # Create the text encoder
            self.text_encoder = JokeTextEncoder(
                text_feature_dim=cfg.text_feature_dim,
                embedding_dim=cfg.embedding_dim,
            )
        else:
            # Fall back if no text features are used
            self.item_text_features = None
            self.text_encoder = None

    # ---------------------------------------------------------
    # 4. Initial item representations
    # Combines learned item embeddings with projected text features.
    # ---------------------------------------------------------
    # Build item representations before graph propagation
    def get_initial_item_repr(self) -> torch.Tensor:
        # Start from the normal learned item embeddings
        learned_item_repr = self.item_emb.weight

        # Return the normal item embeddings if no text is used
        if not self.has_text_features or self.text_encoder is None:
            return learned_item_repr

        # Project joke text features into embedding space
        text_item_repr = self.text_encoder(self.item_text_features)

        # Add projected text features to the learned item embeddings
        return learned_item_repr + text_item_repr

    # ---------------------------------------------------------
    # 5. Graph propagation
    # Runs LightGCN message passing and layer averaging.
    # ---------------------------------------------------------
    # Propagate embeddings through the normalised graph
    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the initial user embeddings
        user_init = self.user_emb.weight

        # Get the initial item embeddings
        item_init = self.get_initial_item_repr()

        # Combine user and item embeddings into one matrix
        all_emb = torch.cat([user_init, item_init], dim=0)

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
    # 6. BPR loss
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