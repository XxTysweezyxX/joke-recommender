from __future__ import annotations

"""
Trains the text-augmented LightGCN recommender.
Builds the graph, prepares joke text features, and optimises the model with BPR loss.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

from joke_reco.text_augmented_lightgcn.text_augmented_lightgcn import LightGCN, LightGCNConfig
from joke_reco.text_augmented_lightgcn.build_joke_text_features import build_item_text_features


# ---------------------------------------------------------
# 1. Training output container
# Stores the trained model, ID mappings, and graph.
# ---------------------------------------------------------
@dataclass
class TrainResult:
    # Store the trained LightGCN model
    model: LightGCN

    # Store the raw user ID to model index mapping
    user_map: Dict[int, int]

    # Store the raw joke ID to model index mapping
    item_map: Dict[int, int]

    # Store the normalised user-item graph
    norm_adj: torch.Tensor


# ---------------------------------------------------------
# 2. ID mapping
# Converts raw user and joke IDs into compact model indices.
# ---------------------------------------------------------
def _build_id_maps(edges_pos: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    # Get the unique users from the positive edges
    users = edges_pos["user_id"].unique().tolist()

    # Get the unique jokes from the positive edges
    items = edges_pos["joke_id"].unique().tolist()

    # Map each raw user ID to a compact index
    user_map = {int(u): i for i, u in enumerate(users)}

    # Map each raw joke ID to a compact index
    item_map = {int(j): i for i, j in enumerate(items)}

    return user_map, item_map


# ---------------------------------------------------------
# 3. Normalised graph construction
# Builds the symmetric normalised adjacency matrix for LightGCN.
# ---------------------------------------------------------
def _build_norm_adj(
    edges_pos: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
) -> torch.Tensor:
    # Get the number of users
    num_users = len(user_map)

    # Get the number of items
    num_items = len(item_map)

    # Total graph nodes = users + items
    n = num_users + num_items

    # Store sparse matrix row indices
    rows: List[int] = []

    # Store sparse matrix column indices
    cols: List[int] = []

    # Loop through each positive user-joke interaction
    for u_raw, j_raw in zip(
        edges_pos["user_id"].astype(int),
        edges_pos["joke_id"].astype(int)
    ):
        # Convert the raw user ID into a model index
        u = user_map.get(u_raw)

        # Convert the raw joke ID into a model index
        it = item_map.get(j_raw)

        # Skip any missing mappings
        if u is None or it is None:
            continue

        # Shift item nodes after all user nodes
        i_node = num_users + it

        # Add the user-to-item edge
        rows.append(u)
        cols.append(i_node)

        # Add the item-to-user edge
        rows.append(i_node)
        cols.append(u)

    # Stop if no valid graph edges were created
    if len(rows) == 0:
        raise ValueError("No positive edges after filtering. Check like_threshold or input data.")

    # Build the sparse edge index tensor
    indices = torch.tensor([rows, cols], dtype=torch.long)

    # Give each edge a value of 1
    values = torch.ones(len(rows), dtype=torch.float32)

    # Create the sparse adjacency matrix
    adj = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    # Compute node degrees
    deg = torch.sparse.sum(adj, dim=1).to_dense()

    # Compute D^(-1/2)
    deg_inv_sqrt = torch.pow(deg, -0.5)

    # Replace infinities for isolated nodes
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Get the row and column indices of each edge
    r, c = adj.indices()

    # Apply symmetric normalisation to each edge value
    norm_values = adj.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]

    # Build the final normalised adjacency matrix
    norm_adj = torch.sparse_coo_tensor(
        adj.indices(),
        norm_values,
        size=adj.size()
    ).coalesce()

    return norm_adj


# ---------------------------------------------------------
# 4. BPR batch sampling
# Samples user, positive item, and negative item triplets.
# ---------------------------------------------------------
def _sample_bpr_batch(
    user_pos_items: Dict[int, np.ndarray],
    num_users: int,
    num_items: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Randomly sample user indices
    users = rng.integers(0, num_users, size=batch_size, endpoint=False)

    # Store sampled positive items
    pos = np.empty(batch_size, dtype=np.int64)

    # Store sampled negative items
    neg = np.empty(batch_size, dtype=np.int64)

    # Build one training triplet per sampled user
    for idx, u in enumerate(users):
        # Get this user's known positive items
        pos_items = user_pos_items.get(u)

        # Fall back to a random item if no positives are stored
        if pos_items is None or len(pos_items) == 0:
            pos[idx] = rng.integers(0, num_items)
        else:
            # Sample one known positive item
            pos[idx] = int(pos_items[rng.integers(0, len(pos_items))])

        # Keep sampling until a negative item is found
        while True:
            # Sample a candidate negative item
            j = int(rng.integers(0, num_items))

            # Accept it if it is not in the user's positives
            if pos_items is None or j not in pos_items:
                neg[idx] = j
                break

    return users.astype(np.int64), pos, neg


# ---------------------------------------------------------
# 5. Main training function
# Filters positives, builds features, and trains the model.
# ---------------------------------------------------------
def train_ta_lightgcn(
    edges_train: pd.DataFrame,
    jokes_df: pd.DataFrame,
    like_threshold: float,
    embedding_dim: int = 64,
    num_layers: int = 3,
    lr: float = 1e-3,
    batch_size: int = 2048,
    epochs: int = 10,
    samples_per_epoch: int = 200_000,
    seed: int = 42,
    device: str = "cpu",
) -> TrainResult:
    # Keep only positive user-joke interactions
    edges_pos = edges_train.loc[
        edges_train["rating"] >= like_threshold,
        ["user_id", "joke_id"]
    ].copy()

    # Stop if no positive interactions remain
    if edges_pos.empty:
        raise ValueError("No positive interactions found. Lower like_threshold or check data.")

    # Build user and item ID mappings
    user_map, item_map = _build_id_maps(edges_pos)

    # Build the normalised user-item graph
    norm_adj = _build_norm_adj(edges_pos, user_map, item_map)

    # ---------------------------------------------------------
    # 5.1 Text feature preparation
    # Builds joke text features for the item side of the model.
    # ---------------------------------------------------------
    # AI-assisted section:
    # ChatGPT helped with this text-feature integration step.
    # Prompt summary: "Help me extend my LightGCN training pipeline
    # so it builds joke text features and passes them into a
    # text-augmented LightGCN model."

    # Build joke text features aligned with the item mapping
    item_text_features, vectorizer = build_item_text_features(
        jokes_df=jokes_df,
        item_map=item_map,
        device=device,
    )

    # Get the number of mapped users
    num_users = len(user_map)

    # Get the number of mapped items
    num_items = len(item_map)

    # Copy positive edges for index conversion
    tmp = edges_pos.copy()

    # Map raw user IDs into user indices
    tmp["u"] = tmp["user_id"].astype(int).map(user_map)

    # Map raw joke IDs into item indices
    tmp["i"] = tmp["joke_id"].astype(int).map(item_map)

    # Drop rows that failed to map
    tmp = tmp.dropna()

    # Store each user's positive item indices
    user_pos_items: Dict[int, np.ndarray] = {}

    # Group item indices by user index
    for u, grp in tmp.groupby("u")["i"]:
        user_pos_items[int(u)] = grp.astype(int).unique()

    # Set the target device
    dev = torch.device(device)

    # Build the model configuration
    cfg = LightGCNConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        text_feature_dim=item_text_features.shape[1],
    )

    # Create the text-augmented LightGCN model
    model = LightGCN(
        cfg,
        item_text_features=item_text_features,
    ).to(dev)

    # Move the graph to the same device as the model
    norm_adj_dev = norm_adj.to(dev)

    # Create the optimiser
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Create a reproducible random number generator
    rng = np.random.default_rng(seed)

    # Compute the number of batches per epoch
    steps_per_epoch = max(1, samples_per_epoch // batch_size)

    # ---------------------------------------------------------
    # 5.2 Training loop
    # Repeatedly samples triplets and updates the model.
    # ---------------------------------------------------------
    # Run the training epochs
    for epoch in range(1, epochs + 1):
        # Put the model in training mode
        model.train()

        # Track the total epoch loss
        total_loss = 0.0

        # Run each batch update for the epoch
        for _ in range(steps_per_epoch):
            # Sample a batch of BPR triplets
            u, pos_i, neg_j = _sample_bpr_batch(
                user_pos_items=user_pos_items,
                num_users=num_users,
                num_items=num_items,
                batch_size=batch_size,
                rng=rng,
            )

            # Convert sampled users to a tensor
            u_t = torch.from_numpy(u).to(dev)

            # Convert positive items to a tensor
            pos_t = torch.from_numpy(pos_i).to(dev)

            # Convert negative items to a tensor
            neg_t = torch.from_numpy(neg_j).to(dev)

            # Clear old gradients
            opt.zero_grad(set_to_none=True)

            # Run graph propagation
            user_emb, item_emb = model.propagate(norm_adj_dev)

            # Compute the BPR ranking loss
            loss = LightGCN.bpr_loss(
                user_emb[u_t],
                item_emb[pos_t],
                item_emb[neg_t]
            )

            # Backpropagate the loss
            loss.backward()

            # Update model parameters
            opt.step()

            # Add this batch loss to the epoch total
            total_loss += float(loss.item())

        # Compute the average loss for the epoch
        avg_loss = total_loss / steps_per_epoch

        # Print training progress
        print(f"[LightGCN] epoch={epoch}/{epochs}  avg_bpr_loss={avg_loss:.4f}")

    # ---------------------------------------------------------
    # 6. Training output
    # Returns the trained model and required lookup objects.
    # ---------------------------------------------------------
    return TrainResult(
        model=model,
        user_map=user_map,
        item_map=item_map,
        norm_adj=norm_adj,
    )