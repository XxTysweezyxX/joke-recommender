from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig


@dataclass
class TrainResult:
    """
    Stores the important outputs from training so they can be
    reused later for evaluation, checkpointing, or recommendation.
    """
    # The trained LightGCN model
    model: LightGCN

    # Maps original user IDs to the model's internal user indices
    user_map: Dict[int, int]

    # Maps original joke/item IDs to the model's internal item indices
    item_map: Dict[int, int]

    # Normalised sparse adjacency matrix of the user-item graph
    norm_adj: torch.Tensor  # torch sparse COO tensor


def _build_id_maps(edges_pos: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Converts raw user IDs and joke IDs into compact index values.

    LightGCN embeddings need indices like 0, 1, 2, 3... rather than
    the original raw IDs from the dataset.
    """
    # Get unique users and unique jokes/items
    users = edges_pos["user_id"].unique().tolist()
    items = edges_pos["joke_id"].unique().tolist()

    # Build dictionaries:
    # raw user ID  -> user index
    # raw joke ID  -> item index
    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(j): i for i, j in enumerate(items)}
    return user_map, item_map


def _build_norm_adj(
    edges_pos: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
) -> torch.Tensor:
    """
    Builds the symmetric normalised adjacency matrix for the
    user-item bipartite graph.

    Graph structure:
        A = [[0, R],
             [R^T, 0]]

    After that, it applies degree normalisation:
        D^{-1/2} A D^{-1/2}

    This is the matrix used during LightGCN propagation.
    """
    num_users = len(user_map)
    num_items = len(item_map)
    n = num_users + num_items  # total number of graph nodes

    # These will store the row and column positions for sparse edges
    rows: List[int] = []
    cols: List[int] = []

    # Loop through every positive user-joke interaction
    for u_raw, j_raw in zip(edges_pos["user_id"].astype(int), edges_pos["joke_id"].astype(int)):
        # Convert raw IDs into internal model indices
        u = user_map.get(u_raw)
        it = item_map.get(j_raw)

        # Skip anything missing just in case
        if u is None or it is None:
            continue

        # Item nodes come after all user nodes in the combined graph
        i_node = num_users + it

        # Add both directions to make the graph undirected:
        # user -> item and item -> user
        rows.extend([u, i_node])
        cols.extend([i_node, u])

    # If no graph edges exist, training cannot continue
    if len(rows) == 0:
        raise ValueError("No positive edges after filtering. Check like_threshold or input data.")

    # Build sparse adjacency matrix with value 1 for every edge
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)

    adj = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    # Degree of each node = number of connected neighbours
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # shape: (n,)

    # Compute D^{-1/2}
    deg_inv_sqrt = torch.pow(deg, -0.5)

    # Replace infinities with 0 for isolated nodes
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Apply symmetric normalisation to each edge value
    # v_ij = v_ij * d_i^{-1/2} * d_j^{-1/2}
    r, c = adj.indices()
    norm_values = adj.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]

    # Final normalised adjacency matrix used in graph propagation
    norm_adj = torch.sparse_coo_tensor(adj.indices(), norm_values, size=adj.size()).coalesce()
    return norm_adj


def _sample_bpr_batch(
    user_pos_items: Dict[int, np.ndarray],
    num_users: int,
    num_items: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Samples training triplets for BPR loss:
    (user, positive_item, negative_item)

    - user = a user index
    - positive_item = an item the user liked
    - negative_item = an item the user did not like
    """
    # Randomly choose users for this batch
    users = rng.integers(0, num_users, size=batch_size, endpoint=False)

    # Arrays for positive and negative item samples
    pos = np.empty(batch_size, dtype=np.int64)
    neg = np.empty(batch_size, dtype=np.int64)

    for idx, u in enumerate(users):
        # Get this user's known positive items
        pos_items = user_pos_items.get(u)

        # If the user has no stored positives, fall back to any random item
        if pos_items is None or len(pos_items) == 0:
            pos[idx] = rng.integers(0, num_items)
        else:
            # Otherwise pick one of the user's positive items
            pos[idx] = int(pos_items[rng.integers(0, len(pos_items))])

        # Pick a negative item that is not in the user's positives
        while True:
            j = int(rng.integers(0, num_items))
            if pos_items is None or j not in pos_items:
                neg[idx] = j
                break

    return users.astype(np.int64), pos, neg


def train_lightgcn(
    edges_train: pd.DataFrame,
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
    """
    Trains the LightGCN model using BPR loss on positive interactions only.

    Expected columns in edges_train:
    - user_id
    - joke_id
    - rating

    Only ratings >= like_threshold are treated as positive interactions.
    """
    # ---------------------------------------------------------
    # 1) Keep only positive interactions
    # ---------------------------------------------------------
    edges_pos = edges_train.loc[edges_train["rating"] >= like_threshold, ["user_id", "joke_id"]].copy()

    # Stop early if no positives exist
    if edges_pos.empty:
        raise ValueError("No positive interactions found. Lower like_threshold or check data.")

    # ---------------------------------------------------------
    # 2) Build ID maps and graph adjacency matrix
    # ---------------------------------------------------------
    user_map, item_map = _build_id_maps(edges_pos)
    norm_adj = _build_norm_adj(edges_pos, user_map, item_map)

    num_users = len(user_map)
    num_items = len(item_map)

    # ---------------------------------------------------------
    # 3) Build user -> positive items lookup (index space)
    # ---------------------------------------------------------
    tmp = edges_pos.copy()

    # Convert raw IDs into model indices
    tmp["u"] = tmp["user_id"].astype(int).map(user_map)
    tmp["i"] = tmp["joke_id"].astype(int).map(item_map)

    # Remove any rows that failed to map
    tmp = tmp.dropna()

    # Dictionary:
    # user index -> array of positive item indices
    user_pos_items: Dict[int, np.ndarray] = {}
    for u, grp in tmp.groupby("u")["i"]:
        user_pos_items[int(u)] = grp.astype(int).unique()

    # ---------------------------------------------------------
    # 4) Create model, move graph to device, set optimiser
    # ---------------------------------------------------------
    dev = torch.device(device)

    cfg = LightGCNConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )

    # Create the LightGCN model
    model = LightGCN(cfg).to(dev)

    # Move adjacency matrix to the same device as the model
    norm_adj_dev = norm_adj.to(dev)

    # Adam optimiser updates the trainable embeddings
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Random generator for reproducible sampling
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 5) Training loop
    # ---------------------------------------------------------
    steps_per_epoch = max(1, samples_per_epoch // batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            # Sample a batch of (user, positive item, negative item)
            u, pos_i, neg_j = _sample_bpr_batch(
                user_pos_items=user_pos_items,
                num_users=num_users,
                num_items=num_items,
                batch_size=batch_size,
                rng=rng,
            )

            # Convert NumPy arrays to PyTorch tensors
            u_t = torch.from_numpy(u).to(dev)
            pos_t = torch.from_numpy(pos_i).to(dev)
            neg_t = torch.from_numpy(neg_j).to(dev)

            # Clear old gradients before the next update
            opt.zero_grad(set_to_none=True)

            # Run LightGCN propagation to get updated user/item embeddings
            user_emb, item_emb = model.propagate(norm_adj_dev)

            # Compute BPR loss:
            # positive items should rank above negative items
            loss = LightGCN.bpr_loss(user_emb[u_t], item_emb[pos_t], item_emb[neg_t])

            # Backpropagation + optimiser step
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        # Average loss for this epoch
        avg_loss = total_loss / steps_per_epoch
        print(f"[LightGCN] epoch={epoch}/{epochs}  avg_bpr_loss={avg_loss:.4f}")

    # ---------------------------------------------------------
    # 6) Return everything needed after training
    # ---------------------------------------------------------
    return TrainResult(
        model=model,
        user_map=user_map,
        item_map=item_map,
        norm_adj=norm_adj,  # keep original on CPU for later saving/loading
    )