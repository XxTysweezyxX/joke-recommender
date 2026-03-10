from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch

from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig


@dataclass
class TrainResult:
    """Return bundle used by the runner to save a checkpoint."""
    model: LightGCN
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    norm_adj: torch.Tensor  # torch sparse COO tensor


def _build_id_maps(edges_pos: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build raw-id -> contiguous index maps for users and items.
    edges_pos must contain columns: user_id, joke_id
    """
    users = edges_pos["user_id"].unique().tolist()
    items = edges_pos["joke_id"].unique().tolist()

    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(j): i for i, j in enumerate(items)}
    return user_map, item_map


def _build_norm_adj(
    edges_pos: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
) -> torch.Tensor:
    """
    Build the symmetric normalized adjacency matrix for the bipartite user-item graph:
      A = [[0, R],
           [R^T, 0]]
    Then returns  D^{-1/2} A D^{-1/2}  as a torch sparse COO tensor.
    """
    num_users = len(user_map)
    num_items = len(item_map)
    n = num_users + num_items  # total nodes

    # Build COO indices (undirected edges between user node and item node)
    rows: List[int] = []
    cols: List[int] = []

    for u_raw, j_raw in zip(edges_pos["user_id"].astype(int), edges_pos["joke_id"].astype(int)):
        u = user_map.get(u_raw)
        it = item_map.get(j_raw)
        if u is None or it is None:
            continue

        i_node = num_users + it
        # user -> item, item -> user (undirected)
        rows.extend([u, i_node])
        cols.extend([i_node, u])

    if len(rows) == 0:
        raise ValueError("No positive edges after filtering. Check like_threshold or input data.")

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)

    adj = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    # Degree: sum of adjacency rows
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # (n,)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize values: v_ij := v_ij * d_i^{-1/2} * d_j^{-1/2}
    r, c = adj.indices()
    norm_values = adj.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]

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
    Sample (u, pos_i, neg_j) triplets for BPR.
    - user indices are 0..num_users-1
    - item indices are 0..num_items-1
    """
    users = rng.integers(0, num_users, size=batch_size, endpoint=False)

    pos = np.empty(batch_size, dtype=np.int64)
    neg = np.empty(batch_size, dtype=np.int64)

    for idx, u in enumerate(users):
        pos_items = user_pos_items.get(u)
        if pos_items is None or len(pos_items) == 0:
            # fallback (rare) - pick a random positive item
            pos[idx] = rng.integers(0, num_items)
        else:
            pos[idx] = int(pos_items[rng.integers(0, len(pos_items))])

        # negative sample not in user's positives
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
    Train LightGCN using BPR loss on positive edges (rating >= like_threshold).

    edges_train columns expected: user_id, joke_id, rating
    Returns TrainResult with model + maps + normalized adjacency.
    """
    # 1) Filter to positive interactions
    edges_pos = edges_train.loc[edges_train["rating"] >= like_threshold, ["user_id", "joke_id"]].copy()
    if edges_pos.empty:
        raise ValueError("No positive interactions found. Lower like_threshold or check data.")

    # 2) Build maps and adjacency
    user_map, item_map = _build_id_maps(edges_pos)
    norm_adj = _build_norm_adj(edges_pos, user_map, item_map)

    num_users = len(user_map)
    num_items = len(item_map)

    # 3) Build user -> positive item list (in index space)
    tmp = edges_pos.copy()
    tmp["u"] = tmp["user_id"].astype(int).map(user_map)
    tmp["i"] = tmp["joke_id"].astype(int).map(item_map)
    tmp = tmp.dropna()

    user_pos_items: Dict[int, np.ndarray] = {}
    for u, grp in tmp.groupby("u")["i"]:
        user_pos_items[int(u)] = grp.astype(int).unique()

    # 4) Model + optimizer
    dev = torch.device(device)
    cfg = LightGCNConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
    model = LightGCN(cfg).to(dev)
    norm_adj_dev = norm_adj.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(seed)

    # 5) Training loop
    steps_per_epoch = max(1, samples_per_epoch // batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            u, pos_i, neg_j = _sample_bpr_batch(
                user_pos_items=user_pos_items,
                num_users=num_users,
                num_items=num_items,
                batch_size=batch_size,
                rng=rng,
            )

            u_t = torch.from_numpy(u).to(dev)
            pos_t = torch.from_numpy(pos_i).to(dev)
            neg_t = torch.from_numpy(neg_j).to(dev)

            opt.zero_grad(set_to_none=True)

            user_emb, item_emb = model.propagate(norm_adj_dev)
            loss = LightGCN.bpr_loss(user_emb[u_t], item_emb[pos_t], item_emb[neg_t])

            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / steps_per_epoch
        print(f"[LightGCN] epoch={epoch}/{epochs}  avg_bpr_loss={avg_loss:.4f}")

    return TrainResult(
        model=model,
        user_map=user_map,
        item_map=item_map,
        norm_adj=norm_adj,  # keep original on CPU; runner moves to CPU anyway
    )