from __future__ import annotations

"""
Evaluate the trained LightGCN model using a shared train/test split.

Run from /src with:
    python -m scripts.result.result_lightgcn
"""

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.metrics import precision_at_k, recall_at_k, ndcg_at_k
from joke_reco import config
from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig


def recommend_for_user_lightgcn(
    user_id: int,
    train_edges: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    k: int,
) -> list[int]:
    """
    Recommend top-k joke_ids for one user using dot-product scores.

    Excludes jokes the user already interacted with in the training set.
    """
    if user_id not in user_map:
        return []

    u_idx = user_map[user_id]
    u_vec = user_emb[u_idx]  # (D,)

    # score all items
    scores = torch.matmul(item_emb, u_vec)  # (num_items,)

    # exclude training interactions for this user
    seen_items = set(
        train_edges.loc[train_edges["user_id"] == user_id, "joke_id"].astype(int).tolist()
    )

    # convert item_map index -> raw joke_id
    idx_to_item = {idx: raw_joke_id for raw_joke_id, idx in item_map.items()}

    candidates = []
    for item_idx in range(len(idx_to_item)):
        raw_joke_id = idx_to_item[item_idx]
        if raw_joke_id in seen_items:
            continue
        candidates.append((raw_joke_id, float(scores[item_idx].item())))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [joke_id for joke_id, _score in candidates[:k]]


def main() -> None:
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"
    model_path = ROOT / "models" / "lightgcn_jester.pt"

    print("[evaluate_lightgcn] Loading edges...")
    edges = pd.read_csv(edges_path)

    print("[evaluate_lightgcn] Rebuilding shared train/test split...")
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    print("[evaluate_lightgcn] Loading checkpoint...")
    ckpt = torch.load(model_path, map_location="cpu")

    user_map = ckpt["user_map"]
    item_map = ckpt["item_map"]
    norm_adj = ckpt["norm_adj"]

    meta = ckpt["meta"]
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
    )

    model = LightGCN(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    # Only result users who have relevant held-out items
    test_relevant = (
        test_edges[test_edges["rating"] >= config.LIKE_THRESHOLD]
        .groupby("user_id")["joke_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )

    eligible_users = [u for u in test_relevant.keys() if u in user_map]

    # Optional speed limit
    eval_users = eligible_users[: config.EVAL_USERS]

    precisions = []
    recalls = []
    ndcgs = []

    for user_id in eval_users:
        relevant = test_relevant[user_id]

        recs = recommend_for_user_lightgcn(
            user_id=user_id,
            train_edges=train_edges,
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
            k=config.K,
        )

        if not recs:
            continue

        precisions.append(precision_at_k(recs, relevant, config.K))
        recalls.append(recall_at_k(recs, relevant, config.K))
        ndcgs.append(ndcg_at_k(recs, relevant, config.K))

    if not precisions:
        print("[evaluate_lightgcn] No users were evaluated.")
        return

    print(
        f"LightGCN Evaluation (users={len(precisions)}, K={config.K}, "
        f"threshold={config.LIKE_THRESHOLD}, holdout={config.HOLDOUT_PER_USER})"
    )
    print(f"Precision@{config.K}: {sum(precisions) / len(precisions):.4f}")
    print(f"Recall@{config.K}:    {sum(recalls) / len(recalls):.4f}")
    print(f"NDCG@{config.K}:      {sum(ndcgs) / len(ndcgs):.4f}")


if __name__ == "__main__":
    main()