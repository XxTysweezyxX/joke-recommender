from __future__ import annotations

"""
Evaluate the trained LightGCN model using the shared train/test split.

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
    Recommend top-k joke IDs for one user using LightGCN scores.

    The score is the dot product between the user embedding and
    each item embedding.

    Jokes already seen in the training set are removed.
    """
    # If the user was not part of training, no recommendation can be made
    if user_id not in user_map:
        return []

    # Get this user's embedding vector
    u_idx = user_map[user_id]
    u_vec = user_emb[u_idx]

    # Score every item against this user
    scores = torch.matmul(item_emb, u_vec)

    # Get jokes this user already interacted with in training
    seen_items = set(
        train_edges.loc[train_edges["user_id"] == user_id, "joke_id"].astype(int).tolist()
    )

    # Convert internal item index back to the original joke_id
    idx_to_item = {idx: raw_joke_id for raw_joke_id, idx in item_map.items()}

    candidates = []
    for item_idx in range(len(idx_to_item)):
        raw_joke_id = idx_to_item[item_idx]

        # Skip jokes already seen during training
        if raw_joke_id in seen_items:
            continue

        # Store candidate joke and its score
        candidates.append((raw_joke_id, float(scores[item_idx].item())))

    # Rank candidates from highest score to lowest
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Return only the joke IDs for the top-k items
    return [joke_id for joke_id, _score in candidates[:k]]


def main() -> None:
    # File paths for interaction data and saved model checkpoint
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

    # Load supporting data from the checkpoint
    user_map = ckpt["user_map"]
    item_map = ckpt["item_map"]
    norm_adj = ckpt["norm_adj"]

    # Rebuild the same model configuration used during training
    meta = ckpt["meta"]
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
    )

    # Recreate the model and load trained weights
    model = LightGCN(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Propagate through the graph to get final user/item embeddings
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    # Keep only users who have relevant held-out test jokes
    test_relevant = (
        test_edges[test_edges["rating"] >= config.LIKE_THRESHOLD]
        .groupby("user_id")["joke_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )

    # Only evaluate users that exist in the trained model
    eligible_users = [u for u in test_relevant.keys() if u in user_map]

    # Optional limit on number of users evaluated
    eval_users = eligible_users[: config.EVAL_USERS]

    precisions = []
    recalls = []
    ndcgs = []

    # Evaluate recommendations user by user
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

        # Skip users with no recommendations
        if not recs:
            continue

        # Compute ranking metrics
        precisions.append(precision_at_k(recs, relevant, config.K))
        recalls.append(recall_at_k(recs, relevant, config.K))
        ndcgs.append(ndcg_at_k(recs, relevant, config.K))

    # If nothing was evaluated, stop early
    if not precisions:
        print("[evaluate_lightgcn] No users were evaluated.")
        return

    # Print final average metrics
    print(
        f"LightGCN Evaluation (users={len(precisions)}, K={config.K}, "
        f"threshold={config.LIKE_THRESHOLD}, holdout={config.HOLDOUT_PER_USER})"
    )
    print(f"Precision@{config.K}: {sum(precisions) / len(precisions):.4f}")
    print(f"Recall@{config.K}:    {sum(recalls) / len(recalls):.4f}")
    print(f"NDCG@{config.K}:      {sum(ndcgs) / len(ndcgs):.4f}")


if __name__ == "__main__":
    main()