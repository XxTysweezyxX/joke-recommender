from __future__ import annotations

"""
Evaluate the trained text-augmented LightGCN model using the shared train/test split.

Run from /src with:
    python -m scripts.result.result_ta_lightgcn
"""

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.metrics import precision_at_k, recall_at_k, ndcg_at_k
from joke_reco import config
from joke_reco.text_augmented_lightgcn.build_joke_text_features import build_item_text_features
from joke_reco.text_augmented_lightgcn.text_augmented_lightgcn import LightGCN, LightGCNConfig


# ---------------------------------------------------------
# Helper: recommend jokes for one user using LightGCN scores
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Main: evaluate trained LightGCN model
# ---------------------------------------------------------
def main() -> None:
    """
    Main evaluation runner for the text-augmented LightGCN model.

    This script:
    1) loads the cleaned interaction data
    2) rebuilds the shared train/test split
    3) loads the trained LightGCN checkpoint
    4) rebuilds joke text features in the same way as training
    5) recreates the model and loads trained weights
    6) evaluates top-k recommendation performance
    """
    # File paths for interaction data, joke text data, and saved model checkpoint
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"
    model_path = ROOT / "models" / "ta_lightgcn_jester.pt"

    print("[evaluate_ta_lightgcn] Loading edges...")
    edges = pd.read_csv(edges_path)

    print("[evaluate_ta_lightgcn] Loading jokes...")
    jokes = pd.read_csv(jokes_path)

    print("[evaluate_ta_lightgcn] Rebuilding shared train/test split...")
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    print("[evaluate_ta_lightgcn] Loading checkpoint...")
    ckpt = torch.load(model_path, map_location="cpu")

    # ---------------------------------------------------------
    # Load supporting data from checkpoint
    # ---------------------------------------------------------
    user_map = ckpt["user_map"]
    item_map = ckpt["item_map"]
    norm_adj = ckpt["norm_adj"]
    meta = ckpt["meta"]

    # ---------------------------------------------------------
    # Rebuild item-side text features (same setup as training)
    # ---------------------------------------------------------
    print("[evaluate_ta_lightgcn] Building item text features...")
    item_text_features, _vectorizer = build_item_text_features(
        jokes_df=jokes,
        item_map=item_map,
        device="cpu",
    )

    # ---------------------------------------------------------
    # Recreate the same model configuration used during training
    # ---------------------------------------------------------
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
        text_feature_dim=item_text_features.shape[1],
    )

    # ---------------------------------------------------------
    # Recreate the model and load trained weights
    # ---------------------------------------------------------
    model = LightGCN(
        model_cfg,
        item_text_features=item_text_features,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---------------------------------------------------------
    # Run propagation to get final user/item embeddings
    # ---------------------------------------------------------
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    # ---------------------------------------------------------
    # Build relevant held-out test jokes per user
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Evaluate recommendations user by user
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Handle empty evaluation case
    # ---------------------------------------------------------
    if not precisions:
        print("[evaluate_ta_lightgcn] No users were evaluated.")
        return

    # ---------------------------------------------------------
    # Print final average metrics
    # ---------------------------------------------------------
    print(
        f"Text Augmented LightGCN Evaluation (users={len(precisions)}, K={config.K}, "
        f"threshold={config.LIKE_THRESHOLD}, holdout={config.HOLDOUT_PER_USER})"
    )
    print(f"Precision@{config.K}: {sum(precisions) / len(precisions):.4f}")
    print(f"Recall@{config.K}:    {sum(recalls) / len(recalls):.4f}")
    print(f"NDCG@{config.K}:      {sum(ndcgs) / len(ndcgs):.4f}")


# ---------------------------------------------------------
# Standard Python entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()