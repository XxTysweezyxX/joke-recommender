from __future__ import annotations

"""
Evaluates the trained text-augmented LightGCN model on the shared train/test split.
Rebuilds the saved model, generates recommendations, and reports ranking metrics.
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
# 1. Recommendation helper
# Scores all unseen jokes for one user and returns top-k IDs.
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
    # Return no recommendations if the user is missing
    if user_id not in user_map:
        return []

    # Convert the raw user ID to its model index
    u_idx = user_map[user_id]

    # Get the user embedding vector
    u_vec = user_emb[u_idx]

    # Score every item against this user
    scores = torch.matmul(item_emb, u_vec)

    # Collect jokes already seen in training
    seen_items = set(
        train_edges.loc[train_edges["user_id"] == user_id, "joke_id"].astype(int).tolist()
    )

    # Convert internal item indices back to raw joke IDs
    idx_to_item = {idx: raw_joke_id for raw_joke_id, idx in item_map.items()}

    # Store unseen candidate jokes and their scores
    candidates = []

    # Score each item in the model
    for item_idx in range(len(idx_to_item)):
        # Convert the internal index back to a raw joke ID
        raw_joke_id = idx_to_item[item_idx]

        # Skip jokes already seen in training
        if raw_joke_id in seen_items:
            continue

        # Store the candidate joke and its score
        candidates.append((raw_joke_id, float(scores[item_idx].item())))

    # Rank candidates from highest to lowest score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Return only the top-k joke IDs
    return [joke_id for joke_id, _score in candidates[:k]]


# ---------------------------------------------------------
# 2. Main execution
# Runs the full text-augmented LightGCN evaluation pipeline.
# ---------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------
    # 2.1 Data loading and split rebuilding
    # Loads the dataset and recreates the shared train/test split.
    # ---------------------------------------------------------
    # Build the path to the cleaned interaction data
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Build the path to the cleaned joke text data
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Build the path to the saved TA-LightGCN checkpoint
    model_path = ROOT / "models" / "ta_lightgcn_jester.pt"

    # Print a progress message
    print("[evaluate_ta_lightgcn] Loading edges...")

    # Load the cleaned interaction data
    edges = pd.read_csv(edges_path)

    # Print a progress message
    print("[evaluate_ta_lightgcn] Loading jokes...")

    # Load the cleaned joke text data
    jokes = pd.read_csv(jokes_path)

    # Print a progress message
    print("[evaluate_ta_lightgcn] Rebuilding shared train/test split...")

    # Rebuild the shared train/test split
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    # ---------------------------------------------------------
    # 2.2 Checkpoint loading
    # Loads the saved model checkpoint and metadata.
    # ---------------------------------------------------------
    # Print a progress message
    print("[evaluate_ta_lightgcn] Loading checkpoint...")

    # Load the saved checkpoint
    ckpt = torch.load(model_path, map_location="cpu")

    # Load the saved user mapping
    user_map = ckpt["user_map"]

    # Load the saved item mapping
    item_map = ckpt["item_map"]

    # Load the saved normalised graph
    norm_adj = ckpt["norm_adj"]

    # Load saved model metadata
    meta = ckpt["meta"]

    # ---------------------------------------------------------
    # 2.3 Text feature rebuilding and model setup
    # Rebuilds item text features and recreates the saved model.
    # ---------------------------------------------------------
    # Print a progress message
    print("[evaluate_ta_lightgcn] Building item text features...")

    # Rebuild item-side text features in the same way as training
    item_text_features, _vectorizer = build_item_text_features(
        jokes_df=jokes,
        item_map=item_map,
        device="cpu",
    )

    # Recreate the saved model configuration
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
        text_feature_dim=item_text_features.shape[1],
    )

    # Rebuild the text-augmented LightGCN model
    model = LightGCN(
        model_cfg,
        item_text_features=item_text_features,
    )

    # Load the trained model weights
    model.load_state_dict(ckpt["state_dict"])

    # Switch to evaluation mode
    model.eval()

    # Run propagation to get final user and item embeddings
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    # ---------------------------------------------------------
    # 2.4 Relevant test set preparation
    # Builds the held-out relevant joke set for each user.
    # ---------------------------------------------------------
    # Build held-out relevant jokes for each user
    test_relevant = (
        test_edges[test_edges["rating"] >= config.LIKE_THRESHOLD]
        .groupby("user_id")["joke_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )

    # Keep only users that exist in the trained model
    eligible_users = [u for u in test_relevant.keys() if u in user_map]

    # Limit the number of evaluated users if needed
    eval_users = eligible_users[: config.EVAL_USERS]

    # Store user-level precision values
    precisions = []

    # Store user-level recall values
    recalls = []

    # Store user-level NDCG values
    ndcgs = []

    # ---------------------------------------------------------
    # 2.5 User-level evaluation
    # Generates recommendations and computes ranking metrics.
    # ---------------------------------------------------------
    # Evaluate recommendations user by user
    for user_id in eval_users:
        # Get the held-out relevant jokes for this user
        relevant = test_relevant[user_id]

        # Generate top-k recommendations
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

        # Compute Precision@K
        precisions.append(precision_at_k(recs, relevant, config.K))

        # Compute Recall@K
        recalls.append(recall_at_k(recs, relevant, config.K))

        # Compute NDCG@K
        ndcgs.append(ndcg_at_k(recs, relevant, config.K))

    # ---------------------------------------------------------
    # 2.6 Final metric reporting
    # Prints the average ranking metrics.
    # ---------------------------------------------------------
    # Stop early if no users were evaluated
    if not precisions:
        print("[evaluate_ta_lightgcn] No users were evaluated.")
        return

    # Print the evaluation summary heading
    print(
        f"Text Augmented LightGCN Evaluation (users={len(precisions)}, K={config.K}, "
        f"threshold={config.LIKE_THRESHOLD}, holdout={config.HOLDOUT_PER_USER})"
    )

    # Print the average precision
    print(f"Precision@{config.K}: {sum(precisions) / len(precisions):.4f}")

    # Print the average recall
    print(f"Recall@{config.K}:    {sum(recalls) / len(recalls):.4f}")

    # Print the average NDCG
    print(f"NDCG@{config.K}:      {sum(ndcgs) / len(ndcgs):.4f}")


if __name__ == "__main__":
    main()