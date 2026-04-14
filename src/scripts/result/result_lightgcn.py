from __future__ import annotations

"""
Evaluates the trained original LightGCN model on the shared train/test split.
Rebuilds the saved model, generates recommendations, and reports ranking metrics.
"""

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.metrics import precision_at_k, recall_at_k, ndcg_at_k
from joke_reco import config
from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig


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
# Runs the full original LightGCN evaluation pipeline.
# ---------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------
    # 2.1 Data loading and split rebuilding
    # Loads the dataset and recreates the shared train/test split.
    # ---------------------------------------------------------
    # Build the path to the cleaned interaction data
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Build the path to the saved LightGCN checkpoint
    model_path = ROOT / "models" / "lightgcn_jester.pt"

    # Print a progress message
    print("[evaluate_lightgcn] Loading edges...")

    # Load the cleaned interaction data
    edges = pd.read_csv(edges_path)

    # Print a progress message
    print("[evaluate_lightgcn] Rebuilding shared train/test split...")

    # Rebuild the shared train/test split
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    # ---------------------------------------------------------
    # 2.2 Checkpoint loading and model rebuilding
    # Loads the saved checkpoint and recreates the original model.
    # ---------------------------------------------------------
    # Print a progress message
    print("[evaluate_lightgcn] Loading checkpoint...")

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

    # Recreate the saved model configuration
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
    )

    # Rebuild the original LightGCN model
    model = LightGCN(model_cfg)

    # Load the full checkpoint state dictionary
    raw_state_dict = ckpt["state_dict"]

    # Get the keys expected by the original model
    model_state_keys = set(model.state_dict().keys())

    # Keep only weights that match the original model
    filtered_state_dict = {
        k: v for k, v in raw_state_dict.items() if k in model_state_keys
    }

    # Track any checkpoint keys that are ignored
    ignored_keys = [k for k in raw_state_dict.keys() if k not in model_state_keys]

    # Warn if extra checkpoint keys were found
    if ignored_keys:
        print(
            "[evaluate_lightgcn] Warning: checkpoint contains extra keys not used by "
            f"the original LightGCN model. Ignoring: {ignored_keys}"
        )

    # Load the filtered model weights
    model.load_state_dict(filtered_state_dict, strict=False)

    # Switch to evaluation mode
    model.eval()

    # Run propagation to get final user and item embeddings
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    # ---------------------------------------------------------
    # 2.3 Relevant test set preparation
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
    # 2.4 User-level evaluation
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
    # 2.5 Final metric reporting
    # Prints the average ranking metrics.
    # ---------------------------------------------------------
    # Stop early if no users were evaluated
    if not precisions:
        print("[evaluate_lightgcn] No users were evaluated.")
        return

    # Print the evaluation summary heading
    print(
        f"LightGCN Evaluation (users={len(precisions)}, K={config.K}, "
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