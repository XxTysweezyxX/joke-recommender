from __future__ import annotations

"""
Evaluates the TF-IDF baseline on the shared train/test split.
Builds recommendations for each user and reports Precision@K, Recall@K, and NDCG@K.
"""

import pandas as pd

from joke_reco import config
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.metrics import precision_at_k, recall_at_k, ndcg_at_k
from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.train_tfidf import build_tfidf_recommender


# ---------------------------------------------------------
# 1. Main execution
# Runs the full TF-IDF evaluation pipeline.
# ---------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------
    # 1.1 Data loading and split rebuilding
    # Loads the dataset and recreates the shared train/test split.
    # ---------------------------------------------------------
    # Build the path to the cleaned interaction data
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Print a progress message
    print("[evaluate_tfidf] Loading data...")

    # Load the cleaned interaction data
    edges = pd.read_csv(edges_path)

    # Print a progress message
    print("[evaluate_tfidf] Rebuilding shared train/test split...")

    # Rebuild the shared train/test split
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    # ---------------------------------------------------------
    # 1.2 TF-IDF model setup
    # Builds the TF-IDF recommender used for evaluation.
    # ---------------------------------------------------------
    # Print a progress message
    print("[evaluate_tfidf] Fitting TF-IDF model...")

    # Build the TF-IDF recommender
    model = build_tfidf_recommender(
        max_features=5000,
        use_bigrams=True,
    )

    # ---------------------------------------------------------
    # 1.3 Relevant test set preparation
    # Builds the held-out relevant joke set for each user.
    # ---------------------------------------------------------
    # Build held-out relevant jokes for each user
    test_relevant = (
        test_edges[test_edges["rating"] >= config.LIKE_THRESHOLD]
        .groupby("user_id")["joke_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )

    # Limit the number of evaluated users if needed
    eval_users = list(test_relevant.keys())[: config.EVAL_USERS]

    # Store user-level precision values
    precisions = []

    # Store user-level recall values
    recalls = []

    # Store user-level NDCG values
    ndcgs = []

    # ---------------------------------------------------------
    # 1.4 User-level evaluation
    # Generates recommendations and computes ranking metrics.
    # ---------------------------------------------------------
    # Evaluate recommendations user by user
    for user_id in eval_users:
        # Get the held-out relevant jokes for this user
        relevant = test_relevant[user_id]

        # Generate top-k recommendations without near-duplicates
        recs_with_scores = model.recommend_for_user_no_duplicates(
            edges_df=train_edges,
            user_id=user_id,
            k=config.K,
            like_threshold=config.LIKE_THRESHOLD,
            candidate_pool=100,
            sim_threshold=0.70,
        )

        # Keep only the recommended joke IDs
        recs = [joke_id for joke_id, _score in recs_with_scores]

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
    # 1.5 Final metric reporting
    # Prints the average ranking metrics.
    # ---------------------------------------------------------
    # Stop early if no users were evaluated
    if not precisions:
        print("[evaluate_tfidf] No users were evaluated.")
        return

    # Print the evaluation summary heading
    print(
        f"TF-IDF Evaluation (users={len(precisions)}, K={config.K}, "
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