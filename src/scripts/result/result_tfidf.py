from __future__ import annotations

"""
Evaluate the TF-IDF baseline using the shared train/test split.

Run from /src with:
    python -m scripts.result.result_tfidf
"""

import pandas as pd

from joke_reco import config
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.metrics import precision_at_k, recall_at_k, ndcg_at_k
from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.train_tfidf import build_tfidf_recommender


def main() -> None:
    # Path to the cleaned interaction dataset
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    print("[evaluate_tfidf] Loading data...")
    edges = pd.read_csv(edges_path)

    print("[evaluate_tfidf] Rebuilding shared train/test split...")
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    print("[evaluate_tfidf] Fitting TF-IDF model...")
    model = build_tfidf_recommender(max_features=5000, use_bigrams=True)

    # Build the set of relevant held-out jokes for each user
    test_relevant = (
        test_edges[test_edges["rating"] >= config.LIKE_THRESHOLD]
        .groupby("user_id")["joke_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )

    # Optional limit on number of users evaluated
    eval_users = list(test_relevant.keys())[: config.EVAL_USERS]

    precisions = []
    recalls = []
    ndcgs = []

    # Evaluate recommendations user by user
    for user_id in eval_users:
        relevant = test_relevant[user_id]

        # Get top-k recommendations using the TF-IDF diversity version
        recs_with_scores = model.recommend_for_user_no_duplicates(
            edges_df=train_edges,
            user_id=user_id,
            k=config.K,
            like_threshold=config.LIKE_THRESHOLD,
            candidate_pool=100,
            sim_threshold=0.70,
        )

        # Keep only joke IDs for metric evaluation
        recs = [joke_id for joke_id, _score in recs_with_scores]

        # Skip users with no recommendations
        if not recs:
            continue

        # Compute ranking metrics
        precisions.append(precision_at_k(recs, relevant, config.K))
        recalls.append(recall_at_k(recs, relevant, config.K))
        ndcgs.append(ndcg_at_k(recs, relevant, config.K))

    # If nothing was evaluated, stop early
    if not precisions:
        print("[evaluate_tfidf] No users were evaluated.")
        return

    # Print final average metrics
    print(
        f"TF-IDF Evaluation (users={len(precisions)}, K={config.K}, "
        f"threshold={config.LIKE_THRESHOLD}, holdout={config.HOLDOUT_PER_USER})"
    )
    print(f"Precision@{config.K}: {sum(precisions) / len(precisions):.4f}")
    print(f"Recall@{config.K}:    {sum(recalls) / len(recalls):.4f}")
    print(f"NDCG@{config.K}:      {sum(ndcgs) / len(ndcgs):.4f}")


if __name__ == "__main__":
    main()