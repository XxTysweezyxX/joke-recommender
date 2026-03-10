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

    test_relevant = (
        test_edges[test_edges["rating"] >= config.LIKE_THRESHOLD]
        .groupby("user_id")["joke_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )

    eval_users = list(test_relevant.keys())[: config.EVAL_USERS]

    precisions = []
    recalls = []
    ndcgs = []

    for user_id in eval_users:
        relevant = test_relevant[user_id]

        recs_with_scores = model.recommend_for_user_no_duplicates(
            edges_df=train_edges,
            user_id=user_id,
            k=config.K,
            like_threshold=config.LIKE_THRESHOLD,
            candidate_pool=100,
            sim_threshold=0.70,
        )

        recs = [joke_id for joke_id, _score in recs_with_scores]
        if not recs:
            continue

        precisions.append(precision_at_k(recs, relevant, config.K))
        recalls.append(recall_at_k(recs, relevant, config.K))
        ndcgs.append(ndcg_at_k(recs, relevant, config.K))

    if not precisions:
        print("[evaluate_tfidf] No users were evaluated.")
        return

    print(
        f"TF-IDF Evaluation (users={len(precisions)}, K={config.K}, "
        f"threshold={config.LIKE_THRESHOLD}, holdout={config.HOLDOUT_PER_USER})"
    )
    print(f"Precision@{config.K}: {sum(precisions) / len(precisions):.4f}")
    print(f"Recall@{config.K}:    {sum(recalls) / len(recalls):.4f}")
    print(f"NDCG@{config.K}:      {sum(ndcgs) / len(ndcgs):.4f}")


if __name__ == "__main__":
    main()