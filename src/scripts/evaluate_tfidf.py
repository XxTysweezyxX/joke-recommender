import pandas as pd

from joke_reco.paths import PROCESSED_DIR
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.metrics import precision_at_k, recall_at_k, ndcg_at_k
from joke_reco.tfidf.tfidf_model import TfidfRecommender


def main() -> None:
    # Load processed data created by prepare_jester.py
    edges = pd.read_csv(PROCESSED_DIR / "jester_edges_clean.csv")
    jokes = pd.read_csv(PROCESSED_DIR / "jester_jokes_clean.csv")

    like_threshold = 5.0
    k = 10
    holdout_per_user = 2

    # Split interactions
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=like_threshold,
        test_size=holdout_per_user,
        seed=42,
    )

    # Fit TF-IDF model on all joke text
    model = TfidfRecommender.fit(jokes_df=jokes, max_features=5000, use_bigrams=True)

    # Evaluate on a subset of users for speed (increase later if you want)
    user_ids = test_edges["user_id"].unique().tolist()
    user_ids = user_ids[:500]

    p_sum = r_sum = n_sum = 0.0
    n_users = 0

    for uid in user_ids:
        relevant = set(
            test_edges.loc[test_edges["user_id"] == uid, "joke_id"].astype(int).tolist()
        )
        if not relevant:
            continue

        recs = model.recommend_for_user(
            edges_df=train_edges,
            user_id=int(uid),
            k=k,
            like_threshold=like_threshold,
        )
        rec_ids = [jid for jid, _ in recs]

        p_sum += precision_at_k(rec_ids, relevant, k)
        r_sum += recall_at_k(rec_ids, relevant, k)
        n_sum += ndcg_at_k(rec_ids, relevant, k)
        n_users += 1

    if n_users == 0:
        print("No evaluable users (try lowering holdout_per_user or like_threshold).")
        return

    print(f"TF-IDF Evaluation (users={n_users}, K={k}, threshold={like_threshold}, holdout={holdout_per_user})")
    print(f"Precision@{k}: {p_sum / n_users:.4f}")
    print(f"Recall@{k}:    {r_sum / n_users:.4f}")
    print(f"NDCG@{k}:      {n_sum / n_users:.4f}")


if __name__ == "__main__":
    main()