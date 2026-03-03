"""
Run TF-IDF baseline recommendations end-to-end.

Expected inputs (already created by your prepare step):
- data/processed/jester_edges_clean.csv  columns: user_id, joke_id, rating
- data/processed/jester_jokes_clean.csv  columns: joke_id, joke_text
"""

import pandas as pd

from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.tfidf_model import TfidfRecommender


def main() -> None:
    # Load processed data
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    edges = pd.read_csv(edges_path)
    jokes = pd.read_csv(jokes_path)

    # Fit TF-IDF model on joke text
    model = TfidfRecommender.fit(jokes_df=jokes, max_features=5000, use_bigrams=True)

    # Choose a user (change this to test other users)
    user_id = 0
    k = 5

    # Get recommendations
    recs = model.recommend_for_user_no_duplicates(
        edges_df=edges,
        user_id=user_id,
        k=k,
        like_threshold=5.0,
        candidate_pool=50,
        sim_threshold=0.70,
    )

    print(f"User {user_id} — Top-{k} TF-IDF recommendations")
    print("-" * 55)

    # Make a quick lookup for joke text
    joke_text_map = dict(zip(jokes["joke_id"].astype(int), jokes["joke_text"].astype(str)))

    # Print recommendations with a short preview
    for rank, (joke_id, score) in enumerate(recs, start=1):
        text = joke_text_map.get(joke_id, "")
        preview = text.replace("\n", " ")[:220]
        print(f"{rank}. Joke {joke_id} | score={score:.4f}")
        print(f"   {preview}...")
        print()


if __name__ == "__main__":
    main()