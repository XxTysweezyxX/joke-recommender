from __future__ import annotations

"""
Build / fit the TF-IDF baseline model.

Run from /src with:
    python -m scripts.train.run_train_tfidf
"""

from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.train_tfidf import build_tfidf_recommender


def main() -> None:
    """
    Load the joke text data, fit the TF-IDF model, and print basic info.
    """
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    print("[run_train_tfidf] Loading jokes...")

    model = build_tfidf_recommender(
        max_features=5000,
        use_bigrams=True,
    )

    print(f"[run_train_tfidf] Loaded {len(model.joke_ids)} jokes from {jokes_path}")
    print("[run_train_tfidf] Fitting TF-IDF vectorizer...")
    print(f"[run_train_tfidf] TF-IDF matrix shape: {model.tfidf_matrix.shape}")
    print("[run_train_tfidf] TF-IDF model ready.")


if __name__ == "__main__":
    main()