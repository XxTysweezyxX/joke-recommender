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
    Main runner for the TF-IDF model.

    It loads the cleaned jokes file, builds the TF-IDF recommender,
    and prints a few useful details so I can confirm it worked.
    """
    # Path to the cleaned jokes dataset
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Progress message so I know the script has started properly
    print("[run_train_tfidf] Loading jokes...")

    # Build the TF-IDF recommender using the helper function
    model = build_tfidf_recommender(
        max_features=5000,
        use_bigrams=True,
    )

    # Print basic information about what was loaded/built
    print(f"[run_train_tfidf] Loaded {len(model.joke_ids)} jokes from {jokes_path}")
    print("[run_train_tfidf] Fitting TF-IDF vectorizer...")
    print(f"[run_train_tfidf] TF-IDF matrix shape: {model.tfidf_matrix.shape}")
    print("[run_train_tfidf] TF-IDF model ready.")


# Standard Python entry point:
# only run main() if this file is executed directly
if __name__ == "__main__":
    main()