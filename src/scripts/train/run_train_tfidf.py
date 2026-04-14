from __future__ import annotations

"""
Runs training for the TF-IDF baseline model.
Loads the cleaned joke text, builds the TF-IDF recommender, and prints basic model details.
"""

from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.train_tfidf import build_tfidf_recommender


# ---------------------------------------------------------
# 1. Main execution
# Loads joke text, builds the TF-IDF model, and prints details.
# ---------------------------------------------------------
def main() -> None:
    # Build the path to the cleaned joke text file
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Print a progress message
    print("[run_train_tfidf] Loading jokes...")

    # Build the TF-IDF recommender
    model = build_tfidf_recommender(
        max_features=5000,
        use_bigrams=True,
    )

    # Print the number of loaded jokes
    print(f"[run_train_tfidf] Loaded {len(model.joke_ids)} jokes from {jokes_path}")

    # Print a progress message
    print("[run_train_tfidf] Fitting TF-IDF vectorizer...")

    # Print the shape of the TF-IDF matrix
    print(f"[run_train_tfidf] TF-IDF matrix shape: {model.tfidf_matrix.shape}")

    # Print a completion message
    print("[run_train_tfidf] TF-IDF model ready.")


if __name__ == "__main__":
    main()