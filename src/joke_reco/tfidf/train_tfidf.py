from __future__ import annotations

"""
Builds and fits the TF-IDF recommender for the joke dataset.
Loads the cleaned joke text and returns a fitted TF-IDF model for later use.
"""

import pandas as pd

from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.tfidf_model import TfidfRecommender


# ---------------------------------------------------------
# 1. TF-IDF recommender builder
# Loads the cleaned joke text and fits the TF-IDF model.
# ---------------------------------------------------------
def build_tfidf_recommender(
    max_features: int = 5000,
    use_bigrams: bool = True,
) -> TfidfRecommender:
    # Build the path to the cleaned jokes file
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Load the cleaned joke text data
    jokes_df = pd.read_csv(jokes_path)

    # Fit the TF-IDF recommender on the joke text
    model = TfidfRecommender.fit(
        jokes_df=jokes_df,
        max_features=max_features,
        use_bigrams=use_bigrams,
    )

    # Return the fitted recommender
    return model