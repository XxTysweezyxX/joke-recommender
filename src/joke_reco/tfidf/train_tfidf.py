from __future__ import annotations

"""
Build / fit the TF-IDF recommender for the joke dataset.

Purpose:
- Load the cleaned jokes dataset from disk
- Fit the TF-IDF recommender on joke text
- Return the fitted model for later recommendation and evaluation

Note:
Initial scaffolding and some implementation support for this module
were developed with AI assistance. The code was then reviewed,
adapted, and validated for use in this project.
"""

import pandas as pd

from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.tfidf_model import TfidfRecommender


# ---------------------------------------------------------
# Helper: build the fitted TF-IDF recommender
# ---------------------------------------------------------
def build_tfidf_recommender(
    max_features: int = 5000,
    use_bigrams: bool = True,
) -> TfidfRecommender:
    """
    Load the cleaned jokes file from disk and fit a TF-IDF recommender.

    max_features controls the vocabulary size.
    use_bigrams decides whether to include two-word phrases as features.
    """
    # Path to the cleaned jokes dataset
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Load the joke text data
    jokes_df = pd.read_csv(jokes_path)

    # Fit the TF-IDF recommender on the joke text
    model = TfidfRecommender.fit(
        jokes_df=jokes_df,
        max_features=max_features,
        use_bigrams=use_bigrams,
    )

    # Return the fitted model
    return model