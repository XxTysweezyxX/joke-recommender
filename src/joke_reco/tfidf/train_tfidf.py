from __future__ import annotations

"""
Build / fit the TF-IDF recommender for the joke dataset.
"""

import pandas as pd

from joke_reco.paths import PROCESSED_DIR
from joke_reco.tfidf.tfidf_model import TfidfRecommender


def build_tfidf_recommender(
    max_features: int = 5000,
    use_bigrams: bool = True,
) -> TfidfRecommender:
    """
    Load jokes from disk and fit a TF-IDF recommender.
    """
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"
    jokes_df = pd.read_csv(jokes_path)

    model = TfidfRecommender.fit(
        jokes_df=jokes_df,
        max_features=max_features,
        use_bigrams=use_bigrams,
    )
    return model