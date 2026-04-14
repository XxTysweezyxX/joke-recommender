from __future__ import annotations

"""
Builds joke text features for the text-augmented LightGCN model.
Cleans joke text, creates TF-IDF vectors, aligns them to item indices, and returns a torch tensor.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------
# 1. Joke text preparation
# Keeps only the needed columns and standardises the order.
# ---------------------------------------------------------
def prepare_jokes_df(jokes_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the joke ID and joke text columns
    jokes_df = jokes_df[["joke_id", "joke_text"]].copy()

    # Convert joke IDs to integers
    jokes_df["joke_id"] = jokes_df["joke_id"].astype(int)

    # Convert joke text to strings
    jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

    # Sort rows by joke ID for stable ordering
    jokes_df = jokes_df.sort_values("joke_id").reset_index(drop=True)

    return jokes_df


# ---------------------------------------------------------
# 2. TF-IDF feature building
# Converts each joke into a numeric text feature vector.
# ---------------------------------------------------------
def build_tfidf_text_matrix(
    jokes_df: pd.DataFrame,
    max_features: int = 5000,
    use_bigrams: bool = True,
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    # Clean and standardise the joke dataframe
    jokes_df = prepare_jokes_df(jokes_df)

    # Use unigrams and bigrams if enabled
    ngram_range = (1, 2) if use_bigrams else (1, 1)

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
    )

    # Fit the vectorizer and transform the joke text
    tfidf_matrix = vectorizer.fit_transform(jokes_df["joke_text"])

    # Convert the sparse matrix into a dense NumPy array
    text_features = tfidf_matrix.toarray().astype(np.float32)

    # Store the joke IDs in the same row order
    joke_ids = jokes_df["joke_id"].to_numpy(dtype=int)

    return vectorizer, text_features, joke_ids


# ---------------------------------------------------------
# 3. Feature alignment
# Reorders TF-IDF rows to match the model's item index order.
# ---------------------------------------------------------
def align_text_features_to_item_map(
    text_features: np.ndarray,
    joke_ids: np.ndarray,
    item_map: Dict[int, int],
) -> np.ndarray:
    # Get the number of items used by the model
    num_items = len(item_map)

    # Get the width of each text feature vector
    text_feature_dim = text_features.shape[1]

    # Create an empty aligned feature matrix
    aligned = np.zeros((num_items, text_feature_dim), dtype=np.float32)

    # Map each raw joke ID to its TF-IDF row index
    joke_id_to_row = {int(jid): idx for idx, jid in enumerate(joke_ids)}

    # Fill each model item row with the correct text features
    for raw_joke_id, item_idx in item_map.items():
        # Look up the TF-IDF row for this joke
        row_idx = joke_id_to_row.get(int(raw_joke_id))

        # Skip missing jokes and leave the row as zeros
        if row_idx is None:
            continue

        # Copy the TF-IDF row into the aligned item row
        aligned[item_idx] = text_features[row_idx]

    return aligned


# ---------------------------------------------------------
# 4. Final tensor builder
# Runs the full text-feature pipeline and returns a torch tensor.
# ---------------------------------------------------------
def build_item_text_features(
    jokes_df: pd.DataFrame,
    item_map: Dict[int, int],
    max_features: int = 5000,
    use_bigrams: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, TfidfVectorizer]:
    # Build the raw TF-IDF text features
    vectorizer, text_features, joke_ids = build_tfidf_text_matrix(
        jokes_df=jokes_df,
        max_features=max_features,
        use_bigrams=use_bigrams,
    )

    # Reorder the features to match model item indices
    aligned_features = align_text_features_to_item_map(
        text_features=text_features,
        joke_ids=joke_ids,
        item_map=item_map,
    )

    # Convert the aligned features into a torch tensor
    item_text_features = torch.tensor(
        aligned_features,
        dtype=torch.float32,
        device=device,
    )

    return item_text_features, vectorizer