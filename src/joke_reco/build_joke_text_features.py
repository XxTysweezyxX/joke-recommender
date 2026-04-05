from __future__ import annotations

"""
Build standalone joke text features for the text-augmented LightGCN model.

Purpose:
- Load / prepare joke text data
- Build TF-IDF text vectors from joke text
- Align those vectors to the LightGCN item index order
- Return a torch tensor that can be passed into the GCN model

Important:
These text vectors are used as item-side text features for the graph model.
They are not used here for direct recommendation by cosine similarity.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------
# Helper: prepare cleaned joke text dataframe
# ---------------------------------------------------------
def prepare_jokes_df(jokes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for text feature building.

    Expected columns:
    - joke_id
    - joke_text
    """
    jokes_df = jokes_df[["joke_id", "joke_text"]].copy()
    jokes_df["joke_id"] = jokes_df["joke_id"].astype(int)
    jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

    # Sort by joke_id so the row order is stable and reproducible
    jokes_df = jokes_df.sort_values("joke_id").reset_index(drop=True)
    return jokes_df


# ---------------------------------------------------------
# Helper: fit TF-IDF vectorizer on joke text
# ---------------------------------------------------------
def build_tfidf_text_matrix(
    jokes_df: pd.DataFrame,
    max_features: int = 5000,
    use_bigrams: bool = True,
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """
    Build TF-IDF text vectors for all jokes.

    Returns:
    - fitted vectorizer
    - dense feature matrix of shape (num_jokes, text_feature_dim)
    - joke_ids aligned with the rows of the matrix
    """
    jokes_df = prepare_jokes_df(jokes_df)

    # Use either unigrams only or unigrams + bigrams
    ngram_range = (1, 2) if use_bigrams else (1, 1)

    # Build TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
    )

    # Fit vectorizer and build sparse matrix
    tfidf_matrix = vectorizer.fit_transform(jokes_df["joke_text"])

    # Convert to dense NumPy array so it can be passed into torch later
    text_features = tfidf_matrix.toarray().astype(np.float32)

    # Keep joke IDs aligned with the matrix rows
    joke_ids = jokes_df["joke_id"].to_numpy(dtype=int)

    return vectorizer, text_features, joke_ids


# ---------------------------------------------------------
# Helper: align text features to LightGCN item index order
# ---------------------------------------------------------
def align_text_features_to_item_map(
    text_features: np.ndarray,
    joke_ids: np.ndarray,
    item_map: Dict[int, int],
) -> np.ndarray:
    """
    Reorder the joke text feature matrix so it matches the LightGCN item index order.

    Input:
    - text_features rows are aligned to joke_ids
    - item_map maps raw joke_id -> internal item index

    Output:
    - aligned matrix of shape (num_items, text_feature_dim)
      where row i matches the LightGCN item index i
    """
    num_items = len(item_map)
    text_feature_dim = text_features.shape[1]

    aligned = np.zeros((num_items, text_feature_dim), dtype=np.float32)

    # Quick lookup from raw joke_id -> row index in text_features
    joke_id_to_row = {int(jid): idx for idx, jid in enumerate(joke_ids)}

    for raw_joke_id, item_idx in item_map.items():
        row_idx = joke_id_to_row.get(int(raw_joke_id))

        # If the joke is missing from the joke text file,
        # leave its row as zeros
        if row_idx is None:
            continue

        aligned[item_idx] = text_features[row_idx]

    return aligned


# ---------------------------------------------------------
# Main helper: build item-side text feature tensor
# ---------------------------------------------------------
def build_item_text_features(
    jokes_df: pd.DataFrame,
    item_map: Dict[int, int],
    max_features: int = 5000,
    use_bigrams: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, TfidfVectorizer]:
    """
    Build standalone item-side text features for the text-augmented LightGCN model.

    Steps:
    1) fit TF-IDF vectors from joke text
    2) align vectors to the LightGCN item index order
    3) convert to torch tensor

    Returns:
    - item_text_features tensor of shape (num_items, text_feature_dim)
    - fitted TF-IDF vectorizer
    """
    vectorizer, text_features, joke_ids = build_tfidf_text_matrix(
        jokes_df=jokes_df,
        max_features=max_features,
        use_bigrams=use_bigrams,
    )

    aligned_features = align_text_features_to_item_map(
        text_features=text_features,
        joke_ids=joke_ids,
        item_map=item_map,
    )

    item_text_features = torch.tensor(
        aligned_features,
        dtype=torch.float32,
        device=device,
    )

    return item_text_features, vectorizer


# ---------------------------------------------------------
# Optional runner: quick standalone check
# ---------------------------------------------------------
def main() -> None:
    """
    Optional local test runner.

    This can be adapted later if you want to quickly inspect:
    - feature matrix shape
    - vocabulary size
    - whether the output looks correct
    """
    print("This module is intended to be imported from LightGCN training code.")


# ---------------------------------------------------------
# Standard Python entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()