"""
Per-user train/test split for recommendation evaluation.

We treat jokes with rating >= like_threshold as "relevant".
We hold out a small number of those relevant jokes per user for testing.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def train_test_split_by_user(
    edges: pd.DataFrame,
    like_threshold: float = 7.0,
    test_size: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user:
    - Find "liked" rows (rating >= like_threshold)
    - Randomly hold out up to test_size liked rows into test
    - Everything else goes to train

    Returns:
        (train_edges, test_edges)
    """
    rng = np.random.default_rng(seed)

    train_parts = []
    test_parts = []

    for user_id, user_rows in edges.groupby("user_id"):
        user_rows = user_rows.copy()

        liked = user_rows[user_rows["rating"] >= like_threshold]
        if len(liked) <= 1:
            # Not enough liked items to hold out
            train_parts.append(user_rows)
            continue

        n_holdout = min(test_size, len(liked))
        holdout_idx = rng.choice(liked.index.to_numpy(), size=n_holdout, replace=False)

        test_part = user_rows.loc[holdout_idx]
        train_part = user_rows.drop(index=holdout_idx)

        train_parts.append(train_part)
        test_parts.append(test_part)

    train_edges = pd.concat(train_parts, ignore_index=True)
    test_edges = pd.concat(test_parts, ignore_index=True) if test_parts else edges.iloc[0:0].copy()

    return train_edges, test_edges