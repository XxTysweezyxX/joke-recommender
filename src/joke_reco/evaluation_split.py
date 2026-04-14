from __future__ import annotations

"""
Builds a per-user train/test split for recommendation evaluation.
Holds out a small number of liked jokes per user for testing.
"""

from typing import Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. Train/test split by user
# Holds out up to a fixed number of liked jokes per user.
# ---------------------------------------------------------
def train_test_split_by_user(
    edges: pd.DataFrame,
    like_threshold: float = 7.0,
    test_size: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Create a reproducible random generator
    rng = np.random.default_rng(seed)

    # Store each user's training rows
    train_parts = []

    # Store each user's test rows
    test_parts = []

    # Process one user at a time
    for user_id, user_rows in edges.groupby("user_id"):
        # Copy this user's rows to avoid modifying the original dataframe
        user_rows = user_rows.copy()

        # Keep only this user's liked jokes
        liked = user_rows[user_rows["rating"] >= like_threshold]

        # Skip holdout if the user has too few liked jokes
        if len(liked) <= 1:
            train_parts.append(user_rows)
            continue

        # Choose how many liked jokes to hold out
        n_holdout = min(test_size, len(liked))

        # Randomly sample liked joke rows for the test set
        holdout_idx = rng.choice(
            liked.index.to_numpy(),
            size=n_holdout,
            replace=False,
        )

        # Build this user's test split
        test_part = user_rows.loc[holdout_idx]

        # Put the remaining rows into the training split
        train_part = user_rows.drop(index=holdout_idx)

        # Store this user's training rows
        train_parts.append(train_part)

        # Store this user's test rows
        test_parts.append(test_part)

    # Combine all user training rows into one dataframe
    train_edges = pd.concat(train_parts, ignore_index=True)

    # Combine all user test rows into one dataframe
    test_edges = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else edges.iloc[0:0].copy()
    )

    return train_edges, test_edges