from __future__ import annotations

"""
Defines the TF-IDF content-based recommender for jokes.
Fits joke text into TF-IDF vectors and generates recommendations using text similarity.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# AI-assisted file:
# The TF-IDF vectorizer itself is provided by scikit-learn.
# ChatGPT was used to help structure and implement this TF-IDF recommender file.
# Prompt summary: "Help me write a Python TF-IDF recommender for jokes, including
# fitting joke text, recommending for a user, and a simple diversity filter."


# ---------------------------------------------------------
# 1. TF-IDF recommender container
# Stores the fitted TF-IDF model and lookup data.
# ---------------------------------------------------------
@dataclass
class TfidfRecommender:
    # Store the fitted TF-IDF vectorizer
    vectorizer: TfidfVectorizer

    # Store the sparse TF-IDF matrix
    tfidf_matrix: any

    # Store joke IDs aligned with matrix rows
    joke_ids: np.ndarray

    # Store joke_id to matrix row lookup
    id_to_idx: Dict[int, int]

    # ---------------------------------------------------------
    # 2. Model fitting
    # Fits the TF-IDF model on joke text.
    # ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        jokes_df: pd.DataFrame,
        max_features: int = 5000,
        use_bigrams: bool = True,
    ) -> "TfidfRecommender":
        # Keep only the columns needed for TF-IDF fitting
        jokes_df = jokes_df[["joke_id", "joke_text"]].copy()

        # Convert joke text to strings
        jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

        # Sort jokes by joke_id for stable ordering
        jokes_df = jokes_df.sort_values("joke_id").reset_index(drop=True)

        # Use unigrams only or unigrams plus bigrams
        ngram_range = (1, 2) if use_bigrams else (1, 1)

        # Create the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
        )

        # Fit the vectorizer and transform joke text
        tfidf_matrix = vectorizer.fit_transform(jokes_df["joke_text"])

        # Store joke IDs in the same order as the matrix rows
        joke_ids = jokes_df["joke_id"].to_numpy(dtype=int)

        # Build a lookup from joke_id to matrix row index
        id_to_idx = {int(jid): int(i) for i, jid in enumerate(joke_ids)}

        # Return the fitted recommender object
        return cls(
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            joke_ids=joke_ids,
            id_to_idx=id_to_idx,
        )

    # ---------------------------------------------------------
    # 3. Basic recommendation
    # Recommends top-k jokes for one user using TF-IDF similarity.
    # ---------------------------------------------------------
    def recommend_for_user(
        self,
        edges_df: pd.DataFrame,
        user_id: int,
        k: int = 5,
        like_threshold: float = 7.0,
        fallback_top_n: int = 3,
    ) -> List[Tuple[int, float]]:
        # Get all rating rows for this user
        user_rows = edges_df.loc[edges_df["user_id"] == user_id]

        # Return nothing if the user is missing
        if user_rows.empty:
            return []

        # Track jokes already seen by this user
        seen_jokes = set(user_rows["joke_id"].astype(int).tolist())

        # Keep only jokes rated above the like threshold
        liked_jokes = (
            user_rows.loc[user_rows["rating"] >= like_threshold, "joke_id"]
            .astype(int)
            .tolist()
        )

        # Fall back to the user's top-rated jokes if needed
        if len(liked_jokes) == 0:
            liked_jokes = (
                user_rows.sort_values("rating", ascending=False)["joke_id"]
                .astype(int)
                .head(fallback_top_n)
                .tolist()
            )

        # Convert liked joke IDs into matrix row indices
        liked_indices = [
            self.id_to_idx[jid]
            for jid in liked_jokes
            if jid in self.id_to_idx
        ]

        # Return nothing if no liked jokes exist in the model
        if len(liked_indices) == 0:
            return []

        # Compute cosine similarity between liked jokes and all jokes
        sims = linear_kernel(
            self.tfidf_matrix[liked_indices],
            self.tfidf_matrix,
        )

        # Average the similarity scores across liked jokes
        scores = np.asarray(sims.mean(axis=0)).ravel()

        # Store unseen candidate jokes and their scores
        candidates: List[Tuple[int, float]] = []

        # Check each joke in the TF-IDF model
        for idx, jid in enumerate(self.joke_ids):
            # Convert the joke ID to int
            jid_int = int(jid)

            # Skip jokes the user has already seen
            if jid_int in seen_jokes:
                continue

            # Store the unseen joke and its score
            candidates.append((jid_int, float(scores[idx])))

        # Sort candidates by score from highest to lowest
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return only the top-k recommendations
        return candidates[:k]

    # ---------------------------------------------------------
    # 4. Diversity-filtered recommendation
    # Recommends top-k jokes while reducing near-duplicates.
    # ---------------------------------------------------------
    def recommend_for_user_no_duplicates(
        self,
        edges_df: pd.DataFrame,
        user_id: int,
        k: int = 5,
        like_threshold: float = 5.0,
        fallback_top_n: int = 3,
        candidate_pool: int = 50,
        sim_threshold: float = 0.70,
    ) -> List[Tuple[int, float]]:
        # Build a larger candidate pool first
        candidates = self.recommend_for_user(
            edges_df=edges_df,
            user_id=user_id,
            k=candidate_pool,
            like_threshold=like_threshold,
            fallback_top_n=fallback_top_n,
        )

        # Return nothing if no candidates exist
        if not candidates:
            return []

        # Store the final selected recommendations
        selected: List[Tuple[int, float]] = []

        # Store the matrix row indices of selected jokes
        selected_indices: List[int] = []

        # Process each candidate in ranked order
        for joke_id, score in candidates:
            # Stop once k jokes have been selected
            if len(selected) >= k:
                break

            # Skip if the joke is missing from the lookup
            if joke_id not in self.id_to_idx:
                continue

            # Look up the matrix row index for this candidate
            cand_idx = self.id_to_idx[joke_id]

            # Always accept the first selected joke
            if not selected_indices:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)
                continue

            # Compare this candidate against already selected jokes
            sims = linear_kernel(
                self.tfidf_matrix[cand_idx],
                self.tfidf_matrix[selected_indices],
            )

            # Get the highest similarity to the selected set
            max_sim = float(sims.max())

            # Keep the candidate if it is not too similar
            if max_sim < sim_threshold:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)

        return selected