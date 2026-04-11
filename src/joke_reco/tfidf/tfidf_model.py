from __future__ import annotations

"""
TF-IDF content-based recommender for jokes.

Purpose:
- Fit a TF-IDF representation from joke text
- Recommend jokes for one user based on text similarity
- Optionally reduce near-duplicate recommendations with a simple diversity filter
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class TfidfRecommender:
    """
    A content-based recommender built using TF-IDF on joke text.

    After fitting:
    - vectorizer stores the learned vocabulary
    - tfidf_matrix stores the TF-IDF vector for each joke
    - joke_ids keeps track of which joke belongs to each row
    - id_to_idx lets me quickly map joke_id -> row index
    """

    # ---------------------------------------------------------
    # Stored TF-IDF model components
    # ---------------------------------------------------------

    # Fitted sklearn TF-IDF vectorizer
    vectorizer: TfidfVectorizer

    # Sparse matrix of joke text vectors
    tfidf_matrix: any

    # Array of joke IDs aligned with tfidf_matrix rows
    joke_ids: np.ndarray

    # Lookup from joke_id to row index in tfidf_matrix
    id_to_idx: Dict[int, int]

    # ---------------------------------------------------------
    # 1. Class method: Fit TF-IDF model from joke text
    # ---------------------------------------------------------
    @classmethod
    def fit(
            cls,
            jokes_df: pd.DataFrame,
            max_features: int = 5000,
            use_bigrams: bool = True,
    ) -> "TfidfRecommender":
        """
        Fit a TF-IDF model on the joke text.

        Expected columns:
        - joke_id
        - joke_text

        max_features limits the vocabulary size.
        use_bigrams controls whether two-word phrases are included.
        """

        # Keep only the columns needed for the TF-IDF model.
        jokes_df = jokes_df[["joke_id", "joke_text"]].copy()

        # Convert all joke text values to strings so the vectorizer can process them safely.
        jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

        # Sort by joke_id so the row order stays stable and reproducible.
        jokes_df = jokes_df.sort_values("joke_id").reset_index(drop=True)

        # Use either single words only, or single words and two-word phrases.
        ngram_range = (1, 2) if use_bigrams else (1, 1)

        # Build the TF-IDF vectorizer.
        vectorizer = TfidfVectorizer(
            stop_words="english",  # Remove very common English words.
            max_features=max_features,  # Limit the vocabulary size.
            ngram_range=ngram_range,  # Include unigrams only, or unigrams + bigrams.
        )

        # Convert all joke texts into TF-IDF vectors.
        tfidf_matrix = vectorizer.fit_transform(jokes_df["joke_text"])

        # Store joke IDs in the same order as the TF-IDF matrix rows.
        joke_ids = jokes_df["joke_id"].to_numpy(dtype=int)

        # Build a lookup from joke_id to its row index in the TF-IDF matrix.
        id_to_idx = {int(jid): int(i) for i, jid in enumerate(joke_ids)}

        # Return the fitted recommender with all TF-IDF components stored.
        return cls(
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            joke_ids=joke_ids,
            id_to_idx=id_to_idx,
        )

    # ---------------------------------------------------------
    # 2. Recommend jokes for one user
    # ---------------------------------------------------------
    def recommend_for_user(
        self,
        edges_df: pd.DataFrame,
        user_id: int,
        k: int = 5,
        like_threshold: float = 7.0,
        fallback_top_n: int = 3,
    ) -> List[Tuple[int, float]]:
        """
        Recommend top-k jokes for one user.

        Strategy:
        1) Find jokes this user liked
        2) Compare those liked jokes against all jokes using TF-IDF similarity
        3) Average the similarity scores
        4) Remove jokes the user has already seen
        5) Return the highest-scoring unseen jokes
        """

        # Get all interaction rows for this user
        user_rows = edges_df.loc[edges_df["user_id"] == user_id]

        # If the user is not found in the ratings data, return nothing
        if user_rows.empty:
            return []

        # Track jokes the user has already rated so they are not recommended again.
        seen_jokes = set(user_rows["joke_id"].astype(int).tolist())

        # Keep only jokes rated above the chosen threshold as liked jokes. (7)
        liked_jokes = (
            user_rows.loc[user_rows["rating"] >= like_threshold, "joke_id"]
            .astype(int)
            .tolist()
        )

        # If the user has no jokes above the threshold,
        # fall back to their top-rated jokes instead
        if len(liked_jokes) == 0:
            liked_jokes = (
                user_rows.sort_values("rating", ascending=False)["joke_id"]
                .astype(int)
                .head(fallback_top_n)
                .tolist()
            )

        # Convert liked joke IDs into TF-IDF row indices
        liked_indices = [
            self.id_to_idx[jid]
            for jid in liked_jokes
            if jid in self.id_to_idx
        ]

        # If none of the liked jokes exist in the TF-IDF model, return nothing
        if len(liked_indices) == 0:
            return []

        # Compute cosine similarity between liked jokes and all jokes
        # Shape = (number_of_liked_jokes, total_number_of_jokes)
        sims = linear_kernel(
            self.tfidf_matrix[liked_indices],
            self.tfidf_matrix,
        )

        # Average similarity scores across the liked jokes
        scores = np.asarray(sims.mean(axis=0)).ravel()

        # Build list of candidate jokes, excluding already seen jokes
        candidates: List[Tuple[int, float]] = []

        for idx, jid in enumerate(self.joke_ids):
            jid_int = int(jid)

            # Skip jokes the user has already rated
            if jid_int in seen_jokes:
                continue

            # Store unseen joke with its similarity score
            candidates.append((jid_int, float(scores[idx])))

        # Sort candidates from highest score to lowest score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return only the top-k recommendations
        return candidates[:k]

    # ---------------------------------------------------------
    # Recommend jokes with simple diversity filtering
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
        """
        Recommend top-k jokes with a simple diversity rule.

        Idea:
        - First get a larger pool of strong candidates
        - Then greedily keep jokes that are not too similar to
          already selected jokes
        """

        # First build a larger ranked candidate list
        candidates = self.recommend_for_user(
            edges_df=edges_df,
            user_id=user_id,
            k=candidate_pool,
            like_threshold=like_threshold,
            fallback_top_n=fallback_top_n,
        )

        # If there are no candidates, return nothing
        if not candidates:
            return []

        # Final selected recommendations
        selected: List[Tuple[int, float]] = []

        # Track TF-IDF row indices of selected jokes
        selected_indices: List[int] = []

        for joke_id, score in candidates:
            # Stop once k jokes have been selected
            if len(selected) >= k:
                break

            # Skip if joke is missing from TF-IDF lookup
            if joke_id not in self.id_to_idx:
                continue

            cand_idx = self.id_to_idx[joke_id]

            # Always accept the first joke
            if not selected_indices:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)
                continue

            # Compare this candidate against already selected jokes
            sims = linear_kernel(
                self.tfidf_matrix[cand_idx],
                self.tfidf_matrix[selected_indices],
            )
            max_sim = float(sims.max())

            # Keep the candidate only if it is not too similar
            if max_sim < sim_threshold:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)

        return selected