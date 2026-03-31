from __future__ import annotations

"""
TF-IDF content-based recommender for jokes.

This version keeps the structure clean and class-based:
- fit() builds the model from joke text
- recommend_for_user() generates recommendations for one user
- recommend_for_user_no_duplicates() adds a simple diversity filter
"""

"""Note:
This TF-IDF baseline is built using scikit-learn's TfidfVectorizer
as the core text feature extraction method. The surrounding class
structure, user-level recommendation logic, and diversity filtering
were developed and adapted for this project. :contentReference[oaicite:1]{index=1}"""

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

    # The fitted sklearn TF-IDF vectorizer
    vectorizer: TfidfVectorizer

    # Sparse matrix of joke text vectors
    tfidf_matrix: any

    # Array of joke IDs aligned with tfidf_matrix rows
    joke_ids: np.ndarray

    # Lookup from joke_id to row index in tfidf_matrix
    id_to_idx: Dict[int, int]

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

        max_features limits the vocabulary size so the model stays manageable.
        use_bigrams decides whether to use single words only, or
        single words + two-word phrases.
        """
        # Keep only the columns needed for TF-IDF
        jokes_df = jokes_df[["joke_id", "joke_text"]].copy()

        # Make sure joke text is treated as strings
        jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

        # Sort by joke_id so row order stays stable and reproducible
        jokes_df = jokes_df.sort_values("joke_id").reset_index(drop=True)

        # Use unigrams only or unigrams + bigrams
        ngram_range = (1, 2) if use_bigrams else (1, 1)

        # Build the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
        )

        # Convert all joke text into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(jokes_df["joke_text"])

        # Store joke IDs in the same order as the TF-IDF rows
        joke_ids = jokes_df["joke_id"].to_numpy(dtype=int)

        # Build a quick lookup from joke_id to matrix row
        id_to_idx = {int(jid): int(i) for i, jid in enumerate(joke_ids)}

        # Return a fitted recommender object
        return cls(
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            joke_ids=joke_ids,
            id_to_idx=id_to_idx,
        )

    def recommend_for_user(
        self,
        edges_df: pd.DataFrame,
        user_id: int,
        k: int = 5,
        like_threshold: float = 5.0,
        fallback_top_n: int = 3,
    ) -> List[Tuple[int, float]]:
        """
        Recommend top-k jokes for one user.

        Strategy:
        1) Find jokes this user liked
        2) Measure similarity from those liked jokes to all jokes
        3) Average the similarity scores
        4) Remove jokes the user has already seen
        5) Return the highest-scoring unseen jokes
        """
        # Get all rows belonging to this user
        user_rows = edges_df.loc[edges_df["user_id"] == user_id]

        # If the user does not exist in the ratings data, return nothing
        if user_rows.empty:
            return []

        # Jokes the user has already rated
        seen_jokes = set(user_rows["joke_id"].astype(int).tolist())

        # Jokes counted as "liked" based on the threshold
        liked_jokes = (
            user_rows.loc[user_rows["rating"] >= like_threshold, "joke_id"]
            .astype(int)
            .tolist()
        )

        # If the user has no strong likes, fall back to their top-rated jokes
        if len(liked_jokes) == 0:
            liked_jokes = (
                user_rows.sort_values("rating", ascending=False)["joke_id"]
                .astype(int)
                .head(fallback_top_n)
                .tolist()
            )

        # Convert liked joke IDs into row indices in the TF-IDF matrix
        liked_indices = [self.id_to_idx[jid] for jid in liked_jokes if jid in self.id_to_idx]

        # If none of the liked jokes appear in the TF-IDF model, return nothing
        if len(liked_indices) == 0:
            return []

        # Compute cosine similarity between liked jokes and all jokes
        # Output shape = (number_of_liked_jokes, total_number_of_jokes)
        sims = linear_kernel(self.tfidf_matrix[liked_indices], self.tfidf_matrix)

        # Average the similarity scores across the liked jokes
        scores = np.asarray(sims.mean(axis=0)).ravel()

        # Build a list of candidate recommendations, excluding already seen jokes
        candidates: List[Tuple[int, float]] = []
        for idx, jid in enumerate(self.joke_ids):
            jid_int = int(jid)

            # Skip jokes the user has already rated
            if jid_int in seen_jokes:
                continue

            # Add unseen joke with its similarity score
            candidates.append((jid_int, float(scores[idx])))

        # Sort by score from highest to lowest
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return the top-k recommendations
        return candidates[:k]

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
        - First get a larger pool of relevant candidates
        - Then greedily keep jokes that are not too similar to
          already selected jokes
        """
        # First get a larger candidate list ranked by relevance
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

        # Final selected jokes
        selected: List[Tuple[int, float]] = []

        # Store indices of already selected jokes for similarity checking
        selected_indices: List[int] = []

        for joke_id, score in candidates:
            # Stop once k jokes have been selected
            if len(selected) >= k:
                break

            # Safety check in case joke_id is missing from the matrix index
            if joke_id not in self.id_to_idx:
                continue

            cand_idx = self.id_to_idx[joke_id]

            # Always accept the first recommendation
            if not selected_indices:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)
                continue

            # Compare this candidate to jokes already selected
            sims = linear_kernel(self.tfidf_matrix[cand_idx], self.tfidf_matrix[selected_indices])
            max_sim = float(sims.max())

            # Only keep it if it is not too similar to selected jokes
            if max_sim < sim_threshold:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)

        return selected