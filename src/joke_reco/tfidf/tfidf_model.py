"""
TF-IDF content-based recommender for jokes.

This module is designed to be "Java-like" in structure:
- A class with a clear fit() method (like a model builder)
- A recommend_for_user() method (like a service method)
- No notebook dependencies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



@dataclass
class TfidfRecommender:
    """
    Content-based recommender using TF-IDF over joke text.

    Fit once on all jokes; recommend jokes for a user based on similarity
    to jokes they rated highly.

    Attributes:
        vectorizer: sklearn TF-IDF vectorizer
        tfidf_matrix: sparse matrix of joke vectors
        joke_ids: numpy array of joke_ids aligned with tfidf_matrix rows
        id_to_idx: mapping joke_id -> row index in tfidf_matrix
    """

    vectorizer: TfidfVectorizer
    tfidf_matrix: any
    joke_ids: np.ndarray
    id_to_idx: Dict[int, int]

    @classmethod
    def fit(
        cls,
        jokes_df: pd.DataFrame,
        max_features: int = 5000,
        use_bigrams: bool = True,
    ) -> "TfidfRecommender":
        """
        Fit a TF-IDF model on the jokes text.

        Args:
            jokes_df: DataFrame with columns ['joke_id', 'joke_text']
            max_features: limit vocabulary size (keeps model small/fast)
            use_bigrams: if True, use unigrams + bigrams; else unigrams only

        Returns:
            A fitted TfidfRecommender instance.
        """
        # Ensure stable ordering so joke_id aligns with row indices
        jokes_df = jokes_df[["joke_id", "joke_text"]].copy()
        jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)
        jokes_df = jokes_df.sort_values("joke_id").reset_index(drop=True)

        ngram_range = (1, 2) if use_bigrams else (1, 1)

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
        )

        tfidf_matrix = vectorizer.fit_transform(jokes_df["joke_text"])
        joke_ids = jokes_df["joke_id"].to_numpy(dtype=int)

        id_to_idx = {int(jid): int(i) for i, jid in enumerate(joke_ids)}

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
        Recommend top-k jokes for a given user.

        Strategy:
        1) Find jokes the user liked (rating >= like_threshold)
        2) Compute similarity from those liked jokes to all jokes
        3) Aggregate similarities (mean)
        4) Remove jokes already seen by the user
        5) Return top-k

        Args:
            edges_df: DataFrame with ['user_id', 'joke_id', 'rating']
            user_id: target user
            k: number of recommendations
            like_threshold: rating threshold for "liked"
            fallback_top_n: if user has no liked jokes, use their top-N rated

        Returns:
            List of (joke_id, score) sorted by score desc.
        """
        # Filter ratings for this user
        user_rows = edges_df.loc[edges_df["user_id"] == user_id]
        if user_rows.empty:
            return []

        # Jokes the user has already rated (seen)
        seen_jokes = set(user_rows["joke_id"].astype(int).tolist())

        # Identify "liked" jokes
        liked_jokes = (
            user_rows.loc[user_rows["rating"] >= like_threshold, "joke_id"]
            .astype(int)
            .tolist()
        )

        # If user has no liked jokes, fall back to their top-rated jokes
        if len(liked_jokes) == 0:
            liked_jokes = (
                user_rows.sort_values("rating", ascending=False)["joke_id"]
                .astype(int)
                .head(fallback_top_n)
                .tolist()
            )

        # Convert liked joke_ids -> indices in tfidf_matrix
        liked_indices = [self.id_to_idx[jid] for jid in liked_jokes if jid in self.id_to_idx]
        if len(liked_indices) == 0:
            return []

        # Compute cosine similarities between liked jokes and all jokes
        # Result shape: (L, N)
        sims = linear_kernel(self.tfidf_matrix[liked_indices], self.tfidf_matrix)

        # Aggregate similarity score per joke across the liked set
        # Convert to flat array
        scores = np.asarray(sims.mean(axis=0)).ravel()

        # Build candidate list excluding seen jokes
        candidates: List[Tuple[int, float]] = []
        for idx, jid in enumerate(self.joke_ids):
            jid_int = int(jid)
            if jid_int in seen_jokes:
                continue
            candidates.append((jid_int, float(scores[idx])))

        # Sort and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
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
        Recommend top-k jokes for a user with a simple diversity rule:
        don't include jokes that are "too similar" to already selected ones.

        Steps:
        1) Get a larger candidate list using recommend_for_user()
        2) Greedily build the final list:
           - accept candidate if its max similarity to selected < sim_threshold
        """

        # Step 1: larger relevance-based pool
        candidates = self.recommend_for_user(
            edges_df=edges_df,
            user_id=user_id,
            k=candidate_pool,
            like_threshold=like_threshold,
            fallback_top_n=fallback_top_n,
        )

        if not candidates:
            return []

        selected: List[Tuple[int, float]] = []
        selected_indices: List[int] = []

        for joke_id, score in candidates:
            if len(selected) >= k:
                break

            # Skip if joke isn't in our TF-IDF index (safety)
            if joke_id not in self.id_to_idx:
                continue

            cand_idx = self.id_to_idx[joke_id]

            # First selection always accepted
            if not selected_indices:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)
                continue

            # Similarity of candidate vs already-selected jokes
            sims = linear_kernel(self.tfidf_matrix[cand_idx], self.tfidf_matrix[selected_indices])
            max_sim = float(sims.max())

            # Diversity rule: reject near-duplicates
            if max_sim < sim_threshold:
                selected.append((joke_id, score))
                selected_indices.append(cand_idx)

        return selected
