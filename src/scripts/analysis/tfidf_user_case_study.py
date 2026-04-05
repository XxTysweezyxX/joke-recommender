from __future__ import annotations

"""
tfidf_user_case_study.py

User-level TF-IDF case study for dissertation examples.

Purpose:
- Use a fixed set of users for consistent analysis
- Show liked vs disliked jokes for the same users used in the GCN case studies
- Compute a TF-IDF score for each selected joke
- Build a clean table that is directly comparable with the GCN case study outputs

Run from /src with:
    python -m scripts.analysis.tfidf_user_case_study
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.tfidf.train_tfidf import build_tfidf_recommender


# ---------------------------------------------------------
# Editable settings
# ---------------------------------------------------------
FIXED_USERS = [2119, 8135, 2856, 2063]

NUM_LIKED = 3
NUM_DISLIKED = 3

LIKE_THRESHOLD = 7.0
DISLIKE_THRESHOLD = 0.0  # jokes rated <= 0 treated as disliked

OUTPUT_FILENAME = "tfidf_user_case_study.csv"


# ---------------------------------------------------------
# Helper: load cleaned ratings
# ---------------------------------------------------------
def load_edges() -> pd.DataFrame:
    """
    Load the cleaned user-joke ratings file.
    """
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    edges = pd.read_csv(edges_path).copy()
    edges["user_id"] = edges["user_id"].astype(int)
    edges["joke_id"] = edges["joke_id"].astype(int)
    edges["rating"] = edges["rating"].astype(float)

    return edges


# ---------------------------------------------------------
# Helper: load cleaned joke text
# ---------------------------------------------------------
def load_joke_text() -> pd.DataFrame:
    """
    Load the cleaned joke text file.
    """
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    if not jokes_path.exists():
        raise FileNotFoundError(f"Jokes file not found: {jokes_path}")

    jokes_df = pd.read_csv(jokes_path).copy()
    jokes_df["joke_id"] = jokes_df["joke_id"].astype(int)
    jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

    return jokes_df[["joke_id", "joke_text"]].copy()


# ---------------------------------------------------------
# Helper: score one user-joke pair with TF-IDF
# ---------------------------------------------------------
def score_user_joke_pair_tfidf(
    user_id: int,
    joke_id: int,
    user_rows: pd.DataFrame,
    recommender,
    like_threshold: float = LIKE_THRESHOLD,
    fallback_top_n: int = 3,
) -> float | None:
    """
    Compute a TF-IDF preference score for one user-joke pair.

    Idea:
    - Build a text profile from the user's liked jokes
    - Compare the target joke against that profile
    - Return the mean cosine similarity

    Important:
    - If the target joke is itself one of the liked jokes used to build the profile,
      exclude it from the profile first. This avoids giving it an artificially high
      self-similarity score.
    """
    # All jokes this user rated positively
    liked_jokes = (
        user_rows.loc[user_rows["rating"] >= like_threshold, "joke_id"]
        .astype(int)
        .tolist()
    )

    # Fallback if no jokes pass the threshold
    if len(liked_jokes) == 0:
        liked_jokes = (
            user_rows.sort_values("rating", ascending=False)["joke_id"]
            .astype(int)
            .head(fallback_top_n)
            .tolist()
        )

    # Remove the target joke from the profile if it is present
    profile_jokes = [jid for jid in liked_jokes if jid != joke_id]

    # If removing the target leaves us with nothing, fall back to original liked list
    if len(profile_jokes) == 0:
        profile_jokes = liked_jokes.copy()

    # Convert joke IDs to TF-IDF row indices
    profile_indices = [
        recommender.id_to_idx[jid]
        for jid in profile_jokes
        if jid in recommender.id_to_idx
    ]

    # Target joke must exist in the TF-IDF matrix too
    if joke_id not in recommender.id_to_idx:
        return None

    if len(profile_indices) == 0:
        return None

    target_idx = recommender.id_to_idx[joke_id]

    # Compute cosine similarity between target joke and the user's profile jokes
    sims = linear_kernel(
        recommender.tfidf_matrix[target_idx],
        recommender.tfidf_matrix[profile_indices],
    )

    # Return mean similarity as the TF-IDF score
    return float(np.asarray(sims).mean())


# ---------------------------------------------------------
# Helper: choose liked and disliked jokes for one user
# ---------------------------------------------------------
def build_user_examples(
    user_id: int,
    user_rows: pd.DataFrame,
    jokes_df: pd.DataFrame,
    recommender,
    num_liked: int = 3,
    num_disliked: int = 3,
) -> pd.DataFrame:
    """
    Build a small table of liked and disliked jokes for one user,
    similar in structure to the GCN case study outputs.
    """
    liked = (
        user_rows[user_rows["rating"] >= LIKE_THRESHOLD]
        .sort_values("rating", ascending=False)
        .head(num_liked)
        .copy()
    )

    disliked = (
        user_rows[user_rows["rating"] <= DISLIKE_THRESHOLD]
        .sort_values("rating", ascending=True)
        .head(num_disliked)
        .copy()
    )

    liked["group"] = "liked"
    disliked["group"] = "disliked"

    examples = pd.concat([liked, disliked], ignore_index=True)

    if examples.empty:
        return pd.DataFrame()

    # Compute a TF-IDF score for each selected joke
    examples["tfidf_score"] = examples["joke_id"].apply(
        lambda jid: score_user_joke_pair_tfidf(
            user_id=user_id,
            joke_id=int(jid),
            user_rows=user_rows,
            recommender=recommender,
            like_threshold=LIKE_THRESHOLD,
            fallback_top_n=3,
        )
    )

    examples = examples.merge(jokes_df, on="joke_id", how="left")

    examples["user_id"] = int(user_id)
    examples = examples[
        ["user_id", "group", "joke_id", "rating", "tfidf_score", "joke_text"]
    ].copy()

    return examples


# ---------------------------------------------------------
# Pretty print one user table
# ---------------------------------------------------------
def print_user_case_study(user_id: int, user_df: pd.DataFrame) -> None:
    """
    Print one user's TF-IDF case study in a format that is closer
    to the GCN case study output.
    """
    print("\n" + "=" * 100)
    print(f"USER {user_id} | liked vs disliked jokes")
    print("=" * 100)

    print(
        user_df[
            ["group", "joke_id", "rating", "tfidf_score", "joke_preview"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("[tfidf_case_study] Loading edges...")
    edges_df = load_edges()

    print("[tfidf_case_study] Loading jokes...")
    jokes_df = load_joke_text()

    print("[tfidf_case_study] Building TF-IDF recommender...")
    recommender = build_tfidf_recommender(
        max_features=5000,
        use_bigrams=True,
    )

    print("[tfidf_case_study] Using fixed users...")
    selected_users = FIXED_USERS
    print(f"[tfidf_case_study] Selected users: {selected_users}")

    all_tables = []

    for user_id in selected_users:
        print(f"[tfidf_case_study] Building case study for user {user_id}...")

        user_rows = edges_df[edges_df["user_id"] == user_id].copy()

        if user_rows.empty:
            print(f"[tfidf_case_study] User {user_id} not found in ratings data.")
            continue

        table = build_user_examples(
            user_id=user_id,
            user_rows=user_rows,
            jokes_df=jokes_df,
            recommender=recommender,
            num_liked=NUM_LIKED,
            num_disliked=NUM_DISLIKED,
        )

        if table.empty:
            print(f"[tfidf_case_study] No usable case-study output for user {user_id}.")
            continue

        all_tables.append(table)

    if not all_tables:
        print("[tfidf_case_study] No case-study tables were created.")
        return

    final_df = pd.concat(all_tables, ignore_index=True)

    # Shortened joke text for terminal printing and dissertation tables
    final_df["joke_preview"] = (
        final_df["joke_text"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    print("\n=== TF-IDF USER CASE STUDY TABLE ===")

    for user_id in selected_users:
        user_df = final_df[final_df["user_id"] == user_id].copy()

        if user_df.empty:
            continue

        print_user_case_study(user_id, user_df)

    out_path = ROOT / "outputs" / OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"\n[tfidf_case_study] Saved case-study table to: {out_path}")


if __name__ == "__main__":
    main()