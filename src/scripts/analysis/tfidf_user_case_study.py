from __future__ import annotations

"""
Builds a user-level case study table for the TF-IDF recommender.
Shows liked and disliked jokes for fixed users and compares them with TF-IDF scores.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.tfidf.train_tfidf import build_tfidf_recommender


# ---------------------------------------------------------
# 1. Settings
# Defines the fixed users, thresholds, and output filename.
# ---------------------------------------------------------
FIXED_USERS = [2119, 8135, 2856, 2063]

NUM_LIKED = 3
NUM_DISLIKED = 3

LIKE_THRESHOLD = 7.0
DISLIKE_THRESHOLD = 0.0

OUTPUT_FILENAME = "tfidf_user_case_study.csv"


# ---------------------------------------------------------
# 2. Ratings loading
# Loads the cleaned user-joke rating data.
# ---------------------------------------------------------
def load_edges() -> pd.DataFrame:
    # Build the path to the cleaned ratings file
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Stop if the file is missing
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    # Load the ratings file
    edges = pd.read_csv(edges_path).copy()

    # Ensure user IDs are integers
    edges["user_id"] = edges["user_id"].astype(int)

    # Ensure joke IDs are integers
    edges["joke_id"] = edges["joke_id"].astype(int)

    # Ensure ratings are floats
    edges["rating"] = edges["rating"].astype(float)

    return edges


# ---------------------------------------------------------
# 3. Joke text loading
# Loads the cleaned joke text used for display and TF-IDF features.
# ---------------------------------------------------------
def load_joke_text() -> pd.DataFrame:
    # Build the path to the cleaned joke text file
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Stop if the file is missing
    if not jokes_path.exists():
        raise FileNotFoundError(f"Jokes file not found: {jokes_path}")

    # Load the joke text file
    jokes_df = pd.read_csv(jokes_path).copy()

    # Ensure joke IDs are integers
    jokes_df["joke_id"] = jokes_df["joke_id"].astype(int)

    # Ensure joke text is stored as strings
    jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

    # Return only the columns needed
    return jokes_df[["joke_id", "joke_text"]].copy()


# ---------------------------------------------------------
# 4. User-joke scoring
# Computes one TF-IDF similarity score for a user-joke pair.
# ---------------------------------------------------------
def score_user_joke_pair_tfidf(
    user_id: int,
    joke_id: int,
    user_rows: pd.DataFrame,
    recommender,
    like_threshold: float = LIKE_THRESHOLD,
    fallback_top_n: int = 3,
) -> float | None:
    # Collect all jokes this user rated positively
    liked_jokes = (
        user_rows.loc[user_rows["rating"] >= like_threshold, "joke_id"]
        .astype(int)
        .tolist()
    )

    # Fall back to top-rated jokes if no jokes meet the threshold
    if len(liked_jokes) == 0:
        liked_jokes = (
            user_rows.sort_values("rating", ascending=False)["joke_id"]
            .astype(int)
            .head(fallback_top_n)
            .tolist()
        )

    # Remove the target joke from the profile if it appears there
    profile_jokes = [jid for jid in liked_jokes if jid != joke_id]

    # Fall back to the full liked list if the profile becomes empty
    if len(profile_jokes) == 0:
        profile_jokes = liked_jokes.copy()

    # Convert profile joke IDs into TF-IDF row indices
    profile_indices = [
        recommender.id_to_idx[jid]
        for jid in profile_jokes
        if jid in recommender.id_to_idx
    ]

    # Return nothing if the target joke is missing from the TF-IDF matrix
    if joke_id not in recommender.id_to_idx:
        return None

    # Return nothing if the user profile has no valid indices
    if len(profile_indices) == 0:
        return None

    # Look up the TF-IDF row index for the target joke
    target_idx = recommender.id_to_idx[joke_id]

    # Compute cosine similarities between the target and the profile jokes
    sims = linear_kernel(
        recommender.tfidf_matrix[target_idx],
        recommender.tfidf_matrix[profile_indices],
    )

    # Return the mean similarity as the TF-IDF score
    return float(np.asarray(sims).mean())


# ---------------------------------------------------------
# 5. Example table building
# Selects liked and disliked jokes and adds TF-IDF scores.
# ---------------------------------------------------------
def build_user_examples(
    user_id: int,
    user_rows: pd.DataFrame,
    jokes_df: pd.DataFrame,
    recommender,
    num_liked: int = 3,
    num_disliked: int = 3,
) -> pd.DataFrame:
    # Select the highest-rated liked jokes
    liked = (
        user_rows[user_rows["rating"] >= LIKE_THRESHOLD]
        .sort_values("rating", ascending=False)
        .head(num_liked)
        .copy()
    )

    # Select the lowest-rated disliked jokes
    disliked = (
        user_rows[user_rows["rating"] <= DISLIKE_THRESHOLD]
        .sort_values("rating", ascending=True)
        .head(num_disliked)
        .copy()
    )

    # Label the liked rows
    liked["group"] = "liked"

    # Label the disliked rows
    disliked["group"] = "disliked"

    # Combine liked and disliked examples
    examples = pd.concat([liked, disliked], ignore_index=True)

    # Return an empty dataframe if no examples were found
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

    # Add the full joke text for each selected joke
    examples = examples.merge(jokes_df, on="joke_id", how="left")

    # Add the raw user ID as a column
    examples["user_id"] = int(user_id)

    # Keep only the final columns needed for output
    examples = examples[
        ["user_id", "group", "joke_id", "rating", "tfidf_score", "joke_text"]
    ].copy()

    return examples


# ---------------------------------------------------------
# 6. Pretty printing
# Prints one user's case-study table in a clean format.
# ---------------------------------------------------------
def print_user_case_study(user_id: int, user_df: pd.DataFrame) -> None:
    # Print a divider line
    print("\n" + "=" * 100)

    # Print the user heading
    print(f"USER {user_id} | liked vs disliked jokes")

    # Print another divider line
    print("=" * 100)

    # Print the selected columns for this user
    print(
        user_df[
            ["group", "joke_id", "rating", "tfidf_score", "joke_preview"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# 7. Main execution
# Loads data, builds the case study, prints it, and saves it.
# ---------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------
    # 7.1 Data and model loading
    # Loads the datasets and builds the TF-IDF recommender.
    # ---------------------------------------------------------
    # Load the cleaned ratings data
    print("[tfidf_case_study] Loading edges...")
    edges_df = load_edges()

    # Load the cleaned joke text
    print("[tfidf_case_study] Loading jokes...")
    jokes_df = load_joke_text()

    # Build the TF-IDF recommender
    print("[tfidf_case_study] Building TF-IDF recommender...")
    recommender = build_tfidf_recommender(
        max_features=5000,
        use_bigrams=True,
    )

    # ---------------------------------------------------------
    # 7.2 Fixed-user case study building
    # Builds liked/disliked example tables for the selected users.
    # ---------------------------------------------------------
    # Use the fixed users for the case study
    print("[tfidf_case_study] Using fixed users...")
    selected_users = FIXED_USERS

    # Print the selected user IDs
    print(f"[tfidf_case_study] Selected users: {selected_users}")

    # Store all user tables
    all_tables = []

    # Build a case-study table for each selected user
    for user_id in selected_users:
        print(f"[tfidf_case_study] Building case study for user {user_id}...")

        # Select all rows for this user
        user_rows = edges_df[edges_df["user_id"] == user_id].copy()

        # Skip users missing from the ratings data
        if user_rows.empty:
            print(f"[tfidf_case_study] User {user_id} not found in ratings data.")
            continue

        # Build the liked/disliked examples table
        table = build_user_examples(
            user_id=user_id,
            user_rows=user_rows,
            jokes_df=jokes_df,
            recommender=recommender,
            num_liked=NUM_LIKED,
            num_disliked=NUM_DISLIKED,
        )

        # Skip empty tables
        if table.empty:
            print(f"[tfidf_case_study] No usable case-study output for user {user_id}.")
            continue

        # Store the completed user table
        all_tables.append(table)

    # Stop if no tables were created
    if not all_tables:
        print("[tfidf_case_study] No case-study tables were created.")
        return

    # ---------------------------------------------------------
    # 7.3 Final formatting and output
    # Formats the final table, prints it, and saves it to CSV.
    # ---------------------------------------------------------
    # Combine all user tables into one dataframe
    final_df = pd.concat(all_tables, ignore_index=True)

    # AI-assisted section:
    # ChatGPT was used to help make the case-study output more presentable,
    # including the shortened preview text and printed table layout.

    # Create shortened joke previews for printing
    final_df["joke_preview"] = (
        final_df["joke_text"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    # Print a heading for the output table
    print("\n=== TF-IDF USER CASE STUDY TABLE ===")

    # Print each user's case-study table
    for user_id in selected_users:
        user_df = final_df[final_df["user_id"] == user_id].copy()

        # Skip users with no rows in the final table
        if user_df.empty:
            continue

        print_user_case_study(user_id, user_df)

    # Build the output CSV path
    out_path = ROOT / "outputs" / OUTPUT_FILENAME

    # Create the output folder if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the final case-study table
    final_df.to_csv(out_path, index=False)

    # Print the save location
    print(f"\n[tfidf_case_study] Saved case-study table to: {out_path}")


if __name__ == "__main__":
    main()