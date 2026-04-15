from __future__ import annotations

"""
Builds a user-level case study table for the text-augmented LightGCN model.
Shows liked and disliked jokes for fixed users and compares them with model scores.
"""

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco import config
from joke_reco.text_augmented_lightgcn.build_joke_text_features import build_item_text_features
from joke_reco.text_augmented_lightgcn.text_augmented_lightgcn import LightGCN, LightGCNConfig

# AI-assisted file:
# ChatGPT was used to help structure and refine this user case-study
# file for the text-augmented LightGCN model.
# Prompt summary: "Help me write a Python script that loads a trained
# text-augmented LightGCN model, rebuilds joke text features,
# selects fixed users, compares liked and disliked jokes, computes
# user-joke scores, and builds a clean case-study table for
# dissertation analysis."

# ---------------------------------------------------------
# 1. Settings
# Defines the fixed users and rating thresholds for the case study.
# ---------------------------------------------------------
FIXED_USERS = [2119, 8135, 2856, 2063]

NUM_LIKED = 3
NUM_DISLIKED = 3

LIKE_THRESHOLD = 7.0
DISLIKE_THRESHOLD = 0.0


# ---------------------------------------------------------
# 2. Joke text loading
# Loads the cleaned joke text used for display in the output table.
# ---------------------------------------------------------
def load_joke_text() -> pd.DataFrame:
    # Build the path to the cleaned joke text file
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Load the joke text file
    jokes_df = pd.read_csv(jokes_path)

    # Ensure joke IDs are integers
    jokes_df["joke_id"] = jokes_df["joke_id"].astype(int)

    # Ensure joke text is stored as strings
    jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

    # Return only the columns needed for the case study
    return jokes_df[["joke_id", "joke_text"]].copy()


# ---------------------------------------------------------
# 3. Trained model loading
# Reloads the saved text-augmented LightGCN model and embeddings.
# ---------------------------------------------------------
def load_trained_ta_lightgcn() -> tuple[dict[int, int], dict[int, int], torch.Tensor, torch.Tensor]:
    # Build the checkpoint path
    model_path = ROOT / "models" / "ta_lightgcn_jester.pt"

    # Build the cleaned joke text path
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Load the saved checkpoint
    print("[ta_case_study] Loading checkpoint...")
    ckpt = torch.load(model_path, map_location="cpu")

    # Load the saved user mapping
    user_map = ckpt["user_map"]

    # Load the saved item mapping
    item_map = ckpt["item_map"]

    # Load the saved normalised graph
    norm_adj = ckpt["norm_adj"]

    # Load saved model metadata
    meta = ckpt["meta"]

    # Reload the joke text file
    print("[ta_case_study] Rebuilding joke text features...")
    jokes_df = pd.read_csv(jokes_path)

    # Rebuild item-side text features in the same way as training
    item_text_features, _ = build_item_text_features(
        jokes_df=jokes_df,
        item_map=item_map,
        device="cpu",
    )

    # Recreate the saved model configuration
    print("[ta_case_study] Recreating model...")
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
        text_feature_dim=item_text_features.shape[1],
    )

    # Rebuild the text-augmented LightGCN model
    model = LightGCN(
        model_cfg,
        item_text_features=item_text_features,
    )

    # Load the trained model weights
    model.load_state_dict(ckpt["state_dict"])

    # Switch to evaluation mode
    model.eval()

    # Run propagation to get final user and item embeddings
    print("[ta_case_study] Running propagation...")
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    return user_map, item_map, user_emb, item_emb


# ---------------------------------------------------------
# 4. User-joke scoring
# Computes one text-augmented LightGCN score for a user-joke pair.
# ---------------------------------------------------------
def score_user_joke_pair(
    user_id: int,
    joke_id: int,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
) -> float | None:
    # Return nothing if the user is missing from the model
    if user_id not in user_map:
        return None

    # Return nothing if the joke is missing from the model
    if joke_id not in item_map:
        return None

    # Convert the raw user ID to its model index
    u_idx = user_map[user_id]

    # Convert the raw joke ID to its model index
    i_idx = item_map[joke_id]

    # Get the user embedding vector
    u_vec = user_emb[u_idx]

    # Get the joke embedding vector
    i_vec = item_emb[i_idx]

    # Compute the preference score using a dot product
    score = torch.dot(u_vec, i_vec).item()

    return float(score)


# ---------------------------------------------------------
# 5. Example table building
# Selects liked and disliked jokes and adds model scores.
# ---------------------------------------------------------
def build_user_examples(
    user_id: int,
    user_rows: pd.DataFrame,
    jokes_df: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
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

    # Score each selected joke with the TA-LightGCN model
    examples["ta_lightgcn_score"] = examples["joke_id"].apply(
        lambda jid: score_user_joke_pair(
            user_id=user_id,
            joke_id=int(jid),
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
        )
    )

    # Add the full joke text for each selected joke
    examples = examples.merge(jokes_df, on="joke_id", how="left")

    # Add the raw user ID as a column
    examples["user_id"] = int(user_id)

    # Keep only the final columns needed for output
    examples = examples[
        ["user_id", "group", "joke_id", "rating", "ta_lightgcn_score", "joke_text"]
    ].copy()

    return examples


# ---------------------------------------------------------
# 6. Main execution
# Loads data, builds the case study, prints it, and saves it.
# ---------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------
    # 6.1 Data and model loading
    # Loads the ratings, joke text, and trained TA-LightGCN model.
    # ---------------------------------------------------------
    # Build the path to the cleaned ratings file
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Load the ratings data
    print("[ta_case_study] Loading edges...")
    edges = pd.read_csv(edges_path)

    # Rebuild the shared train/test split for consistency
    print("[ta_case_study] Rebuilding shared train/test split...")
    _train_edges, _test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    # Load the cleaned joke text
    print("[ta_case_study] Loading jokes...")
    jokes_df = load_joke_text()

    # Reload the trained TA-LightGCN model
    print("[ta_case_study] Loading trained text-augmented LightGCN...")
    user_map, item_map, user_emb, item_emb = load_trained_ta_lightgcn()

    # ---------------------------------------------------------
    # 6.2 Fixed-user case study building
    # Builds liked/disliked example tables for the selected users.
    # ---------------------------------------------------------
    # Use the fixed users for the case study
    print("[ta_case_study] Using fixed users...")
    selected_users = FIXED_USERS

    # Print the selected user IDs
    print(f"[ta_case_study] Selected users: {selected_users}")

    # Store all user tables
    all_tables = []

    # Build a case-study table for each selected user
    for user_id in selected_users:
        print(f"[ta_case_study] Building examples for user {user_id}...")

        # Select all rows for this user
        user_rows = edges[edges["user_id"] == user_id].copy()

        # Skip users missing from the ratings data
        if user_rows.empty:
            print(f"[ta_case_study] User {user_id} not found in ratings data.")
            continue

        # Build the liked/disliked examples table
        table = build_user_examples(
            user_id=user_id,
            user_rows=user_rows,
            jokes_df=jokes_df,
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
            num_liked=NUM_LIKED,
            num_disliked=NUM_DISLIKED,
        )

        # Skip empty tables
        if table.empty:
            print(f"[ta_case_study] No suitable examples found for user {user_id}.")
            continue

        # Store the completed user table
        all_tables.append(table)

    # Stop if no tables were created
    if not all_tables:
        print("[ta_case_study] No user case-study tables were created.")
        return

    # ---------------------------------------------------------
    # 6.3 Final formatting and output
    # Formats the final table, prints it, and saves it to CSV.
    # ---------------------------------------------------------
    # Combine all user tables into one dataframe
    final_df = pd.concat(all_tables, ignore_index=True)



    # Create shortened joke previews for printing
    final_df["joke_preview"] = (
        final_df["joke_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    # Print a heading for the output table
    print("\n=== TEXT-AUGMENTED LIGHTGCN USER CASE STUDY TABLE ===")

    # Print each user's case study table
    for user_id in selected_users:
        user_df = final_df[final_df["user_id"] == user_id].copy()

        print("\n" + "=" * 100)
        print(f"USER {user_id} | liked vs disliked jokes")
        print("=" * 100)

        print(
            user_df[
                ["group", "joke_id", "rating", "ta_lightgcn_score", "joke_preview"]
            ].to_string(index=False)
        )

    # Build the output CSV path
    out_path = ROOT / "outputs" / "ta_lightgcn_user_case_study.csv"

    # Create the output folder if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the final case-study table
    final_df.to_csv(out_path, index=False)

    # Print the save location
    print(f"\n[ta_case_study] Saved case-study table to: {out_path}")


if __name__ == "__main__":
    main()