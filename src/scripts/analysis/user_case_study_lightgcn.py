from __future__ import annotations

"""
Builds a user-level case study table for the original LightGCN model.
Shows liked and disliked jokes for fixed users and compares them with model scores.
"""
# AI-assisted file:
# ChatGPT was used to help structure and refine this user case-study
# file for the original LightGCN model.
# Prompt summary: "Help me write a Python script that loads a trained
# LightGCN model, selects fixed users, compares liked and disliked
# jokes, computes user-joke scores, and builds a clean case-study
# table with short joke previews for dissertation analysis."

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig


# ---------------------------------------------------------
# 1. Settings
# Defines the fixed users, thresholds, and output filename.
# ---------------------------------------------------------
FIXED_USERS = [2119, 8135, 2856, 2063]

NUM_LIKED = 3
NUM_DISLIKED = 3

LIKE_THRESHOLD = 7.0
DISLIKE_THRESHOLD = 0.0

OUTPUT_FILENAME = "lightgcn_user_case_study.csv"


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
# Loads the cleaned joke text used for display in the output table.
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
# 4. Trained model loading
# Reloads the saved original LightGCN model and embeddings.
# ---------------------------------------------------------
def load_trained_lightgcn() -> tuple[dict[int, int], dict[int, int], torch.Tensor, torch.Tensor]:
    # Build the checkpoint path
    model_path = ROOT / "models" / "lightgcn_jester.pt"

    # Stop if the checkpoint is missing
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Load the saved checkpoint
    print(f"[case_study] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")

    # Load the saved user mapping
    user_map = ckpt["user_map"]

    # Load the saved item mapping
    item_map = ckpt["item_map"]

    # Load the saved normalised graph
    norm_adj = ckpt["norm_adj"]

    # Load saved model metadata
    meta = ckpt["meta"]

    # Recreate the original LightGCN model configuration
    print("[case_study] Recreating original LightGCN model...")
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
    )

    # Rebuild the original LightGCN model
    model = LightGCN(model_cfg)

    # Load the saved state dictionary
    raw_state_dict = ckpt["state_dict"]

    # Get the keys used by the plain LightGCN model
    model_state_keys = set(model.state_dict().keys())

    # Keep only checkpoint weights that match this model
    filtered_state_dict = {
        k: v for k, v in raw_state_dict.items() if k in model_state_keys
    }

    # Record any keys that will be ignored
    ignored_keys = [k for k in raw_state_dict.keys() if k not in model_state_keys]

    # Warn if the checkpoint contains extra keys
    if ignored_keys:
        print(
            "[case_study] Warning: checkpoint contains extra keys not used by "
            f"the original LightGCN model. Ignoring: {ignored_keys}"
        )

    # Load the filtered model weights
    model.load_state_dict(filtered_state_dict, strict=False)

    # Switch to evaluation mode
    model.eval()

    # Run propagation to get final user and item embeddings
    print("[case_study] Running propagation...")
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    return user_map, item_map, user_emb, item_emb


# ---------------------------------------------------------
# 5. User-joke scoring
# Computes one LightGCN score for a user-joke pair.
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

    # Compute the preference score with a dot product
    with torch.no_grad():
        score = torch.dot(user_emb[u_idx], item_emb[i_idx]).item()

    return float(score)


# ---------------------------------------------------------
# 6. Example table building
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

    # Score each selected joke with the LightGCN model
    examples["lightgcn_score"] = examples["joke_id"].apply(
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
        ["user_id", "group", "joke_id", "rating", "lightgcn_score", "joke_text"]
    ].copy()

    return examples


# ---------------------------------------------------------
# 7. Pretty printing
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
            ["group", "joke_id", "rating", "lightgcn_score", "joke_preview"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# 8. Main execution
# Loads data, builds the case study, prints it, and saves it.
# ---------------------------------------------------------
def main() -> None:
    # Load the cleaned ratings data
    print("[case_study] Loading edges...")
    edges = load_edges()

    # Load the cleaned joke text
    print("[case_study] Loading jokes...")
    jokes_df = load_joke_text()

    # Reload the trained original LightGCN model
    print("[case_study] Loading trained original LightGCN...")
    user_map, item_map, user_emb, item_emb = load_trained_lightgcn()

    # Use the fixed users for the case study
    print("[case_study] Using fixed users...")
    selected_users = FIXED_USERS

    # Print the selected user IDs
    print(f"[case_study] Selected users: {selected_users}")

    # Store all user tables
    all_tables = []

    # Build a case-study table for each selected user
    for user_id in selected_users:
        print(f"[case_study] Building examples for user {user_id}...")

        # Select all rows for this user
        user_rows = edges[edges["user_id"] == user_id].copy()

        # Skip users missing from the ratings data
        if user_rows.empty:
            print(f"[case_study] User {user_id} not found in ratings data.")
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
            print(f"[case_study] No suitable examples found for user {user_id}.")
            continue

        # Store the completed user table
        all_tables.append(table)

    # Stop if no tables were created
    if not all_tables:
        print("[case_study] No user case-study tables were created.")
        return

    # Combine all user tables into one dataframe
    final_df = pd.concat(all_tables, ignore_index=True)

    # Create shortened joke previews for printing
    final_df["joke_preview"] = (
        final_df["joke_text"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    # Print a heading for the output table
    print("\n=== ORIGINAL LIGHTGCN USER CASE STUDY TABLE ===")

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
    print(f"\n[case_study] Saved case-study table to: {out_path}")


if __name__ == "__main__":
    main()