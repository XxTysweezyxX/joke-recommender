from __future__ import annotations

"""
User-level case study for dissertation examples.

Purpose:
- Show liked vs disliked jokes for selected users
- Compare actual Jester ratings with LightGCN preference scores
- Build a clean table for dissertation analysis

Run from /src with:
    python -m scripts.analysis.user_case_study_lightgcn
"""

import random
import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco import config
from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig


# ---------------------------------------------------------
# Settings you can tweak
# ---------------------------------------------------------
CASE_STUDY_USERS = 2
CASE_STUDY_SEED = 42

NUM_LIKED = 3
NUM_DISLIKED = 3

# Suggested dissertation thresholds for examples
LIKE_THRESHOLD = 7.0
DISLIKE_THRESHOLD = 0.0  # jokes rated <= 0 treated as disliked


# ---------------------------------------------------------
# Helper: load trained LightGCN and final embeddings
# ---------------------------------------------------------
def load_trained_lightgcn():
    model_path = ROOT / "models" / "lightgcn_jester.pt"

    ckpt = torch.load(model_path, map_location="cpu")

    user_map = ckpt["user_map"]
    item_map = ckpt["item_map"]
    norm_adj = ckpt["norm_adj"]
    meta = ckpt["meta"]

    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
    )

    model = LightGCN(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    return model, user_map, item_map, user_emb, item_emb


# ---------------------------------------------------------
# Helper: score one user-joke pair
# ---------------------------------------------------------
def score_user_joke_pair(
    user_id: int,
    joke_id: int,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
) -> float | None:
    """
    Returns the LightGCN preference score for one user-joke pair.

    Important:
    This is NOT a predicted Jester rating.
    It is a learned ranking / relevance score.
    """
    if user_id not in user_map:
        return None
    if joke_id not in item_map:
        return None

    u_idx = user_map[user_id]
    i_idx = item_map[joke_id]

    u_vec = user_emb[u_idx]
    i_vec = item_emb[i_idx]

    score = torch.dot(u_vec, i_vec).item()
    return float(score)


# ---------------------------------------------------------
# Helper: load joke text
# ---------------------------------------------------------
def load_joke_text() -> pd.DataFrame:
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"
    jokes_df = pd.read_csv(jokes_path)

    jokes_df["joke_id"] = jokes_df["joke_id"].astype(int)
    jokes_df["joke_text"] = jokes_df["joke_text"].astype(str)

    return jokes_df[["joke_id", "joke_text"]].copy()


# ---------------------------------------------------------
# Helper: randomly pick valid users
# ---------------------------------------------------------
def pick_random_valid_users(
    edges: pd.DataFrame,
    num_users: int = 2,
    num_liked: int = 3,
    num_disliked: int = 3,
    like_threshold: float = 7.0,
    dislike_threshold: float = 0.0,
    seed: int = 42,
) -> list[int]:
    """
    Randomly picks users who have enough liked and disliked jokes
    for the case-study comparison.
    """
    valid_users = []

    for user_id, group in edges.groupby("user_id"):
        liked_count = (group["rating"] >= like_threshold).sum()
        disliked_count = (group["rating"] <= dislike_threshold).sum()

        if liked_count >= num_liked and disliked_count >= num_disliked:
            valid_users.append(int(user_id))

    if len(valid_users) < num_users:
        raise ValueError(
            f"Not enough valid users found. Needed {num_users}, found {len(valid_users)}."
        )

    rng = random.Random(seed)
    selected = rng.sample(valid_users, num_users)
    return selected


# ---------------------------------------------------------
# Helper: choose liked and disliked jokes for one user
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
    """
    Builds a small table of liked and disliked jokes for one user.
    """

    # Strong positive examples
    liked = (
        user_rows[user_rows["rating"] >= LIKE_THRESHOLD]
        .sort_values("rating", ascending=False)
        .head(num_liked)
        .copy()
    )

    # Clear negative / low-score examples
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

    # Score each joke with LightGCN
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

    # Add joke text
    examples = examples.merge(jokes_df, on="joke_id", how="left")

    # Keep output tidy
    examples["user_id"] = int(user_id)
    examples = examples[
        ["user_id", "group", "joke_id", "rating", "lightgcn_score", "joke_text"]
    ].copy()

    return examples


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    print("[case_study] Loading edges...")
    edges = pd.read_csv(edges_path)

    print("[case_study] Rebuilding shared train/test split...")
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    print("[case_study] Loading jokes...")
    jokes_df = load_joke_text()

    print("[case_study] Loading trained LightGCN...")
    _model, user_map, item_map, user_emb, item_emb = load_trained_lightgcn()

    print("[case_study] Selecting random valid users...")
    selected_users = pick_random_valid_users(
        edges=edges,
        num_users=CASE_STUDY_USERS,
        num_liked=NUM_LIKED,
        num_disliked=NUM_DISLIKED,
        like_threshold=LIKE_THRESHOLD,
        dislike_threshold=DISLIKE_THRESHOLD,
        seed=CASE_STUDY_SEED,
    )

    print(f"[case_study] Selected users: {selected_users}")

    all_tables = []

    for user_id in selected_users:
        print(f"[case_study] Building examples for user {user_id}...")

        # Use full interaction history for liked/disliked examples
        user_rows = edges[edges["user_id"] == user_id].copy()

        if user_rows.empty:
            print(f"[case_study] User {user_id} not found in ratings data.")
            continue

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

        if table.empty:
            print(f"[case_study] No suitable examples found for user {user_id}.")
            continue

        all_tables.append(table)

    if not all_tables:
        print("[case_study] No user case-study tables were created.")
        return

    final_df = pd.concat(all_tables, ignore_index=True)

    # Shorten joke text for easier reading in terminal / dissertation export
    final_df["joke_preview"] = (
        final_df["joke_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    print("\n=== USER CASE STUDY TABLE ===")
    print(
        final_df[
            ["user_id", "group", "joke_id", "rating", "lightgcn_score", "joke_preview"]
        ].to_string(index=False)
    )

    # Save full table for dissertation use
    out_path = ROOT / "outputs" / "lightgcn_user_case_study.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"\n[case_study] Saved case-study table to: {out_path}")


if __name__ == "__main__":
    main()