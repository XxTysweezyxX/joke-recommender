from __future__ import annotations

"""
User-level case study for dissertation examples.

Purpose:
- Show liked vs disliked jokes for fixed users
- Compare actual Jester ratings with LightGCN preference scores
- Build a clean table for dissertation analysis

Run from /src with:
    python -m scripts.analysis.user_case_study_lightgcn
"""

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig


# ---------------------------------------------------------
# Editable settings
# ---------------------------------------------------------
FIXED_USERS = [2119, 8135, 2856, 2063]

NUM_LIKED = 3
NUM_DISLIKED = 3

LIKE_THRESHOLD = 7.0
DISLIKE_THRESHOLD = 0.0  # jokes rated <= 0 treated as disliked

OUTPUT_FILENAME = "lightgcn_user_case_study.csv"


# ---------------------------------------------------------
# Helper: load cleaned ratings
# ---------------------------------------------------------
def load_edges() -> pd.DataFrame:
    """
    Load the cleaned Jester interaction data.
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
# Helper: load joke text
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
# Helper: load trained original LightGCN
# ---------------------------------------------------------
def load_trained_lightgcn() -> tuple[dict[int, int], dict[int, int], torch.Tensor, torch.Tensor]:
    """
    Reload the trained original LightGCN model and return
    propagated user/item embeddings.
    """
    model_path = ROOT / "models" / "lightgcn_jester.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"[case_study] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")

    user_map = ckpt["user_map"]
    item_map = ckpt["item_map"]
    norm_adj = ckpt["norm_adj"]
    meta = ckpt["meta"]

    print("[case_study] Recreating original LightGCN model...")
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
    )

    model = LightGCN(model_cfg)

    # Safer checkpoint loading:
    # keep only keys that belong to the plain/original LightGCN model
    raw_state_dict = ckpt["state_dict"]
    model_state_keys = set(model.state_dict().keys())

    filtered_state_dict = {
        k: v for k, v in raw_state_dict.items() if k in model_state_keys
    }

    ignored_keys = [k for k in raw_state_dict.keys() if k not in model_state_keys]

    if ignored_keys:
        print(
            "[case_study] Warning: checkpoint contains extra keys not used by "
            f"the original LightGCN model. Ignoring: {ignored_keys}"
        )

    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    print("[case_study] Running propagation...")
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    return user_map, item_map, user_emb, item_emb


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
    Return the LightGCN preference score for one user-joke pair.

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

    with torch.no_grad():
        score = torch.dot(user_emb[u_idx], item_emb[i_idx]).item()

    return float(score)


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
    Build a small table of liked and disliked jokes for one user.
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

    examples = examples.merge(jokes_df, on="joke_id", how="left")

    examples["user_id"] = int(user_id)
    examples = examples[
        ["user_id", "group", "joke_id", "rating", "lightgcn_score", "joke_text"]
    ].copy()

    return examples


# ---------------------------------------------------------
# Pretty print one user table
# ---------------------------------------------------------
def print_user_case_study(user_id: int, user_df: pd.DataFrame) -> None:
    """
    Print one user's case study in a cleaner format.
    """
    print("\n" + "=" * 100)
    print(f"USER {user_id} | liked vs disliked jokes")
    print("=" * 100)

    print(
        user_df[
            ["group", "joke_id", "rating", "lightgcn_score", "joke_preview"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("[case_study] Loading edges...")
    edges = load_edges()

    print("[case_study] Loading jokes...")
    jokes_df = load_joke_text()

    print("[case_study] Loading trained original LightGCN...")
    user_map, item_map, user_emb, item_emb = load_trained_lightgcn()

    print("[case_study] Using fixed users...")
    selected_users = FIXED_USERS
    print(f"[case_study] Selected users: {selected_users}")

    all_tables = []

    for user_id in selected_users:
        print(f"[case_study] Building examples for user {user_id}...")

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

    final_df["joke_preview"] = (
        final_df["joke_text"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    print("\n=== ORIGINAL LIGHTGCN USER CASE STUDY TABLE ===")

    for user_id in selected_users:
        user_df = final_df[final_df["user_id"] == user_id].copy()

        if user_df.empty:
            continue

        print_user_case_study(user_id, user_df)

    out_path = ROOT / "outputs" / OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"\n[case_study] Saved case-study table to: {out_path}")


if __name__ == "__main__":
    main()