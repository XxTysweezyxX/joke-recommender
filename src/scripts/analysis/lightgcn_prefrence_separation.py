
from __future__ import annotations

"""
lightgcn_preference_separation.py

Purpose:
- Analyse whether the original LightGCN model separates liked jokes from disliked jokes
- Use a fixed set of users so the analysis stays consistent across the dissertation
- Summarise user-level preference separation in a cleaner way than within-group ordering

Run from /src with:
    python -m scripts.analysis.lightgcn_preference_separation
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
DISLIKE_THRESHOLD = 0.0

OUTPUT_FILENAME = "lightgcn_preference_separation.csv"


# ---------------------------------------------------------
# Helper: load cleaned ratings
# ---------------------------------------------------------
def load_edges() -> pd.DataFrame:
    """
    Load the cleaned Jester interaction data.
    """
    # Build the path to the processed ratings file
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Stop early with a clear error if the file is missing
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    # Read the cleaned ratings data
    edges = pd.read_csv(edges_path).copy()

    # Make sure the core columns use the expected data types
    # so later filtering, grouping, and scoring work properly
    edges["user_id"] = edges["user_id"].astype(int)
    edges["joke_id"] = edges["joke_id"].astype(int)
    edges["rating"] = edges["rating"].astype(float)

    # Return the cleaned interaction dataframe
    return edges


# ---------------------------------------------------------
# Helper: load trained original LightGCN
# ---------------------------------------------------------
def load_trained_lightgcn():
    """
    Reload the trained original LightGCN model.
    """

    model_path = ROOT / "models" / "lightgcn_jester.pt" # Path to the saved original LightGCN checkpoint

    # Stop early with a clear error if the checkpoint file is missing
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Load the saved checkpoint onto CPU
    # map_location="cpu" makes sure it loads even if the model was trained elsewhere
    ckpt = torch.load(model_path, map_location="cpu")

    # Recover the main objects saved during training
    user_map = ckpt["user_map"]      # raw user ID -> model user index
    item_map = ckpt["item_map"]      # raw joke ID -> model item index
    norm_adj = ckpt["norm_adj"]      # normalised adjacency matrix for propagation
    meta = ckpt["meta"]              # saved training/model settings

    # Rebuild the same LightGCN configuration used during training
    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
    )

    # Recreate the original LightGCN model
    model = LightGCN(model_cfg)

    # Get the saved weights from the checkpoint
    raw_state_dict = ckpt["state_dict"]

    # Collect the parameter names that belong to the current original LightGCN model
    model_state_keys = set(model.state_dict().keys())

    # Keep only weights that match the original LightGCN structure
    # This is useful in case the checkpoint accidentally contains
    # extra text-augmented keys that do not belong to the plain model
    filtered_state_dict = {
        k: v for k, v in raw_state_dict.items() if k in model_state_keys
    }

    # Track any ignored keys so they can be reported clearly
    ignored_keys = [k for k in raw_state_dict.keys() if k not in model_state_keys]

    # Warn if extra checkpoint weights were found and ignored
    if ignored_keys:
        print(
            "[analysis] Warning: checkpoint contains extra keys not used by "
            f"the original LightGCN model. Ignoring: {ignored_keys}"
        )

    # Load the matching weights into the model
    # strict=False allows the load to continue even if some extra keys were ignored
    model.load_state_dict(filtered_state_dict, strict=False)

    # Switch to evaluation mode since this script is only scoring, not training
    model.eval()

    # Run graph propagation once to get the final user and item embeddings
    # torch.no_grad() avoids storing gradients and keeps the analysis lighter
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    # Return everything needed for later user-joke scoring
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
    It is a learned ranking / preference score.
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
# Helper: pick fixed liked/disliked examples for one user
# ---------------------------------------------------------
def select_user_examples(
    user_rows: pd.DataFrame,
    num_liked: int = 3,
    num_disliked: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the strongest liked and disliked jokes for one user.

    Liked jokes:
    - rating >= LIKE_THRESHOLD
    - sorted from highest rating down

    Disliked jokes:
    - rating <= DISLIKE_THRESHOLD
    - sorted from lowest rating up
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

    return liked, disliked


# ---------------------------------------------------------
# Helper: pairwise separation accuracy
# ---------------------------------------------------------
def compute_pairwise_accuracy(
    liked_scores: list[float],
    disliked_scores: list[float],
) -> float | None:
    """
    Compute the proportion of liked-vs-disliked score pairs
    where the liked joke scored higher.

    With 3 liked and 3 disliked jokes, this gives 9 comparisons.
    """
    if not liked_scores or not disliked_scores:
        return None

    total_pairs = 0
    correct_pairs = 0

    for liked_score in liked_scores:
        for disliked_score in disliked_scores:
            total_pairs += 1
            if liked_score > disliked_score:
                correct_pairs += 1

    if total_pairs == 0:
        return None

    return correct_pairs / total_pairs


# ---------------------------------------------------------
# Helper: build one summary row per user
# ---------------------------------------------------------
def build_user_summary(
    user_id: int,
    user_rows: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
) -> dict | None:
    """
    Build a preference separation summary for one user.
    """
    liked_df, disliked_df = select_user_examples(
        user_rows=user_rows,
        num_liked=NUM_LIKED,
        num_disliked=NUM_DISLIKED,
    )

    if liked_df.empty or disliked_df.empty:
        return None

    liked_scores = []
    disliked_scores = []

    for joke_id in liked_df["joke_id"].astype(int).tolist():
        score = score_user_joke_pair(
            user_id=user_id,
            joke_id=joke_id,
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
        )
        if score is not None:
            liked_scores.append(score)

    for joke_id in disliked_df["joke_id"].astype(int).tolist():
        score = score_user_joke_pair(
            user_id=user_id,
            joke_id=joke_id,
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
        )
        if score is not None:
            disliked_scores.append(score)

    if not liked_scores or not disliked_scores:
        return None

    mean_liked = sum(liked_scores) / len(liked_scores)
    mean_disliked = sum(disliked_scores) / len(disliked_scores)
    score_gap = mean_liked - mean_disliked

    min_liked = min(liked_scores)
    max_disliked = max(disliked_scores)

    strict_separation = min_liked > max_disliked

    pairwise_accuracy = compute_pairwise_accuracy(
        liked_scores=liked_scores,
        disliked_scores=disliked_scores,
    )

    return {
        "user_id": int(user_id),
        "model": "original_lightgcn",
        "num_liked": len(liked_scores),
        "num_disliked": len(disliked_scores),
        "mean_liked_score": float(mean_liked),
        "mean_disliked_score": float(mean_disliked),
        "score_gap": float(score_gap),
        "min_liked_score": float(min_liked),
        "max_disliked_score": float(max_disliked),
        "strict_separation": bool(strict_separation),
        "pairwise_accuracy": float(pairwise_accuracy) if pairwise_accuracy is not None else None,
    }


# ---------------------------------------------------------
# Helper: print readable summary
# ---------------------------------------------------------
def print_summary_table(df: pd.DataFrame) -> None:
    """
    Print the final summary in a clean dissertation-friendly format.
    """
    print("\n=== ORIGINAL LIGHTGCN PREFERENCE SEPARATION ANALYSIS ===")
    print(
        df[
            [
                "user_id",
                "mean_liked_score",
                "mean_disliked_score",
                "score_gap",
                "min_liked_score",
                "max_disliked_score",
                "strict_separation",
                "pairwise_accuracy",
            ]
        ].to_string(index=False)
    )

    print("\n[analysis] Interpretation guide:")
    print("- score_gap > 0 means liked jokes scored higher on average than disliked jokes")
    print("- strict_separation=True means every liked joke scored above every disliked joke")
    print("- pairwise_accuracy is the proportion of liked-vs-disliked pairs ordered correctly")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("[analysis] Loading edges...")
    edges = load_edges()

    print("[analysis] Loading trained original LightGCN...")
    user_map, item_map, user_emb, item_emb = load_trained_lightgcn()

    rows = []

    print(f"[analysis] Using fixed users: {FIXED_USERS}")

    for user_id in FIXED_USERS:
        print(f"[analysis] Building preference separation summary for user {user_id}...")

        user_rows = edges[edges["user_id"] == user_id].copy()

        if user_rows.empty:
            print(f"[analysis] User {user_id} not found in ratings data.")
            continue

        row = build_user_summary(
            user_id=user_id,
            user_rows=user_rows,
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
        )

        if row is None:
            print(f"[analysis] Could not build summary for user {user_id}.")
            continue

        rows.append(row)

    if not rows:
        print("[analysis] No preference separation rows were created.")
        return

    final_df = pd.DataFrame(rows)

    print_summary_table(final_df)

    mean_gap = final_df["score_gap"].mean()
    mean_pairwise = final_df["pairwise_accuracy"].mean()
    strict_rate = final_df["strict_separation"].mean()

    print(f"\n[analysis] Mean score gap:          {mean_gap:.6f}")
    print(f"[analysis] Mean pairwise acc:      {mean_pairwise:.2%}")
    print(f"[analysis] Strict separation rate: {strict_rate:.2%}")

    out_path = ROOT / "outputs" / OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"[analysis] Saved report to: {out_path}")


if __name__ == "__main__":
    main()