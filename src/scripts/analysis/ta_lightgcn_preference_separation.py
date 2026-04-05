# ta_lightgcn_preference_separation.py

from __future__ import annotations

"""
ta_lightgcn_preference_separation.py

Purpose:
- Analyse whether the text-augmented LightGCN model separates liked jokes from disliked jokes
- Use a fixed set of users so the analysis stays consistent across the dissertation
- Summarise user-level preference separation in a cleaner way than within-group ordering

Run from /src with:
    python -m scripts.analysis.ta_lightgcn_preference_separation
"""

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.build_joke_text_features import build_item_text_features
from joke_reco.text_augmented_lightgcn.text_augmented_lightgcn import (
    LightGCN,
    LightGCNConfig,
)


# ---------------------------------------------------------
# Editable settings
# ---------------------------------------------------------
FIXED_USERS = [2119, 8135, 2856, 2063]

NUM_LIKED = 3
NUM_DISLIKED = 3

LIKE_THRESHOLD = 7.0
DISLIKE_THRESHOLD = 0.0

OUTPUT_FILENAME = "ta_lightgcn_preference_separation.csv"


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
# Helper: load cleaned jokes
# ---------------------------------------------------------
def load_jokes() -> pd.DataFrame:
    """
    Load cleaned joke text data.
    """
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    if not jokes_path.exists():
        raise FileNotFoundError(f"Jokes file not found: {jokes_path}")

    jokes = pd.read_csv(jokes_path).copy()
    jokes["joke_id"] = jokes["joke_id"].astype(int)
    jokes["joke_text"] = jokes["joke_text"].astype(str)

    return jokes[["joke_id", "joke_text"]]


# ---------------------------------------------------------
# Helper: load trained text-augmented LightGCN
# ---------------------------------------------------------
def load_trained_ta_lightgcn():
    """
    Reload the trained text-augmented LightGCN model.
    """
    model_path = ROOT / "models" / "ta_lightgcn_jester.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    user_map = ckpt["user_map"]
    item_map = ckpt["item_map"]
    norm_adj = ckpt["norm_adj"]
    meta = ckpt["meta"]

    jokes_df = load_jokes()

    item_text_features, _vectorizer = build_item_text_features(
        jokes_df=jokes_df,
        item_map=item_map,
        device="cpu",
    )

    model_cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=meta["embedding_dim"],
        num_layers=meta["num_layers"],
        text_feature_dim=item_text_features.shape[1],
        item_init_mode=meta.get("item_init_mode", "add"),
    )

    model = LightGCN(
        model_cfg,
        item_text_features=item_text_features,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

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
    Return the text-augmented LightGCN preference score for one user-joke pair.

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
        "model": "text_augmented_lightgcn",
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
    print("\n=== TEXT-AUGMENTED LIGHTGCN PREFERENCE SEPARATION ANALYSIS ===")
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

    print("\n[ta_analysis] Interpretation guide:")
    print("- score_gap > 0 means liked jokes scored higher on average than disliked jokes")
    print("- strict_separation=True means every liked joke scored above every disliked joke")
    print("- pairwise_accuracy is the proportion of liked-vs-disliked pairs ordered correctly")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("[ta_analysis] Loading edges...")
    edges = load_edges()

    print("[ta_analysis] Loading trained text-augmented LightGCN...")
    user_map, item_map, user_emb, item_emb = load_trained_ta_lightgcn()

    rows = []

    print(f"[ta_analysis] Using fixed users: {FIXED_USERS}")

    for user_id in FIXED_USERS:
        print(f"[ta_analysis] Building preference separation summary for user {user_id}...")

        user_rows = edges[edges["user_id"] == user_id].copy()

        if user_rows.empty:
            print(f"[ta_analysis] User {user_id} not found in ratings data.")
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
            print(f"[ta_analysis] Could not build summary for user {user_id}.")
            continue

        rows.append(row)

    if not rows:
        print("[ta_analysis] No preference separation rows were created.")
        return

    final_df = pd.DataFrame(rows)

    print_summary_table(final_df)

    mean_gap = final_df["score_gap"].mean()
    mean_pairwise = final_df["pairwise_accuracy"].mean()
    strict_rate = final_df["strict_separation"].mean()

    print(f"\n[ta_analysis] Mean score gap:          {mean_gap:.6f}")
    print(f"[ta_analysis] Mean pairwise acc:      {mean_pairwise:.2%}")
    print(f"[ta_analysis] Strict separation rate: {strict_rate:.2%}")

    out_path = ROOT / "outputs" / OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"[ta_analysis] Saved report to: {out_path}")


if __name__ == "__main__":
    main()