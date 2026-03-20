from __future__ import annotations

"""
lightgcn_test_match_analysis.py

Purpose:
- Use real held-out jokes from the test set
- Score them with the trained LightGCN model
- Compare them against low-rated jokes for the same user
- Check whether the held-out liked joke gets the higher score

Important:
- The LightGCN output is a learned preference score
- It is NOT an exact predicted Jester rating
"""

import random
import pandas as pd
import torch

from joke_reco import config
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.lightgcn.lightgcn_model import LightGCN, LightGCNConfig
from joke_reco.paths import PROCESSED_DIR, ROOT


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
SEED = 42
NUM_USERS_TO_SAMPLE = 10
NEGATIVE_THRESHOLD = 0.0
OUTPUT_FILENAME = "lightgcn_test_match_analysis.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
def load_edges() -> pd.DataFrame:
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    edges = pd.read_csv(edges_path).copy()
    edges["user_id"] = edges["user_id"].astype(int)
    edges["joke_id"] = edges["joke_id"].astype(int)
    edges["rating"] = edges["rating"].astype(float)

    return edges


def load_jokes() -> pd.DataFrame:
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    if not jokes_path.exists():
        raise FileNotFoundError(f"Jokes file not found: {jokes_path}")

    jokes = pd.read_csv(jokes_path).copy()
    jokes["joke_id"] = jokes["joke_id"].astype(int)
    jokes["joke_text"] = jokes["joke_text"].astype(str)

    return jokes[["joke_id", "joke_text"]]


# ---------------------------------------------------------
# Load trained LightGCN
# ---------------------------------------------------------
def load_trained_lightgcn():
    model_path = ROOT / "models" / "lightgcn_jester.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

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

    return user_map, item_map, user_emb, item_emb


# ---------------------------------------------------------
# Score a user-joke pair
# ---------------------------------------------------------
def score_user_joke_pair(
    user_id: int,
    joke_id: int,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
) -> float | None:
    if user_id not in user_map:
        return None
    if joke_id not in item_map:
        return None

    u_idx = user_map[user_id]
    j_idx = item_map[joke_id]

    with torch.no_grad():
        score = torch.dot(user_emb[u_idx], item_emb[j_idx]).item()

    return float(score)


# ---------------------------------------------------------
# Build test-set comparison rows
# ---------------------------------------------------------
def build_test_match_rows(
    edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    num_users_to_sample: int,
    negative_threshold: float,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)

    candidate_users = list(test_edges["user_id"].unique())
    rng.shuffle(candidate_users)

    rows = []
    chosen_users = 0

    for user_id in candidate_users:
        user_test = test_edges[test_edges["user_id"] == user_id].copy()
        user_full = edges[edges["user_id"] == user_id].copy()

        if user_test.empty or user_full.empty:
            continue

        # Pick one held-out positive joke from test set
        pos_row = user_test.sample(n=1, random_state=seed).iloc[0]

        # Pick one low-rated joke from full Jester history
        user_low = user_full[user_full["rating"] <= negative_threshold].copy()
        if user_low.empty:
            continue

        neg_row = user_low.sample(n=1, random_state=seed).iloc[0]

        positive_score = score_user_joke_pair(
            user_id=int(user_id),
            joke_id=int(pos_row["joke_id"]),
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
        )

        negative_score = score_user_joke_pair(
            user_id=int(user_id),
            joke_id=int(neg_row["joke_id"]),
            user_map=user_map,
            item_map=item_map,
            user_emb=user_emb,
            item_emb=item_emb,
        )

        if positive_score is None or negative_score is None:
            continue

        rows.append(
            {
                "user_id": int(user_id),
                "positive_joke_id": int(pos_row["joke_id"]),
                "positive_rating": float(pos_row["rating"]),
                "positive_score": float(positive_score),
                "negative_joke_id": int(neg_row["joke_id"]),
                "negative_rating": float(neg_row["rating"]),
                "negative_score": float(negative_score),
                "score_gap": float(positive_score - negative_score),
                "matched_direction": bool(positive_score > negative_score),
            }
        )

        chosen_users += 1
        if chosen_users >= num_users_to_sample:
            break

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Add joke text
# ---------------------------------------------------------
def attach_joke_text(report: pd.DataFrame, jokes_df: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return report

    pos_text = jokes_df.rename(
        columns={
            "joke_id": "positive_joke_id",
            "joke_text": "positive_joke_text",
        }
    )
    neg_text = jokes_df.rename(
        columns={
            "joke_id": "negative_joke_id",
            "joke_text": "negative_joke_text",
        }
    )

    report = report.merge(pos_text, on="positive_joke_id", how="left")
    report = report.merge(neg_text, on="negative_joke_id", how="left")

    report["positive_preview"] = (
        report["positive_joke_text"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    report["negative_preview"] = (
        report["negative_joke_text"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 140)
    )

    return report


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    print("[analysis] Loading edges...")
    edges = load_edges()

    print("[analysis] Loading jokes...")
    jokes_df = load_jokes()

    print("[analysis] Rebuilding shared train/test split...")
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )

    print("[analysis] Loading trained LightGCN...")
    user_map, item_map, user_emb, item_emb = load_trained_lightgcn()

    print("[analysis] Building test-set score comparisons...")
    report = build_test_match_rows(
        edges=edges,
        test_edges=test_edges,
        user_map=user_map,
        item_map=item_map,
        user_emb=user_emb,
        item_emb=item_emb,
        num_users_to_sample=NUM_USERS_TO_SAMPLE,
        negative_threshold=NEGATIVE_THRESHOLD,
        seed=SEED,
    )

    if report.empty:
        print("[analysis] No comparison rows could be created.")
        return

    report = attach_joke_text(report, jokes_df)

    print("\n=== TEST-SET MATCH ANALYSIS ===")
    print(
        report[
            [
                "user_id",
                "positive_joke_id",
                "positive_rating",
                "positive_score",
                "negative_joke_id",
                "negative_rating",
                "negative_score",
                "score_gap",
                "matched_direction",
            ]
        ].to_string(index=False)
    )

    match_rate = report["matched_direction"].mean()
    print(f"\n[analysis] Match rate: {match_rate:.2%}")
    print(
        "[analysis] matched_direction=True means the held-out liked joke "
        "got a higher LightGCN score than the low-rated joke."
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / OUTPUT_FILENAME
    report.to_csv(out_path, index=False)

    print(f"[analysis] Saved report to: {out_path}")


if __name__ == "__main__":
    main()