from __future__ import annotations

# Runner script: loads data, trains LightGCN, and saves a checkpoint.
# Run from /src with:
#     python -m scripts.run_train_ta_lightgcn

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco import config
from joke_reco.text_augmented_lightgcn.train_ta_lightgcn import train_ta_lightgcn


def main() -> None:
    """
    Main runner for LightGCN training.

    This script loads the interaction data, rebuilds the train/test split,
    trains the LightGCN model, and saves the final checkpoint to disk.
    """
    print("[run_train_ta_lightgcn] Loading edges...")

    # Path to the cleaned user-joke interaction file
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Load all interaction rows
    edges = pd.read_csv(edges_path)
    print(f"[run_train_ta_lightgcn] Loaded {len(edges):,} rows from {edges_path}")

    print("[run_train_ta_lightgcn] Splitting train/test...")

    # Path to the cleaned joke text file
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Load joke text data
    jokes = pd.read_csv(jokes_path)
    print(f"[run_train_ta_lightgcn] Loaded {len(jokes):,} jokes from {jokes_path}")

    # Recreate the same train/test split used in evaluation
    train_edges, _test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )
    print(f"[run_train_ta_lightgcn] Train rows: {len(train_edges):,}")

    print("[run_train_ta_lightgcn] Training LightGCN...")

    # Train the LightGCN model using the training split
    result = train_ta_lightgcn(
        edges_train=train_edges,
        jokes_df=jokes,
        like_threshold=config.LIKE_THRESHOLD,
        embedding_dim=64,
        num_layers=3,
        lr=1e-3,
        batch_size=2048,
        epochs=10,
        samples_per_epoch=200_000,
        seed=config.SEED,
        device="cpu",  # can switch to "cuda" later if using a GPU
    )

    # Create a models folder if it does not already exist
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Output path for the saved LightGCN checkpoint
    out_path = models_dir / "ta_lightgcn_jester.pt"

    # Package everything needed to reload the trained model later
    ckpt = {
        "state_dict": result.model.state_dict(),   # trained model weights
        "user_map": result.user_map,               # raw user ID -> model index
        "item_map": result.item_map,               # raw joke ID -> model index
        "norm_adj": result.norm_adj.to("cpu"),     # graph adjacency matrix
        "meta": {
            "embedding_dim": 64,
            "num_layers": 3,
            "like_threshold": config.LIKE_THRESHOLD,
            "holdout_per_user": config.HOLDOUT_PER_USER,
            "seed": config.SEED,
            "text_features": "tfidf",
            #"item_init_mode": "text_only",#
            "item_init_mode": "add",
        },
    }

    # Save the checkpoint to disk
    torch.save(ckpt, out_path)
    print("[run_train_ta_lightgcn] Saved checkpoint to:", out_path)


# Standard Python entry point
if __name__ == "__main__":
    main()