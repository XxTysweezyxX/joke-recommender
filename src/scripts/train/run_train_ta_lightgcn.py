from __future__ import annotations

"""
Runs training for the text-augmented LightGCN model.
Loads ratings and joke text, rebuilds the train split, trains the model, and saves a checkpoint.
"""

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco import config
from joke_reco.text_augmented_lightgcn.train_ta_lightgcn import train_ta_lightgcn


# ---------------------------------------------------------
# 1. Main execution
# Runs the full text-augmented LightGCN training pipeline.
# ---------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------
    # 1.1 Data loading
    # Loads the cleaned interaction and joke text datasets.
    # ---------------------------------------------------------
    # Print a progress message
    print("[run_train_ta_lightgcn] Loading edges...")

    # Build the path to the cleaned ratings file
    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"

    # Load the cleaned ratings data
    edges = pd.read_csv(edges_path)

    # Build the path to the cleaned joke text file
    jokes_path = PROCESSED_DIR / "jester_jokes_clean.csv"

    # Load the cleaned joke text data
    jokes = pd.read_csv(jokes_path)

    # Print the number of loaded jokes
    print(f"[run_train_ta_lightgcn] Loaded {len(jokes):,} jokes from {jokes_path}")

    # ---------------------------------------------------------
    # 1.2 Train/test split rebuilding
    # Recreates the shared train split used by the project.
    # ---------------------------------------------------------
    # Print a progress message
    print("[run_train_ta_lightgcn] Splitting train/test...")

    # Rebuild the shared train/test split
    train_edges, _test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )


    # ---------------------------------------------------------
    # 1.3 Model training
    # Trains the text-augmented LightGCN model on the training split.
    # ---------------------------------------------------------
    # Print a progress message
    print("[run_train_ta_lightgcn] Training LightGCN...")

    # Train the text-augmented LightGCN model
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
        device="cpu",
    )

    # ---------------------------------------------------------
    # 1.4 Checkpoint packaging and saving
    # Stores the trained model, graph, text settings, and metadata to disk.
    # ---------------------------------------------------------
    # Build the models folder path
    models_dir = ROOT / "models"

    # Create the models folder if needed
    models_dir.mkdir(parents=True, exist_ok=True)

    # Build the output checkpoint path
    out_path = models_dir / "ta_lightgcn_jester.pt"

    # Package the trained model and metadata
    ckpt = {
        "state_dict": result.model.state_dict(),
        "user_map": result.user_map,
        "item_map": result.item_map,
        "norm_adj": result.norm_adj.to("cpu"),
        "meta": {
            "embedding_dim": 64,
            "num_layers": 3,
            "like_threshold": config.LIKE_THRESHOLD,
            "holdout_per_user": config.HOLDOUT_PER_USER,
            "seed": config.SEED,
            "text_features": "tfidf",
            "item_init_mode": "add",
        },
    }

    # Save the checkpoint to disk
    torch.save(ckpt, out_path)

    # Print the save location
    print("[run_train_ta_lightgcn] Saved checkpoint to:", out_path)


if __name__ == "__main__":
    main()