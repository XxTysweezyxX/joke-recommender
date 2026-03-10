from __future__ import annotations

# Runner script: loads data, trains LightGCN, saves checkpoint.
# Run from /src with:  python -m scripts.run_train_lightgcn

import pandas as pd
import torch

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco import config
from joke_reco.lightgcn.train_lightgcn import train_lightgcn


def main() -> None:
    print("[run_train_lightgcn] Loading edges...")

    edges_path = PROCESSED_DIR / "jester_edges_clean.csv"
    edges = pd.read_csv(edges_path)
    print(f"[run_train_lightgcn] Loaded {len(edges):,} rows from {edges_path}")

    print("[run_train_lightgcn] Splitting train/test...")
    train_edges, _test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=config.LIKE_THRESHOLD,
        test_size=config.HOLDOUT_PER_USER,
        seed=config.SEED,
    )
    print(f"[run_train_lightgcn] Train rows: {len(train_edges):,}")

    print("[run_train_lightgcn] Training LightGCN...")
    result = train_lightgcn(
        edges_train=train_edges,
        like_threshold=config.LIKE_THRESHOLD,
        embedding_dim=64,
        num_layers=3,
        lr=1e-3,
        batch_size=2048,
        epochs=10,
        samples_per_epoch=200_000,
        seed=config.SEED,
        device="cpu",  # switch to "cuda" later if you want
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "lightgcn_jester.pt"

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
        },
    }

    torch.save(ckpt, out_path)
    print("[run_train_lightgcn] Saved checkpoint to:", out_path)


if __name__ == "__main__":
    main()