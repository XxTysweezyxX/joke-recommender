import torch
import pandas as pd

from joke_reco.paths import PROCESSED_DIR, ROOT
from joke_reco.evaluation_split import train_test_split_by_user
from joke_reco.metrics import precision_at_k, recall_at_k, ndcg_at_k
from joke_reco.lightgcn.model import LightGCN, LightGCNConfig


def main() -> None:
    edges = pd.read_csv(PROCESSED_DIR / "jester_edges_clean.csv")

    # Keep evaluation identical to TF-IDF
    like_threshold = 5.0
    holdout_per_user = 2
    k = 10
    eval_users = 500
    seed = 42

    # Split (same as TF-IDF)
    train_edges, test_edges = train_test_split_by_user(
        edges=edges,
        like_threshold=like_threshold,
        test_size=holdout_per_user,
        seed=seed,
    )

    # Load saved model artifacts
    model_path = ROOT / "models" / "lightgcn_jester.pt"
    ckpt = torch.load(model_path, map_location="cpu")

    user_map = ckpt["user_map"]
    item_map = ckpt["item_map"]
    norm_adj = ckpt["norm_adj"]  # sparse tensor on CPU

    meta = ckpt.get("meta", {})
    embedding_dim = int(meta.get("embedding_dim", 64))
    num_layers = int(meta.get("num_layers", 3))

    cfg = LightGCNConfig(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
    model = LightGCN(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Precompute propagated embeddings once for evaluation
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)

    # Evaluate on a subset of users for speed
    user_ids = test_edges["user_id"].unique().tolist()[:eval_users]

    p_sum = r_sum = n_sum = 0.0
    n_users = 0

    # Build quick per-user seen set from train_edges (so we don't recommend seen items)
    train_seen = (
        train_edges.groupby("user_id")["joke_id"]
        .apply(lambda s: set(s.astype(int).tolist()))
        .to_dict()
    )

    # Candidate items are only those in item_map (contiguous training items)
    all_item_ids = list(item_map.keys())  # raw joke_ids present in training positives
    all_item_idx = torch.tensor([item_map[jid] for jid in all_item_ids], dtype=torch.long)

    for raw_uid in user_ids:
        raw_uid = int(raw_uid)

        # Relevant = held-out liked jokes (raw joke_ids)
        relevant = set(
            test_edges.loc[test_edges["user_id"] == raw_uid, "joke_id"]
            .astype(int)
            .tolist()
        )
        if not relevant:
            continue

        # If user not in training mapping (rare), skip
        if raw_uid not in user_map:
            continue

        seen = train_seen.get(raw_uid, set())
        u_idx = user_map[raw_uid]

        # Score all candidate items for this user
        with torch.no_grad():
            u_vec = user_emb[u_idx].unsqueeze(0)  # (1, D)
            scores = (u_vec * item_emb[all_item_idx]).sum(dim=1)  # (N,)

        # Build ranked list excluding seen
        scored = []
        for jid, s in zip(all_item_ids, scores.tolist()):
            if jid in seen:
                continue
            scored.append((jid, float(s)))

        scored.sort(key=lambda x: x[1], reverse=True)
        rec_ids = [jid for jid, _ in scored[:k]]

        p_sum += precision_at_k(rec_ids, relevant, k)
        r_sum += recall_at_k(rec_ids, relevant, k)
        n_sum += ndcg_at_k(rec_ids, relevant, k)
        n_users += 1

    if n_users == 0:
        print("No evaluable users. Check model training and split settings.")
        return

    print(f"LightGCN Evaluation (users={n_users}, K={k}, threshold={like_threshold}, holdout={holdout_per_user})")
    print(f"Precision@{k}: {p_sum / n_users:.4f}")
    print(f"Recall@{k}:    {r_sum / n_users:.4f}")
    print(f"NDCG@{k}:      {n_sum / n_users:.4f}")


if __name__ == "__main__":
    main()