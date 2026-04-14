from __future__ import annotations

"""
Computes ranking metrics for Top-K recommendation.
Includes Precision@K, Recall@K, and NDCG@K for binary relevance.
"""


# AI-assisted file:
# ChatGPT was used to help structure and implement this metrics file.
# Prompt summary: "Help me write a simple Python metrics file for Top-K recommendation,
# including Precision@K, Recall@K, and NDCG@K."
from typing import List, Set
import math


# ---------------------------------------------------------
# 1. Precision@K
# Measures how many top-k recommendations are relevant.
# ---------------------------------------------------------
def precision_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    # Return zero if k is invalid
    if k <= 0:
        return 0.0

    # Keep only the top-k recommendations
    topk = recs[:k]

    # Return zero if there are no recommendations
    if not topk:
        return 0.0

    # Count how many recommended items are relevant
    hits = sum(1 for jid in topk if jid in relevant)

    # Divide hits by k
    return hits / float(k)


# ---------------------------------------------------------
# 2. Recall@K
# Measures how many relevant items appear in the top-k list.
# ---------------------------------------------------------
def recall_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    # Return zero if there are no relevant items
    if not relevant:
        return 0.0

    # Keep only the top-k recommendations
    topk = recs[:k]

    # Count how many recommended items are relevant
    hits = sum(1 for jid in topk if jid in relevant)

    # Divide hits by the total number of relevant items
    return hits / float(len(relevant))


# ---------------------------------------------------------
# 3. NDCG@K
# Measures ranking quality by rewarding relevant items near the top.
# ---------------------------------------------------------
def ndcg_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    # Keep only the top-k recommendations
    topk = recs[:k]

    # Compute discounted cumulative gain for a ranked list
    def dcg(items: List[int]) -> float:
        # Start the DCG score at zero
        score = 0.0

        # Loop through ranked items starting from position 1
        for i, jid in enumerate(items, start=1):
            # Add discounted gain if the item is relevant
            if jid in relevant:
                score += 1.0 / math.log2(i + 1)

        return score

    # Compute DCG for the recommended ranking
    dcg_val = dcg(topk)

    # Build an ideal ranking with relevant items first
    ideal = list(relevant)[:k]

    # Compute DCG for the ideal ranking
    idcg_val = dcg(ideal)

    # Return zero if the ideal score is zero
    if idcg_val == 0.0:
        return 0.0

    # Normalize DCG by the ideal DCG
    return dcg_val / idcg_val