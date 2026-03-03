"""
Ranking metrics for Top-K recommendation.

Binary relevance metrics:
- Precision@K
- Recall@K
- NDCG@K
"""

from __future__ import annotations

from typing import List, Set
import math


def precision_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    """Precision@K = (# relevant in top K) / K"""
    if k <= 0:
        return 0.0
    topk = recs[:k]
    if not topk:
        return 0.0
    hits = sum(1 for jid in topk if jid in relevant)
    return hits / float(k)


def recall_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    """Recall@K = (# relevant in top K) / (# relevant total)"""
    if not relevant:
        return 0.0
    topk = recs[:k]
    hits = sum(1 for jid in topk if jid in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(recs: List[int], relevant: Set[int], k: int) -> float:
    """
    NDCG@K for binary relevance.
    DCG = sum_{i=1..K} rel_i / log2(i+1)
    NDCG = DCG / IDCG
    """
    topk = recs[:k]

    def dcg(items: List[int]) -> float:
        score = 0.0
        for i, jid in enumerate(items, start=1):
            if jid in relevant:
                score += 1.0 / math.log2(i + 1)
        return score

    dcg_val = dcg(topk)

    # Ideal ranking: all relevant items first (up to k)
    ideal = list(relevant)[:k]
    idcg_val = dcg(ideal)

    if idcg_val == 0.0:
        return 0.0
    return dcg_val / idcg_val