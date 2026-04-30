"""Ranking and timing metrics for momentum prediction.

All ranking metrics treat higher score as more likely to raise. NaN scores are
ranked last. Ties are broken arbitrarily (pandas rank with method='first').
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _top_k_mask(scores: pd.Series, k: int) -> pd.Series:
    """Boolean mask: True for the top-k scored rows."""
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(scores)
    k = min(k, n)
    ranks = scores.rank(method="first", ascending=False, na_option="bottom")
    return ranks <= k


def precision_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    """Fraction of the top-k scored items that are positive."""
    mask = _top_k_mask(scores, k)
    if mask.sum() == 0:
        return float("nan")
    return float(labels[mask].mean())


def recall_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    """Fraction of all positives captured in the top-k."""
    total_pos = int(labels.sum())
    if total_pos == 0:
        return float("nan")
    mask = _top_k_mask(scores, k)
    return float(labels[mask].sum() / total_pos)


def lift_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    """precision_at_k divided by base rate (labels.mean())."""
    base = float(labels.mean())
    if base == 0:
        return float("nan")
    return precision_at_k(scores, labels, k) / base


def pr_auc(scores: pd.Series, labels: pd.Series) -> float:
    """Area under the precision-recall curve (sklearn average_precision)."""
    s = scores.fillna(scores.min() - 1.0) if scores.isna().any() else scores
    return float(average_precision_score(labels.values, s.values))


def median_lead_time(
    score_history: pd.DataFrame,
    funding_dates: pd.Series,
    threshold: float,
    score_col: str = "score",
    date_col: str = "as_of_date",
    company_id_col: str = "company_id",
) -> float:
    """Median days between first above-threshold score and the actual round.

    Parameters
    ----------
    score_history : DataFrame
        Long-form (company_id, as_of_date, score) covering multiple cutoffs.
    funding_dates : Series
        Indexed by company_id, value = actual funding round date (the true
        positive event); only true positives should appear here.
    threshold : float
        Score threshold considered "above" - typically the score of the k-th
        highest item, or a calibrated probability cutoff.

    Returns
    -------
    float
        Median lead time in days. NaN if no true positives crossed threshold
        before their funding date.
    """
    h = score_history[[company_id_col, date_col, score_col]].copy()
    h[date_col] = pd.to_datetime(h[date_col])
    above = h[h[score_col] >= threshold]
    first_above = above.groupby(company_id_col)[date_col].min()

    fd = pd.to_datetime(funding_dates)
    joined = first_above.to_frame("first_above").join(fd.rename("round_date"), how="inner")
    # only count cases where the model fired BEFORE the round actually happened
    joined = joined[joined["first_above"] <= joined["round_date"]]
    if joined.empty:
        return float("nan")
    leads = (joined["round_date"] - joined["first_above"]).dt.days
    return float(leads.median())


def evaluate_all(
    scores: pd.Series,
    labels: pd.Series,
    ks: Iterable[int] = (10, 20, 50),
) -> dict:
    """Compute the standard metric bundle for a single fold."""
    out = {"pr_auc": pr_auc(scores, labels), "base_rate": float(labels.mean())}
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(scores, labels, k)
        out[f"recall_at_{k}"] = recall_at_k(scores, labels, k)
        out[f"lift_at_{k}"] = lift_at_k(scores, labels, k)
    return out
