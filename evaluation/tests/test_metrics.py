"""Tests for ranking and lead-time metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    evaluate_all,
    lift_at_k,
    median_lead_time,
    pr_auc,
    precision_at_k,
    recall_at_k,
)


# PRE: scores are unique, k <= n
# POST: precision_at_k counts only the top-k items
def test_precision_at_k_basic():
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = pd.Series([1, 1, 0, 0, 1])
    assert precision_at_k(scores, labels, 2) == 1.0
    assert precision_at_k(scores, labels, 5) == 0.6


# POST: recall_at_k <= 1, recall at n equals 1 when there are positives
def test_recall_at_k_full():
    scores = pd.Series([0.1, 0.2, 0.3, 0.4])
    labels = pd.Series([1, 0, 1, 0])
    assert recall_at_k(scores, labels, 4) == 1.0
    # top-1 by score is index 3 (label 0), so zero positives are captured
    assert recall_at_k(scores, labels, 1) == 0.0
    # top-2 captures index 3 (0) and index 2 (1) -> half the positives
    assert recall_at_k(scores, labels, 2) == 0.5


# POST: lift_at_k = precision_at_k / base_rate
def test_lift_at_k_matches_definition():
    scores = pd.Series(np.linspace(0, 1, 10))
    labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    base = labels.mean()
    assert lift_at_k(scores, labels, 5) == pytest.approx(precision_at_k(scores, labels, 5) / base)


# POST: pr_auc on a perfect ranking equals 1
def test_pr_auc_perfect():
    scores = pd.Series([5, 4, 3, 2, 1])
    labels = pd.Series([1, 1, 0, 0, 0])
    assert pr_auc(scores, labels) == pytest.approx(1.0)


# POST: pr_auc on a random ranking ~ base rate
def test_pr_auc_inverted_low():
    scores = pd.Series([1, 2, 3, 4, 5])
    labels = pd.Series([1, 1, 0, 0, 0])
    # ranking is inverted, so PR-AUC should be far below 1
    assert pr_auc(scores, labels) < 0.6


# PRE: scores rise toward the funding date
# POST: lead time equals the gap between the first crossing and the round date
def test_median_lead_time_simple():
    history = pd.DataFrame(
        {
            "company_id": ["a"] * 4,
            "as_of_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "score": [0.1, 0.2, 0.6, 0.7],
        }
    )
    funding = pd.Series(
        {"a": pd.Timestamp("2024-05-15")}, name="round_date"
    )
    # threshold 0.5 first crossed on 2024-03-01; round on 2024-05-15 -> 75 days
    out = median_lead_time(history, funding, threshold=0.5)
    assert out == 75


# POST: NaN when nothing crosses the threshold before the event
def test_median_lead_time_no_crossings():
    history = pd.DataFrame(
        {"company_id": ["a"], "as_of_date": [pd.Timestamp("2024-01-01")], "score": [0.1]}
    )
    funding = pd.Series({"a": pd.Timestamp("2024-02-01")})
    assert np.isnan(median_lead_time(history, funding, threshold=0.9))


def test_evaluate_all_keys():
    scores = pd.Series(np.linspace(0, 1, 100))
    labels = pd.Series((scores > 0.5).astype(int))
    out = evaluate_all(scores, labels, ks=(10, 20, 50))
    for k in (10, 20, 50):
        for prefix in ("precision_at_", "recall_at_", "lift_at_"):
            assert f"{prefix}{k}" in out
    assert "pr_auc" in out and "base_rate" in out
