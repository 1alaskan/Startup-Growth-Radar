"""Tests for baseline rankers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.baselines import JobPostingRanker, RandomRanker, RecencyRanker


# POST: deterministic given a seed; output length matches input
def test_random_ranker_deterministic():
    X = pd.DataFrame({"jobs_90d": np.zeros(50)})
    a = RandomRanker(seed=7).fit(X).predict_proba(X)
    b = RandomRanker(seed=7).fit(X).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 50


# POST: ordering matches the underlying feature
def test_job_posting_ranker_orders_by_feature():
    X = pd.DataFrame({"jobs_90d": [0, 5, 12, 3]})
    scores = JobPostingRanker().fit(X).predict_proba(X)
    np.testing.assert_array_equal(np.argsort(-scores), np.array([2, 1, 3, 0]))


# PRE: required column missing -> raises immediately
def test_job_posting_ranker_requires_column():
    with pytest.raises(KeyError):
        JobPostingRanker().fit(pd.DataFrame({"other": [1]}))


# POST: more recently funded -> higher score
def test_recency_ranker_orders_recent_first():
    X = pd.DataFrame({"days_since_last_round": [10, 100, 5, np.nan]})
    scores = RecencyRanker().fit(X).predict_proba(X)
    # company at index 2 (5 days) should rank first; NaN should rank last
    order = np.argsort(-scores)
    assert order[0] == 2
    assert order[-1] == 3
