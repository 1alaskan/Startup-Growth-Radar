"""Reference rankers used as floors for the XGBoost model.

Each baseline implements .fit(X, y) (no-op for stateless ones) and
.predict_proba(X) returning a 1D ndarray of scores (higher = more likely).
This matches the slice of the sklearn API the runner needs and keeps the
swap-in / swap-out simple.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RandomRanker:
    """Uniform random scores. Seeded for reproducibility within a fold."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random(len(X))


class JobPostingRanker:
    """Single-feature ranker on trailing job posting count.

    Defaults to the 90-day count column. NaN -> 0 so companies with no jobs
    data get the lowest possible signal.
    """

    def __init__(self, feature: str = "jobs_90d"):
        self.feature = feature

    def fit(self, X, y=None):
        if self.feature not in X.columns:
            raise KeyError(f"JobPostingRanker requires column '{self.feature}'")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return X[self.feature].fillna(0).to_numpy(dtype=float)


class RecencyRanker:
    """Most-recently-funded first.

    Score = -days_since_last_round. Companies with no observed round get a
    very low score so they sort to the bottom.
    """

    def __init__(self, feature: str = "days_since_last_round"):
        self.feature = feature

    def fit(self, X, y=None):
        if self.feature not in X.columns:
            raise KeyError(f"RecencyRanker requires column '{self.feature}'")
        return self

    def predict_proba(self, X) -> np.ndarray:
        v = X[self.feature].to_numpy(dtype=float)
        # NaN treated as "very stale" -> very negative score
        v = np.where(np.isnan(v), np.nanmax(v) + 1e9 if np.isfinite(np.nanmax(v)) else 1e9, v)
        return -v
