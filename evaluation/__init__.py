"""Walk-forward evaluation harness for the startup momentum pipeline.

Parallel module that consumes the same feature definitions as the production
training pipeline but runs offline backtests with strict temporal filtering.
"""

from evaluation.splits import WalkForwardSplit, generate_walk_forward_splits
from evaluation.features import build_features, FeatureSpec
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    lift_at_k,
    pr_auc,
    median_lead_time,
)
from evaluation.baselines import RandomRanker, JobPostingRanker, RecencyRanker
from evaluation.runner import run_evaluation

__all__ = [
    "WalkForwardSplit",
    "generate_walk_forward_splits",
    "build_features",
    "FeatureSpec",
    "precision_at_k",
    "recall_at_k",
    "lift_at_k",
    "pr_auc",
    "median_lead_time",
    "RandomRanker",
    "JobPostingRanker",
    "RecencyRanker",
    "run_evaluation",
]
