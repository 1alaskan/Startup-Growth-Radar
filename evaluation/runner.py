"""Walk-forward evaluation runner.

For each cutoff it:
  1. Builds train features at train_cutoff and test features at cutoff.
  2. Builds train labels for [train_cutoff, train_cutoff+90d] and test labels
     for [cutoff, cutoff+90d].
  3. Fits XGBoost + each baseline on the train split and scores the test split.
  4. Records ranking metrics per (cutoff, model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from evaluation.baselines import JobPostingRanker, RandomRanker, RecencyRanker
from evaluation.features import FeatureSpec, build_features
from evaluation.metrics import evaluate_all
from evaluation.splits import (
    LABEL_HORIZON_DAYS,
    WalkForwardSplit,
    build_labels,
    generate_walk_forward_splits,
)

DEFAULT_KS = (10, 20, 50)


def _make_xgb():
    """Lazy-import so the module imports without xgboost installed."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=0,
    )


@dataclass
class FoldResult:
    cutoff: pd.Timestamp
    n_train: int
    n_test: int
    base_rate: float
    metrics: Dict[str, Dict[str, float]]  # model_name -> metric_name -> value
    scores: pd.DataFrame                  # company_id, as_of_date, model, score, label


def _drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ("company_id", "as_of_date") if c in df.columns]
    return df.drop(columns=drop)


def _run_one_fold(
    split: WalkForwardSplit,
    spec: FeatureSpec,
    funding_events: pd.DataFrame,
    ks: Iterable[int],
    label_horizon_days: int,
) -> FoldResult:
    # Train uses an earlier as-of date so its label window is fully observed at T.
    train_feats = build_features(split.train_cutoff, spec)
    test_feats = build_features(split.cutoff, spec)

    train_labels = build_labels(
        train_feats["company_id"],
        funding_events,
        window_start=split.train_cutoff,
        window_end=split.train_cutoff + pd.Timedelta(days=label_horizon_days),
    )
    test_labels = build_labels(
        test_feats["company_id"],
        funding_events,
        window_start=split.cutoff,
        window_end=split.label_window_end,
    )

    X_train = _drop_non_features(train_feats.set_index("company_id"))
    y_train = train_labels.loc[X_train.index, "label"]
    X_test = _drop_non_features(test_feats.set_index("company_id"))
    y_test = test_labels.loc[X_test.index, "label"]

    # Align columns in case feature filtering produced different sets
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    models = {
        "random": RandomRanker(seed=int(split.cutoff.value % (2**31))),
        "job_posting": JobPostingRanker(),
        "recency": RecencyRanker(),
    }
    try:
        models["xgboost"] = _make_xgb()
    except ImportError:
        # xgboost not installed; the harness still works for baselines
        pass

    fold_metrics: Dict[str, Dict[str, float]] = {}
    score_rows: List[pd.DataFrame] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X_test)
            # XGBClassifier returns (n, 2); baselines return (n,)
            scores = raw[:, 1] if getattr(raw, "ndim", 1) == 2 else raw
        else:
            scores = model.decision_function(X_test)

        scores = pd.Series(scores, index=X_test.index, name="score")
        fold_metrics[name] = evaluate_all(scores, y_test, ks=ks)
        score_rows.append(
            pd.DataFrame(
                {
                    "company_id": X_test.index,
                    "as_of_date": split.cutoff,
                    "model": name,
                    "score": scores.values,
                    "label": y_test.values,
                }
            )
        )

    return FoldResult(
        cutoff=split.cutoff,
        n_train=len(X_train),
        n_test=len(X_test),
        base_rate=float(y_test.mean()),
        metrics=fold_metrics,
        scores=pd.concat(score_rows, ignore_index=True),
    )


def run_evaluation(
    cutoffs: Iterable[pd.Timestamp],
    spec: FeatureSpec,
    funding_events: pd.DataFrame,
    ks: Iterable[int] = DEFAULT_KS,
    label_horizon_days: int = LABEL_HORIZON_DAYS,
) -> Dict[str, pd.DataFrame]:
    """Run the full walk-forward backtest.

    Returns a dict with:
      - 'per_fold' : tidy frame (cutoff, model, metric, value)
      - 'summary'  : pivot of mean & std across folds per (model, metric)
      - 'scores'   : concatenated score history for downstream calibration
                     and lead-time analysis
    """
    splits = generate_walk_forward_splits(cutoffs, spec.spine)
    fold_results = [_run_one_fold(s, spec, funding_events, ks, label_horizon_days) for s in splits]

    rows = []
    score_frames = []
    for fr in fold_results:
        score_frames.append(fr.scores)
        for model_name, m in fr.metrics.items():
            for metric_name, v in m.items():
                rows.append(
                    {
                        "cutoff": fr.cutoff,
                        "model": model_name,
                        "metric": metric_name,
                        "value": v,
                        "n_train": fr.n_train,
                        "n_test": fr.n_test,
                    }
                )
    per_fold = pd.DataFrame(rows)

    summary = (
        per_fold.groupby(["model", "metric"])["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    return {
        "per_fold": per_fold,
        "summary": summary,
        "scores": pd.concat(score_frames, ignore_index=True),
    }
