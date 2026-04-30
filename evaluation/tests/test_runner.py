"""Smoke tests for the walk-forward runner.

We don't assert XGBoost beats baselines on synthetic data; we just check the
shapes, no-leakage invariants, and that the per-fold/summary frames carry
the expected metric coverage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.runner import run_evaluation


# PRE: spec has spine + dynamic sources, funding events span the test window
# POST: per_fold frame covers every (cutoff, model, metric) cell
def test_run_evaluation_shapes(spec):
    cutoffs = pd.date_range("2023-04-01", periods=3, freq="3MS")
    out = run_evaluation(cutoffs, spec, spec.funding_events)

    assert set(out.keys()) == {"per_fold", "summary", "scores"}
    pf = out["per_fold"]

    # POST: every fold produced metrics for at least the three baselines
    models = set(pf["model"].unique())
    assert {"random", "job_posting", "recency"}.issubset(models)

    # POST: per_fold has rows = folds * models * metrics
    expected_metrics_per_model = {
        "pr_auc",
        "base_rate",
        "precision_at_10",
        "precision_at_20",
        "precision_at_50",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "lift_at_10",
        "lift_at_20",
        "lift_at_50",
    }
    actual = set(pf["metric"].unique())
    assert expected_metrics_per_model.issubset(actual)

    # POST: summary has mean/std columns
    assert {"mean", "std"}.issubset(set(out["summary"].columns))

    # POST: scores frame is keyed (company_id, as_of_date, model)
    s = out["scores"]
    assert s.duplicated(subset=["company_id", "as_of_date", "model"]).sum() == 0


# POST: scores frame's as_of_date never exceeds cutoff (no test leakage)
def test_runner_respects_cutoff(spec):
    cutoffs = [pd.Timestamp("2023-07-01")]
    out = run_evaluation(cutoffs, spec, spec.funding_events)
    assert (out["scores"]["as_of_date"] <= cutoffs[0]).all()
