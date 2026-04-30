# Evaluation Harness

Walk-forward backtest for the startup momentum prediction pipeline. Runs
parallel to the production training code and never touches it.

## Layout

```
evaluation/
  splits.py          walk-forward fold generator + label builder
  features.py        as-of-date feature builder (pandas, port-friendly to PySpark)
  metrics.py         precision/recall/lift @ k, PR-AUC, median lead time
  baselines.py       random / single-feature / recency rankers
  runner.py          loops over folds, fits xgb + baselines, returns tidy frames
  tests/             pytest suite with pre/post-condition style checks
  notebooks/
    calibration.qmd  reliability diagrams + Platt vs isotonic
    results.qmd      headline table, lead-time distribution, per-fold stability
```

## Quick start

```python
import pandas as pd
from evaluation.features import FeatureSpec
from evaluation.runner import run_evaluation

spec = FeatureSpec(
    spine=pd.read_parquet("companies_clean.parquet"),
    job_postings=pd.read_parquet("jobspy.parquet"),
    github_events=pd.read_parquet("github_activity.parquet"),
    news_events=pd.read_parquet("news.parquet"),
    form_d=pd.read_parquet("sec_edgar.parquet"),
    google_trends=pd.read_parquet("google_trends.parquet"),
    funding_events=pd.read_parquet("funding_events.parquet"),
)

cutoffs = pd.date_range("2023-01-01", periods=6, freq="3MS")
out = run_evaluation(cutoffs, spec, spec.funding_events)

out["summary"]   # mean/std of each metric per model
out["per_fold"]  # tidy (cutoff, model, metric, value)
out["scores"]    # long-form score history -> calibration / lead-time
```

## Tests

```
python -m pytest evaluation/tests -q
```

The test suite uses synthetic fixtures so it runs without any of the
production parquet files.

## Notes for porting feature builder to PySpark

`features.py` keeps every aggregation as a flat groupby on rows already
filtered by an as-of cutoff. The transformation pattern is

  `source.where(ts <= as_of).groupBy(company_id).agg(...)`

which is a 1:1 PySpark translation. Keep the same `FeatureSpec` keys when
porting; the runner only depends on the resulting frame schema, not the
pandas-specific internals.
