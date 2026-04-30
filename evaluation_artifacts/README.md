# Evaluation artifacts

Drop the three parquet files produced by `evaluation.runner.run_evaluation`
here, then upload to S3 under the `evaluation/` prefix.

Expected files:
- `per_fold.parquet`  — tidy (cutoff, model, metric, value)
- `summary.parquet`   — mean/std/min/max per (model, metric)
- `scores.parquet`    — long-form (company_id, as_of_date, model, score, label)

## Generating

```python
import pandas as pd
from evaluation.features import FeatureSpec
from evaluation.runner import run_evaluation

spec = FeatureSpec(
    spine=pd.read_parquet("path/to/companies_clean.parquet"),
    job_postings=pd.read_parquet("path/to/jobspy.parquet"),
    github_events=pd.read_parquet("path/to/github_activity.parquet"),
    news_events=pd.read_parquet("path/to/news.parquet"),
    form_d=pd.read_parquet("path/to/sec_edgar.parquet"),
    google_trends=pd.read_parquet("path/to/google_trends.parquet"),
    funding_events=pd.read_parquet("path/to/funding_events.parquet"),
)

cutoffs = pd.date_range("2023-01-01", periods=6, freq="3MS")
out = run_evaluation(cutoffs, spec, spec.funding_events)

out["per_fold"].to_parquet("evaluation_artifacts/per_fold.parquet")
out["summary"].to_parquet("evaluation_artifacts/summary.parquet")
out["scores"].to_parquet("evaluation_artifacts/scores.parquet")
```

## Uploading to S3

```bash
aws s3 cp evaluation_artifacts/per_fold.parquet s3://startup-momentum-pipeline/evaluation/
aws s3 cp evaluation_artifacts/summary.parquet  s3://startup-momentum-pipeline/evaluation/
aws s3 cp evaluation_artifacts/scores.parquet   s3://startup-momentum-pipeline/evaluation/
```

After upload, click "🔄 Refresh data from S3" in the dashboard sidebar.
