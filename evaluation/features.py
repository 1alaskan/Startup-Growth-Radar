"""Feature builder with as-of-date semantics.

Every dynamic source is filtered to events timestamped on or before the
as-of-date before any aggregation runs. The aggregations are kept in plain
pandas groupby form so they map 1:1 to PySpark window/group operations when
the feature builder is ported to EMR Serverless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    """Bundle of source DataFrames used by build_features.

    Each source has a timestamp column that is required for as-of filtering.
    Spine carries the static company metadata and is the join axis.
    """

    spine: pd.DataFrame  # company_id, name, founded_date, stage, ...
    job_postings: Optional[pd.DataFrame] = None  # company_id, posted_date, role
    github_events: Optional[pd.DataFrame] = None  # company_id, event_date, commits, stars
    news_events: Optional[pd.DataFrame] = None    # company_id, published_date, source
    form_d: Optional[pd.DataFrame] = None         # company_id, filing_date, amount
    google_trends: Optional[pd.DataFrame] = None  # company_id, week, score
    funding_events: Optional[pd.DataFrame] = None # company_id, round_date, amount

    # mapping: source name -> timestamp column (used in pre/post-condition checks)
    timestamp_cols: Dict[str, str] = field(
        default_factory=lambda: {
            "job_postings": "posted_date",
            "github_events": "event_date",
            "news_events": "published_date",
            "form_d": "filing_date",
            "google_trends": "week",
            "funding_events": "round_date",
        }
    )


def _filter_as_of(df: Optional[pd.DataFrame], date_col: str, as_of: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Return rows with date_col <= as_of (NaT rows dropped)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out[out[date_col].notna() & (out[date_col] <= as_of)]


def _job_features(df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    # Counts within trailing 30 / 90 day windows + total observed.
    if df is None or df.empty:
        return pd.DataFrame(columns=["company_id", "jobs_30d", "jobs_90d", "jobs_total"])
    df = df.copy()
    df["posted_date"] = pd.to_datetime(df["posted_date"])
    w30 = as_of - timedelta(days=30)
    w90 = as_of - timedelta(days=90)
    g = df.groupby("company_id")
    out = pd.DataFrame(
        {
            "jobs_30d": g.apply(lambda x: (x["posted_date"] >= w30).sum()),
            "jobs_90d": g.apply(lambda x: (x["posted_date"] >= w90).sum()),
            "jobs_total": g.size(),
        }
    ).reset_index()
    return out


def _github_features(df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    # Engineering activity proxies. Use most-recent rolling sums.
    if df is None or df.empty:
        return pd.DataFrame(columns=["company_id", "gh_commits_90d", "gh_stars_latest"])
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    w90 = as_of - timedelta(days=90)
    recent = df[df["event_date"] >= w90]
    commits = recent.groupby("company_id")["commits"].sum().rename("gh_commits_90d")
    stars = (
        df.sort_values("event_date").groupby("company_id")["stars"].last().rename("gh_stars_latest")
    )
    return commits.to_frame().join(stars, how="outer").reset_index()


def _news_features(df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["company_id", "news_30d", "news_90d"])
    df = df.copy()
    df["published_date"] = pd.to_datetime(df["published_date"])
    w30 = as_of - timedelta(days=30)
    w90 = as_of - timedelta(days=90)
    g = df.groupby("company_id")
    return pd.DataFrame(
        {
            "news_30d": g.apply(lambda x: (x["published_date"] >= w30).sum()),
            "news_90d": g.apply(lambda x: (x["published_date"] >= w90).sum()),
        }
    ).reset_index()


def _form_d_features(df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["company_id", "formd_count", "days_since_formd"])
    df = df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    counts = df.groupby("company_id").size().rename("formd_count")
    last = df.groupby("company_id")["filing_date"].max()
    days_since = (as_of - last).dt.days.rename("days_since_formd")
    return counts.to_frame().join(days_since, how="outer").reset_index()


def _trends_features(df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    # Mean trend score over the trailing 12 weeks.
    if df is None or df.empty:
        return pd.DataFrame(columns=["company_id", "trends_12w_mean"])
    df = df.copy()
    df["week"] = pd.to_datetime(df["week"])
    cutoff = as_of - timedelta(weeks=12)
    recent = df[df["week"] >= cutoff]
    mean = recent.groupby("company_id")["score"].mean().rename("trends_12w_mean")
    return mean.reset_index()


def _funding_recency_feature(df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    # Days since last observed funding round at as_of_date.
    if df is None or df.empty:
        return pd.DataFrame(columns=["company_id", "days_since_last_round"])
    df = df.copy()
    df["round_date"] = pd.to_datetime(df["round_date"])
    last = df.groupby("company_id")["round_date"].max()
    days = (as_of - last).dt.days.rename("days_since_last_round")
    return days.reset_index()


def build_features(as_of_date, spec: FeatureSpec) -> pd.DataFrame:
    """Build a per-company feature frame as observed at as_of_date.

    Pre-conditions
    --------------
    - as_of_date is a parseable date.
    - spec.spine has company_id (unique).
    - For every non-None source, its declared timestamp column exists.

    Post-conditions (enforced by tests, not asserted at runtime here for speed)
    --------------------------------------------------------------------------
    - No row in any intermediate filtered source has timestamp > as_of_date.
    - Output is uniquely keyed by company_id.
    - Output rows == number of eligible companies in the spine
      (founded_date <= as_of_date OR null).
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    if "company_id" not in spec.spine.columns:
        raise ValueError("spine must contain a 'company_id' column")

    spine = spec.spine.copy()
    if "founded_date" in spine.columns:
        founded = pd.to_datetime(spine["founded_date"], errors="coerce")
        spine = spine[founded.isna() | (founded <= as_of)].copy()

    # Filter all dynamic sources to <= as_of *before* any aggregation
    jp = _filter_as_of(spec.job_postings, "posted_date", as_of)
    gh = _filter_as_of(spec.github_events, "event_date", as_of)
    nw = _filter_as_of(spec.news_events, "published_date", as_of)
    fd = _filter_as_of(spec.form_d, "filing_date", as_of)
    gt = _filter_as_of(spec.google_trends, "week", as_of)
    fe = _filter_as_of(spec.funding_events, "round_date", as_of)

    parts = [
        spine[["company_id"]].drop_duplicates(),
        _job_features(jp, as_of),
        _github_features(gh, as_of),
        _news_features(nw, as_of),
        _form_d_features(fd, as_of),
        _trends_features(gt, as_of),
        _funding_recency_feature(fe, as_of),
    ]

    out = parts[0]
    for p in parts[1:]:
        if p is None or p.empty:
            continue
        out = out.merge(p, on="company_id", how="left")

    # Numeric fill: missing implies "no observed activity"
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(0)
    out["as_of_date"] = as_of
    return out
