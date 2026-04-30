"""Tests for the as-of-date feature builder.

The single most important property is "no future leakage": for any chosen
as_of_date, the aggregated features must be identical to those computed on
a copy of the source data with all rows after as_of_date deleted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.features import FeatureSpec, build_features


def _strip_future(df: pd.DataFrame, col: str, as_of: pd.Timestamp) -> pd.DataFrame:
    return df[pd.to_datetime(df[col]) <= as_of].copy()


# PRE: spine has company_id; sources have their declared timestamp columns.
# POST: features only depend on rows with timestamp <= as_of_date.
def test_no_future_leakage(spec):
    as_of = pd.Timestamp("2023-06-01")

    full = build_features(as_of, spec)

    censored = FeatureSpec(
        spine=spec.spine,
        job_postings=_strip_future(spec.job_postings, "posted_date", as_of),
        github_events=_strip_future(spec.github_events, "event_date", as_of),
        news_events=_strip_future(spec.news_events, "published_date", as_of),
        form_d=_strip_future(spec.form_d, "filing_date", as_of),
        google_trends=_strip_future(spec.google_trends, "week", as_of),
        funding_events=_strip_future(spec.funding_events, "round_date", as_of),
    )
    censored_features = build_features(as_of, censored)

    # POST: feeding the future to build_features does not change its output
    pd.testing.assert_frame_equal(
        full.sort_values("company_id").reset_index(drop=True),
        censored_features.sort_values("company_id").reset_index(drop=True),
    )


# PRE: as_of_date is later than all events
# POST: jobs_total equals total rows in source (no events were silently dropped)
def test_full_history_aggregation(spec):
    as_of = pd.Timestamp("2030-01-01")
    feats = build_features(as_of, spec)

    expected_total = (
        spec.job_postings.groupby("company_id").size().reindex(feats["company_id"]).fillna(0)
    )
    pd.testing.assert_series_equal(
        feats.set_index("company_id")["jobs_total"].astype(int).sort_index(),
        expected_total.astype(int).sort_index(),
        check_names=False,
    )


# PRE: as_of_date is before any event
# POST: every dynamic feature is zero (no rows pass the as-of filter)
def test_pre_history_yields_zeros(spec):
    as_of = pd.Timestamp("2000-01-01")
    feats = build_features(as_of, spec)
    # nothing eligible because all founded_dates are after 2000
    assert feats.empty or (feats.drop(columns=["company_id", "as_of_date"]).sum().sum() == 0)


# POST: output is uniquely keyed by company_id
def test_unique_key(spec):
    as_of = pd.Timestamp("2024-01-01")
    feats = build_features(as_of, spec)
    assert feats["company_id"].is_unique


# POST: as_of_date column is set on every row
def test_as_of_date_stamped(spec):
    as_of = pd.Timestamp("2024-01-01")
    feats = build_features(as_of, spec)
    assert (feats["as_of_date"] == as_of).all()


def test_missing_spine_column_raises(spec):
    bad = FeatureSpec(spine=spec.spine.drop(columns=["company_id"]))
    with pytest.raises(ValueError, match="company_id"):
        build_features(pd.Timestamp("2024-01-01"), bad)
