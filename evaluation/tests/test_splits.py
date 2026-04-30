"""Tests for walk-forward split generation."""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.splits import (
    LABEL_HORIZON_DAYS,
    build_labels,
    generate_walk_forward_splits,
)


# PRE-CONDITIONS: cutoffs sortable, spine has company_id and founded_date.
# POST-CONDITIONS: every fold's label_window_end == cutoff + 90d, eligible
# universe is monotonically non-decreasing across cutoffs.
def test_splits_basic_shape(spine):
    cutoffs = pd.date_range("2023-01-01", periods=6, freq="3MS")
    splits = generate_walk_forward_splits(cutoffs, spine)

    assert len(splits) == 6
    for s, c in zip(splits, cutoffs):
        # POST: label window is exactly the configured horizon
        assert s.cutoff == c
        assert (s.label_window_end - s.cutoff).days == LABEL_HORIZON_DAYS
        assert (s.cutoff - s.train_cutoff).days == LABEL_HORIZON_DAYS

    # POST: eligible universe never shrinks as time advances
    sizes = [len(s.company_ids) for s in splits]
    assert sizes == sorted(sizes)


def test_eligibility_excludes_unfounded_companies(spine):
    # PRE: pick a cutoff before every founded_date
    cutoffs = [pd.Timestamp("2019-01-01")]
    splits = generate_walk_forward_splits(cutoffs, spine)
    # POST: nothing is eligible at a cutoff before the earliest founding
    assert len(splits[0].company_ids) == 0


def test_build_labels_window_inclusivity(spine, funding_events):
    ids = pd.Index(spine["company_id"].unique())
    # PRE: window covers all funding events in the fixture
    out = build_labels(
        ids,
        funding_events,
        window_start=pd.Timestamp("2022-01-01"),
        window_end=pd.Timestamp("2025-01-01"),
    )
    # POST: label is 0/1, indexed by company, no extra rows
    assert set(out["label"].unique()).issubset({0, 1})
    assert out.index.equals(ids)
    # half the synthetic universe raised a follow-on (i % 2 == 0)
    assert out["label"].sum() == sum(1 for i in range(len(ids)) if i % 2 == 0)


def test_build_labels_excludes_outside_window(spine, funding_events):
    ids = pd.Index(spine["company_id"].unique())
    # PRE: window before any funding events
    out = build_labels(
        ids,
        funding_events,
        window_start=pd.Timestamp("2010-01-01"),
        window_end=pd.Timestamp("2010-06-01"),
    )
    # POST: zero positives when no event falls inside the window
    assert out["label"].sum() == 0
