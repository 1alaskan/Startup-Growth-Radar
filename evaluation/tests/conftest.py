"""Synthetic fixtures for evaluation tests.

Builds a small, deterministic universe (20 companies, ~2 years of activity)
that exercises every dynamic source. Kept self-contained so tests don't
depend on the production parquet files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.features import FeatureSpec


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def spine():
    n = 20
    return pd.DataFrame(
        {
            "company_id": [f"c{i:02d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "founded_date": pd.to_datetime(
                ["2020-01-01", "2020-06-15", "2021-01-10", "2021-04-01"] * 5
            ),
            "stage": (["seed"] * 10) + (["series_a"] * 10),
        }
    )


@pytest.fixture(scope="session")
def job_postings(spine, rng):
    rows = []
    for cid in spine["company_id"]:
        n_posts = int(rng.integers(0, 30))
        for _ in range(n_posts):
            rows.append(
                {
                    "company_id": cid,
                    "posted_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 800))),
                    "role": rng.choice(["eng", "sales", "ops"]),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def github_events(spine, rng):
    rows = []
    for cid in spine["company_id"][:15]:
        for d in pd.date_range("2022-01-01", "2024-01-01", freq="30D"):
            rows.append(
                {
                    "company_id": cid,
                    "event_date": d,
                    "commits": int(rng.integers(0, 50)),
                    "stars": int(rng.integers(0, 200)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def news_events(spine, rng):
    rows = []
    for cid in spine["company_id"][:10]:
        for _ in range(int(rng.integers(0, 5))):
            rows.append(
                {
                    "company_id": cid,
                    "published_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 400))),
                    "source": "newsapi",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def form_d(spine, rng):
    rows = []
    for cid in spine["company_id"][::3]:
        rows.append(
            {
                "company_id": cid,
                "filing_date": pd.Timestamp("2023-06-01") + pd.Timedelta(days=int(rng.integers(-200, 200))),
                "amount": float(rng.integers(500_000, 10_000_000)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def google_trends(spine, rng):
    rows = []
    for cid in spine["company_id"]:
        for w in pd.date_range("2022-01-01", "2024-06-01", freq="W"):
            rows.append({"company_id": cid, "week": w, "score": float(rng.integers(0, 100))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def funding_events(spine, rng):
    # Half the universe raises a follow-on at a deterministic offset.
    rows = []
    for i, cid in enumerate(spine["company_id"]):
        if i % 2 == 0:
            rows.append(
                {
                    "company_id": cid,
                    "round_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i * 30),
                    "amount": 5_000_000.0,
                    "round_type": "series_a",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def spec(spine, job_postings, github_events, news_events, form_d, google_trends, funding_events):
    return FeatureSpec(
        spine=spine,
        job_postings=job_postings,
        github_events=github_events,
        news_events=news_events,
        form_d=form_d,
        google_trends=google_trends,
        funding_events=funding_events,
    )
