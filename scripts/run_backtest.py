"""One-shot script to run the walk-forward backtest on real data.

Usage (from the repo root):

    python scripts/run_backtest.py

Outputs three parquet files into evaluation_artifacts/ that you then upload
to s3://startup-momentum-pipeline/evaluation/. Edit DATA_DIR and the
COLUMN_MAP below to match your local files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the evaluation package importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.features import FeatureSpec
from evaluation.runner import run_evaluation


# ── Configuration: edit these to match your local data ───────────────────────

# Folder on your hard drive where all the cleaned parquets live.
# Adjust this to your actual path.
DATA_DIR = Path(
    r"C:\Users\spink\OneDrive\Desktop\Personal Projects"
    r"\Startup Momentum Prediction Pipeline\Data"
)

# Filename for each source. Set to None for sources you don't have.
FILES = {
    "spine":          "companies_clean.parquet",
    "job_postings":   "job_postings_cleaned.parquet",
    "github_events":  "github_activity_cleaned.parquet",
    "news_events":    "news_gdelt_articles.parquet",
    "form_d":         "sec_edgar_filings.parquet",
    "google_trends":  "google_trends_cleaned.parquet",
    "funding_events": None,  # often a derived column on the spine; see below
}

# The feature builder expects a fixed schema. If your column names differ,
# rename them here. Left side = expected name, right side = name in your file.
COLUMN_MAP = {
    "job_postings": {"company_id": "company_id", "posted_date": "posted_date"},
    "github_events": {
        "company_id": "company_id",
        "event_date": "event_date",
        "commits": "commits",
        "stars": "stars",
    },
    "news_events": {"company_id": "company_id", "published_date": "published_date"},
    "form_d": {"company_id": "company_id", "filing_date": "filing_date"},
    "google_trends": {"company_id": "company_id", "week": "week", "score": "score"},
    "funding_events": {"company_id": "company_id", "round_date": "round_date"},
}

# Six quarterly cutoffs, latest one is 90 days ago so its labels are
# observable today. Adjust if your data history is shorter.
CUTOFFS = pd.date_range(end=pd.Timestamp.today() - pd.Timedelta(days=90),
                        periods=6, freq="-3MS")[::-1]

OUT_DIR = ROOT / "evaluation_artifacts"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load(name: str) -> pd.DataFrame | None:
    """Load a source parquet, applying COLUMN_MAP renames."""
    fname = FILES.get(name)
    if not fname:
        print(f"  [skip] {name}: no file configured")
        return None
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  [skip] {name}: {path} not found")
        return None
    df = pd.read_parquet(path)
    # rename: COLUMN_MAP[name] is {expected_col: actual_col_in_file}
    rename = {actual: expected for expected, actual in COLUMN_MAP.get(name, {}).items()}
    df = df.rename(columns=rename)
    print(f"  [ok]   {name}: {len(df):,} rows from {fname}")
    return df


def derive_funding_events_from_spine(spine: pd.DataFrame) -> pd.DataFrame | None:
    """If we don't have a separate funding_events file, fall back to spine columns.

    The spine carries last_funding_date / last_funding_type per company, which
    is enough to label a single follow-on round. For multiple rounds per
    company you'll want a real funding events table.
    """
    if "last_funding_date" not in spine.columns:
        return None
    fe = spine[["company_id", "last_funding_date"]].copy()
    fe = fe.rename(columns={"last_funding_date": "round_date"})
    fe["round_date"] = pd.to_datetime(fe["round_date"], errors="coerce")
    fe = fe.dropna(subset=["round_date"])
    print(f"  [ok]   funding_events (derived from spine): {len(fe):,} rows")
    return fe


def main() -> int:
    print(f"Loading sources from: {DATA_DIR}")
    if not DATA_DIR.exists():
        print(f"ERROR: DATA_DIR does not exist. Edit scripts/run_backtest.py.")
        return 1

    spine = load("spine")
    if spine is None or spine.empty:
        print("ERROR: spine is required.")
        return 1
    if "company_id" not in spine.columns:
        print("ERROR: spine must contain a 'company_id' column. "
              "Edit COLUMN_MAP['spine'] if your file uses a different name.")
        return 1

    spec = FeatureSpec(
        spine=spine,
        job_postings=load("job_postings"),
        github_events=load("github_events"),
        news_events=load("news_events"),
        form_d=load("form_d"),
        google_trends=load("google_trends"),
        funding_events=load("funding_events") if FILES.get("funding_events")
                       else derive_funding_events_from_spine(spine),
    )

    if spec.funding_events is None or spec.funding_events.empty:
        print("ERROR: no funding events available. Cannot build labels.")
        return 1

    print(f"\nCutoffs ({len(CUTOFFS)}):")
    for c in CUTOFFS:
        print(f"  {c.date()}")

    print("\nRunning backtest...")
    out = run_evaluation(CUTOFFS, spec, spec.funding_events)

    OUT_DIR.mkdir(exist_ok=True)
    out["per_fold"].to_parquet(OUT_DIR / "per_fold.parquet")
    out["summary"].to_parquet(OUT_DIR / "summary.parquet")
    out["scores"].to_parquet(OUT_DIR / "scores.parquet")

    print(f"\nWrote 3 files to {OUT_DIR}")
    print("\nNext: upload them to s3://startup-momentum-pipeline/evaluation/")
    print("Then click 'Refresh data from S3' in the dashboard sidebar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
