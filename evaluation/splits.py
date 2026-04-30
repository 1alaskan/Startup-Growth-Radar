"""Walk-forward split generator.

For each cutoff T:
  - train rows are companies whose label horizon [T_train, T_train+90d] ended
    at or before T (so labels are observable),
  - test rows are companies labeled on whether they raised in [T, T+90d],
  - dynamic features are filtered to <= T (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, List

import pandas as pd

LABEL_HORIZON_DAYS = 90


@dataclass(frozen=True)
class WalkForwardSplit:
    """One fold of a walk-forward backtest.

    Attributes
    ----------
    cutoff : pd.Timestamp
        As-of date T. Features are computed using data observable at <= T.
    label_window_end : pd.Timestamp
        T + 90 days. Test companies are labeled on whether they raised in
        [T, label_window_end].
    train_cutoff : pd.Timestamp
        Train fold uses cutoff T - 90 days so train labels are observable
        at scoring time T. Train features must use data <= train_cutoff.
    company_ids : pd.Index
        Set of companies eligible to score (must exist on or before T).
    """

    cutoff: pd.Timestamp
    label_window_end: pd.Timestamp
    train_cutoff: pd.Timestamp
    company_ids: pd.Index


def generate_walk_forward_splits(
    cutoffs: Iterable[pd.Timestamp],
    spine: pd.DataFrame,
    label_horizon_days: int = LABEL_HORIZON_DAYS,
    company_id_col: str = "company_id",
    founded_col: str = "founded_date",
) -> List[WalkForwardSplit]:
    """Build a list of walk-forward splits from a set of cutoffs.

    Eligible companies for a fold are those founded on or before the cutoff.
    """
    cutoffs = [pd.Timestamp(c).normalize() for c in cutoffs]
    horizon = timedelta(days=label_horizon_days)

    if founded_col in spine.columns:
        founded = pd.to_datetime(spine[founded_col], errors="coerce")
    else:
        founded = pd.Series(pd.NaT, index=spine.index)

    splits: List[WalkForwardSplit] = []
    for t in cutoffs:
        # company is eligible if it existed on or before T (or founded date unknown)
        eligible_mask = founded.isna() | (founded <= t)
        ids = spine.loc[eligible_mask, company_id_col]
        splits.append(
            WalkForwardSplit(
                cutoff=t,
                label_window_end=t + horizon,
                train_cutoff=t - horizon,
                company_ids=pd.Index(ids.unique()),
            )
        )
    return splits


def build_labels(
    company_ids: pd.Index,
    funding_events: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    company_id_col: str = "company_id",
    date_col: str = "round_date",
) -> pd.DataFrame:
    """Label companies on whether they raised a follow-on round in a window.

    Returns a DataFrame indexed by company_id with columns:
      - label : 0/1
      - first_round_date : pd.Timestamp or NaT (date of first qualifying round)
    """
    fe = funding_events[[company_id_col, date_col]].copy()
    fe[date_col] = pd.to_datetime(fe[date_col], errors="coerce")
    in_window = fe[(fe[date_col] >= window_start) & (fe[date_col] <= window_end)]

    first = in_window.groupby(company_id_col)[date_col].min()
    out = pd.DataFrame(index=company_ids)
    out.index.name = company_id_col
    out["first_round_date"] = first.reindex(out.index)
    out["label"] = out["first_round_date"].notna().astype(int)
    return out
