"""
Startup Growth Radar — Streamlit application.

Reads scored data from S3 (refreshed weekly by the AWS pipeline)
and presents a ranked, filterable view of ~846 startups.
"""

import io
from datetime import datetime, timezone

import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit command) ────────────────────────────

st.set_page_config(
    page_title="Startup Growth Radar",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom theme ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Light-blue accent palette */
:root {
    --blue-50: #eff6ff;
    --blue-100: #dbeafe;
    --blue-200: #bfdbfe;
    --blue-400: #60a5fa;
    --blue-500: #3b82f6;
    --blue-600: #2563eb;
    --blue-700: #1d4ed8;
}

/* Header area */
header[data-testid="stHeader"] {
    background: linear-gradient(90deg, var(--blue-600), var(--blue-400));
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--blue-50);
    border: 1px solid var(--blue-200);
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label {
    color: var(--blue-700) !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--blue-50) 0%, #ffffff 100%);
    border-right: 2px solid var(--blue-200);
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--blue-600) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: var(--blue-500) !important;
}

/* Top-right toolbar icons (GitHub, share, etc.) */
ul[data-testid="stToolbar"] {
    gap: 0.75rem !important;
}
ul[data-testid="stToolbar"] button,
ul[data-testid="stToolbar"] a {
    transform: scale(1.4);
}

/* Weekly banner */
.weekly-banner {
    background: linear-gradient(90deg, var(--blue-100), var(--blue-50));
    border-left: 4px solid var(--blue-500);
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 16px;
    font-size: 0.9rem;
    color: var(--blue-700);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────

BUCKET = "startup-momentum-pipeline"

S3_KEYS = {
    "scores": "modeling/company_scores.csv",
    "hiring": "modeling/hiring_friendliness_scores.csv",
    "features": "modeling/features.parquet",
    "metadata": "modeling/feature_metadata.csv",
    "spine": "cleaned/spine_cleaned.parquet",
}

# Walk-forward evaluation artifacts written by evaluation.runner.run_evaluation.
# Tabs that depend on these degrade gracefully when the files don't exist yet.
EVAL_S3_KEYS = {
    "per_fold": "evaluation/per_fold.parquet",
    "summary": "evaluation/summary.parquet",
    "scores": "evaluation/scores.parquet",
}

FUNDING_STAGE_LABELS = {
    0.0: "Pre-Seed / Angel",
    1.0: "Seed",
    2.0: "Series A",
    3.0: "Series B",
    4.0: "Post-IPO",
}

INDUSTRY_MAP = {
    "ind_ai": "AI",
    "ind_software": "Software",
    "ind_it": "IT",
    "ind_saas": "SaaS",
    "ind_healthcare": "Healthcare",
    "ind_fintech": "FinTech",
    "ind_financial": "Financial Services",
    "ind_ml": "Machine Learning",
    "ind_manufacturing": "Manufacturing",
    "ind_biotech": "Biotech",
    "ind_genai": "Generative AI",
    "ind_devtools": "Developer Tools",
}

TIER_COLORS = {
    "Very High": "#22c55e",
    "High": "#84cc16",
    "Moderate": "#eab308",
    "Low": "#f97316",
    "Very Low": "#ef4444",
}

SIGNAL_LABELS = {
    "signal_job_posting": "Job Posting (30%)",
    "signal_funding_recency": "Funding Recency (25%)",
    "signal_headcount_proxy": "Headcount Proxy (20%)",
    "signal_github_activity": "GitHub Activity (15%)",
    "signal_company_trajectory": "Trajectory (10%)",
}

# ── S3 data loading ──────────────────────────────────────────────────────────


def _get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["aws"]["access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["secret_access_key"],
        region_name="us-east-1",
    )


def _resolve_latest_key(key: str, extension: str) -> tuple[str, "datetime | None"]:
    """Return (S3 key, LastModified) of the newest file matching the prefix.

    Handles both flat files at `key` and Spark directory output under `key/`.
    Picks the most recently modified part file so weekly Spark reruns are
    picked up even when prior runs leave older part files behind.
    """
    s3 = _get_s3_client()
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=key)
    contents = resp.get("Contents", [])

    matches = [
        obj for obj in contents
        if obj["Key"].endswith(extension) and "/_" not in obj["Key"]
    ]

    if not matches:
        return key, None

    newest = max(matches, key=lambda o: o["LastModified"])
    return newest["Key"], newest["LastModified"]


@st.cache_data(ttl=3600)
def read_csv_from_s3(key: str) -> tuple[pd.DataFrame, "datetime | None"]:
    """Read newest CSV under `key`, returning (df, last_modified)."""
    target_key, last_modified = _resolve_latest_key(key, ".csv")
    s3 = _get_s3_client()
    obj = s3.get_object(Bucket=BUCKET, Key=target_key)
    return pd.read_csv(io.BytesIO(obj["Body"].read())), last_modified


@st.cache_data(ttl=3600)
def read_parquet_from_s3(key: str) -> tuple[pd.DataFrame, "datetime | None"]:
    """Read newest Parquet under `key`, returning (df, last_modified)."""
    target_key, last_modified = _resolve_latest_key(key, ".parquet")
    s3 = _get_s3_client()
    obj = s3.get_object(Bucket=BUCKET, Key=target_key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read())), last_modified


@st.cache_data(ttl=3600)
def load_all_data() -> dict:
    """Load all datasets from S3 and return as a dictionary."""
    scores, scores_mtime = read_csv_from_s3(S3_KEYS["scores"])
    hiring, _ = read_csv_from_s3(S3_KEYS["hiring"])
    features, _ = read_parquet_from_s3(S3_KEYS["features"])
    metadata, _ = read_csv_from_s3(S3_KEYS["metadata"])
    spine, _ = read_parquet_from_s3(S3_KEYS["spine"])
    return {
        "scores": scores,
        "hiring": hiring,
        "features": features,
        "metadata": metadata,
        "spine": spine,
        "last_updated": scores_mtime,
    }


@st.cache_data(ttl=3600)
def load_evaluation_artifacts() -> dict:
    """Load walk-forward backtest output. Missing files yield None entries."""
    out: dict = {"per_fold": None, "summary": None, "scores": None, "last_updated": None}
    for name, key in EVAL_S3_KEYS.items():
        try:
            df, mtime = read_parquet_from_s3(key)
            out[name] = df
            if out["last_updated"] is None or (mtime and mtime > out["last_updated"]):
                out["last_updated"] = mtime
        except Exception:
            # File not uploaded yet — tab will render an empty-state message.
            out[name] = None
    return out


# ── Data preparation ─────────────────────────────────────────────────────────


def derive_primary_industry(row: pd.Series) -> str:
    """Return the first matching industry label from ind_* columns."""
    for col, label in INDUSTRY_MAP.items():
        if row.get(col, 0) == 1:
            return label
    return "Other"


def format_usd(value) -> str:
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def prepare_master_table(data: dict) -> pd.DataFrame:
    """Merge hiring scores + momentum scores + spine metadata into one table."""
    hiring = data["hiring"].copy()
    scores = data["scores"][
        ["company_id", "momentum_score", "momentum_tier", "momentum_rank"]
    ].copy()

    spine_cols = [
        "company_id", "name", "industries", "hq_location", "description_combined",
        "city", "state", "country", "total_funding_usd", "last_funding_date",
        "last_funding_type", "num_employees", "website", "linkedin",
        "top_investors", "num_investors", "founded_date",
    ]
    spine = data["spine"][
        [c for c in spine_cols if c in data["spine"].columns]
    ].copy()

    features = data["features"].copy()

    # Merge
    df = hiring.merge(scores, on="company_id", how="left")
    df = df.merge(spine, on="company_id", how="left", suffixes=("", "_spine"))

    # Use spine name if hiring name is missing
    if "name_spine" in df.columns:
        df["name"] = df["name"].fillna(df["name_spine"])
        df.drop(columns=["name_spine"], inplace=True)

    # Derive funding stage label from scores data
    scores_full = data["scores"][["company_id", "funding_stage"]].copy()
    df = df.merge(scores_full, on="company_id", how="left", suffixes=("", "_dup"))

    df["funding_stage_label"] = df["funding_stage"].map(FUNDING_STAGE_LABELS).fillna("Unknown")

    # Derive primary industry from features
    ind_cols = ["company_id"] + [c for c in features.columns if c.startswith("ind_")]
    ind_df = features[ind_cols].copy()
    ind_df["primary_industry"] = ind_df.apply(derive_primary_industry, axis=1)
    df = df.merge(ind_df[["company_id", "primary_industry"]], on="company_id", how="left")

    # Has job postings flag
    if "has_job_postings" in features.columns:
        df = df.merge(
            features[["company_id", "has_job_postings"]],
            on="company_id", how="left",
        )

    # Format total funding for display
    df["total_funding_display"] = df["total_funding_usd"].apply(format_usd)

    # Sort by hiring rank
    df = df.sort_values("hiring_rank").reset_index(drop=True)

    return df


# ── Filters ──────────────────────────────────────────────────────────────────


def apply_filters(
    df: pd.DataFrame,
    min_score: int,
    tiers: list,
    industries: list,
    stages: list,
    search: str,
) -> pd.DataFrame:
    filtered = df.copy()

    if min_score > 0:
        filtered = filtered[filtered["hiring_score"] >= min_score]

    if tiers:
        filtered = filtered[filtered["momentum_tier"].isin(tiers)]

    if industries:
        filtered = filtered[filtered["primary_industry"].isin(industries)]

    if stages:
        filtered = filtered[filtered["funding_stage_label"].isin(stages)]

    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
        ]

    return filtered


# ── Rendering functions ──────────────────────────────────────────────────────


def render_kpi_cards(df: pd.DataFrame, total_count: int):
    c1, c2, c3, c4 = st.columns(4)
    n = len(df)

    with c1:
        st.metric("Companies", f"{n:,}", help=f"of {total_count:,} total")
    with c2:
        avg_score = df["hiring_score"].mean() if n > 0 else 0
        st.metric("Avg Hiring Score", f"{avg_score:.1f}")
    with c3:
        high_mom = (
            df["momentum_tier"].isin(["Very High", "High"]).sum() if n > 0 else 0
        )
        pct = high_mom / n * 100 if n > 0 else 0
        st.metric("High Momentum", f"{high_mom}", delta=f"{pct:.0f}%")
    with c4:
        hiring = df["has_job_postings"].sum() if "has_job_postings" in df.columns and n > 0 else 0
        pct_h = hiring / n * 100 if n > 0 else 0
        st.metric("Actively Hiring", f"{int(hiring)}", delta=f"{pct_h:.0f}%")


def render_company_table(df: pd.DataFrame) -> pd.DataFrame:
    """Render the ranked company table and return display DataFrame."""
    display_cols = {
        "hiring_rank": "Rank",
        "name": "Company",
        "hiring_score": "Hiring Score",
        "hiring_tier": "Hiring Tier",
        "momentum_score": "Momentum",
        "momentum_tier": "Momentum Tier",
        "primary_industry": "Industry",
        "funding_stage_label": "Stage",
        "total_funding_display": "Total Funding",
    }

    display_df = df[list(display_cols.keys())].rename(columns=display_cols)

    st.dataframe(
        display_df,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Company": st.column_config.TextColumn(width="medium"),
            "Hiring Score": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Momentum": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f", width="small",
            ),
        },
        hide_index=True,
        use_container_width=True,
        height=500,
    )

    return display_df


def render_signal_chart(company_row: pd.Series, median_signals: dict):
    """Render radar chart of hiring signals for a selected company."""
    signal_cols = list(SIGNAL_LABELS.keys())
    labels = list(SIGNAL_LABELS.values())

    company_vals = [company_row.get(c, 0) for c in signal_cols]
    median_vals = [median_signals.get(c, 0) for c in signal_cols]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=company_vals + [company_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=company_row.get("name", "Selected"),
        fillcolor="rgba(59, 130, 246, 0.2)",
        line=dict(color="#3b82f6"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=median_vals + [median_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="Median",
        fillcolor="rgba(200, 200, 200, 0.15)",
        line=dict(color="#BBBBBB", dash="dot"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_company_detail(company_id: str, data: dict, master_df: pd.DataFrame):
    """Render the full detail view for a selected company."""
    row = master_df[master_df["company_id"] == company_id].iloc[0]

    st.markdown("---")
    st.subheader(row["name"])

    left, right = st.columns([2, 1])

    with left:
        # Description
        desc = row.get("description_combined", "")
        if pd.notna(desc) and desc:
            st.write(desc)

        # Location
        parts = [row.get("city"), row.get("state"), row.get("country")]
        location = ", ".join([str(p) for p in parts if pd.notna(p)])
        if location:
            st.write(f"**Location:** {location}")

        # Industries
        industries = row.get("industries")
        if pd.notna(industries) and industries:
            st.write(f"**Industries:** {industries}")

        # Links
        links = []
        if pd.notna(row.get("website")):
            links.append(f"[Website]({row['website']})")
        if pd.notna(row.get("linkedin")):
            links.append(f"[LinkedIn]({row['linkedin']})")
        if links:
            st.write(" | ".join(links))

    with right:
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Hiring Score", f"{row['hiring_score']:.1f}")
            st.caption(f"Tier: {row.get('hiring_tier', 'N/A')}")
        with r2:
            mom = row.get("momentum_score")
            st.metric("Momentum", f"{mom:.1%}" if pd.notna(mom) else "N/A")
            st.caption(f"Tier: {row.get('momentum_tier', 'N/A')}")

        st.write(f"**Funding:** {format_usd(row.get('total_funding_usd'))}")
        st.write(f"**Stage:** {row.get('funding_stage_label', 'N/A')}")
        st.write(f"**Employees:** {row.get('num_employees', 'N/A')}")

        investors = row.get("top_investors")
        if pd.notna(investors) and investors:
            st.write(f"**Investors:** {investors}")

    # Signal radar chart
    st.markdown("#### Hiring Signal Breakdown")
    signal_cols = list(SIGNAL_LABELS.keys())
    median_signals = {
        c: master_df[c].median() for c in signal_cols if c in master_df.columns
    }
    render_signal_chart(row, median_signals)

    # Feature detail (expandable, grouped by category)
    features_df = data["features"]
    metadata_df = data["metadata"]

    company_features = features_df[features_df["company_id"] == company_id]
    if len(company_features) > 0:
        company_features = company_features.iloc[0]

        with st.expander("All Features"):
            groups = metadata_df["group"].unique() if "group" in metadata_df.columns else []
            tabs = st.tabs([g.title() for g in groups]) if len(groups) > 0 else []

            for tab, group in zip(tabs, groups):
                with tab:
                    group_feats = metadata_df[metadata_df["group"] == group]
                    rows = []
                    for _, feat in group_feats.iterrows():
                        fname = feat["feature_name"]
                        val = company_features.get(fname)
                        rows.append({
                            "Feature": fname,
                            "Value": f"{val:.4f}" if isinstance(val, float) else str(val),
                            "Mean": feat.get("mean", ""),
                            "Min": feat.get("min", ""),
                            "Max": feat.get("max", ""),
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Evaluation tabs (walk-forward backtest) ──────────────────────────────────


def _eval_empty_state(missing: list[str]):
    """Render a placeholder when evaluation parquets aren't uploaded yet."""
    st.info(
        "Walk-forward backtest results aren't available yet. "
        f"Missing S3 files: `{', '.join(missing)}`.\n\n"
        "Run `evaluation.runner.run_evaluation(...)` and upload the resulting "
        "parquets to `s3://startup-momentum-pipeline/evaluation/`."
    )


def render_backtest_tab(eval_data: dict):
    """Headline mean±std table + per-fold stability lines."""
    summary = eval_data.get("summary")
    per_fold = eval_data.get("per_fold")
    if summary is None or per_fold is None:
        missing = [k for k in ("summary", "per_fold") if eval_data.get(k) is None]
        _eval_empty_state(missing)
        return

    headline_metrics = ["pr_auc", "precision_at_10", "precision_at_20", "lift_at_10", "recall_at_50"]
    headline = summary[summary["metric"].isin(headline_metrics)].copy()
    headline["cell"] = headline.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1
    )
    table = (
        headline.pivot(index="model", columns="metric", values="cell")
        .reindex(columns=[m for m in headline_metrics if m in headline["metric"].values])
    )
    st.markdown("#### Mean ± Std across folds")
    st.dataframe(table, use_container_width=True)

    st.markdown("#### Per-fold stability")
    metric_choice = st.selectbox(
        "Metric",
        options=sorted(per_fold["metric"].unique()),
        index=sorted(per_fold["metric"].unique()).index("precision_at_20")
        if "precision_at_20" in per_fold["metric"].unique() else 0,
    )
    pivot = (
        per_fold[per_fold["metric"] == metric_choice]
        .pivot(index="cutoff", columns="model", values="value")
        .reset_index()
    )
    fig = px.line(
        pivot.melt(id_vars="cutoff", var_name="model", value_name=metric_choice),
        x="cutoff", y=metric_choice, color="model", markers=True,
    )
    fig.update_layout(height=400, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)


def render_hits_tab(eval_data: dict, master_df: pd.DataFrame, data: dict):
    """Feed of true positives: companies the model flagged that then raised."""
    scores = eval_data.get("scores")
    if scores is None:
        _eval_empty_state(["scores"])
        return

    model_options = sorted(scores["model"].unique())
    default_model = "xgboost" if "xgboost" in model_options else model_options[0]
    chosen_model = st.selectbox("Model", options=model_options, index=model_options.index(default_model))
    top_k = st.slider("Surface top-K per fold", min_value=5, max_value=50, value=20, step=5)

    s = scores[scores["model"] == chosen_model].copy()
    s["as_of_date"] = pd.to_datetime(s["as_of_date"])
    # Within each fold, take the top-k by score.
    s["rank_in_fold"] = s.groupby("as_of_date")["score"].rank(method="first", ascending=False)
    surfaced = s[s["rank_in_fold"] <= top_k]

    # A "hit" = surfaced AND label==1 (raised in the 90-day window).
    hits = surfaced[surfaced["label"] == 1].copy()
    if hits.empty:
        st.warning("No true-positive hits in the current selection.")
        return

    # Compute lead time: first cutoff at which this company crossed into top-k,
    # joined to the actual round date from the spine.
    first_surfaced = (
        surfaced[surfaced["label"] == 1]
        .groupby("company_id")["as_of_date"].min()
        .rename("first_surfaced")
    )

    spine = data["spine"][["company_id"]].copy()
    if "last_funding_date" in data["spine"].columns:
        spine["round_date"] = pd.to_datetime(
            data["spine"]["last_funding_date"], errors="coerce"
        )
    else:
        spine["round_date"] = pd.NaT

    if "last_funding_type" in data["spine"].columns:
        spine["round_type"] = data["spine"]["last_funding_type"]

    feed = (
        first_surfaced.to_frame()
        .join(
            hits.sort_values("score", ascending=False)
            .drop_duplicates("company_id")
            .set_index("company_id")[["score"]]
            .rename(columns={"score": "best_score"}),
            how="left",
        )
        .join(spine.set_index("company_id"), how="left")
        .reset_index()
    )

    # Add company name + funding stage from the master table.
    feed = feed.merge(
        master_df[["company_id", "name", "primary_industry", "funding_stage_label", "website"]],
        on="company_id", how="left",
    )

    feed["lead_time_days"] = (feed["round_date"] - feed["first_surfaced"]).dt.days
    feed = feed.sort_values("first_surfaced", ascending=False)

    # KPI strip
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Hit count", len(feed))
    with c2:
        med_lead = feed["lead_time_days"].dropna()
        st.metric("Median lead time", f"{med_lead.median():.0f} d" if len(med_lead) else "N/A")
    with c3:
        st.metric("Best score", f"{feed['best_score'].max():.3f}")

    st.markdown("#### Hits feed")
    show_cols = {
        "first_surfaced": "First surfaced",
        "name": "Company",
        "best_score": "Score",
        "round_date": "Round date",
        "round_type": "Round",
        "lead_time_days": "Lead (d)",
        "primary_industry": "Industry",
        "funding_stage_label": "Stage",
        "website": "Website",
    }
    available = {k: v for k, v in show_cols.items() if k in feed.columns}
    display = feed[list(available.keys())].rename(columns=available)
    st.dataframe(
        display,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.3f"),
            "Lead (d)": st.column_config.NumberColumn(format="%d"),
            "Website": st.column_config.LinkColumn(),
        },
        hide_index=True, use_container_width=True, height=500,
    )


def render_calibration_tab(eval_data: dict):
    """Decile reliability diagram on the held-out scores frame."""
    scores = eval_data.get("scores")
    if scores is None:
        _eval_empty_state(["scores"])
        return

    model_options = sorted(scores["model"].unique())
    default_model = "xgboost" if "xgboost" in model_options else model_options[0]
    chosen = st.selectbox(
        "Model", options=model_options,
        index=model_options.index(default_model), key="calib_model",
    )
    n_bins = st.slider("Bins", min_value=5, max_value=20, value=10)

    s = scores[scores["model"] == chosen].copy()
    if s["score"].nunique() < n_bins:
        st.warning("Not enough unique scores to bin — try fewer bins or a different model.")
        return

    edges = pd.qcut(s["score"], q=n_bins, duplicates="drop", retbins=True)[1]
    s["bin"] = pd.cut(s["score"], bins=edges, include_lowest=True, labels=False)
    curve = (
        s.groupby("bin")
        .agg(predicted=("score", "mean"), observed=("label", "mean"), n=("label", "size"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, curve["predicted"].max()], y=[0, curve["predicted"].max()],
        mode="lines", line=dict(dash="dash", color="gray"), name="perfect",
    ))
    fig.add_trace(go.Scatter(
        x=curve["predicted"], y=curve["observed"], mode="lines+markers",
        name=chosen, line=dict(color="#2563eb"),
    ))
    fig.update_layout(
        xaxis_title="Mean predicted score (bin)",
        yaxis_title="Observed positive rate",
        height=450, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Bin detail")
    st.dataframe(curve, hide_index=True, use_container_width=True)


def render_overview_tab(data: dict, master_df: pd.DataFrame):
    """Render the Data Overview tab with distribution charts."""
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Hiring Score Distribution")
        fig = px.histogram(
            master_df, x="hiring_score", color="hiring_tier",
            nbins=20,
            color_discrete_map=TIER_COLORS,
            category_orders={"hiring_tier": ["Very High", "High", "Moderate", "Low", "Very Low"]},
        )
        fig.update_layout(
            xaxis_title="Hiring Score",
            yaxis_title="Count",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            legend_title="Tier",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Momentum Score Distribution")
        fig = px.histogram(
            master_df, x="momentum_score", color="momentum_tier",
            nbins=20,
            color_discrete_map=TIER_COLORS,
            category_orders={"momentum_tier": ["Very High", "High", "Moderate", "Low", "Very Low"]},
        )
        fig.update_layout(
            xaxis_title="Momentum Score",
            yaxis_title="Count",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            legend_title="Tier",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Industry Breakdown")
        ind_counts = []
        for col, label in INDUSTRY_MAP.items():
            if col in data["features"].columns:
                count = int(data["features"][col].sum())
                ind_counts.append({"Industry": label, "Count": count})
        ind_df = pd.DataFrame(ind_counts).sort_values("Count", ascending=True)
        fig = px.bar(ind_df, x="Count", y="Industry", orientation="h",
                     color_discrete_sequence=["#60a5fa"])
        fig.update_layout(
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### Funding Stage Breakdown")
        stage_counts = master_df["funding_stage_label"].value_counts().reset_index()
        stage_counts.columns = ["Stage", "Count"]
        fig = px.pie(
            stage_counts, values="Count", names="Stage", hole=0.4,
            color_discrete_sequence=["#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe"],
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    st.title("Startup Growth Radar")

    # Load data
    try:
        data = load_all_data()
    except Exception as e:
        st.error(f"Could not load data from S3: {e}")
        st.info("Check that AWS credentials are configured in .streamlit/secrets.toml")
        st.stop()

    last_updated = data.get("last_updated")
    if last_updated is not None:
        age = datetime.now(timezone.utc) - last_updated
        age_days = age.days
        ts_str = last_updated.strftime("%Y-%m-%d %H:%M UTC")
        freshness = (
            f"Data last refreshed by AWS pipeline: <b>{ts_str}</b> "
            f"({age_days} day{'s' if age_days != 1 else ''} ago)"
        )
    else:
        freshness = "Data freshness unknown — could not read S3 LastModified."

    st.markdown(
        '<div class="weekly-banner">'
        f"{freshness}<br>"
        "An AWS pipeline (Step Functions + EMR Serverless) rescores all companies "
        "each Monday and writes fresh results to S3."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Scoring 871 startups by growth momentum using funding, hiring, "
        "and news signals. Check out the "
        "[Startup Growth Radar GitHub repo](https://github.com/1alaskan/Launchpad-Scout) "
        "to run it on your own list of companies!"
    )
    st.caption("Ranked by hiring friendliness score")

    master_df = prepare_master_table(data)
    total_count = len(master_df)

    # ── Sidebar filters ──────────────────────────────────────────────────

    st.sidebar.title("Filters")

    if st.sidebar.button("🔄 Refresh data from S3", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    min_score = st.sidebar.slider("Minimum Hiring Score", 0, 100, 0, step=5)

    all_tiers = ["Very High", "High", "Moderate", "Low", "Very Low"]
    tiers = st.sidebar.multiselect("Momentum Tier", options=all_tiers, default=all_tiers)

    all_industries = sorted(master_df["primary_industry"].dropna().unique())
    industries = st.sidebar.multiselect("Industry", options=all_industries)

    all_stages = sorted(master_df["funding_stage_label"].dropna().unique())
    stages = st.sidebar.multiselect("Funding Stage", options=all_stages)

    search = st.sidebar.text_input("Search company name")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data refreshes every Monday via AWS pipeline "
        "(Step Functions + EMR Serverless + Lambda)"
    )
    st.sidebar.caption("Built with Streamlit")

    # Apply filters
    filtered_df = apply_filters(master_df, min_score, tiers, industries, stages, search)

    # ── Tabs ─────────────────────────────────────────────────────────────

    eval_data = load_evaluation_artifacts()

    (
        tab_rankings,
        tab_overview,
        tab_backtest,
        tab_hits,
        tab_calibration,
    ) = st.tabs([
        "Company Rankings",
        "Data Overview",
        "Backtest Results",
        "Hits Feed",
        "Calibration",
    ])

    with tab_rankings:
        render_kpi_cards(filtered_df, total_count)

        if len(filtered_df) == 0:
            st.info("No companies match the selected filters.")
        else:
            render_company_table(filtered_df)

            # Company detail selector
            company_names = filtered_df[["company_id", "name"]].drop_duplicates()
            company_options = dict(
                zip(company_names["name"], company_names["company_id"])
            )
            selected_name = st.selectbox(
                "Select a company for details",
                options=[""] + list(company_options.keys()),
            )

            if selected_name:
                selected_id = company_options[selected_name]
                render_company_detail(selected_id, data, master_df)

    with tab_overview:
        render_overview_tab(data, master_df)

    with tab_backtest:
        render_backtest_tab(eval_data)

    with tab_hits:
        render_hits_tab(eval_data, master_df, data)

    with tab_calibration:
        render_calibration_tab(eval_data)


if __name__ == "__main__":
    main()
