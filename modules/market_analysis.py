from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from modules.data_loader import (
    get_comps_for_property,
    get_performance_for_property,
    get_property_list,
    get_property_name_map,
    load_properties,
)


# ============================================================================
# CONSTANTS
# ============================================================================

POSITIONING_WEIGHT_RENT = 0.35
POSITIONING_WEIGHT_OCCUPANCY = 0.35
POSITIONING_WEIGHT_GROWTH = 0.15
POSITIONING_WEIGHT_DISTANCE = 0.15

RENT_PREMIUM_STRONG_PCT = 0.05
RENT_DISCOUNT_STRONG_PCT = -0.05

OCCUPANCY_OUTPERFORM_STRONG_PTS = 0.020
OCCUPANCY_UNDERPERFORM_STRONG_PTS = -0.020

GROWTH_OUTPERFORM_STRONG_PTS = 0.010
GROWTH_UNDERPERFORM_STRONG_PTS = -0.010

DISTANCE_NEAR_THRESHOLD = 2.0
DISTANCE_FAR_THRESHOLD = 5.0

POSITIONING_SCORE_STRONG = 75.0
POSITIONING_SCORE_AVERAGE = 55.0
POSITIONING_SCORE_WEAK = 40.0

POSITIONING_LEADER = "Leader"
POSITIONING_COMPETITIVE = "Competitive"
POSITIONING_AT_RISK = "At Risk"
POSITIONING_WEAK = "Weak"

RAG_GREEN = "Green"
RAG_AMBER = "Amber"
RAG_RED = "Red"

TREND_PERIODS = 12

_NO_GEO_COORDS = {"lat": 39.8283, "lon": -98.5795}

_REQUIRED_PROPERTY_COLUMNS = {"Property_ID"}
_REQUIRED_PERFORMANCE_COLUMNS = {"Year_Month", "Occupancy"}
_REQUIRED_COMP_COLUMNS = set()

_EXPECTED_WEIGHT_SUM = 1.0
_WEIGHT_TOLERANCE = 1e-9


# ============================================================================
# VALIDATION / DATA CONTRACTS
# ============================================================================

@dataclass(frozen=True)
class SubjectMetrics:
    current_occupancy: float
    subject_rent: float
    rent_growth: float


def _assert_weight_configuration() -> None:
    """Ensure positioning weights remain valid."""
    total = (
        POSITIONING_WEIGHT_RENT
        + POSITIONING_WEIGHT_OCCUPANCY
        + POSITIONING_WEIGHT_GROWTH
        + POSITIONING_WEIGHT_DISTANCE
    )
    if abs(total - _EXPECTED_WEIGHT_SUM) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"Positioning weights must sum to 1.0; got {total:.12f}"
        )


def _require_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{df_name} missing required columns: {sorted(missing)}"
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a scalar value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_ratio(value: Any, default: float = 0.0) -> float:
    """
    Normalize a ratio-like value to decimal form.

    Rules:
    - values in [-1, 1] are assumed already decimal
    - values outside that interval are assumed percentage points and divided by 100
    """
    numeric = _safe_float(value, default=default)
    if abs(numeric) > 1.0:
        return numeric / 100.0
    return numeric


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float to [lower, upper]."""
    return max(lower, min(upper, value))


def _get_properties_table() -> pd.DataFrame:
    """Load and validate canonical properties table."""
    props = load_properties().copy()
    _require_columns(props, _REQUIRED_PROPERTY_COLUMNS, "properties")
    return props


def _get_property_row(property_id: str) -> pd.Series:
    """Return a single validated property row."""
    props = _get_properties_table()
    subset = props.loc[props["Property_ID"] == property_id]
    if subset.empty:
        raise ValueError(f"Property not found: {property_id}")
    if len(subset) > 1:
        raise ValueError(f"Duplicate Property_ID found in properties table: {property_id}")
    return subset.iloc[0]


def _get_performance_frame(property_id: str) -> pd.DataFrame:
    """Load and validate property performance history."""
    perf = get_performance_for_property(property_id).copy()
    if perf.empty:
        raise ValueError(f"No performance data found for property: {property_id}")
    _require_columns(perf, _REQUIRED_PERFORMANCE_COLUMNS, f"performance[{property_id}]")
    perf = perf.sort_values("Year_Month").reset_index(drop=True)
    return perf


def _get_comp_frame(property_id: str) -> pd.DataFrame:
    """
    Load comp data and normalize schema.

    Normalized columns guaranteed in returned frame:
    - Avg_Rent
    - Comp_Occupancy
    - Rent_Growth_YoY
    - Distance_Miles
    """
    comps = get_comps_for_property(property_id).copy()
    if comps.empty:
        raise ValueError(f"No market comps found for property: {property_id}")

    if "Avg_Rent" not in comps.columns and "Market_Rent" in comps.columns:
        comps["Avg_Rent"] = comps["Market_Rent"]

    if "Comp_Occupancy" not in comps.columns and "Occupancy" in comps.columns:
        comps["Comp_Occupancy"] = comps["Occupancy"]

    for col in ["Avg_Rent", "Comp_Occupancy", "Rent_Growth_YoY", "Distance_Miles"]:
        if col not in comps.columns:
            comps[col] = np.nan

    comps["Avg_Rent"] = pd.to_numeric(comps["Avg_Rent"], errors="coerce")
    comps["Comp_Occupancy"] = pd.to_numeric(comps["Comp_Occupancy"], errors="coerce")
    comps["Rent_Growth_YoY"] = pd.to_numeric(comps["Rent_Growth_YoY"], errors="coerce")
    comps["Distance_Miles"] = pd.to_numeric(comps["Distance_Miles"], errors="coerce")

    comps["Comp_Occupancy"] = comps["Comp_Occupancy"].map(
        lambda x: _normalize_ratio(x, default=np.nan)
    )
    comps["Rent_Growth_YoY"] = comps["Rent_Growth_YoY"].map(
        lambda x: _normalize_ratio(x, default=np.nan)
    )

    return comps.reset_index(drop=True)


def _score_to_rag(score: float) -> str:
    """Map score to RAG label."""
    if score >= POSITIONING_SCORE_STRONG:
        return RAG_GREEN
    if score >= POSITIONING_SCORE_WEAK:
        return RAG_AMBER
    return RAG_RED


def _score_to_label(score: float) -> str:
    """Map score to positioning label."""
    if score >= POSITIONING_SCORE_STRONG:
        return POSITIONING_LEADER
    if score >= POSITIONING_SCORE_AVERAGE:
        return POSITIONING_COMPETITIVE
    if score >= POSITIONING_SCORE_WEAK:
        return POSITIONING_AT_RISK
    return POSITIONING_WEAK


def _weighted_mean(
    values: pd.Series,
    weights: pd.Series | None = None,
    default: float = np.nan,
) -> float:
    """Safely compute weighted mean."""
    values_num = pd.to_numeric(values, errors="coerce")

    if weights is None:
        return float(values_num.mean()) if values_num.notna().any() else default

    weights_num = pd.to_numeric(weights, errors="coerce")
    mask = values_num.notna() & weights_num.notna() & (weights_num > 0)

    if not mask.any():
        return float(values_num.mean()) if values_num.notna().any() else default

    return float(np.average(values_num[mask], weights=weights_num[mask]))


def _comp_weights(comps: pd.DataFrame) -> pd.Series:
    """
    Build deterministic relevance weights.

    Weight drivers:
    - closer distance => higher weight
    - higher occupancy => modestly higher weight
    - stronger growth => modestly higher weight
    """
    distance = pd.to_numeric(comps["Distance_Miles"], errors="coerce")
    occ = pd.to_numeric(comps["Comp_Occupancy"], errors="coerce")
    growth = pd.to_numeric(comps["Rent_Growth_YoY"], errors="coerce")

    distance_weight = (1.0 / distance.clip(lower=0.25)).fillna(1.0)
    occ_weight = (1.0 + occ.clip(lower=0.80, upper=1.00).fillna(0.93) - 0.90)
    growth_weight = (1.0 + growth.clip(lower=-0.05, upper=0.10).fillna(0.02))

    weights = distance_weight * occ_weight * growth_weight
    return weights.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _get_subject_rent(perf: pd.DataFrame, prop: pd.Series) -> float:
    """Get latest subject rent using documented fallback order."""
    if "Avg_Actual_Rent" in perf.columns:
        latest = _safe_float(perf.iloc[-1].get("Avg_Actual_Rent"), default=np.nan)
        if pd.notna(latest) and latest > 0:
            return latest
    return _safe_float(prop.get("Avg_Rent_per_Unit"), default=0.0)


def _compute_subject_rent_growth(perf: pd.DataFrame) -> float:
    """
    Compute subject rent growth over available trailing window.

    Uses rent-to-rent growth, not revenue-to-rent growth, to preserve semantic consistency
    with comp Rent_Growth_YoY comparisons.
    """
    if "Avg_Actual_Rent" not in perf.columns:
        return 0.0

    trailing = perf.tail(TREND_PERIODS)
    if len(trailing) < 2:
        return 0.0

    start_rent = _safe_float(trailing.iloc[0].get("Avg_Actual_Rent"), default=np.nan)
    end_rent = _safe_float(trailing.iloc[-1].get("Avg_Actual_Rent"), default=np.nan)

    if pd.notna(start_rent) and start_rent > 0 and pd.notna(end_rent):
        return (end_rent / start_rent) - 1.0
    return 0.0


def _latest_subject_metrics(property_id: str) -> SubjectMetrics:
    """Return latest validated subject metrics."""
    prop = _get_property_row(property_id)
    perf = _get_performance_frame(property_id)

    latest = perf.iloc[-1]

    current_occupancy = _normalize_ratio(latest.get("Occupancy"), default=0.0)
    subject_rent = _get_subject_rent(perf, prop)
    rent_growth = _compute_subject_rent_growth(perf)

    return SubjectMetrics(
        current_occupancy=current_occupancy,
        subject_rent=subject_rent,
        rent_growth=rent_growth,
    )


def _percentile_rank(value: float, series: pd.Series) -> float:
    """
    Percentile rank in [0, 100].

    Definition:
    proportion of non-null observations less than or equal to value.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).mean() * 100.0)


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def get_subject_vs_comp_summary(property_id: str) -> dict[str, Any]:
    """Compare validated subject metrics against weighted comp averages."""
    _assert_weight_configuration()

    prop = _get_property_row(property_id)
    comps = _get_comp_frame(property_id)
    subject = _latest_subject_metrics(property_id)
    weights = _comp_weights(comps)

    comp_avg_rent = _weighted_mean(comps["Avg_Rent"], weights=weights, default=0.0)
    comp_avg_occ = _weighted_mean(comps["Comp_Occupancy"], weights=weights, default=0.0)
    comp_avg_growth = _weighted_mean(comps["Rent_Growth_YoY"], weights=weights, default=0.0)

    rent_premium_pct = ((subject.subject_rent / comp_avg_rent) - 1.0) if comp_avg_rent > 0 else 0.0
    occupancy_gap = subject.current_occupancy - comp_avg_occ
    rent_growth_gap = subject.rent_growth - comp_avg_growth

    return {
        "Property_ID": property_id,
        "Property_Name": prop.get("Property_Name", property_id),
        "Market": prop.get("Market", ""),
        "Region": prop.get("Region", prop.get("Market", "")),
        "Value_Add_Phase": prop.get("Value_Add_Phase", ""),
        "Asset_Phase": prop.get("Asset_Phase", prop.get("Value_Add_Phase", "")),
        "Subject_Rent": float(subject.subject_rent),
        "Comp_Avg_Rent": float(comp_avg_rent),
        "Rent_Premium_Pct": float(rent_premium_pct),
        "Subject_Occupancy": float(subject.current_occupancy),
        "Comp_Avg_Occupancy": float(comp_avg_occ),
        "Occupancy_Gap": float(occupancy_gap),
        "Subject_Rent_Growth": float(subject.rent_growth),
        "Comp_Avg_Rent_Growth": float(comp_avg_growth),
        "Rent_Growth_Gap": float(rent_growth_gap),
        "Comp_Count": int(len(comps)),
    }


@st.cache_data(show_spinner=False)
def get_comp_table(property_id: str) -> pd.DataFrame:
    """Build display-ready comp table with deterministic column contract."""
    prop = _get_property_row(property_id)
    comps = _get_comp_frame(property_id)
    subject = _latest_subject_metrics(property_id)

    subject_rent = subject.subject_rent
    subject_occ = subject.current_occupancy

    comps = comps.copy()
    comps["Rent_Premium_vs_Comp"] = subject_rent - comps["Avg_Rent"]
    comps["Rent_Premium_Pct"] = np.where(
        comps["Avg_Rent"] > 0,
        (subject_rent / comps["Avg_Rent"]) - 1.0,
        np.nan,
    )
    comps["Occupancy_Gap_vs_Comp"] = subject_occ - comps["Comp_Occupancy"]
    comps["Comp_Relevance_Score"] = _comp_weights(comps)

    comps = comps.sort_values(
        by=["Comp_Relevance_Score", "Comp_Occupancy", "Rent_Growth_YoY"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    comps["Comp_Rank"] = np.arange(1, len(comps) + 1)
    comps["Subject_Property_Name"] = prop.get("Property_Name", property_id)

    for col in ["Comp_Name", "Unit_Class", "Year_Built"]:
        if col not in comps.columns:
            comps[col] = np.nan

    ordered_cols = [
        "Comp_Rank",
        "Comp_Name",
        "Unit_Class",
        "Year_Built",
        "Distance_Miles",
        "Avg_Rent",
        "Comp_Occupancy",
        "Rent_Growth_YoY",
        "Rent_Premium_vs_Comp",
        "Rent_Premium_Pct",
        "Occupancy_Gap_vs_Comp",
        "Comp_Relevance_Score",
        "Subject_Property_Name",
    ]
    output = comps[ordered_cols].copy()
    output["Market_Rent"] = output["Avg_Rent"]
    output["Occupancy"] = output["Comp_Occupancy"]
    return output


@st.cache_data(show_spinner=False)
def get_rent_distribution_data(property_id: str) -> pd.DataFrame:
    """Build subject-vs-comps monthly rent comparison frame."""
    prop = _get_property_row(property_id)
    subject = _latest_subject_metrics(property_id)
    comp_table = get_comp_table(property_id)

    rows = [{
        "Label": prop.get("Property_Name", property_id),
        "Type": "Subject",
        "Monthly_Rent": float(subject.subject_rent),
    }]

    for _, row in comp_table.iterrows():
        rows.append({
            "Label": row["Comp_Name"] if pd.notna(row["Comp_Name"]) else f"Comp_{int(row['Comp_Rank'])}",
            "Type": "Comp",
            "Monthly_Rent": _safe_float(row["Avg_Rent"], default=0.0),
        })

    result = pd.DataFrame(rows).sort_values("Monthly_Rent", ascending=False).reset_index(drop=True)
    return result


@st.cache_data(show_spinner=False)
def get_occupancy_comparison_data(property_id: str) -> pd.DataFrame:
    """Build subject-vs-comps occupancy comparison frame."""
    prop = _get_property_row(property_id)
    subject = _latest_subject_metrics(property_id)
    comp_table = get_comp_table(property_id)

    rows = [{
        "Label": prop.get("Property_Name", property_id),
        "Type": "Subject",
        "Occupancy": float(subject.current_occupancy),
    }]

    for _, row in comp_table.iterrows():
        rows.append({
            "Label": row["Comp_Name"] if pd.notna(row["Comp_Name"]) else f"Comp_{int(row['Comp_Rank'])}",
            "Type": "Comp",
            "Occupancy": _normalize_ratio(row["Comp_Occupancy"], default=0.0),
        })

    result = pd.DataFrame(rows).sort_values("Occupancy", ascending=False).reset_index(drop=True)
    return result


@st.cache_data(show_spinner=False)
def get_competitive_positioning_score(property_id: str) -> dict[str, Any]:
    """Calculate composite competitive positioning score."""
    _assert_weight_configuration()

    summary = get_subject_vs_comp_summary(property_id)
    comps = _get_comp_frame(property_id)

    rent_premium_pct = _safe_float(summary["Rent_Premium_Pct"], default=0.0)
    occupancy_gap = _safe_float(summary["Occupancy_Gap"], default=0.0)
    growth_gap = _safe_float(summary["Rent_Growth_Gap"], default=0.0)

    rent_score = _clamp(50.0 + (rent_premium_pct / 0.10) * 50.0, 0.0, 100.0)
    occ_score = _clamp(50.0 + (occupancy_gap / 0.05) * 50.0, 0.0, 100.0)
    growth_score = _clamp(50.0 + (growth_gap / 0.03) * 50.0, 0.0, 100.0)

    avg_distance = float(pd.to_numeric(comps["Distance_Miles"], errors="coerce").mean())
    if np.isnan(avg_distance):
        distance_score = 60.0
    elif avg_distance <= DISTANCE_NEAR_THRESHOLD:
        distance_score = 85.0
    elif avg_distance <= DISTANCE_FAR_THRESHOLD:
        distance_score = 65.0
    else:
        distance_score = 45.0

    positioning_score = (
        rent_score * POSITIONING_WEIGHT_RENT
        + occ_score * POSITIONING_WEIGHT_OCCUPANCY
        + growth_score * POSITIONING_WEIGHT_GROWTH
        + distance_score * POSITIONING_WEIGHT_DISTANCE
    )

    return {
        "Property_ID": property_id,
        "Positioning_Score": float(positioning_score),
        "Positioning_Label": _score_to_label(positioning_score),
        "RAG_Status": _score_to_rag(positioning_score),
        "Rent_Subscore": float(rent_score),
        "Occupancy_Subscore": float(occ_score),
        "Growth_Subscore": float(growth_score),
        "Distance_Subscore": float(distance_score),
        "Average_Comp_Distance": float(avg_distance) if not np.isnan(avg_distance) else np.nan,
    }


@st.cache_data(show_spinner=False)
def generate_market_narrative(property_id: str) -> str:
    """Generate a deterministic narrative paragraph."""
    prop = _get_property_row(property_id)
    summary = get_subject_vs_comp_summary(property_id)
    score = get_competitive_positioning_score(property_id)

    property_name = prop.get("Property_Name", property_id)
    market = prop.get("Market", "")
    phase = prop.get("Value_Add_Phase", "")

    rent_premium_pct = _safe_float(summary["Rent_Premium_Pct"], default=0.0)
    occupancy_gap = _safe_float(summary["Occupancy_Gap"], default=0.0)
    growth_gap = _safe_float(summary["Rent_Growth_Gap"], default=0.0)
    comp_count = int(summary["Comp_Count"])

    if rent_premium_pct >= RENT_PREMIUM_STRONG_PCT:
        rent_text = "is achieving a meaningful rent premium to the competitive set"
    elif rent_premium_pct <= RENT_DISCOUNT_STRONG_PCT:
        rent_text = "is leasing at a meaningful discount to the competitive set"
    else:
        rent_text = "is broadly in line with the competitive set on rent"

    if occupancy_gap >= OCCUPANCY_OUTPERFORM_STRONG_PTS:
        occ_text = "while maintaining occupancy ahead of market peers"
    elif occupancy_gap <= OCCUPANCY_UNDERPERFORM_STRONG_PTS:
        occ_text = "but occupancy trails comparable properties"
    else:
        occ_text = "with occupancy broadly in line with market peers"

    if growth_gap >= GROWTH_OUTPERFORM_STRONG_PTS:
        growth_text = "Recent rent growth is outperforming the observed competitive set."
    elif growth_gap <= GROWTH_UNDERPERFORM_STRONG_PTS:
        growth_text = "Recent rent growth is lagging the observed competitive set."
    else:
        growth_text = "Recent rent growth is generally tracking the observed competitive set."

    return (
        f"{property_name} is positioned in the {market} market and is currently in {phase} phase. "
        f"Based on {comp_count} comparable properties, the asset {rent_text} {occ_text}. "
        f"{growth_text} Competitive positioning score: {score['Positioning_Score']:.1f}/100 "
        f"({score['Positioning_Label']})."
    )


@st.cache_data(show_spinner=False)
def get_market_watchlist_flags(property_id: str) -> list[str]:
    """Generate deterministic market-related watchlist flags."""
    summary = get_subject_vs_comp_summary(property_id)
    score = get_competitive_positioning_score(property_id)

    flags: list[str] = []

    if _safe_float(summary["Rent_Premium_Pct"]) <= RENT_DISCOUNT_STRONG_PCT:
        flags.append("Subject is leasing at a meaningful discount to comps.")

    if _safe_float(summary["Occupancy_Gap"]) <= OCCUPANCY_UNDERPERFORM_STRONG_PTS:
        flags.append("Occupancy trails the comp set by more than 200 bps.")

    if _safe_float(summary["Rent_Growth_Gap"]) <= GROWTH_UNDERPERFORM_STRONG_PTS:
        flags.append("Recent rent growth is lagging the comp set.")

    if score["RAG_Status"] == RAG_RED:
        flags.append("Overall competitive positioning is weak versus the market.")

    avg_distance = score["Average_Comp_Distance"]
    if pd.notna(avg_distance) and avg_distance > DISTANCE_FAR_THRESHOLD:
        flags.append("Comp set is geographically dispersed; interpret market signals with caution.")

    if not flags:
        flags.append("No material market positioning concerns identified.")

    return flags


@st.cache_data(show_spinner=False)
def get_market_summary_table() -> pd.DataFrame:
    """
    Build portfolio-level market summary table.

    This function does not silently skip errors. If one property fails,
    the caller should know.
    """
    rows: list[dict[str, Any]] = []
    property_name_map = get_property_name_map()

    for property_id in get_property_list():
        summary = get_subject_vs_comp_summary(property_id)
        score = get_competitive_positioning_score(property_id)

        rows.append({
            "Property_ID": property_id,
            "Property_Name": property_name_map.get(property_id, property_id),
            "Market": summary["Market"],
            "Subject_Rent": summary["Subject_Rent"],
            "Comp_Avg_Rent": summary["Comp_Avg_Rent"],
            "Rent_Premium_Pct": summary["Rent_Premium_Pct"],
            "Subject_Occupancy": summary["Subject_Occupancy"],
            "Comp_Avg_Occupancy": summary["Comp_Avg_Occupancy"],
            "Occupancy_Gap": summary["Occupancy_Gap"],
            "Rent_Growth_Gap": summary["Rent_Growth_Gap"],
            "Positioning_Score": score["Positioning_Score"],
            "Positioning_Label": score["Positioning_Label"],
            "RAG_Status": score["RAG_Status"],
        })

    return pd.DataFrame(rows).sort_values("Positioning_Score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_market_trends(property_id: str) -> dict[str, Any]:
    """
    Build trend data from subject performance history.

    Note:
    This is subject trend data used as a local market proxy, not true external market history.
    """
    perf = _get_performance_frame(property_id).tail(TREND_PERIODS).reset_index(drop=True)
    if len(perf) < 2:
        return {"trend_data": pd.DataFrame(columns=["Period", "Market_Rent", "Occupancy", "Rent_Growth", "Revenue_Growth"])}

    rows: list[dict[str, Any]] = []

    for i, row in perf.iterrows():
        period = str(row.get("Year_Month", ""))
        market_rent = _safe_float(row.get("Avg_Actual_Rent"), default=np.nan)
        occupancy = _normalize_ratio(row.get("Occupancy"), default=np.nan)
        revenue = _safe_float(row.get("Actual_Revenue"), default=np.nan)

        if i == 0:
            rent_growth = np.nan
            revenue_growth = np.nan
        else:
            prev_rent = _safe_float(perf.iloc[i - 1].get("Avg_Actual_Rent"), default=np.nan)
            prev_rev = _safe_float(perf.iloc[i - 1].get("Actual_Revenue"), default=np.nan)

            rent_growth = (
                (market_rent / prev_rent) - 1.0
                if pd.notna(market_rent) and pd.notna(prev_rent) and prev_rent > 0
                else np.nan
            )
            revenue_growth = (
                (revenue / prev_rev) - 1.0
                if pd.notna(revenue) and pd.notna(prev_rev) and prev_rev > 0
                else np.nan
            )

        rows.append({
            "Period": period,
            "Market_Rent": market_rent,
            "Occupancy": occupancy,
            "Rent_Growth": rent_growth,
            "Revenue_Growth": revenue_growth,
        })

    return {"trend_data": pd.DataFrame(rows)}


def _compute_benchmark_score(
    rent: float,
    occupancy: float,
    growth: float,
    comps: pd.DataFrame,
) -> float:
    """Score one property against comp distributions using percentile ranks."""
    rent_score = _percentile_rank(rent, comps["Avg_Rent"])
    occ_score = _percentile_rank(occupancy, comps["Comp_Occupancy"])
    growth_score = _percentile_rank(growth, comps["Rent_Growth_YoY"])

    return (
        rent_score * POSITIONING_WEIGHT_RENT
        + occ_score * POSITIONING_WEIGHT_OCCUPANCY
        + growth_score * POSITIONING_WEIGHT_GROWTH
        + 60.0 * POSITIONING_WEIGHT_DISTANCE
    )


@st.cache_data(show_spinner=False)
def get_peer_benchmarking(property_id: str) -> pd.DataFrame:
    """Build peer benchmarking table including subject row."""
    prop = _get_property_row(property_id)
    comps = _get_comp_frame(property_id)
    subject = _latest_subject_metrics(property_id)

    rows: list[dict[str, Any]] = [{
        "Property_Name": prop.get("Property_Name", property_id),
        "Property_ID": property_id,
        "Avg_Rent": float(subject.subject_rent),
        "Occupancy": float(subject.current_occupancy),
        "Rent_Growth_YoY": float(subject.rent_growth),
        "Benchmark_Score": float(
            _compute_benchmark_score(
                subject.subject_rent,
                subject.current_occupancy,
                subject.rent_growth,
                comps,
            )
        ),
        "Is_Subject": True,
    }]

    for _, comp_row in comps.iterrows():
        c_rent = _safe_float(comp_row.get("Avg_Rent"), default=0.0)
        c_occ = _safe_float(comp_row.get("Comp_Occupancy"), default=0.0)
        c_growth = _safe_float(comp_row.get("Rent_Growth_YoY"), default=0.0)

        rows.append({
            "Property_Name": comp_row.get("Comp_Name", "Unknown"),
            "Property_ID": comp_row.get("Comp_ID", ""),
            "Avg_Rent": float(c_rent),
            "Occupancy": float(c_occ),
            "Rent_Growth_YoY": float(c_growth),
            "Benchmark_Score": float(_compute_benchmark_score(c_rent, c_occ, c_growth, comps)),
            "Is_Subject": False,
        })

    return pd.DataFrame(rows).sort_values("Benchmark_Score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_market_heatmap_data(property_id: str) -> dict[str, Any]:
    """Build normalized metric-by-property matrix for heatmap display."""
    prop = _get_property_row(property_id)
    comps = _get_comp_frame(property_id)
    subject = _latest_subject_metrics(property_id)

    subject_name = prop.get("Property_Name", property_id)
    comp_names = [
        row.get("Comp_Name", f"Comp_{i+1}")
        for i, (_, row) in enumerate(comps.iterrows())
    ]
    all_names = [subject_name] + comp_names

    all_rents = pd.concat(
        [pd.Series([subject.subject_rent]), pd.to_numeric(comps["Avg_Rent"], errors="coerce")],
        ignore_index=True,
    )
    all_occs = pd.concat(
        [pd.Series([subject.current_occupancy]), pd.to_numeric(comps["Comp_Occupancy"], errors="coerce")],
        ignore_index=True,
    )
    all_growths = pd.concat(
        [pd.Series([subject.rent_growth]), pd.to_numeric(comps["Rent_Growth_YoY"], errors="coerce")],
        ignore_index=True,
    )

    def _normalize_series(s: pd.Series) -> pd.Series:
        valid = pd.to_numeric(s, errors="coerce")
        lo = valid.min()
        hi = valid.max()
        if pd.isna(lo) or pd.isna(hi):
            return pd.Series([50.0] * len(valid), index=valid.index, dtype=float)
        if hi == lo:
            return pd.Series([50.0] * len(valid), index=valid.index, dtype=float)
        return ((valid - lo) / (hi - lo) * 100.0).round(1)

    matrix = pd.DataFrame(
        {
            "Rent": _normalize_series(all_rents).values,
            "Occupancy": _normalize_series(all_occs).values,
            "Rent Growth": _normalize_series(all_growths).values,
        },
        index=all_names,
    ).T

    return {"matrix": matrix}


@st.cache_data(show_spinner=False)
def generate_market_sentiment(property_id: str) -> dict[str, Any]:
    """Generate bounded sentiment indicators in [-1, 1]."""
    summary = get_subject_vs_comp_summary(property_id)
    score = get_competitive_positioning_score(property_id)

    rent_prem = _safe_float(summary["Rent_Premium_Pct"], default=0.0)
    occ_gap = _safe_float(summary["Occupancy_Gap"], default=0.0)
    growth_gap = _safe_float(summary["Rent_Growth_Gap"], default=0.0)
    pos_score = _safe_float(score["Positioning_Score"], default=50.0)

    rent_sent = _clamp(rent_prem / 0.05, -1.0, 1.0)
    occ_sent = _clamp(occ_gap / 0.02, -1.0, 1.0)
    growth_sent = _clamp(growth_gap / 0.01, -1.0, 1.0)
    pos_sent = _clamp((pos_score / 50.0) - 1.0, -1.0, 1.0)

    return {
        "Rent_Premium_Sentiment": round(rent_sent, 3),
        "Occupancy_Gap_Sentiment": round(occ_sent, 3),
        "Growth_Gap_Sentiment": round(growth_sent, 3),
        "Positioning_Score_Sentiment": round(pos_sent, 3),
    }


@st.cache_data(show_spinner=False)
def get_market_forecast_indicators(property_id: str) -> dict[str, Any]:
    """Project one step ahead via linear trend on recent series."""
    trends = get_market_trends(property_id)
    trend_df = trends["trend_data"].copy()

    if trend_df.empty or len(trend_df) < 3:
        return {}

    result: dict[str, Any] = {}

    for metric, label in [
        ("Market_Rent", "Market_Rent_Forecast"),
        ("Occupancy", "Occupancy_Forecast"),
    ]:
        series = pd.to_numeric(trend_df[metric], errors="coerce").dropna()
        if len(series) < 3:
            continue

        x = np.arange(len(series), dtype=float)
        y = series.to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)

        current = float(y[-1])
        forecast = float(intercept + slope * len(series))
        change = (forecast / current - 1.0) if current != 0 else 0.0

        y_hat = intercept + slope * x
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        result[label] = {
            "current": round(current, 4),
            "forecast": round(forecast, 4),
            "change": round(change, 4),
            "confidence": round(r2, 4),
        }

    return result


@st.cache_data(show_spinner=False)
def get_geographic_competition_data(property_id: str) -> dict[str, Any]:
    """Build geographic coordinate data for subject and comps."""
    prop = _get_property_row(property_id)
    comps = _get_comp_frame(property_id)

    s_lat = _safe_float(prop.get("Latitude"), default=_NO_GEO_COORDS["lat"])
    s_lon = _safe_float(prop.get("Longitude"), default=_NO_GEO_COORDS["lon"])

    competitors: list[dict[str, Any]] = []
    for _, comp_row in comps.iterrows():
        if "Latitude" not in comp_row.index or "Longitude" not in comp_row.index:
            continue

        c_lat = _safe_float(comp_row.get("Latitude"), default=np.nan)
        c_lon = _safe_float(comp_row.get("Longitude"), default=np.nan)
        if np.isnan(c_lat) or np.isnan(c_lon):
            continue

        competitors.append({
            "name": comp_row.get("Comp_Name", "Unknown"),
            "lat": float(c_lat),
            "lon": float(c_lon),
            "rent": float(_safe_float(comp_row.get("Avg_Rent"), default=0.0)),
        })

    return {
        "coordinates": {
            "subject": {"lat": float(s_lat), "lon": float(s_lon)},
            "competitors": competitors,
        }
    }


# ============================================================================
# SELF-TESTS
# ============================================================================

def _run_self_tests() -> None:
    """Run deterministic integration-oriented self-tests."""
    print("Running market_analysis.py self-tests...\n")

    _assert_weight_configuration()

    property_ids = get_property_list()
    assert isinstance(property_ids, list), "get_property_list() must return a list."
    assert len(property_ids) >= 1, "Expected at least one property."

    rows: list[dict[str, Any]] = []

    for property_id in property_ids:
        summary = get_subject_vs_comp_summary(property_id)
        comp_table = get_comp_table(property_id)
        rent_dist = get_rent_distribution_data(property_id)
        occ_comp = get_occupancy_comparison_data(property_id)
        score = get_competitive_positioning_score(property_id)
        narrative = generate_market_narrative(property_id)
        flags = get_market_watchlist_flags(property_id)
        trends = get_market_trends(property_id)
        peers = get_peer_benchmarking(property_id)
        heatmap = get_market_heatmap_data(property_id)
        sentiment = generate_market_sentiment(property_id)
        forecast = get_market_forecast_indicators(property_id)
        geo = get_geographic_competition_data(property_id)

        assert isinstance(summary, dict), f"Summary must be dict for {property_id}"
        assert summary["Comp_Count"] > 0, f"Comp count must be positive for {property_id}"

        assert isinstance(comp_table, pd.DataFrame), f"Comp table must be DataFrame for {property_id}"
        assert len(comp_table) > 0, f"Comp table must not be empty for {property_id}"

        assert isinstance(rent_dist, pd.DataFrame), f"Rent distribution must be DataFrame for {property_id}"
        assert len(rent_dist) >= 2, f"Rent distribution must include subject + at least one comp for {property_id}"

        assert isinstance(occ_comp, pd.DataFrame), f"Occupancy comparison must be DataFrame for {property_id}"
        assert len(occ_comp) >= 2, f"Occupancy comparison must include subject + at least one comp for {property_id}"

        assert isinstance(score, dict), f"Score must be dict for {property_id}"
        assert 0.0 <= score["Positioning_Score"] <= 100.0, f"Score out of bounds for {property_id}"

        assert isinstance(narrative, str) and len(narrative) > 20, f"Narrative invalid for {property_id}"
        assert isinstance(flags, list) and len(flags) >= 1, f"Flags invalid for {property_id}"

        assert isinstance(trends, dict), f"Trends must be dict for {property_id}"
        assert "trend_data" in trends, f"trend_data missing for {property_id}"
        assert isinstance(trends["trend_data"], pd.DataFrame), f"trend_data must be DataFrame for {property_id}"

        assert isinstance(peers, pd.DataFrame), f"Peers must be DataFrame for {property_id}"
        assert len(peers) >= 2, f"Peers must include subject + comps for {property_id}"

        assert isinstance(heatmap, dict), f"Heatmap must be dict for {property_id}"
        assert "matrix" in heatmap, f"Heatmap missing matrix for {property_id}"
        assert isinstance(heatmap["matrix"], pd.DataFrame), f"Heatmap matrix must be DataFrame for {property_id}"

        assert isinstance(sentiment, dict), f"Sentiment must be dict for {property_id}"
        for value in sentiment.values():
            assert -1.0 <= value <= 1.0, f"Sentiment out of bounds for {property_id}"

        assert isinstance(forecast, dict), f"Forecast must be dict for {property_id}"
        assert isinstance(geo, dict), f"Geo data must be dict for {property_id}"
        assert "coordinates" in geo, f"Geo coordinates missing for {property_id}"

        rows.append({
            "Property_ID": property_id,
            "Positioning_Score": score["Positioning_Score"],
            "RAG_Status": score["RAG_Status"],
            "Rent_Premium_Pct": summary["Rent_Premium_Pct"],
            "Occupancy_Gap": summary["Occupancy_Gap"],
        })

        print(
            f"✓ {property_id:<8} | "
            f"Score: {score['Positioning_Score']:>5.1f} | "
            f"Label: {score['Positioning_Label']:<12} | "
            f"Rent Premium: {summary['Rent_Premium_Pct']:+.1%} | "
            f"Occ Gap: {summary['Occupancy_Gap']:+.1%}"
        )

    portfolio_summary = get_market_summary_table()
    assert len(portfolio_summary) == len(property_ids), "Portfolio summary row count mismatch."

    print("\nPortfolio market positioning summary:")
    print(
        portfolio_summary[
            [
                "Property_ID",
                "Positioning_Score",
                "Positioning_Label",
                "Rent_Premium_Pct",
                "Occupancy_Gap",
                "RAG_Status",
            ]
        ].to_string(index=False)
    )

    print("\nALL MARKET ANALYSIS TESTS PASSED")


if __name__ == "__main__":
    _run_self_tests()