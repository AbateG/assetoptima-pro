from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from modules.data_loader import (
    get_business_plan_for_property,
    get_property_list,
    get_property_name_map,
    load_business_plan,
    load_business_plan_with_properties,
    load_properties,
)
from modules.kpi_calculations import return_on_cost


# ============================================================================
# CONSTANTS
# ============================================================================

PROGRESS_ON_TRACK_THRESHOLD = 0.75
PROGRESS_AT_RISK_THRESHOLD = 0.40

BUDGET_OVERAGE_AMBER = 0.05
BUDGET_OVERAGE_RED = 0.15

DAYS_TO_COMPLETION_WARNING = 30
DAYS_OVERDUE_RED = 30

ROI_STRONG_THRESHOLD = 0.20
ROI_WEAK_THRESHOLD = 0.08

RAG_GREEN = "Green"
RAG_AMBER = "Amber"
RAG_RED = "Red"


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "").replace("%", "")
            if cleaned == "":
                return default
            return float(cleaned)
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_datetime(value: Any) -> pd.Timestamp:
    """Safely convert a value to pandas Timestamp."""
    try:
        if pd.isna(value):
            return pd.NaT
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert a value to bool without treating NaN as True."""
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", ""}:
            return False
    return default


def _normalize_percent(value: Any) -> float:
    """
    Normalize percent-like values to decimal form.

    Examples:
    - 75 -> 0.75
    - "75%" -> 0.75
    - 0.75 -> 0.75
    - 1.2 -> 1.2
    """
    if isinstance(value, str) and "%" in value:
        return _safe_float(value, default=0.0) / 100.0

    numeric = _safe_float(value, default=0.0)

    # Treat values between 1 and 100 as likely whole percents.
    if 1.0 < abs(numeric) <= 100.0:
        return numeric / 100.0

    return numeric


def _assign_budget_rag(budget_var_pct: float) -> str:
    """Assign budget RAG status from budget variance percentage."""
    if budget_var_pct <= BUDGET_OVERAGE_AMBER:
        return RAG_GREEN
    if budget_var_pct <= BUDGET_OVERAGE_RED:
        return RAG_AMBER
    return RAG_RED


def _assign_progress_rag(percent_complete: float, is_overdue: bool) -> str:
    """Assign progress RAG status from completion and overdue status."""
    if is_overdue:
        if percent_complete >= 0.95:
            return RAG_AMBER
        return RAG_RED
    if percent_complete >= PROGRESS_ON_TRACK_THRESHOLD:
        return RAG_GREEN
    if percent_complete >= PROGRESS_AT_RISK_THRESHOLD:
        return RAG_AMBER
    return RAG_RED


def _assign_roi_rag(roi: float) -> str:
    """Assign ROI RAG status."""
    if roi >= ROI_STRONG_THRESHOLD:
        return RAG_GREEN
    if roi >= ROI_WEAK_THRESHOLD:
        return RAG_AMBER
    return RAG_RED


def _ensure_columns(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    """Ensure required columns exist, filling missing columns with defaults."""
    df = df.copy()
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def _coerce_business_plan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize expected business plan columns and derived fields."""
    df = _ensure_columns(
        df,
        {
            "Property_ID": None,
            "Property_Name": None,
            "Initiative": None,
            "Category": None,
            "Status": None,
            "Percent_Complete": 0.0,
            "Budget": 0.0,
            "Actual_Spend": 0.0,
            "Expected_NOI_Lift": 0.0,
            "Budget_Variance": np.nan,
            "Budget_Var_Pct": np.nan,
            "Target_Completion": pd.NaT,
            "Start_Date": pd.NaT,
            "Days_to_Completion": np.nan,
            "Is_Overdue": False,
            "Risk_Level": None,
        },
    )

    df["Percent_Complete_Dec"] = df["Percent_Complete"].apply(_normalize_percent)
    df["Budget"] = df["Budget"].apply(_safe_float)
    df["Actual_Spend"] = df["Actual_Spend"].apply(_safe_float)
    df["Expected_NOI_Lift"] = df["Expected_NOI_Lift"].apply(_safe_float)
    df["Target_Completion"] = df["Target_Completion"].apply(_safe_datetime)
    df["Start_Date"] = df["Start_Date"].apply(_safe_datetime)
    df["Is_Overdue"] = df["Is_Overdue"].apply(_safe_bool)

    if df["Budget_Variance"].isna().all():
        df["Budget_Variance"] = df["Actual_Spend"] - df["Budget"]
    else:
        df["Budget_Variance"] = df["Budget_Variance"].apply(_safe_float)

    if df["Budget_Var_Pct"].isna().all():
        df["Budget_Var_Pct"] = np.where(df["Budget"] > 0, df["Budget_Variance"] / df["Budget"], 0.0)
    else:
        df["Budget_Var_Pct"] = df["Budget_Var_Pct"].apply(_normalize_percent)

    today = pd.Timestamp.today().normalize()
    if df["Days_to_Completion"].isna().all():
        df["Days_to_Completion"] = (df["Target_Completion"] - today).dt.days
    else:
        df["Days_to_Completion"] = pd.to_numeric(df["Days_to_Completion"], errors="coerce")

    return df


def _mean_return_on_cost(df: pd.DataFrame) -> float:
    """Mean initiative-level return on cost excluding zero/negative spend rows."""
    valid = df["Actual_Spend"] > 0
    if not valid.any():
        return 0.0
    roc = df.loc[valid, "Expected_NOI_Lift"] / df.loc[valid, "Actual_Spend"]
    return float(roc.mean())


def _portfolio_return_on_cost(df: pd.DataFrame) -> float:
    """Portfolio-style return on cost using total NOI lift over total spend."""
    total_spend = _safe_float(df["Actual_Spend"].sum(), default=0.0)
    total_noi = _safe_float(df["Expected_NOI_Lift"].sum(), default=0.0)
    return float(total_noi / total_spend) if total_spend > 0 else 0.0


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def get_property_business_plan_summary(property_id: str) -> Dict[str, Any]:
    """Build a summary of business plan execution for a single property."""
    df = _coerce_business_plan_columns(get_business_plan_for_property(property_id).copy())

    if df.empty:
        return {
            "Property_ID": property_id,
            "Initiative_Count": 0,
            "Completed_Count": 0,
            "In_Progress_Count": 0,
            "Delayed_Count": 0,
            "Total_Budget": 0.0,
            "Total_Actual_Spend": 0.0,
            "Budget_Variance": 0.0,
            "Budget_Variance_Pct": 0.0,
            "Expected_NOI_Lift": 0.0,
            "Weighted_Completion_Pct": 0.0,
            "Average_Return_on_Cost": 0.0,
            "Portfolio_Return_on_Cost": 0.0,
            "Overall_RAG": RAG_GREEN,
        }

    total_budget = _safe_float(df["Budget"].sum(), default=0.0)
    total_actual_spend = _safe_float(df["Actual_Spend"].sum(), default=0.0)
    budget_variance = total_actual_spend - total_budget
    budget_variance_pct = budget_variance / total_budget if total_budget > 0 else 0.0
    expected_noi_lift = _safe_float(df["Expected_NOI_Lift"].sum(), default=0.0)

    completed_count = int((df["Percent_Complete_Dec"] >= 1.0).sum())
    in_progress_count = int(((df["Percent_Complete_Dec"] > 0.0) & (df["Percent_Complete_Dec"] < 1.0)).sum())
    delayed_count = int(df["Is_Overdue"].fillna(False).sum())

    if total_budget > 0:
        weighted_completion_pct = float((df["Percent_Complete_Dec"] * df["Budget"]).sum() / total_budget)
    else:
        weighted_completion_pct = float(df["Percent_Complete_Dec"].mean()) if len(df) > 0 else 0.0

    average_return_on_cost = _mean_return_on_cost(df)
    portfolio_return_on_cost = _portfolio_return_on_cost(df)

    if budget_variance_pct > BUDGET_OVERAGE_RED or (delayed_count > 0 and weighted_completion_pct < PROGRESS_AT_RISK_THRESHOLD):
        overall_rag = RAG_RED
    elif delayed_count > 0 or budget_variance_pct > BUDGET_OVERAGE_AMBER:
        overall_rag = RAG_AMBER
    else:
        overall_rag = RAG_GREEN

    return {
        "Property_ID": property_id,
        "Initiative_Count": int(len(df)),
        "Completed_Count": completed_count,
        "In_Progress_Count": in_progress_count,
        "Delayed_Count": delayed_count,
        "Total_Budget": total_budget,
        "Total_Actual_Spend": total_actual_spend,
        "Budget_Variance": budget_variance,
        "Budget_Variance_Pct": budget_variance_pct,
        "Expected_NOI_Lift": expected_noi_lift,
        "Weighted_Completion_Pct": weighted_completion_pct,
        "Average_Return_on_Cost": average_return_on_cost,
        "Portfolio_Return_on_Cost": portfolio_return_on_cost,
        "Overall_RAG": overall_rag,
    }


def get_initiative_progress_table(property_id: str) -> pd.DataFrame:
    """Build a display-ready initiative progress table for one property."""
    df = _coerce_business_plan_columns(get_business_plan_for_property(property_id).copy())
    if df.empty:
        return pd.DataFrame()

    df["Progress_RAG"] = df.apply(
        lambda row: _assign_progress_rag(
            percent_complete=_safe_float(row["Percent_Complete_Dec"]),
            is_overdue=_safe_bool(row.get("Is_Overdue", False)),
        ),
        axis=1,
    )
    df["Budget_RAG"] = df["Budget_Var_Pct"].apply(_assign_budget_rag)

    df["Return_on_Cost_Calc"] = np.where(
        df["Actual_Spend"] > 0,
        df["Expected_NOI_Lift"] / df["Actual_Spend"],
        0.0,
    )
    df["ROI_RAG"] = df["Return_on_Cost_Calc"].apply(_assign_roi_rag)

    sort_cols: List[str] = []
    ascending: List[bool] = []

    if "Is_Overdue" in df.columns:
        sort_cols.append("Is_Overdue")
        ascending.append(False)
    if "Days_to_Completion" in df.columns:
        sort_cols.append("Days_to_Completion")
        ascending.append(True)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last")

    preferred_cols = [
        "Initiative",
        "Category",
        "Status",
        "Percent_Complete",
        "Budget",
        "Actual_Spend",
        "Budget_Variance",
        "Budget_Var_Pct",
        "Expected_NOI_Lift",
        "Return_on_Cost_Calc",
        "Days_to_Completion",
        "Is_Overdue",
        "Progress_RAG",
        "Budget_RAG",
        "ROI_RAG",
    ]
    existing_cols = [col for col in preferred_cols if col in df.columns]
    return df[existing_cols].reset_index(drop=True)


def get_budget_vs_actual_summary(property_id: str | None = None) -> pd.DataFrame:
    """Build budget versus actual spend summary by property or portfolio-wide."""
    if property_id is None:
        df = load_business_plan_with_properties().copy()
    else:
        df = get_business_plan_for_property(property_id).copy()

    df = _coerce_business_plan_columns(df)
    if df.empty:
        return pd.DataFrame()

    group_cols = ["Property_ID"]
    if "Property_Name" in df.columns and not df["Property_Name"].isna().all():
        group_cols.append("Property_Name")

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            Initiative_Count=("Property_ID", "count"),
            Total_Budget=("Budget", "sum"),
            Total_Actual_Spend=("Actual_Spend", "sum"),
            Expected_NOI_Lift=("Expected_NOI_Lift", "sum"),
        )
        .reset_index()
    )

    summary["Budget_Variance"] = summary["Total_Actual_Spend"] - summary["Total_Budget"]
    summary["Budget_Variance_Pct"] = np.where(
        summary["Total_Budget"] > 0,
        summary["Budget_Variance"] / summary["Total_Budget"],
        0.0,
    )
    summary["Budget_RAG"] = summary["Budget_Variance_Pct"].apply(_assign_budget_rag)
    summary["Portfolio_Return_on_Cost"] = np.where(
        summary["Total_Actual_Spend"] > 0,
        summary["Expected_NOI_Lift"] / summary["Total_Actual_Spend"],
        0.0,
    )

    return summary.sort_values("Budget_Variance_Pct", ascending=False, na_position="last").reset_index(drop=True)


def get_gantt_timeline_data(property_id: str | None = None) -> pd.DataFrame:
    """Build Gantt-style timeline data for initiative tracking."""
    if property_id is None:
        df = load_business_plan_with_properties().copy()
    else:
        df = get_business_plan_for_property(property_id).copy()

    df = _coerce_business_plan_columns(df)
    if df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()

    missing_start = df["Start_Date"].isna()
    inferred_days = np.where(
        df["Percent_Complete_Dec"] > 0,
        np.maximum(df["Percent_Complete_Dec"] * 180.0, 30.0),
        60.0,
    )
    df.loc[missing_start, "Start_Date"] = df.loc[missing_start, "Target_Completion"] - pd.to_timedelta(
        inferred_days[missing_start], unit="D"
    )

    df["Display_End"] = df["Target_Completion"].fillna(today)
    df["Display_Start"] = df["Start_Date"].fillna(today)

    df["Timeline_RAG"] = df.apply(
        lambda row: _assign_progress_rag(
            percent_complete=_safe_float(row["Percent_Complete_Dec"]),
            is_overdue=_safe_bool(row.get("Is_Overdue", False)),
        ),
        axis=1,
    )

    preferred_cols = [
        "Property_ID",
        "Property_Name",
        "Initiative",
        "Category",
        "Status",
        "Display_Start",
        "Display_End",
        "Percent_Complete",
        "Percent_Complete_Dec",
        "Is_Overdue",
        "Timeline_RAG",
        "Risk_Level",
    ]
    existing_cols = [col for col in preferred_cols if col in df.columns]
    return df[existing_cols].sort_values(["Display_Start", "Display_End"], na_position="last").reset_index(drop=True)


def calculate_renovation_roi(property_id: str) -> Dict[str, Any]:
    """Calculate renovation / business plan ROI metrics for a property."""
    prop = load_properties().copy()
    prop = _ensure_columns(prop, {"Property_ID": None, "Units": 0.0})

    prop_row = prop[prop["Property_ID"] == property_id]
    if prop_row.empty:
        raise ValueError(f"Property not found: {property_id}")
    prop_row = prop_row.iloc[0]

    df = _coerce_business_plan_columns(get_business_plan_for_property(property_id).copy())

    units = max(_safe_float(prop_row.get("Units"), default=0.0), 1.0)

    if df.empty:
        return {
            "Property_ID": property_id,
            "Units": units,
            "Total_Spend": 0.0,
            "Spend_Per_Unit": 0.0,
            "Expected_NOI_Lift": 0.0,
            "NOI_Lift_Per_Unit": 0.0,
            "Return_on_Cost": 0.0,
            "Implied_Value_Creation": 0.0,
            "ROI_RAG": RAG_GREEN,
        }

    total_spend = _safe_float(df["Actual_Spend"].sum(), default=0.0)
    expected_noi_lift = _safe_float(df["Expected_NOI_Lift"].sum(), default=0.0)

    spend_per_unit = total_spend / units
    noi_lift_per_unit = expected_noi_lift / units
    roc = return_on_cost(expected_noi_lift, total_spend) if total_spend > 0 else 0.0

    implied_cap_rate = 0.055
    implied_value_creation = expected_noi_lift / implied_cap_rate if implied_cap_rate > 0 else 0.0

    return {
        "Property_ID": property_id,
        "Units": units,
        "Total_Spend": total_spend,
        "Spend_Per_Unit": spend_per_unit,
        "Expected_NOI_Lift": expected_noi_lift,
        "NOI_Lift_Per_Unit": noi_lift_per_unit,
        "Return_on_Cost": roc,
        "Implied_Value_Creation": implied_value_creation,
        "ROI_RAG": _assign_roi_rag(roc),
    }


def get_delayed_initiatives(property_id: str | None = None) -> pd.DataFrame:
    """Return initiatives flagged as delayed or at immediate schedule risk."""
    if property_id is None:
        df = load_business_plan_with_properties().copy()
    else:
        df = get_business_plan_for_property(property_id).copy()

    df = _coerce_business_plan_columns(df)
    if df.empty:
        return pd.DataFrame()

    delayed = df[
        (df["Is_Overdue"].fillna(False))
        | (df["Days_to_Completion"] <= DAYS_TO_COMPLETION_WARNING)
        | ((df["Percent_Complete_Dec"] < PROGRESS_AT_RISK_THRESHOLD) & (df["Days_to_Completion"] < 0))
    ].copy()

    if delayed.empty:
        return delayed

    delayed["Delay_Severity"] = np.where(
        delayed["Days_to_Completion"] < -DAYS_OVERDUE_RED,
        "Critical",
        np.where(delayed["Days_to_Completion"] < 0, "Delayed", "Upcoming"),
    )

    preferred_cols = [
        "Property_ID",
        "Property_Name",
        "Initiative",
        "Category",
        "Status",
        "Percent_Complete",
        "Days_to_Completion",
        "Is_Overdue",
        "Risk_Level",
        "Delay_Severity",
    ]
    existing_cols = [col for col in preferred_cols if col in delayed.columns]
    return delayed[existing_cols].sort_values(
        ["Is_Overdue", "Days_to_Completion"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def get_portfolio_business_plan_kpis() -> Dict[str, Any]:
    """Build portfolio-level KPI summary for all business plan initiatives."""
    df = _coerce_business_plan_columns(load_business_plan().copy())
    if df.empty:
        return {
            "Initiative_Count": 0,
            "Property_Count": 0,
            "Completed_Count": 0,
            "In_Progress_Count": 0,
            "Delayed_Count": 0,
            "Total_Budget": 0.0,
            "Total_Actual_Spend": 0.0,
            "Budget_Variance": 0.0,
            "Budget_Variance_Pct": 0.0,
            "Expected_NOI_Lift": 0.0,
            "Weighted_Completion_Pct": 0.0,
            "Average_Return_on_Cost": 0.0,
            "Portfolio_Return_on_Cost": 0.0,
        }

    total_budget = _safe_float(df["Budget"].sum(), default=0.0)
    total_actual_spend = _safe_float(df["Actual_Spend"].sum(), default=0.0)
    expected_noi_lift = _safe_float(df["Expected_NOI_Lift"].sum(), default=0.0)

    budget_variance = total_actual_spend - total_budget
    budget_variance_pct = budget_variance / total_budget if total_budget > 0 else 0.0

    completed_count = int((df["Percent_Complete_Dec"] >= 1.0).sum())
    in_progress_count = int(((df["Percent_Complete_Dec"] > 0.0) & (df["Percent_Complete_Dec"] < 1.0)).sum())
    delayed_count = int(df["Is_Overdue"].fillna(False).sum())

    weighted_completion_pct = (
        float((df["Percent_Complete_Dec"] * df["Budget"]).sum() / total_budget)
        if total_budget > 0
        else float(df["Percent_Complete_Dec"].mean()) if len(df) > 0 else 0.0
    )

    average_return_on_cost = _mean_return_on_cost(df)
    portfolio_return_on_cost = _portfolio_return_on_cost(df)

    return {
        "Initiative_Count": int(len(df)),
        "Property_Count": int(df["Property_ID"].nunique()) if "Property_ID" in df.columns else 0,
        "Completed_Count": completed_count,
        "In_Progress_Count": in_progress_count,
        "Delayed_Count": delayed_count,
        "Total_Budget": total_budget,
        "Total_Actual_Spend": total_actual_spend,
        "Budget_Variance": budget_variance,
        "Budget_Variance_Pct": budget_variance_pct,
        "Expected_NOI_Lift": expected_noi_lift,
        "Weighted_Completion_Pct": weighted_completion_pct,
        "Average_Return_on_Cost": average_return_on_cost,
        "Portfolio_Return_on_Cost": portfolio_return_on_cost,
    }


def get_property_initiative_ranking() -> pd.DataFrame:
    """Rank properties by expected value creation and execution quality."""
    rows = []
    property_name_map = get_property_name_map()

    for property_id in get_property_list():
        try:
            summary = get_property_business_plan_summary(property_id)
            roi = calculate_renovation_roi(property_id)
        except Exception:
            continue

        execution_score = (
            (summary["Weighted_Completion_Pct"] * 40.0)
            + (max(0.0, 1.0 - max(summary["Budget_Variance_Pct"], 0.0)) * 30.0)
            + (min(summary["Average_Return_on_Cost"], 0.30) / 0.30 * 30.0)
        )

        rows.append(
            {
                "Property_ID": property_id,
                "Property_Name": property_name_map.get(property_id, property_id),
                "Initiative_Count": summary["Initiative_Count"],
                "Delayed_Count": summary["Delayed_Count"],
                "Weighted_Completion_Pct": summary["Weighted_Completion_Pct"],
                "Budget_Variance_Pct": summary["Budget_Variance_Pct"],
                "Expected_NOI_Lift": summary["Expected_NOI_Lift"],
                "Average_Return_on_Cost": summary["Average_Return_on_Cost"],
                "Implied_Value_Creation": roi["Implied_Value_Creation"],
                "Execution_Score": execution_score,
                "Overall_RAG": summary["Overall_RAG"],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Property_ID",
                "Property_Name",
                "Initiative_Count",
                "Delayed_Count",
                "Weighted_Completion_Pct",
                "Budget_Variance_Pct",
                "Expected_NOI_Lift",
                "Average_Return_on_Cost",
                "Implied_Value_Creation",
                "Execution_Score",
                "Overall_RAG",
            ]
        )

    return pd.DataFrame(rows).sort_values("Execution_Score", ascending=False).reset_index(drop=True)


# ============================================================================
# SELF-TESTS
# ============================================================================

def _run_self_tests() -> None:
    """Run smoke tests for business_plan_tracker module."""
    print("Running business_plan_tracker.py self-tests...\n")

    property_ids = get_property_list()
    assert len(property_ids) > 0, "Expected at least one property."

    for property_id in property_ids:
        summary = get_property_business_plan_summary(property_id)
        progress_table = get_initiative_progress_table(property_id)
        budget_summary = get_budget_vs_actual_summary(property_id)
        gantt_data = get_gantt_timeline_data(property_id)
        roi = calculate_renovation_roi(property_id)
        delayed = get_delayed_initiatives(property_id)

        assert isinstance(summary, dict), f"Summary must be dict for {property_id}"
        assert summary["Initiative_Count"] >= 0, f"Invalid initiative count for {property_id}"

        assert isinstance(progress_table, pd.DataFrame), f"Progress table must be DataFrame for {property_id}"
        assert isinstance(budget_summary, pd.DataFrame), f"Budget summary must be DataFrame for {property_id}"
        assert isinstance(gantt_data, pd.DataFrame), f"Gantt data must be DataFrame for {property_id}"
        assert isinstance(delayed, pd.DataFrame), f"Delayed initiatives must be DataFrame for {property_id}"

        assert isinstance(roi, dict), f"ROI result must be dict for {property_id}"
        assert roi["Return_on_Cost"] >= 0.0, f"ROI should be non-negative for {property_id}"

        print(
            f"✓ {property_id:<12} | "
            f"Initiatives: {summary['Initiative_Count']:>2} | "
            f"Delayed: {summary['Delayed_Count']:>2} | "
            f"Completion: {summary['Weighted_Completion_Pct']:.1%} | "
            f"Budget Var: {summary['Budget_Variance_Pct']:+.1%} | "
            f"RAG: {summary['Overall_RAG']}"
        )

    portfolio_kpis = get_portfolio_business_plan_kpis()
    assert portfolio_kpis["Initiative_Count"] >= 0, "Portfolio initiative count must be non-negative."
    assert portfolio_kpis["Property_Count"] >= 0, "Portfolio property count must be non-negative."

    ranking = get_property_initiative_ranking()
    assert isinstance(ranking, pd.DataFrame), "Ranking table must be a DataFrame."

    delayed_all = get_delayed_initiatives()
    assert isinstance(delayed_all, pd.DataFrame), "Portfolio delayed initiatives output must be DataFrame."

    print("\nPortfolio business plan KPIs:")
    print(pd.DataFrame([portfolio_kpis]).to_string(index=False))

    print("\nProperty execution ranking:")
    if not ranking.empty:
        cols = [
            "Property_ID",
            "Initiative_Count",
            "Delayed_Count",
            "Weighted_Completion_Pct",
            "Budget_Variance_Pct",
            "Execution_Score",
            "Overall_RAG",
        ]
        cols = [c for c in cols if c in ranking.columns]
        print(ranking[cols].to_string(index=False))
    else:
        print("(no ranking rows)")

    print("\nALL BUSINESS PLAN TRACKER TESTS PASSED")


if __name__ == "__main__":
    _run_self_tests()