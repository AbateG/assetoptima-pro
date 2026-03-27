from __future__ import annotations

import logging
import warnings
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from modules.data_loader import (
    get_performance_for_property,
    get_underwriting_for_property,
    load_monthly_performance,
    load_properties,
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# =============================================================================
# CONSTANTS
# =============================================================================

EXPENSE_LINE_ITEMS: list[str] = [
    "Repairs_Maintenance",
    "Payroll",
    "Marketing",
    "Utilities",
    "Insurance",
    "Admin",
]

EXPENSE_LABELS: Dict[str, str] = {
    "Repairs_Maintenance": "Repairs & Maintenance",
    "Payroll": "Payroll",
    "Marketing": "Marketing",
    "Utilities": "Utilities",
    "Insurance": "Insurance",
    "Admin": "Admin & General",
}


# =============================================================================
# NUMERIC / PERIOD UTILITIES
# =============================================================================

def safe_numeric(value: Any, target_type: type = float, default: float | int = 0):
    """Safely convert a value to int or float."""
    try:
        if value is None or pd.isna(value):
            return default
        numeric = float(value)
        return int(numeric) if target_type is int else float(numeric)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely cast value to int."""
    return int(safe_numeric(value, target_type=int, default=default))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast value to float."""
    return float(safe_numeric(value, target_type=float, default=default))


def normalize_ratio(value: Any, default: float = 0.0) -> float:
    """
    Normalize a ratio/rate/percent-like value to decimal form.

    Examples:
        0.92 -> 0.92
        92   -> 0.92
        5    -> 0.05
    """
    val = safe_float(value, default)
    if abs(val) > 1.0:
        return val / 100.0
    return val


def safe_period_str(value: Any) -> str:
    """Safely convert a date-like value to YYYY-MM."""
    dt = pd.to_datetime(value, errors="coerce")
    return dt.strftime("%Y-%m") if not pd.isna(dt) else "N/A"


def sort_by_period(df: pd.DataFrame, period_col: str = "Year_Month") -> pd.DataFrame:
    """Chronologically sort a DataFrame by period with robust datetime parsing."""
    if df.empty or period_col not in df.columns:
        return df

    out = df.copy()
    out["_period_dt"] = pd.to_datetime(out[period_col], errors="coerce")

    if out["_period_dt"].notna().any():
        out = out.sort_values("_period_dt")
    else:
        out = out.sort_values(period_col)

    return out.drop(columns=["_period_dt"], errors="ignore")


# =============================================================================
# RAG UTILITIES
# =============================================================================

def assign_rag_status(variance_ratio: float, line_type: str) -> str:
    """
    Assign RAG status to a variance ratio in decimal form.

    Examples:
        -0.02 = -2.0%
         0.05 = +5.0%
    """
    if line_type == "revenue":
        if variance_ratio >= -0.03:
            return "Green"
        if variance_ratio >= -0.08:
            return "Amber"
        return "Red"

    if variance_ratio <= 0.03:
        return "Green"
    if variance_ratio <= 0.08:
        return "Amber"
    return "Red"


def rag_to_emoji(status: str) -> str:
    """Convert RAG status to emoji."""
    return {"Green": "🟢", "Amber": "🟡", "Red": "🔴"}.get(status, "⚪")


# =============================================================================
# SECTION 1 — PORTFOLIO ROLLUP
# =============================================================================

def get_portfolio_noi_variance_summary() -> pd.DataFrame:
    """
    Return T12 NOI variance summary by property.

    Output percentage fields are decimals.
    """
    try:
        perf = load_monthly_performance()
        props = load_properties()

        if perf.empty or props.empty:
            return pd.DataFrame()

        perf = sort_by_period(perf, "Year_Month")
        t12 = perf.groupby("Property_ID", group_keys=False).tail(12)

        summary = (
            t12.groupby("Property_ID", dropna=False)
            .agg(
                T12_NOI_Actual=("Actual_NOI", "sum"),
                T12_NOI_Budget=("Budgeted_NOI", "sum"),
            )
            .reset_index()
        )

        summary["NOI_Variance_Dollar"] = summary["T12_NOI_Actual"] - summary["T12_NOI_Budget"]
        summary["NOI_Variance_Pct"] = np.where(
            summary["T12_NOI_Budget"] > 0,
            summary["NOI_Variance_Dollar"] / summary["T12_NOI_Budget"],
            0.0,
        )

        summary["NOI_RAG"] = summary["NOI_Variance_Pct"].apply(
            lambda pct: assign_rag_status(safe_float(pct), "revenue")
        )
        summary["NOI_Trend"] = summary["Property_ID"].apply(
            lambda pid: get_noi_trend_direction(pid, 6)
        )
        summary["Consecutive_Miss_Months"] = summary["Property_ID"].apply(
            lambda pid: get_consecutive_variance_months(pid, "noi", "unfavorable", 0.05)
        )

        if "Property_ID" in props.columns and "Property_Name" in props.columns:
            name_map = dict(zip(props["Property_ID"], props["Property_Name"]))
            summary["Property_Name"] = summary["Property_ID"].map(name_map)
        else:
            summary["Property_Name"] = summary["Property_ID"]

        return summary.sort_values("NOI_Variance_Pct", ascending=True).reset_index(drop=True)

    except Exception as e:
        logger.exception("Error in get_portfolio_noi_variance_summary: %s", e)
        return pd.DataFrame()


# =============================================================================
# SECTION 2 — PROPERTY T12 SUMMARY
# =============================================================================

def get_t12_summary(property_id: str) -> Dict[str, Any]:
    """
    Return trailing 12-month operating summary for one property.

    Percentage fields are decimals.
    """
    try:
        perf = get_performance_for_property(property_id)
        if perf.empty:
            return _empty_t12(property_id)

        perf = sort_by_period(perf, "Year_Month")
        t12 = perf.tail(12)
        if t12.empty:
            return _empty_t12(property_id)

        actual_rev = safe_float(t12.get("Actual_Revenue", pd.Series(dtype=float)).sum())
        budget_rev = safe_float(t12.get("Budgeted_Revenue", pd.Series(dtype=float)).sum())
        actual_exp = safe_float(t12.get("Actual_Expenses", pd.Series(dtype=float)).sum())
        budget_exp = safe_float(t12.get("Budgeted_Expenses", pd.Series(dtype=float)).sum())
        actual_noi = safe_float(t12.get("Actual_NOI", pd.Series(dtype=float)).sum())
        budget_noi = safe_float(t12.get("Budgeted_NOI", pd.Series(dtype=float)).sum())

        rev_var_pct = ((actual_rev - budget_rev) / budget_rev) if budget_rev > 0 else 0.0
        exp_var_pct = ((actual_exp - budget_exp) / budget_exp) if budget_exp > 0 else 0.0
        noi_var_pct = ((actual_noi - budget_noi) / budget_noi) if budget_noi > 0 else 0.0

        occ_series = pd.to_numeric(t12.get("Occupancy", pd.Series(dtype=float)), errors="coerce").dropna()
        avg_occ = normalize_ratio(occ_series.mean()) if not occ_series.empty else 0.0

        return {
            "property_id": property_id,
            "months_included": len(t12),
            "period_start": safe_period_str(t12["Year_Month"].iloc[0]) if "Year_Month" in t12.columns else "N/A",
            "period_end": safe_period_str(t12["Year_Month"].iloc[-1]) if "Year_Month" in t12.columns else "N/A",
            "actual_revenue": actual_rev,
            "budget_revenue": budget_rev,
            "revenue_variance": actual_rev - budget_rev,
            "revenue_var_pct": rev_var_pct,
            "revenue_rag": assign_rag_status(rev_var_pct, "revenue"),
            "actual_expenses": actual_exp,
            "budget_expenses": budget_exp,
            "expense_variance": actual_exp - budget_exp,
            "expense_var_pct": exp_var_pct,
            "expense_rag": assign_rag_status(exp_var_pct, "expense"),
            "actual_noi": actual_noi,
            "budget_noi": budget_noi,
            "noi_variance": actual_noi - budget_noi,
            "noi_variance_pct": noi_var_pct,
            "noi_var_pct": noi_var_pct,
            "noi_rag": assign_rag_status(noi_var_pct, "revenue"),
            "avg_occupancy_pct": avg_occ,
            "expense_ratio": (actual_exp / actual_rev) if actual_rev > 0 else 0.0,
            "noi_margin": (actual_noi / actual_rev) if actual_rev > 0 else 0.0,
        }

    except Exception as e:
        logger.exception("Error in get_t12_summary for property_id=%s: %s", property_id, e)
        return _empty_t12(property_id)


def _empty_t12(property_id: str) -> Dict[str, Any]:
    """Return empty T12 summary payload."""
    return {
        "property_id": property_id,
        "months_included": 0,
        "period_start": "N/A",
        "period_end": "N/A",
        "actual_revenue": 0.0,
        "budget_revenue": 0.0,
        "revenue_variance": 0.0,
        "revenue_var_pct": 0.0,
        "revenue_rag": "Green",
        "actual_expenses": 0.0,
        "budget_expenses": 0.0,
        "expense_variance": 0.0,
        "expense_var_pct": 0.0,
        "expense_rag": "Green",
        "actual_noi": 0.0,
        "budget_noi": 0.0,
        "noi_variance": 0.0,
        "noi_variance_pct": 0.0,
        "noi_var_pct": 0.0,
        "noi_rag": "Green",
        "avg_occupancy_pct": 0.0,
        "expense_ratio": 0.0,
        "noi_margin": 0.0,
    }


# =============================================================================
# SECTION 3 — STREAKS AND TRENDS
# =============================================================================

def get_consecutive_variance_months(
    property_id: str,
    line_item_type: str,
    direction: str,
    threshold_pct: float,
) -> int:
    """
    Count consecutive months meeting variance condition from most recent backward.

    Expects threshold_pct as decimal:
        0.05 = 5%
    """
    try:
        perf = get_performance_for_property(property_id)
        if perf.empty:
            return 0

        perf = sort_by_period(perf, "Year_Month")
        pct_col = {
            "revenue": "Revenue_Var_Pct",
            "expense": "Expense_Var_Pct",
            "noi": "NOI_Var_Pct",
        }.get(line_item_type)

        if pct_col not in perf.columns:
            return 0

        variances = pd.to_numeric(perf[pct_col], errors="coerce").tolist()
        consecutive = 0

        for pct in reversed(variances):
            if pd.isna(pct):
                break

            # Backward compatibility if upstream source still stores whole-percent values
            pct = normalize_ratio(pct)

            if direction == "unfavorable":
                condition_met = pct > threshold_pct if line_item_type == "expense" else pct < -threshold_pct
            else:
                condition_met = pct < -threshold_pct if line_item_type == "expense" else pct > threshold_pct

            if condition_met:
                consecutive += 1
            else:
                break

        return consecutive

    except Exception as e:
        logger.exception("Error in get_consecutive_variance_months for property_id=%s: %s", property_id, e)
        return 0


def get_noi_trend_direction(property_id: str, window_months: int = 6) -> str:
    """Determine NOI trend direction."""
    try:
        perf = get_performance_for_property(property_id)
        if perf.empty:
            return "flat"

        perf = sort_by_period(perf, "Year_Month")
        noi_series = pd.to_numeric(
            perf.tail(max(window_months, 3)).get("Actual_NOI", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()

        if len(noi_series) < 3:
            return "flat"

        x = np.arange(len(noi_series), dtype=float)
        y = noi_series.values.astype(float)
        slope, _ = np.polyfit(x, y, 1)

        mean_noi = np.mean(np.abs(y))
        if mean_noi == 0:
            return "flat"

        normalized_slope = slope / mean_noi
        if normalized_slope > 0.005:
            return "improving"
        if normalized_slope < -0.005:
            return "declining"
        return "flat"

    except Exception as e:
        logger.exception("Error in get_noi_trend_direction for property_id=%s: %s", property_id, e)
        return "flat"


# =============================================================================
# SECTION 4 — BREAKDOWNS & CHARTS
# =============================================================================

def build_variance_summary_table(
    property_id: str,
    trailing_months: int = 3,
) -> pd.DataFrame:
    """
    Build high-level variance summary table.

    Pct_Variance is returned as decimal.
    """
    try:
        perf = get_performance_for_property(property_id)
        if perf.empty:
            return pd.DataFrame()

        perf = sort_by_period(perf, "Year_Month").tail(trailing_months)

        actual_rev = safe_float(perf.get("Actual_Revenue", pd.Series(dtype=float)).sum())
        budget_rev = safe_float(perf.get("Budgeted_Revenue", pd.Series(dtype=float)).sum())
        var_pct_rev = ((actual_rev - budget_rev) / budget_rev) if budget_rev > 0 else 0.0
        rag_rev = assign_rag_status(var_pct_rev, "revenue")

        actual_exp = safe_float(perf.get("Actual_Expenses", pd.Series(dtype=float)).sum())
        budget_exp = safe_float(perf.get("Budgeted_Expenses", pd.Series(dtype=float)).sum())
        var_pct_exp = ((actual_exp - budget_exp) / budget_exp) if budget_exp > 0 else 0.0
        rag_exp = assign_rag_status(var_pct_exp, "expense")

        actual_noi = safe_float(perf.get("Actual_NOI", pd.Series(dtype=float)).sum())
        budget_noi = safe_float(perf.get("Budgeted_NOI", pd.Series(dtype=float)).sum())
        var_pct_noi = ((actual_noi - budget_noi) / budget_noi) if budget_noi > 0 else 0.0
        rag_noi = assign_rag_status(var_pct_noi, "revenue")

        rows = [
            {
                "Line_Item": "Total Revenue",
                "Category": "Revenue",
                "Budgeted": budget_rev,
                "Actual": actual_rev,
                "Dollar_Variance": actual_rev - budget_rev,
                "Pct_Variance": var_pct_rev,
                "RAG_Status": rag_rev,
                "RAG_Emoji": rag_to_emoji(rag_rev),
            },
            {
                "Line_Item": "Total Operating Expenses",
                "Category": "Expense",
                "Budgeted": budget_exp,
                "Actual": actual_exp,
                "Dollar_Variance": actual_exp - budget_exp,
                "Pct_Variance": var_pct_exp,
                "RAG_Status": rag_exp,
                "RAG_Emoji": rag_to_emoji(rag_exp),
            },
            {
                "Line_Item": "Net Operating Income (NOI)",
                "Category": "NOI",
                "Budgeted": budget_noi,
                "Actual": actual_noi,
                "Dollar_Variance": actual_noi - budget_noi,
                "Pct_Variance": var_pct_noi,
                "RAG_Status": rag_noi,
                "RAG_Emoji": rag_to_emoji(rag_noi),
            },
        ]

        return pd.DataFrame(rows).reset_index(drop=True)

    except Exception as e:
        logger.exception("Error in build_variance_summary_table for property_id=%s: %s", property_id, e)
        return pd.DataFrame()


def build_line_item_breakdown(
    property_id: str,
    trailing_months: int = 3,
) -> pd.DataFrame:
    """
    Build expense line-item breakdown table.

    Pct_Variance and Share_of_Total_Pct are returned as decimals.
    """
    try:
        perf = get_performance_for_property(property_id)
        if perf.empty:
            return pd.DataFrame()

        window = sort_by_period(perf, "Year_Month").tail(trailing_months)

        total_actual_exp = safe_float(window.get("Actual_Expenses", pd.Series(dtype=float)).sum())
        total_budget_exp = safe_float(window.get("Budgeted_Expenses", pd.Series(dtype=float)).sum())

        window_totals = pd.to_numeric(window.get("Actual_Expenses", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        window_budgets = pd.to_numeric(window.get("Budgeted_Expenses", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        valid_mask = (window_totals > 0) & (window_budgets > 0)

        rows: list[dict[str, Any]] = []

        for col in EXPENSE_LINE_ITEMS:
            if col not in window.columns:
                continue

            col_actuals = pd.to_numeric(window[col], errors="coerce").fillna(0.0)
            line_actual = safe_float(col_actuals.sum())

            line_budget = (
                total_budget_exp * (line_actual / total_actual_exp)
                if total_actual_exp > 0 else 0.0
            )

            denominator = window_totals.replace(0, np.nan)
            row_budget_line = window_budgets * (col_actuals / denominator)
            months_over = int(((col_actuals > row_budget_line.fillna(0.0)) & valid_mask).sum())

            var_pct = ((line_actual - line_budget) / line_budget) if line_budget > 0 else 0.0
            rag = assign_rag_status(var_pct, "expense")

            rows.append(
                {
                    "Line_Item": EXPENSE_LABELS.get(col, col),
                    "Column_Name": col,
                    "Actual": line_actual,
                    "Budgeted": line_budget,
                    "Dollar_Variance": line_actual - line_budget,
                    "Pct_Variance": var_pct,
                    "Share_of_Total_Pct": (line_actual / total_actual_exp) if total_actual_exp > 0 else 0.0,
                    "RAG_Status": rag,
                    "RAG_Emoji": rag_to_emoji(rag),
                    "Months_Over_Budget": months_over,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        return df.sort_values("Actual", ascending=False).reset_index(drop=True)

    except Exception as e:
        logger.exception("Error in build_line_item_breakdown for property_id=%s: %s", property_id, e)
        return pd.DataFrame()


# =============================================================================
# SECTION 5 — ADDITIONAL HELPERS
# =============================================================================

def get_monthly_noi_trend(
    property_id: str,
    trailing_months: Optional[int] = None,
) -> pd.DataFrame:
    """Return monthly NOI trend table."""
    try:
        perf = get_performance_for_property(property_id)
        if perf.empty:
            return pd.DataFrame()

        perf = sort_by_period(perf, "Year_Month")
        if trailing_months:
            perf = perf.tail(trailing_months)

        cols = [
            "Year_Month",
            "Actual_NOI",
            "Budgeted_NOI",
            "Actual_Revenue",
            "Actual_Expenses",
        ]
        existing_cols = [col for col in cols if col in perf.columns]
        return perf[existing_cols].reset_index(drop=True)

    except Exception as e:
        logger.exception("Error in get_monthly_noi_trend for property_id=%s: %s", property_id, e)
        return pd.DataFrame()


def get_expense_trend(
    property_id: str,
    trailing_months: Optional[int] = None,
) -> pd.DataFrame:
    """Return expense trend table."""
    try:
        perf = get_performance_for_property(property_id)
        if perf.empty:
            return pd.DataFrame()

        perf = sort_by_period(perf, "Year_Month")
        if trailing_months:
            perf = perf.tail(trailing_months)

        cols = ["Year_Month", "Actual_Expenses", "Budgeted_Expenses"] + [
            c for c in EXPENSE_LINE_ITEMS if c in perf.columns
        ]
        existing_cols = [col for col in cols if col in perf.columns]
        return perf[existing_cols].reset_index(drop=True)

    except Exception as e:
        logger.exception("Error in get_expense_trend for property_id=%s: %s", property_id, e)
        return pd.DataFrame()


def compare_actual_vs_underwriting(property_id: str) -> Dict[str, Any]:
    """
    Compare actual trailing performance against underwriting assumptions.

    All percentage-like outputs are decimals.
    """
    try:
        perf = get_performance_for_property(property_id)
        uw = get_underwriting_for_property(property_id)

        if perf.empty or uw.empty:
            return get_empty_uw(property_id)

        perf = sort_by_period(perf, "Year_Month")
        t12 = perf.tail(12)
        if t12.empty:
            return get_empty_uw(property_id)

        uw_row = uw.iloc[0]

        t12_noi = safe_float(t12.get("Actual_NOI", pd.Series(dtype=float)).sum())
        uw_noi = safe_float(uw_row.get("Underwritten_NOI_Year1", 0.0))
        uw_occ = normalize_ratio(uw_row.get("Underwritten_Occupancy", 0.0))
        uw_rent_g = normalize_ratio(uw_row.get("Underwritten_Rent_Growth_Pct", 0.0))
        uw_cap_rate = normalize_ratio(uw_row.get("Underwritten_Purchase_Cap_Rate", 0.0))
        target_irr = normalize_ratio(uw_row.get("Underwritten_IRR", 0.0))

        latest_row = perf.iloc[-1] if not perf.empty else pd.Series(dtype=object)
        actual_occ = normalize_ratio(latest_row.get("Occupancy", 0.0))

        actual_rg = 0.0
        if len(t12) >= 2 and "Actual_Revenue" in t12.columns:
            first_rev = safe_float(t12["Actual_Revenue"].iloc[0])
            last_rev = safe_float(t12["Actual_Revenue"].iloc[-1])
            if first_rev > 0:
                actual_rg = (last_rev / first_rev) - 1.0

        props = load_properties()
        prop_row = props[props["Property_ID"] == property_id] if "Property_ID" in props.columns else pd.DataFrame()
        acq_value = safe_float(prop_row["Purchase_Price"].iloc[0]) if (
            not prop_row.empty and "Purchase_Price" in prop_row.columns
        ) else 0.0

        implied_cap = (t12_noi / acq_value) if acq_value > 0 else 0.0
        noi_variance_pct = ((t12_noi - uw_noi) / uw_noi) if uw_noi > 0 else 0.0
        occupancy_variance_ppt = actual_occ - uw_occ

        return {
            "property_id": property_id,
            "underwritten_year1_noi": uw_noi,
            "actual_t12_noi": t12_noi,
            "noi_variance_dollar": t12_noi - uw_noi,
            "noi_variance_pct": noi_variance_pct,
            "noi_gap_pct": noi_variance_pct,
            "underwritten_occupancy_pct": uw_occ,
            "actual_occupancy_pct": actual_occ,
            "occupancy_variance_ppt": occupancy_variance_ppt,
            "occupancy_rag": (
                "Green" if occupancy_variance_ppt >= 0
                else "Amber" if occupancy_variance_ppt >= -0.05
                else "Red"
            ),
            "underwritten_rent_growth_pct": uw_rent_g,
            "actual_rent_growth_pct": actual_rg,
            "rent_growth_variance_ppt": actual_rg - uw_rent_g,
            "purchase_cap_rate_pct": uw_cap_rate,
            "implied_current_cap_rate_pct": implied_cap,
            "cap_rate_compression_ppt": uw_cap_rate - implied_cap,
            "target_irr_pct": target_irr,
            "hold_years": safe_int(uw_row.get("Target_Hold_Years", 5)),
            "outperforming_thesis": bool(t12_noi > (uw_noi * 1.05)) if uw_noi > 0 else False,
        }

    except Exception as e:
        logger.exception("Error in compare_actual_vs_underwriting for property_id=%s: %s", property_id, e)
        return get_empty_uw(property_id)


def get_empty_uw(property_id: str) -> Dict[str, Any]:
    """Return empty underwriting comparison payload."""
    return {
        "property_id": property_id,
        "underwritten_year1_noi": 0.0,
        "actual_t12_noi": 0.0,
        "noi_variance_dollar": 0.0,
        "noi_variance_pct": 0.0,
        "noi_gap_pct": 0.0,
        "noi_rag": "Green",
        "underwritten_occupancy_pct": 0.0,
        "actual_occupancy_pct": 0.0,
        "occupancy_variance_ppt": 0.0,
        "occupancy_rag": "Green",
        "underwritten_rent_growth_pct": 0.0,
        "actual_rent_growth_pct": 0.0,
        "rent_growth_variance_ppt": 0.0,
        "purchase_cap_rate_pct": 0.0,
        "implied_current_cap_rate_pct": 0.0,
        "cap_rate_compression_ppt": 0.0,
        "target_irr_pct": 0.0,
        "hold_years": 5,
        "outperforming_thesis": False,
    }


def build_portfolio_variance_heatmap_data() -> pd.DataFrame:
    """
    Build property x month NOI variance heatmap data.

    NOI_Var_Pct is assumed to be decimal; if upstream stores whole-percent,
    values are normalized defensively.
    """
    try:
        perf = load_monthly_performance()
        props = load_properties()

        if perf.empty or props.empty:
            return pd.DataFrame()

        perf = perf.copy()

        if "NOI_Var_Pct" in perf.columns:
            perf["NOI_Var_Pct"] = pd.to_numeric(perf["NOI_Var_Pct"], errors="coerce").map(normalize_ratio)

        if "Property_ID" in props.columns and "Property_Name" in props.columns:
            name_map = dict(zip(props["Property_ID"], props["Property_Name"]))
            perf["Property_Name"] = perf["Property_ID"].map(name_map)
        else:
            perf["Property_Name"] = perf.get("Property_ID", "")

        pivot = perf.pivot_table(
            index="Property_Name",
            columns="Year_Month",
            values="NOI_Var_Pct",
            aggfunc="mean",
        )

        if pivot.empty:
            return pivot

        sorted_cols = sorted(pivot.columns, key=lambda x: pd.to_datetime(x, errors="coerce"))
        return pivot.reindex(sorted_cols, axis=1)

    except Exception as e:
        logger.exception("Error in build_portfolio_variance_heatmap_data: %s", e)
        return pd.DataFrame()


def calculate_variance_statistics(property_id: str) -> Dict[str, Any]:
    """Placeholder for future variance statistics."""
    return {}


def identify_variance_patterns(property_id: str) -> Dict[str, Any]:
    """Placeholder for future variance pattern detection."""
    return {}